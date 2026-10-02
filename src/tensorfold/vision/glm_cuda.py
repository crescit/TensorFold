"""GLM-5.3-Flash image features for the two-rank CUDA engine; its NoPE language layers take no image positions."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .qwen_cuda import MAX_PATCHES, WORKSPACE_BYTES, EncodedVision, float_headers


def vision_config(model_dir: str | Path) -> dict:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    config = raw.get("vision_config")
    if raw.get("model_type") != "glm5_next" or not isinstance(config, dict):
        raise ValueError("CUDA GLM vision requires a GLM-5.3-Flash checkpoint with its vision_config")
    for key in ("hidden_size", "out_hidden_size", "depth", "patch_size", "temporal_patch_size", "spatial_merge_size",
                "in_channels", "intermediate_size", "num_heads", "projection_intermediate_size"):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"invalid vision configuration: {key}")
    if config["hidden_size"] % config["num_heads"] or config["hidden_size"] // config["num_heads"] % 4:
        raise ValueError("vision attention requires head widths divisible by four")
    if config["out_hidden_size"] != (raw.get("text_config") or raw).get("hidden_size"):
        raise ValueError("vision output width differs from the language embedding width")
    if not isinstance(raw.get("image_token_id"), int):
        raise ValueError("this GLM checkpoint is missing its image token")
    return config


def tower_shapes(config: dict) -> dict[str, list[int]]:
    """Every tensor of the GLM5-Next tower in the checkpoint's (PyTorch) layout."""

    h, mid, out, inner = (config[k] for k in ("hidden_size", "intermediate_size", "out_hidden_size",
                                              "projection_intermediate_size"))
    merge, head = config["spatial_merge_size"], h // config["num_heads"]
    bias = config.get("attention_bias", True)
    shapes = {"patch_embed.proj.weight": [h, config["in_channels"], config["temporal_patch_size"],
                                          config["patch_size"], config["patch_size"]],
              "patch_embed.proj.bias": [h], "post_layernorm.weight": [h],
              "downsample.weight": [out, h, merge, merge], "downsample.bias": [out],
              "merger.proj.weight": [out, out], "merger.post_projection_norm.weight": [out],
              "merger.post_projection_norm.bias": [out], "merger.gate_proj.weight": [inner, out],
              "merger.up_proj.weight": [inner, out], "merger.down_proj.weight": [out, inner]}
    for layer in range(config["depth"]):
        for part, width, inputs in (("norm1", h, None), ("norm2", h, None), ("attn.q_norm", head, None),
                                    ("attn.k_norm", head, None), ("attn.qkv", 3 * h, h), ("attn.proj", h, h),
                                    ("mlp.gate_proj", mid, h), ("mlp.up_proj", mid, h), ("mlp.down_proj", h, mid)):
            shapes[f"blocks.{layer}.{part}.weight"] = [width] if inputs is None else [width, inputs]
            if inputs is not None and bias:
                shapes[f"blocks.{layer}.{part}.bias"] = [width]
    return shapes


def _channels_last(name: str, shape: list[int]) -> list[int] | None:
    """An MLX conversion's convolution layout (channels last) of a PyTorch-layout shape, else None."""

    if name in ("patch_embed.proj.weight", "downsample.weight"):
        return [shape[0], *shape[2:], shape[1]]
    return None


def checkpoint_vision(model_dir: str | Path) -> tuple[dict, int]:
    """Validate the tower's headers before any model or accelerator allocation; its bytes once loaded as BF16."""

    config = vision_config(model_dir)
    from .glm_exl3 import layout
    shapes = tower_shapes(config)
    sources, entries = layout(model_dir, shapes)
    return config, sum(math.prod(v)*2 for v in shapes.values())


def weight_transform(base, enabled: bool, rank: int, model_dir=None):
    """The startup estimate counts the tower on rank zero, the only rank that loads it."""

    from .qwen_checkpoint import vision_key

    charges = None
    if enabled and rank == 0 and model_dir is not None:
        from .glm_exl3 import layout
        _, entries = layout(model_dir, tower_shapes(vision_config(model_dir)))
        charges = {}
        for _, (kind, parts, shape) in entries.items():
            for part in parts:
                charges[part + '.trellis' if kind == 'quant' else part] = math.prod(shape)*2//len(parts)

    def transform(name, info):
        key = vision_key(name)
        if enabled and rank == 0 and key is not None:
            return (charges.get(key, 0), 0) if charges is not None else (math.prod(info['shape'])*2, 0)
        return base(name, info)

    return transform


def capacity_geometry(base, model_dir, enabled: bool, rank: int):
    def geometry(text):
        from tensorfold.cuda.capacity import Geometry

        result = base(text)
        if not enabled:
            return result
        if rank == 0:
            checkpoint_vision(model_dir)
        reserve = WORKSPACE_BYTES if rank == 0 else 128 * 1024**2     # rank one holds the received features
        return Geometry(lambda slots: result.bytes_at(slots) + reserve, result.reserve, result.minimum_slots)
    return geometry


def placeholder_rows(prompt, image_token: int) -> tuple[int, ...]:
    return tuple(i for i, token in enumerate(prompt) if token == image_token)


class GLMCudaVision:
    """Only the image tower is loaded, on rank zero; the CUDA engine keeps all language computation."""

    def __init__(self, model_dir, device, allow_urls: bool = False):
        import torch
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

        from .glm_processing import GLMImageProcessor
        from .qwen_checkpoint import vision_tensors

        self.allow_urls = allow_urls
        self.config, self.weight_bytes = checkpoint_vision(model_dir)
        self.frontend = GLMImageProcessor.from_directory(model_dir)
        self.image_token = self.frontend.image_token_id
        self.device = device
        config = Glm5NextVisionConfig(**self.config)
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            tower = Glm5NextVisionModel(config)
        shapes = tower_shapes(self.config)
        from .glm_exl3 import load
        tensors = load(model_dir, shapes, device)
        tower.load_state_dict(tensors, strict=True, assign=True)
        rotary = tower.rotary_pos_emb                    # a meta build leaves the frequency buffers empty
        inv_freq, _ = rotary.compute_axial_rope_parameters(config)
        for buffer in ("inv_freq", "original_inv_freq"):
            rotary._buffers[buffer] = inv_freq.to(device).clone()
        self.tower = tower.eval()

    def prepare(self, *args, **kwargs):
        return self.frontend.prepare(*args, **kwargs)

    def continued(self, prepared, tokens):
        return self.frontend.continued(prepared, tokens)

    def encode(self, prepared, prompt) -> EncodedVision:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if tuple(prompt) != tuple(prepared.token_ids):
            raise ValueError("vision preparation belongs to different prompt tokens")
        grid = prepared.image_grid_thw
        merge = self.config["spatial_merge_size"]
        if len(grid.shape) != 2 or grid.shape[1] != 3 or any(int(t) != 1 for t in grid[:, 0]):
            raise ValueError("CUDA vision accepts images with one temporal grid, not video")
        if any(int(v) <= 0 for row in grid for v in row) or any(int(h) % merge or int(w) % merge for _, h, w in grid):
            raise ValueError("image grids must contain positive merge-aligned dimensions")
        patches = sum(int(t) * int(h) * int(w) for t, h, w in grid)
        if patches > MAX_PATCHES:
            raise ValueError(f"image request exceeds the CUDA vision budget of {MAX_PATCHES} patches")
        width = self.config["in_channels"] * self.config["temporal_patch_size"] * self.config["patch_size"] ** 2
        if tuple(prepared.pixel_values.shape) != (patches, width):
            raise ValueError("image patch tensor has an invalid shape")
        rows = tuple(i for start, end in prepared.image_spans for i in range(start, end))
        if rows != placeholder_rows(prompt, self.image_token) or len(rows) != patches // merge**2:
            raise ValueError("vision feature rows must match every image placeholder exactly")
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            # a copy: the prepared arrays are read-only, and a tensor may not share them
            pixels = torch.tensor(prepared.pixel_values, dtype=torch.bfloat16, device=self.device)
            grids = torch.tensor(grid, dtype=torch.int64, device=self.device)
            features = self.tower(pixels, grid_thw=grids, return_dict=True).pooler_output
            features = features.to(dtype=torch.bfloat16).contiguous()
        if tuple(features.shape) != (len(rows), self.config["out_hidden_size"]):
            raise ValueError("vision tower returned a different number of image features")
        return EncodedVision(rows, features, None, 0)


def share_encoded(payload: EncodedVision | None, rank: int, comm, prompt, image_token: int, hidden: int,
                  device) -> EncodedVision:
    """Rank zero's features on both ranks; each finds the image rows in the prompt both already hold."""

    import torch

    rows = placeholder_rows(prompt, image_token)
    if not rows or (payload is not None and tuple(payload.rows) != rows):
        raise ValueError("vision feature rows must match every image placeholder exactly")
    send = (payload.features if rank == 0 else
            torch.zeros((len(rows), hidden), dtype=torch.bfloat16, device=device))
    if tuple(send.shape) != (len(rows), hidden) or send.dtype != torch.bfloat16:
        raise ValueError("vision features differ from the image placeholders or the embedding width")
    got = torch.empty((2 * len(rows), hidden), dtype=torch.bfloat16, device=device)
    comm.all_gather(send.contiguous().view(-1), got.view(-1))
    return EncodedVision(rows, got[:len(rows)], None, 0)       # the same bits on both ranks


def replace_rows(x, payload: EncodedVision, start: int, end: int, copies: int = 1):
    """x [end - start, copies * D] holds prompt rows start..end: each image row's feature goes into every copy."""

    import torch

    selected = [(i, row - start) for i, row in enumerate(payload.rows) if start <= row < end]
    if selected:
        source, target = zip(*selected)
        source = torch.tensor(source, dtype=torch.int64, device=x.device)
        target = torch.tensor(target, dtype=torch.int64, device=x.device)
        rows = payload.features.index_select(0, source)
        x.view(x.shape[0], copies, -1).index_copy_(0, target, rows[:, None, :].expand(-1, copies, -1).contiguous())
    return x
