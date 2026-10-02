"""GLM-5.3-Flash's CUDA image contracts without accelerator runtimes: headers, rows, transport and the tower build."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.glm_cuda import capacity_geometry, checkpoint_vision, tower_shapes, weight_transform

VISION = {"model_type": "glm5_next_vision", "hidden_size": 8, "out_hidden_size": 16, "depth": 1, "patch_size": 2,
          "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3, "intermediate_size": 12,
          "num_heads": 2, "projection_intermediate_size": 20, "attention_bias": True}


def _checkpoint(path, layout=None):
    config = {"model_type": "glm5_next", "image_token_id": 99, "vision_config": VISION,
              "text_config": {"hidden_size": 16}}
    (path / "config.json").write_text(json.dumps(config))
    shapes = {**tower_shapes(VISION), **(layout or {})}
    offset, entries = 0, {}
    for name, shape in shapes.items():
        size = int(np.prod(shape)) * 2
        entries["model.visual." + name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    _write(path, entries, offset)
    return entries, offset


def _write(path, entries, size):
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(size))


def test_the_tower_is_sized_from_headers_alone(tmp_path):
    _, size = _checkpoint(tmp_path)
    config, resident = checkpoint_vision(tmp_path)
    assert config["out_hidden_size"] == 16 and resident == size
    assert len(tower_shapes({**VISION, "depth": 24})) == 347       # GLM-5.3-Flash's tower, tensor for tensor


def test_an_mlx_conversion_keeps_its_channels_last_convolutions(tmp_path):
    _checkpoint(tmp_path, {"patch_embed.proj.weight": [8, 2, 2, 2, 3], "downsample.weight": [16, 2, 2, 8]})
    checkpoint_vision(tmp_path)


@pytest.mark.parametrize("damage", ["missing", "quantized", "range", "shape", "width"])
def test_incomplete_or_incompatible_towers_refuse_before_loading(tmp_path, damage):
    entries, size = _checkpoint(tmp_path)
    key = "model.visual.blocks.0.attn.qkv.weight"
    if damage == "missing":
        del entries[key]
    elif damage == "quantized":
        entries[key]["dtype"] = "U32"
    elif damage == "range":
        entries[key]["data_offsets"] = [size, size + 24 * 8 * 2]
    elif damage == "shape":
        entries[key]["shape"] = [12, 16]
    else:
        config = json.loads((tmp_path / "config.json").read_text())
        config["text_config"]["hidden_size"] = 32
        (tmp_path / "config.json").write_text(json.dumps(config))
    _write(tmp_path, entries, size)
    with pytest.raises(ValueError):
        checkpoint_vision(tmp_path)


def test_the_tower_and_its_workspace_are_counted_on_rank_zero_only(tmp_path):
    from tensorfold.cuda.capacity import Geometry

    _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    zero = capacity_geometry(base, tmp_path, True, 0)({})
    one = capacity_geometry(base, tmp_path, True, 1)({})
    plain = capacity_geometry(base, tmp_path, False, 0)({})
    assert zero.needed(32) > one.needed(32) > plain.needed(32)
    dropped = lambda name, info: (0, 0)               # the language split rule drops model.visual.*
    info = {"shape": [8, 8], "dtype": "BF16"}
    assert weight_transform(dropped, True, 0)("model.visual.x", info) == (128, 0)
    assert weight_transform(dropped, True, 1)("model.visual.x", info) == (0, 0)
    assert weight_transform(dropped, True, 0)("model.language_model.x", info) == (0, 0)


def test_each_image_row_goes_into_every_hyper_connection_copy():
    torch = pytest.importorskip("torch")
    from tensorfold.vision.glm_cuda import replace_rows
    from tensorfold.vision.qwen_cuda import EncodedVision

    features = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) + 100
    payload = EncodedVision((5, 6, 9), features, None, 0)
    x = torch.zeros((4, 2 * 4))                        # prompt rows 4..8, two copies of width 4
    replace_rows(x, payload, 4, 8, 2)
    want = torch.zeros((4, 2, 4))
    want[1], want[2] = features[0], features[1]        # row 9 belongs to the next chunk
    assert torch.equal(x.view(4, 2, 4), want)


def test_prefill_skips_chunks_that_hold_no_image_rows():
    from tensorfold.families.glm5_next.cuda.decode import _images
    from tensorfold.vision.qwen_cuda import EncodedVision

    payload = EncodedVision((10, 11, 30), None, None, 0)
    assert _images(None, 0, 16) is None
    assert _images(payload, 0, 10) is None and _images(payload, 31, 16) is None
    assert all(_images(payload, s, n) is not None for s, n in ((0, 11), (11, 1), (12, 16), (30, 4)))


def test_a_long_prompt_writes_every_image_row_once_in_each_chunk_on_both_ranks():
    torch = pytest.importorskip("torch")
    from tensorfold.families.glm5_next.cuda.decode import PREFILL_ROWS, _images
    from tensorfold.vision.glm_cuda import share_encoded
    from tensorfold.vision.qwen_cuda import EncodedVision

    # a 12,164-token prompt whose 4,096 image rows straddle prefill chunks and the 2,051-token dense limit
    n, image, rows = 12_164, 7, tuple(range(1_900, 1_900 + 4_096))
    prompt = [image if i in rows else 1 + i % 5 for i in range(n)]
    features = (torch.arange(len(rows), dtype=torch.float32)[:, None] + 1).expand(-1, 2).to(torch.bfloat16)
    sent = []

    def all_gather(send, recv):
        sent.append(send.clone()) if not sent else None
        recv[:send.numel()], recv[send.numel():] = sent[0], send

    comm = SimpleNamespace(all_gather=all_gather)
    ranks = [share_encoded(EncodedVision(rows, features, None, 0), 0, comm, prompt, image, 2, "cpu"),
             share_encoded(None, 1, comm, prompt, image, 2, "cpu")]
    for payload in ranks:
        for shift in (0, 1):                           # the main model's rows, the MTP head's next-token rows
            x = torch.zeros((n, 2), dtype=torch.bfloat16)
            for start in range(0, n, PREFILL_ROWS):
                begin, end = start + shift, min(start + PREFILL_ROWS + shift, n)
                write = _images(payload, begin, end - begin)
                if write is not None:
                    write(x[begin:end], 1)
            want = torch.zeros((n, 2), dtype=torch.bfloat16)
            want[list(rows)] = features
            assert torch.equal(x, want), shift


def test_both_ranks_receive_the_same_features_and_find_the_rows_themselves():
    torch = pytest.importorskip("torch")
    from tensorfold.vision.glm_cuda import share_encoded
    from tensorfold.vision.qwen_cuda import EncodedVision

    sent = []

    def all_gather(send, recv):
        sent.append(send.clone()) if rank[0] == 0 else None
        recv[:send.numel()] = sent[-1]
        recv[send.numel():] = send

    rank = [0]
    comm = SimpleNamespace(all_gather=all_gather)
    prompt = [1, 99, 99, 2, 99]
    features = torch.randn(3, 4).to(torch.bfloat16)
    got0 = share_encoded(EncodedVision((1, 2, 4), features, None, 0), 0, comm, prompt, 99, 4, "cpu")
    rank[0] = 1
    got1 = share_encoded(None, 1, comm, prompt, 99, 4, "cpu")
    assert got0.rows == got1.rows == (1, 2, 4)
    assert torch.equal(got0.features, features) and torch.equal(got1.features, features)
    with pytest.raises(ValueError):                    # rows that are not the placeholders never reach the other rank
        share_encoded(EncodedVision((1, 2), features[:2], None, 0), 0, comm, prompt, 99, 4, "cpu")
    with pytest.raises(ValueError):
        share_encoded(None, 1, comm, [1, 2, 3], 99, 4, "cpu")


def test_a_continued_image_prompt_adds_text_only():
    from tensorfold.vision.glm_processing import GLMImageProcessor, PreparedGLMVisionPrompt

    processor = object.__new__(GLMImageProcessor)
    processor.image_token_id = 99
    prepared = PreparedGLMVisionPrompt((1, 99, 99, 2), np.zeros((8, 4)), np.array([[1, 2, 4]]), ((1, 3),), ("h",))
    assert processor.continued(prepared, [1, 99, 99, 2, 5, 6]).token_ids == (1, 99, 99, 2, 5, 6)
    with pytest.raises(ValueError):
        processor.continued(prepared, [1, 99, 99, 2, 99])
    with pytest.raises(ValueError):
        processor.continued(prepared, [1, 99, 2, 2, 5])


def test_a_meta_built_tower_matches_a_normally_built_one_in_the_installed_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    config = Glm5NextVisionConfig(**{**VISION, "hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
                                     "projection_intermediate_size": 48, "patch_size": 4})
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    built = Glm5NextVisionModel(config).eval()
    with torch.device("meta"):
        meta = Glm5NextVisionModel(config)
    meta.load_state_dict(built.state_dict(), strict=True, assign=True)
    rotary = meta.rotary_pos_emb                       # as GLMCudaVision fills it
    inv_freq, _ = rotary.compute_axial_rope_parameters(config)
    for name in ("inv_freq", "original_inv_freq"):
        rotary._buffers[name] = inv_freq.clone()
    for name, buffer in built.rotary_pos_emb.named_buffers():
        assert torch.equal(dict(rotary.named_buffers())[name], buffer)
    pixels = torch.randn(32, 3 * 2 * 4 * 4)
    grid = torch.tensor([[1, 4, 4], [1, 2, 8]])
    with torch.inference_mode():
        want = built(pixels, grid_thw=grid, return_dict=True).pooler_output
        got = meta.eval()(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert got.shape == (8, 32) and torch.equal(got, want)


@pytest.mark.parametrize("channels_last", [False, True])
def test_the_loader_reads_the_tower_from_checkpoint_bytes(tmp_path, monkeypatch, channels_last):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from safetensors.torch import save_file
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    from tensorfold.vision import glm_cuda, glm_processing

    vision = {**VISION, "hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
              "projection_intermediate_size": 48, "patch_size": 4}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "image_token_id": 99,
                                                      "vision_config": vision, "text_config": {"hidden_size": 32}}))
    torch.manual_seed(1)
    built = Glm5NextVisionModel(Glm5NextVisionConfig(**vision)).eval()
    frequencies = {k: v.clone() for k, v in built.rotary_pos_emb.named_buffers()}
    built.to(torch.bfloat16)
    built.rotary_pos_emb._buffers.update(frequencies)  # float32, as a checkpoint load keeps them (not saved)
    weights = {"model.visual." + k: v.contiguous() for k, v in built.state_dict().items()}
    if channels_last:                                  # as an MLX conversion stores its convolutions
        for name in ("patch_embed.proj.weight", "downsample.weight"):
            weights["model.visual." + name] = weights["model.visual." + name].movedim(1, -1).contiguous()
    weights["model.language_model.embed_tokens.weight"] = torch.zeros(4, 32, dtype=torch.bfloat16)
    save_file(weights, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(glm_processing.GLMImageProcessor, "from_directory",
                        classmethod(lambda cls, path: SimpleNamespace(image_token_id=99)))
    loaded = glm_cuda.GLMCudaVision(tmp_path, torch.device("cpu"))
    assert loaded.image_token == 99 and loaded.weight_bytes == sum(v.numel() * 2 for k, v in weights.items()
                                                                     if k.startswith("model.visual."))
    pixels = torch.randn(32, 3 * 2 * 4 * 4).to(torch.bfloat16)
    grid = torch.tensor([[1, 4, 4], [1, 2, 8]])
    with torch.inference_mode():
        want = built(pixels, grid_thw=grid, return_dict=True).pooler_output
        got = loaded.tower(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert torch.equal(got, want)


def test_the_thinking_off_template_passes_the_image_switch_on():
    from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate

    seen = []
    inner = SimpleNamespace(render=lambda messages, **kw: seen.append(kw) or "<|assistant|><think></think>")
    ThinkingOffTemplate(inner).render([], tools=None, enable_thinking=True, allow_images=True)
    assert seen[0]["allow_images"] is True
def test_text_only_template_receives_native_glm_image_markers():
    from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate
    class TextOnly:
        def render(self, messages, **kwargs):
            return messages[0]['content']
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Look: '}, {'type': 'image'},
        {'type': 'text', 'text': ' please.'}]}]
    rendered = ThinkingOffTemplate(TextOnly()).render(
        messages, tools=None, enable_thinking=True, allow_images=True)
    assert rendered == 'Look: <|begin_of_image|><|image|><|end_of_image|> please.'
    assert isinstance(messages[0]['content'], list)
