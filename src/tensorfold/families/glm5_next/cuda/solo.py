"""One-rank full EXL3 GLM: direct checkpoint reads and bounded layer staging."""
from pathlib import Path
import json
import math
import torch

from tensorfold.cuda.capacity import itemsize, headers
from tensorfold.cuda.exl3 import experts
from tensorfold.cuda.exl3.linear import Exl3Linear
from . import dense_exl3, latent
from .qmm import make_b16
from .split import RankReader, read_header


class LocalComm:
    """The collectives' one-rank identity; no NCCL or worker thread."""
    world = 1
    store = None

    def barrier(self):
        pass

    def ready(self, phase):
        pass

    def all_gather(self, source, dest):
        dest.view(-1).copy_(source.reshape(-1))


class FullReader(RankReader):
    def _span(self, name):
        file = str(self.dir / self.index[name])
        if file not in self.files:
            self.files[file] = read_header(file)
        header, base = self.files[file]
        info = header[name]
        a, b = info['data_offsets']
        return file, base + a, base + b, 'rep', list(info['shape']), info['dtype']


def weight_transform(layers):
    """Stored tensors plus bounded latent absorption copies; vision/MTP excluded."""
    def transform(name, info):
        if 'visual.' in name or f'.layers.{layers}.' in name:
            return 0, 0
        size = math.prod(info['shape']) * itemsize(info, name)
        if name.endswith('kv_b_proj.weight'):
            size *= 4  # stored, BF16 key/value matrices, FP32 absorption (conservative)
        return size, 0
    return transform


def validate(model_dir):
    """Validate actual trellis widths/scales from headers before CUDA allocation."""
    root = Path(model_dir)
    raw = json.loads((root / 'config.json').read_text())
    cfg = raw.get('text_config') or raw
    quant = raw.get('quantization_config') or {}
    cb = quant.get('codebook')
    if quant.get('quant_method') != 'exl3' or cb not in ('mul1', 'mcg', '3inst'):
        raise ValueError('TP=1 requires a supported EXL3 codebook')
    h = headers(root)
    layers = int(cfg['num_hidden_layers'])
    groups = 0
    for name, info in h.items():
        if 'visual.' in name or f'.layers.{layers}.' in name or not name.endswith('.trellis'):
            continue
        shape = info['shape']
        if info['dtype'] != 'I16' or len(shape) != 3 or shape[-1] not in range(16, 129, 8):
            raise ValueError(f'invalid EXL3 trellis: {name}')
        k, n = shape[0] * 16, shape[1] * 16
        if not k or not n or k % 128 or n % 128:
            raise ValueError(f'off-grid EXL3 matrix: {name}')
        if shape[-1] % 16 and cb != 'mul1':
            raise ValueError(f'half-bit trellis requires mul1: {name}')
        prefix = name[:-8]
        for suffix, width in [('suh', k), ('svh', n)]:
            scale = h.get(prefix + '.' + suffix)
            if scale is None or scale['dtype'] != 'F16' or scale['shape'] != [width]:
                raise ValueError(f'invalid EXL3 {suffix}: {prefix}')
        actual = next((c for c in ('mul1', 'mcg') if prefix + '.' + c in h), '3inst')
        if actual != cb:
            raise ValueError(f'EXL3 codebook mismatch: {prefix} ({actual} != {cb})')
        groups += 1
    if not groups:
        raise ValueError('checkpoint has no EXL3 groups')
    return {'groups': groups, 'codebook': cb, 'layers': layers}


def load(model_dir, device='cuda'):
    from .weights import Config, HCW, KDAW, IndexW, DSAW, MLPW, MoEW, LayerW, Weights, PREFIX
    root = Path(model_dir)
    validate(root)
    cfg = Config.read(root)
    quant = json.loads((root / 'config.json').read_text())['quantization_config']
    if quant.get('quant_method') != 'exl3' or quant.get('codebook') not in ('mul1', 'mcg', '3inst'):
        raise ValueError('one-rank GLM requires a supported full EXL3 checkpoint')
    dev = torch.device(device)
    rd = FullReader(root, 0)

    def t(name, dtype=None):
        x = rd.get(PREFIX + name).to(dev)
        return x if dtype is None else x.to(dtype)

    def group(module):
        name = PREFIX + module
        cb = next((c for c in ('mul1', 'mcg') if name + '.' + c in rd.index), '3inst')
        bias = rd.get(name + '.bias') if name + '.bias' in rd.index else None
        return Exl3Linear.from_tensors(rd.get(name + '.trellis'), rd.get(name + '.suh'),
                                      rd.get(name + '.svh'), cb, bias=bias, device=dev)

    def matrix(name):
        if PREFIX + name + '.trellis' in rd.index:
            part = group(name)
            return dense_exl3.Dense3([part], [part.n], part.k)
        return make_b16(t(name + '.weight'))

    def stack(names):
        parts = [matrix(n) for n in names]
        parts = [p.parts[0] if isinstance(p, dense_exl3.Dense3) else p for p in parts]
        return dense_exl3.Dense3(parts, [p.n for p in parts], parts[0].k)

    def hc(i, site):
        return HCW(t(f'layers.{i}.hc_{site}_fn').contiguous(),
                   t(f'layers.{i}.hc_{site}_base', torch.float32),
                   t(f'layers.{i}.hc_{site}_scale', torch.float32))

    def kda(i):
        p = f'layers.{i}.self_attn.'
        if PREFIX + p + 'qkv_proj.trellis' in rd.index:
            names = ['qkv_proj', 'f_a_proj', 'g_a_proj', 'b_proj']
            conv = t(p + 'conv1d.weight').reshape(3 * cfg.lin_heads * 128, cfg.conv)
        else:
            names = ['q_proj', 'k_proj', 'v_proj', 'f_a_proj', 'g_a_proj', 'b_proj']
            conv = torch.cat([t(p + f'{x}_conv1d.weight') for x in 'qkv']).reshape(3 * cfg.lin_heads * 128, cfg.conv)
        return KDAW(stack([p + n for n in names]), matrix(p + 'f_b_proj'), matrix(p + 'g_b_proj'),
                    conv.contiguous(), t(p + 'A_log', torch.float32).contiguous(),
                    t(p + 'dt_bias', torch.float32).contiguous(), t(p + 'o_norm.weight'),
                    matrix(p + 'o_proj'), cfg.lin_heads)

    def dsa(i):
        p = f'layers.{i}.self_attn.'
        # This checkpoint stores kv_b in FP16. Absorb the exact stored rows;
        # do not quantize them or expand routed expert matrices.
        w = t(p + 'kv_b_proj.weight').reshape(cfg.heads, cfg.qk_dim + cfg.v_dim, cfg.kv_lora)
        keys, vals = w[:, :cfg.qk_dim].reshape(-1, cfg.kv_lora), w[:, cfg.qk_dim:].reshape(-1, cfg.kv_lora)
        absorb = latent.AbsorbW.from_rows(keys.float(), vals.float(), cfg.heads) if latent.ENABLED else None
        ix = IndexW(stack([p + 'indexer.wk', p + 'indexer.weights_proj']), matrix(p + 'indexer.wq_b'),
                    t(p + 'indexer.k_norm.weight'), t(p + 'indexer.k_norm.bias'),
                    t(p + 'indexer.index_kpool_compress_gate', torch.bfloat16).contiguous(),
                    t(p + 'indexer.index_kpool_compress_ape', torch.bfloat16).contiguous())
        return DSAW(stack([p + 'q_a_proj', p + 'kv_a_proj_with_mqa']), t(p + 'q_a_layernorm.weight'),
                    t(p + 'kv_a_layernorm.weight'), matrix(p + 'q_b_proj'), make_b16(keys), make_b16(vals),
                    matrix(p + 'o_proj'), cfg.heads, ix, absorb)

    def mlp(p):
        gu = stack([p + 'gate_proj', p + 'up_proj'])
        return MLPW(gu, matrix(p + 'down_proj'), gu.n // 2)

    def moe(i):
        p = f'layers.{i}.mlp.'
        mats = []
        for proj in ('gate', 'up', 'down'):
            triples = []
            for e in range(cfg.experts):
                name = PREFIX + p + f'experts.{e}.{proj}_proj'
                triples.append(tuple(rd.get(name + '.' + suffix).to(dev) for suffix in ('trellis', 'suh', 'svh')))
            mats.append(triples)
        routed = experts.prepare(*mats, quant['codebook'], device=dev)
        return MoEW(t(p + 'gate.weight', torch.bfloat16), t(p + 'gate.e_score_correction_bias', torch.float32),
                    routed, mlp(p + 'shared_experts.'))

    try:
        embed = rd.get(PREFIX + 'embed_tokens.weight').to(dev, torch.bfloat16)
        built = []
        for i in range(cfg.layers):
            layer = LayerW(i, cfg.kinds[i], hc(i, 'attn'), hc(i, 'ffn'),
                           t(f'layers.{i}.input_layernorm.weight'), t(f'layers.{i}.post_attention_layernorm.weight'))
            if layer.kind == 'kda':
                layer.kda = kda(i)
            else:
                layer.dsa = dsa(i)
            if cfg.mlp_kinds[i] == 'dense':
                layer.mlp = mlp(f'layers.{i}.mlp.')
            else:
                layer.moe = moe(i)
            built.append(layer)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f'[tensorfold] TP=1 full EXL3 layer {i+1}/{cfg.layers}', flush=True)
        name = 'lm_head'
        if name + '.trellis' in rd.index:
            cb = next((c for c in ('mul1', 'mcg') if name + '.' + c in rd.index), '3inst')
            part = Exl3Linear.from_tensors(rd.get(name + '.trellis'), rd.get(name + '.suh'),
                                          rd.get(name + '.svh'), cb, device=dev)
            if part.n != cfg.vocab:
                raise ValueError('head width must match tokenizer vocabulary')
            head = dense_exl3.Dense3([part], [part.n], part.k)
        else:
            head = make_b16(rd.get('lm_head.weight').to(dev))
        w = Weights(cfg, embed, built, t('norm.weight'), head, None, 0, 1, dev)
        w.meta.update(layers=list(range(cfg.layers)))
        return w
    finally:
        rd.close()
