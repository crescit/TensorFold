"""Full low-bit/fused checkpoint on one GPU: real draft and cache equality."""
import json
import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_glm_engine import _checkpoint, _drafter, _generate
from tensorfold.engine.exact_sampling import Sampling


def full_checkpoint(path):
    _checkpoint(path, exl3=True, mtp=False)
    file = path / 'model-00001-of-00001.safetensors'
    with safe_open(str(file), framework='pt') as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    rng = np.random.default_rng(13)
    for name in list(tensors):
        if name.endswith('.mlp.gate.weight'):
            tensors[name] = tensors[name].to(torch.float16)
            continue
        if name.endswith('.mcg'):
            tensors[name[:-4] + '.mul1'] = tensors.pop(name)
        elif name.endswith('.trellis'):
            tensors[name] = tensors[name][..., :32].contiguous()
        elif name.endswith('.weight') and tensors[name].ndim == 2:
            n, k = tensors[name].shape
            if min(n, k) < 128 or n % 128 or k % 128 or any(x in name for x in ('embed_tokens', 'kv_b_proj', 'gate.weight', 'compress_gate')):
                continue
            module = name[:-7]
            bits = 5 if module == 'lm_head' else 4
            tensors.pop(name)
            tensors[module + '.trellis'] = torch.from_numpy(rng.integers(-32768, 32767, (k//16, n//16, 16*bits), dtype=np.int16))
            tensors[module + '.suh'] = torch.ones(k, dtype=torch.float16)
            tensors[module + '.svh'] = torch.full((n,), .01, dtype=torch.float16)
            tensors[module + '.mul1'] = torch.empty(0, dtype=torch.int16)
    prefix = 'model.language_model.layers.0.self_attn.'
    for suffix in ('trellis', 'svh'):
        axis = 1 if suffix == 'trellis' else 0
        tensors[prefix + 'qkv_proj.' + suffix] = torch.cat([tensors.pop(prefix + x + '_proj.' + suffix) for x in 'qkv'], axis)
    tensors[prefix + 'qkv_proj.suh'] = tensors.pop(prefix + 'q_proj.suh')
    tensors[prefix + 'qkv_proj.mul1'] = tensors.pop(prefix + 'q_proj.mul1')
    for x in 'kv':
        tensors.pop(prefix + x + '_proj.suh')
        tensors.pop(prefix + x + '_proj.mul1')
    tensors[prefix + 'conv1d.weight'] = torch.cat([tensors.pop(prefix + x + '_conv1d.weight') for x in 'qkv'])
    save_file(tensors, str(file))
    cfg = json.loads((path / 'config.json').read_text())
    cfg['quantization_config'].update(codebook='mul1', bits=2.05)
    (path / 'config.json').write_text(json.dumps(cfg))


@pytest.fixture(scope='module')
def solo(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    root = tmp_path_factory.mktemp('glm_solo')
    full_checkpoint(root / 'model')
    _drafter(root / 'draft')
    return GlmEngine(root / 'model', rank=0, master='', port=0, world=1,
                     drafter=root / 'draft', context=4096, prefill_rows=128)


@pytest.mark.parametrize('temperature', [0.0, 0.8])
def test_one_rank_drafts_equal_serial(solo, temperature):
    sampling = Sampling(temperature=temperature, seed=123)
    prompt = [11, 29, 4, 23, 99]
    serial, _ = _generate(solo, prompt, sampling, draft=False, tokens=16)
    solo.cache.clear(); solo.live = []
    drafted, _ = _generate(solo, prompt, sampling, draft=True, tokens=16)
    assert drafted == serial
    assert solo.w.layers[1].moe.router.dtype == torch.bfloat16
    assert solo.w.world == 1 and solo.w.comm is None
    assert solo.w.layers[1].moe.experts.cb == 2
    assert solo.w.head.n == solo.w.cfg.vocab


def test_one_rank_resume_equals_fresh(solo):
    sampling = Sampling(temperature=0.0, seed=123)
    prompt = [10, 22, 30, 44]
    _generate(solo, prompt, sampling, tokens=8)
    follow = prompt + [31, 45, 7]
    cached, _ = _generate(solo, follow, sampling, tokens=16)
    solo.cache.clear(); solo.live = []
    fresh, _ = _generate(solo, follow, sampling, tokens=16)
    assert cached == fresh
