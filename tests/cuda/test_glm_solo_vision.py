"""One-rank DFlash2 consumes replaced image rows exactly as text embeddings."""
from types import SimpleNamespace
import pytest
import torch
from test_glm_solo import solo
from test_glm_engine import _generate
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.vision.qwen_cuda import EncodedVision
from tensorfold.vision.glm_exl3 import decoded_weight
from tensorfold.cuda.exl3.linear import Exl3Linear


def test_one_rank_vision_constructor(tmp_path, monkeypatch):
    import json
    from test_glm_solo import full_checkpoint
    from test_glm_engine import _drafter
    from tensorfold.vision import glm_cuda
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    full_checkpoint(tmp_path / 'model')
    _drafter(tmp_path / 'draft')
    path = tmp_path / 'model/config.json'
    cfg = json.loads(path.read_text()); cfg['image_token_id'] = 9
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(glm_cuda, 'capacity_geometry', lambda base, *args: base)
    monkeypatch.setattr(glm_cuda, 'weight_transform', lambda base, *args: base)
    monkeypatch.setattr(glm_cuda, 'GLMCudaVision', lambda *args, **kw: SimpleNamespace(weight_bytes=0))
    engine = GlmEngine(tmp_path / 'model', rank=0, master='', port=0, world=1, vision=True,
                       drafter=tmp_path / 'draft', context=4096, prefill_rows=128)
    assert engine.vision is not None and engine.image_token == 9


@pytest.mark.parametrize('temperature', [0, .8])
def test_single_rank_vision_dflash_equals_text(solo, monkeypatch, temperature):
    sampling = Sampling(temperature=temperature, seed=123)
    prompt = [10, 22, 30, 44]
    expected, _ = _generate(solo, prompt, sampling, tokens=16)
    payload = EncodedVision((1,), solo.w.embed[22:23].clone(), None, 0)
    monkeypatch.setattr(solo, 'image_token', 9)
    monkeypatch.setattr(solo, 'vision', SimpleNamespace(encode=lambda *args: payload))
    image = [10, 9, 30, 44]
    for draft in (False, True):
        out = []
        stats = solo.generate(image, 16, sampling, lambda new: out.extend(new), draft=draft, vision=object())
        assert out == expected
        assert stats['cached'] == 0
        assert solo.live[1] == -1
    following, _ = _generate(solo, prompt, sampling, tokens=16)
    assert following == expected


def test_materialized_exl3_weight_matches_quantized_linear():
    g = torch.Generator().manual_seed(15)
    trellis = torch.randint(-32768, 32767, (16, 16, 80), dtype=torch.int16, generator=g)
    suh = torch.rand(256, dtype=torch.float16, generator=g) * .2
    svh = torch.rand(256, dtype=torch.float16, generator=g) * .2
    layer = Exl3Linear.from_tensors(trellis, suh, svh, 'mul1')
    x = torch.randn((7, 256), generator=g).cuda().bfloat16()
    decoded = decoded_weight(layer)
    got = x.float() @ decoded.T
    ref = layer(x, out_dtype=torch.float32)
    torch.testing.assert_close(got, ref, rtol=.03, atol=.003)
