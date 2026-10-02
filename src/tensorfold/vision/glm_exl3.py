"""Decode only the quantized vision tower, one matrix at a time."""
import math
from pathlib import Path
from .qwen_checkpoint import vision_tensors


def layout(model_dir, shapes):
    from tensorfold.cuda.capacity import SIZES
    sources = vision_tensors(Path(model_dir))
    for name, (path, info, begin) in sources.items():
        shape, offsets = info.get('shape', []), info.get('data_offsets', [])
        dtype = info.get('dtype')
        if (dtype not in SIZES or len(offsets) != 2 or any(type(x) is not int or x < 0 for x in shape)
                or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * SIZES[dtype]
                or begin + offsets[1] > path.stat().st_size):
            raise ValueError(f'invalid vision tensor range: {name}')
    used, entries = set(), {}
    for name, shape in shapes.items():
        module, suffix = name.rsplit('.', 1)
        parts = [module]
        split = module.endswith('.attn.qkv') and module[:-3] + 'q_proj.trellis' in sources
        if split:
            parts = [module[:-3] + x + '_proj' for x in 'qkv']
            used.update(k for k in (module + '.weight', module + '.bias') if k in sources)
        if suffix == 'weight' and all(p + '.trellis' in sources for p in parts):
            dims = [shape[0] // len(parts), math.prod(shape[1:])]
            for p in parts:
                tr = sources[p + '.trellis'][1]
                if (tr['dtype'] != 'I16' or len(tr['shape']) != 3 or tr['shape'][-1] not in range(16, 129, 8)
                        or [tr['shape'][1]*16, tr['shape'][0]*16] != [math.ceil(d/128)*128 for d in dims]):
                    raise ValueError(f'invalid vision trellis dimensions: {p}')
                for key, size in [('suh', tr['shape'][0]*16), ('svh', tr['shape'][1]*16)]:
                    info = sources.get(p + '.' + key, (None, {}, None))[1]
                    if info.get('dtype') != 'F16' or info.get('shape') != [size]:
                        raise ValueError(f'invalid vision EXL3 scale: {p}.{key}')
                if p + '.mul1' not in sources:
                    raise ValueError(f'quantized GLM vision requires mul1: {p}')
                used.update(p + '.' + key for key in ('trellis', 'suh', 'svh', 'mul1'))
            entries[name] = ('quant', parts, shape)
        else:
            names = [p + '.' + suffix for p in parts]
            expected = [shape[0] // len(parts), *shape[1:]]
            for key in names:
                info = sources.get(key, (None, {}, None))[1]
                alt = [expected[0], *expected[2:], expected[1]] if len(expected) > 2 else None
                if info.get('dtype') not in ('F16', 'BF16', 'F32') or info.get('shape') not in (expected, alt):
                    raise ValueError(f'invalid vision tensor shape or dtype: {key}')
                used.add(key)
            entries[name] = ('float', names, shape)
    if used != set(sources):
        raise ValueError(f'unexpected vision tensor: {sorted(set(sources)-used)[:3]}')
    return sources, entries


def hadamard(x, axis):
    """Normalized 128-wide Walsh-Hadamard blocks, computed in FP32."""
    x = x.movedim(axis, -1).contiguous()
    shape = x.shape
    x = x.reshape(-1, 128).float()
    for width in (1, 2, 4, 8, 16, 32, 64):
        v = x.reshape(-1, 128//(2*width), 2, width)
        a, b = v[:, :, 0], v[:, :, 1]
        import torch
        x = torch.stack((a+b, a-b), dim=2).reshape(-1, 128)
    return (x / math.sqrt(128)).reshape(shape).movedim(-1, axis)


def decoded_weight(layer):
    w = layer.unpack().float()
    w = hadamard(hadamard(w, 0), 1)
    w *= layer.suh.float()[:, None]
    w *= layer.svh.float()[None, :]
    return w.T.contiguous()


def load(model_dir, shapes, device):
    import torch
    from tensorfold.cuda.exl3.linear import Exl3Linear
    sources, entries = layout(model_dir, shapes)
    kinds = {'I16': torch.int16, 'F16': torch.float16, 'BF16': torch.bfloat16, 'F32': torch.float32}
    def read(name):
        path, info, begin = sources[name]
        a, b = info['data_offsets']
        with path.open('rb') as stream:
            stream.seek(begin+a)
            raw = bytearray(stream.read(b-a))
        return torch.frombuffer(raw, dtype=kinds[info['dtype']]).reshape(info['shape']).to(device)
    result = {}
    for name, (kind, parts, shape) in entries.items():
        values = []
        for part in parts:
            if kind == 'quant':
                layer = Exl3Linear.from_tensors(read(part+'.trellis'), read(part+'.suh'), read(part+'.svh'),
                                               'mul1', device=device)
                value = decoded_weight(layer)[:shape[0]//len(parts), :math.prod(shape[1:])]
                value = value.reshape(shape[0]//len(parts), *shape[1:])
                del layer
            else:
                value = read(part)
                expected = [shape[0]//len(parts), *shape[1:]]
                if list(value.shape) != expected:
                    value = value.movedim(-1, 1)
            values.append(value.to(torch.bfloat16).contiguous())
            del value
        result[name] = values[0] if len(values) == 1 else torch.cat(values)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return result
