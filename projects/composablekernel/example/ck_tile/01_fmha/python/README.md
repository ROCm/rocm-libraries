# CK Tile FMHA Python

`ck_tile_fmha` provides a FlashAttention-compatible Python API backed by CK Tile.
It supports FP16/BF16 fixed-length and variable-length attention, packed inputs,
forward/backward, MHA/GQA/MQA, causal/local masks, ALiBi, dropout, deterministic
backward, and KV-cache operations, subject to CK backend restrictions.

Validated on **gfx1250**. Nonzero `softcap`, FP8, attention sinks, and unequal Q/V
head dimensions are unsupported. The package can coexist with `flash_attn` and
does not fall back to other backends.

## Build and install

Requires Linux, ROCm PyTorch >= 2.4, a ROCm compiler supporting the target GPU,
Python >= 3.9, CMake >= 3.21, Ninja, setuptools >= 64, and wheel. Install PyTorch
first; `--no-build-isolation` builds against that installation.

From this directory:

```bash
GPU_ARCHS=gfx1250 MAX_JOBS=64 python -m pip install --no-build-isolation '.[test]'
```

`GPU_ARCHS` defaults to the detected GPU; set it for cross-compilation.
`CK_ROOT` selects an alternate CK checkout. The build uses the C++ example's
architecture-specific kernels, including deterministic backward and fixed-length
bias support.

## Development and incremental builds

```bash
GPU_ARCHS=gfx1250 MAX_JOBS=64 python -m pip install --no-build-isolation -e '.[test]'
# After editing a kernel, code generator, or binding:
GPU_ARCHS=gfx1250 MAX_JOBS=64 python setup.py build_ext --inplace
```

Keep `python/build/` to reuse compiled objects, or set `CK_TILE_FMHA_BUILD_DIR`
to relocate the cache. Editable installs pick up Python edits immediately;
restart Python after rebuilding a loaded native extension.

## Use

```python
import torch
from ck_tile_fmha import flash_attn_func

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)
out.sum().backward()
```

Import FlashAttention function names from `ck_tile_fmha`; existing positional
arguments are preserved. ROCm PyTorch uses the `cuda` device name.

Fixed-length `flash_attn_func`, `flash_attn_kvpacked_func`, and
`flash_attn_qkvpacked_func` additionally accept an optional trailing `bias` argument:

```python
bias = torch.randn(128, 128, device="cuda", dtype=q.dtype, requires_grad=True)
out = flash_attn_func(q, k, v, bias=bias, causal=True)
out.sum().backward()  # Computes gradients for q, k, v, and bias.
```

`bias` has shape `(seqlen_q, seqlen_k)`, matches Q/K/V's device and FP16/BF16 dtype,
and requires a contiguous last dimension. It is shared across batches and heads
and added to `Q @ K.T * softmax_scale` before softmax. It cannot be combined with
`alibi_slopes` and is unavailable in variable-length and KV-cache APIs.

Bias gradients require a `(batch_size, nheads, seqlen_q, seqlen_k)` temporary,
reduced across batches and heads.

## Tests

| File | Coverage |
| --- | --- |
| `tests/test_fixedlen.py` | Fixed-length/packed forward and backward, bias |
| `tests/test_varlen.py` | Variable-length/packed forward and backward, paged attention |
| `tests/test_kvcache.py` | Cache updates, decode, rotary embeddings, cache indexing |
| `tests/test_interface.py` | Validation, native compatibility, fake tensors, `torch.compile` |

Tests use GPU PyTorch references in `tests/reference.py`; FlashAttention and
Triton are not required. Shared helpers live in `tests/utils.py`.

```bash
# Full suite (add --junitxml=test-results/fmha.xml to save a report)
HIP_VISIBLE_DEVICES=0 python -m pytest tests

# Fixed-length bias tests
HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_fixedlen.py -k bias
```
