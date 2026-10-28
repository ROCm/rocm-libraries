# CK Tile FMHA Python

`ck_tile_fmha` provides PyTorch bindings for CK Tile FMHA kernels.
It supports FP16/BF16 fixed-length and variable-length attention, packed inputs,
forward/backward, MHA/GQA/MQA, causal/local masks, ALiBi, dropout, deterministic
backward, and KV-cache operations, subject to CK backend restrictions.

Validated on **gfx1250**. Nonzero `softcap`, attention sinks, and unequal Q/V
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
architecture-specific kernels with receipt 5, which generates only the variants
needed by the current Python API, including bias and deterministic backward.

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

## FP8 fixed-length forward

`flash_attn_fp8_func` takes prequantized E4M3 Q/K/V and returns BF16 output.
Use `torch.float8_e4m3fn` on gfx1250/gfx950 and `torch.float8_e4m3fnuz` on gfx94x.
Descales are FP32 tensors on the same GPU; dequantization is `input.float() * descale`.

| `qscale` | Q descale | K/V descales |
| --- | --- | --- |
| `no_scale` | None | None |
| `pertensor` (default) | One element | One element each |
| `perhead` | `[batch, nheads]` | `[batch, nheads_kv]` |
| `blockscale` | `[batch, nheads, ceil(seqlen_q / block_size_q)]` | `[batch, nheads_kv, ceil(seqlen_k / block_size_kv)]` |

```python
from ck_tile_fmha import flash_attn_fp8_func

# Quantize each tensor independently; reduce over sequence/features for per-head.
def quantize_perhead(x):
    descale = x.float().abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    quantized = (x.float() / descale[:, None, :, None]).to(torch.float8_e4m3fn)
    return quantized, descale

q8, qs = quantize_perhead(q.detach())
k8, ks = quantize_perhead(k.detach())
v8, vs = quantize_perhead(v.detach())
out, lse = flash_attn_fp8_func(q8, k8, v8, qs, ks, vs, qscale="perhead", return_lse=True)
```

Supports MHA/GQA/MQA and causal/local masks. FP8 backward, dropout, bias and ALiBi
are not exposed. On gfx1250, FP8 head dimensions must be <= 128. Per-head is
currently available only on gfx1250; other modes and LSE availability depend on
CK's target kernels. `return_lse=True` returns
`(out, lse)` with FP32 LSE `[batch, nheads, seqlen_q]`.

Block sizes default to 128 and must satisfy the kernel's tile alignment. Block
scales require a contiguous last dimension. On gfx1250, **blockscale V descales
must be positive powers of two** (E8M0); Q/K descales may be general FP32 values.
`kv_blockscale` is reserved for a future paged-KV API.

## Tests

| File | Coverage |
| --- | --- |
| `tests/test_fixedlen.py` | Fixed-length/packed forward and backward, bias |
| `tests/test_fp8.py` | FP8 fixed-length quantization, validation and `torch.compile` |
| `tests/test_varlen.py` | Variable-length/packed forward and backward, paged attention |
| `tests/test_kvcache.py` | Cache updates, decode, rotary embeddings, cache indexing |
| `tests/test_interface.py` | Validation, native compatibility, fake tensors, `torch.compile` |

FP8 tests compare against `flash_attn_func` using dequantized inputs cast to BF16;
other numerical tests use GPU PyTorch references in `tests/reference.py`.
FlashAttention and Triton are not required. Shared helpers live in `tests/utils.py`.

```bash
# Full suite (add --junitxml=test-results/fmha.xml to save a report)
HIP_VISIBLE_DEVICES=0 python -m pytest tests

# Fixed-length bias tests
HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_fixedlen.py -k bias
```

To run the suite manually in Jenkins, enable `RUN_FMHA_PYTHON_TESTS` in
**Build with Parameters**. It defaults to off for PR and scheduled builds.
This builds `Dockerfile.fmha_python` from the current checkout and publishes
JUnit results and the pytest log. A physical `gfx1250` GPU runner is required;
the existing gfx1250 simulator CI does not run this suite.
Before merging changes to `groovy/vars/ck.groovy`, also enable
`USE_CURRENT_BRANCH_FOR_CK_GROOVY`.
