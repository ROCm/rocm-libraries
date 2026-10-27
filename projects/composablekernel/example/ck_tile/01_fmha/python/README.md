# CK Tile FMHA Python

`ck_tile_fmha` exposes CK Tile FMHA through the FlashAttention Python API and
validates it with the FlashAttention CK pytest matrix and GPU PyTorch references.
It is a separate package, native extension, and `torch.ops` namespace, so it can
coexist with `flash_attn`. It never falls back to another attention backend.

The initial validation target is **gfx1250**. This integration covers the
FlashAttention CK backend's existing FP16/BF16 functionality: fixed and variable
length attention, packed inputs, forward and backward, MHA/GQA/MQA, causal/local
masks, ALiBi, dropout, deterministic backward, and KV-cache operations. Existing
backend restrictions still apply. Nonzero `softcap` is explicitly rejected; CK
extensions such as FP8, attention sinks and unequal Q/V head dimensions are
outside this initial API.

## Build and install

Use a Linux environment with ROCm PyTorch >= 2.4, a ROCm compiler supporting the
target GPU, Python >= 3.9, CMake >= 3.21, Ninja, setuptools >= 64 and wheel.
The installation directory must be writable; a virtual environment is suitable.
Install the ROCm PyTorch build appropriate to your environment first. Build in
that environment using `--no-build-isolation`, so the extension uses the same
PyTorch headers and ABI as the runtime.

From this directory:

```bash
GPU_ARCHS=gfx1250 MAX_JOBS=64 python -m pip install --no-build-isolation '.[test]'
```

The setuptools entry point is also supported:

```bash
GPU_ARCHS=gfx1250 MAX_JOBS=64 python setup.py install
python -m pip install pytest einops
```

If `GPU_ARCHS` is unset, the current GPU is detected. Set it explicitly for
cross-compilation. Other architectures are configurable but are not part of the
initial gfx1250 validation. The compiler must support every requested target.
`CK_ROOT` optionally selects another CK checkout. A source distribution includes
the CK headers and FMHA generator needed to build without the original checkout.

The build reuses `01_fmha/CMakeLists.txt`, including CK's per-architecture forward
head-dimension list, so the Python tests exercise the same specialized kernels as
the C++ example (for gfx1250, including head dimensions 96, 160 and 192). It
generates `--receipt 2` for forward and backward, including deterministic kernels,
and does not use the example's reduced testing filters. Per-instance scheduling
and per-direction BF16 conversion settings come from the existing CK targets.

## Development and incremental builds

```bash
GPU_ARCHS=gfx1250 MAX_JOBS=64 python -m pip install --no-build-isolation -e '.[test]'
# After editing a kernel, code generator, or binding:
GPU_ARCHS=gfx1250 MAX_JOBS=64 python setup.py build_ext --inplace
```

CMake/Ninja track CK headers, generated kernels and compiler options. Codegen
preserves unchanged generated files. The PyTorch binding build uses private
copies for hipification and relinks whenever the CK kernel archive changes.
Keep the build directory to reuse compiled objects. Python-only edits take effect
immediately with an editable installation. Restart the Python process after
rebuilding a loaded native extension.

All installation methods share a persistent cache under `python/build/`, including
editable pip installs. Set `CK_TILE_FMHA_BUILD_DIR` to relocate that cache.

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

The public function names and signatures match FlashAttention. Import them from
`ck_tile_fmha` instead of `flash_attn`. ROCm PyTorch uses the `cuda` device name.

## Test before a PR

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests
```

This runs the original parameter matrix, without a default time budget or reduced
case selection. It contains 412,756 combinations before the original runtime
skips. Original numeric thresholds, seeds, race repetition counts and determinism
checks are retained. The FP32 reference uses GPU PyTorch tensor operations rather
than SDPA or another fused attention implementation. Padding and rotary helpers
are local, so neither FlashAttention nor its Triton kernels are needed.
Additional integration tests check unsupported softcap, rotary offsets, and
native/fake tensor metadata for PyTorch's custom operators.

For a specific failure, use pytest's normal node IDs or `-k` selection. Save a
complete report when desired:

```bash
mkdir -p test-results
HIP_VISIBLE_DEVICES=0 python -m pytest tests --junitxml=test-results/fmha.xml
```

Failures remain failures. Missing generated kernels are not automatically skipped,
and tolerances are not relaxed to hide kernel issues. The original skips cover
unsupported cache combinations and reference-memory restrictions. Kernel fixes
are separate follow-up work. The C++ tests continue to cover CK functionality
outside this Python API; the existing performance scripts are unchanged.
