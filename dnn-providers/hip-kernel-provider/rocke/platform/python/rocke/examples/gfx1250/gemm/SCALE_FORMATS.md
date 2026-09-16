# Independent scale formats on gfx1250

The [block-scaled GEMM builder](../../../instances/gfx1250/block_scaled_gemm.py)
accepts independent A/B scale-format selections on its native SCALE/SCALE16
paths. See [packed FP6 and mixed matrix formats](FP6.md) for matrix storage.

## Contents

- [Scale formats](#scale-formats)
- [Evidence and planned experiment](#evidence-and-planned-experiment)

## Scale formats

`scale_dtype` remains the shared default. Optional `scale_dtype_a` and
`scale_dtype_b` override it independently. All native scale buffers contain
encoded bytes; choosing a format does not quantize or convert the buffer.

| A data | A scale | B data | B scale |
| --- | --- | --- | --- |
| FP8 / FP6 / FP4 | E8M0 | FP8 / FP6 / FP4 | E8M0 |
| FP8 / FP6 | E8M0 | FP4 | E5M3 or E4M3 |
| FP4 | E5M3 or E4M3 | FP8 / FP6 | E8M0 |
| FP4 | E5M3 | FP4 | E5M3 |
| FP4 | E4M3 | FP4 | E4M3 |

Here FP8 and FP6 each include both exponent/mantissa formats. E8M0 has selector
0, E5M3 selector 1, and E4M3 selector 2. The reference decoder interprets E5M3
as an unsigned eight-bit scale format with bias 15, distinct from E5M2.
The implementation requires matching scale formats for FP4 x FP4.
The `i8` scale spelling remains an alias for E8M0 for existing callers.

```python
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec, build_block_scaled_gemm,
)

spec = BlockScaledGemmSpec(
    name="mixed_fp6_fp4", M=32, N=48, K=256,
    dtype_a="fp6e2m3", dtype_b="fp4",
    matrix_path="wmma_scale16", block_k=16,
    scale_dtype="e8m0", scale_dtype_b="e4m3",
)
kernel = build_block_scaled_gemm(spec)
```

The atom API exposes the same selectors through
`IRBuilder.mma(..., scale_dtype_a=..., scale_dtype_b=...)` and
`rocke_b_mma_scaled`. Existing calls keep E8M0 and unchanged emitted code.
Nondefault scale types are included in generated kernel names.

## Evidence and planned experiment

The public [machine-readable ISA](https://gpuopen.com/download/machine-readable-isa/latest/)
2026-08-06 snapshot's `amdgpu_isa_cdna5.xml` defines SCALE/SCALE16 operand widths
and block sizes, but does not enumerate E5M3 scale decoding or the complete
scale legality table. LLVM's `WMMA::MatrixScaleFmt` defines the selector names
and values used here. The table above describes this implementation's accepted
combinations; compiler acceptance alone does not establish numerical behavior.

Existing mixed-format fixtures cover bounded finite values and compare against
an independent host calculation. They do not establish arbitrary-input rounding,
scale NaN behavior, or performance. The [scale-format discrimination experiment](SCALE_FORMAT_EXPERIMENT.md)
is staged separately and has not been run. It compares identical raw scale bytes
against literal predictions and verifies the emitted instruction fields.

```sh
ROCKE_LLVM_FLAVOR=llvm23 ROCKE_BACKEND=both ROCKE_CPP_STRICT=1 \
ROCKE_REQUIRE_GFX1250=1 python -m pytest -q -s \
    tests/instances/test_gfx1250_scale_formats_numeric.py
```

Use a matching gfx1250 device, LLVM 23 COMGR/HIP, NumPy/ml_dtypes, and a freshly
built C++ extension. COMGR checks Python/C++ LLVM identity; HIP fixtures compile
Python-generated HIP. Choosing a scale format does not quantize or convert input
buffers.
