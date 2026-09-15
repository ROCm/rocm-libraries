# Packed FP6 and mixed matrix formats

The [block-scaled GEMM builder](../../../instances/gfx1250/block_scaled_gemm.py)
supports both six-bit formats on gfx1250: `fp6` / `fp6e2m3` means E2M3,
and `bf6` / `fp6e3m2` means E3M2. A and B can independently select FP8 E4M3,
FP8 E5M2, FP6 E2M3, FP6 E3M2, or FP4 E2M1. Native paths use K=128 atoms;
`wmma_scale` groups scales by K=32 and `wmma_scale16` by K=16.

## Contents

- [Packed storage](#packed-storage)
- [E8M0 scales](#e8m0-scales)
- [ISA and validation references](#isa-and-validation-references)

## Packed storage

FP6 buffers are byte arrays with shapes `[M, 3*K/4]` and `[N, 3*K/4]`.
Four consecutive six-bit codes occupy three bytes, low bits first. Codes
cross byte and word boundaries; they are not stored one per byte. Both formats
have a sign at bit 5, finite exponent encodings, subnormals, and signed zero.
E2M3 has exponent bias 1; E3M2 has bias 3.

A lane with half index `h = lane // 16` consumes K ranges
`[32*h, 32*h+32)` and `[64+32*h, 96+32*h)` for each atom. Each range occupies
24 bytes. The resulting twelve i32 words are padded with four zero words for
the sixteen-word compiler builtin interface. FP8 and FP4 keep their existing
packing. Mixed operands use their own row strides and fragment layouts.

## E8M0 scales

Native paths use packed E8M0 scale bytes for every matrix pair. The shared
`scale_dtype="e8m0"` selects this contract; `i8` remains its storage alias.

```python
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec, build_block_scaled_gemm,
)

spec = BlockScaledGemmSpec(
    name="mixed_fp6_fp4", M=32, N=48, K=256,
    dtype_a="fp6e2m3", dtype_b="fp4",
    matrix_path="wmma_scale16", block_k=16, scale_dtype="e8m0",
)
kernel = build_block_scaled_gemm(spec)
```

## ISA and validation references

The [AMD machine-readable ISA](https://gpuopen.com/download/machine-readable-isa/latest/) snapshot
dated 2026-08-06 contains `amdgpu_isa_cdna5.xml`, which defines `FMT_NUM_FP6`,
`FMT_NUM_BF6`, their packed fields, and SCALE/SCALE16 operand widths. The atom
ABI uses sixteen i32 words for each matrix input. The scale operands are B32
for SCALE and B64 for SCALE16.

CPU tests compare all 64 encodings of each FP6 format with an independent
numeric dtype and check packed-byte boundaries. GPU tests exercise all code
products at lane/K-group boundaries, neutral and independent A/B scales,
isolated K groups, K=256, and mixed matrix formats with E8M0 scales. These are
bounded correctness fixtures; they do not establish arbitrary-input rounding,
E8M0 NaN behavior, or performance.

```sh
ROCKE_LLVM_FLAVOR=llvm23 ROCKE_BACKEND=both ROCKE_CPP_STRICT=1 \
ROCKE_REQUIRE_GFX1250=1 python -m pytest -q -s \
    tests/instances/test_gfx1250_scaled_wmma_numeric.py
```

Use a matching gfx1250 device, LLVM 23 COMGR/HIP, NumPy/ml_dtypes, and a freshly
built C++ extension. COMGR checks Python/C++ LLVM identity; the HIP verifier
compiles the Python HIP lowerer output. Matrix quantization and conversion,
partial tiles, logical six-bit IR scalar types, and a dynamic K loop remain
outside this packed-input builder.
