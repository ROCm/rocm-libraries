# Packed FP6 GEMM

The [block-scaled GEMM builder](../../../instances/gfx1250/block_scaled_gemm.py)
supports both six-bit formats on gfx1250: `fp6` / `fp6e2m3` means E2M3,
and `bf6` / `fp6e3m2` means E3M2. Both operands use the same six-bit format. Mixed A/B formats are a separate extension. Native paths use K=128 atoms;
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
the sixteen-word compiler builtin interface. A/B use independent storage descriptors with the same dense format.

The builder uses [`TensorStorage`](../../../core/storage.py),
[`ScaledWmmaOp.matrix_layout`](../../../core/arch/wmma_scale.py), and
[`load_matrix_fragment`](../../../helpers/mma_io.py). Dense six-bit packing
places sixteen elements across three i32 words. The shared loader derives
alignment and assembles each 24-byte chunk from 16-byte and 8-byte loads.
Numerical encoders and the independent reference decoder remain separate from
these unsigned bit-pattern operations.

## E8M0 scales

Native paths use packed E8M0 scale bytes for every matrix pair. The shared
`scale_dtype="e8m0"` selects this contract; `i8` remains its storage alias.

```python
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec, build_block_scaled_gemm,
)

spec = BlockScaledGemmSpec(
    name="mxfp6", M=32, N=48, K=256,
    dtype_a="fp6e2m3", dtype_b="fp6e2m3",
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
isolated K groups, K=256, with E8M0 scales. These are
bounded correctness fixtures; they do not establish arbitrary-input rounding,
E8M0 NaN behavior, or performance.

```sh
ROCKE_LLVM_FLAVOR=llvm23 ROCKE_BACKEND=both ROCKE_CPP_STRICT=1 \
ROCKE_REQUIRE_GFX1250=1 python -m pytest -q -s \
    tests/instances/test_gfx1250_mxfp6_numeric.py
```

Use a matching gfx1250 device, LLVM 23 COMGR/HIP, NumPy/ml_dtypes, and a freshly
built C++ extension. COMGR checks Python/C++ LLVM identity; the HIP verifier
compiles the Python HIP lowerer output. Matrix quantization and conversion,
partial tiles, scalar six-bit arithmetic/conversions, and a dynamic K loop remain
outside this packed-input builder. Logical FP6 types already exist, but the
storage descriptors expand into byte loads and integer carrier vectors before
IR serialization; packed TensorView integration remains separate work.

Run the focused example with:

```sh
python -m rocke.examples.gfx1250.gemm.mxfp6_gemm --dtype fp6
python -m rocke.examples.gfx1250.gemm.mxfp6_gemm --dtype bf6 --matrix-path wmma_scale16 --compile-route hip
```

`mxfp6_gemm` runs both `fp6` (E2M3) and `bf6` (E3M2) by default, as
separate homogeneous cases. Use `--dtype fp6`, `--dtype bf6`, or explicit
`--dtype both` to select the encodings.
