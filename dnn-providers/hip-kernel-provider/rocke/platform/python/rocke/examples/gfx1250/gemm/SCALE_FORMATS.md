# Independent scale formats on gfx1250

The [block-scaled GEMM builder](../../../instances/gfx1250/block_scaled_gemm.py)
accepts independent A/B scale-format selections on its native SCALE/SCALE16
paths. See [packed FP6](FP6.md) and [packed FP4](FP4_SCALE.md) for matrix storage.

## Contents

- [Scale formats](#scale-formats)
- [Evidence and planned experiment](#evidence-and-planned-experiment)

## Scale formats

`scale_dtype` remains the shared default. Optional `scale_dtype_a` and
`scale_dtype_b` override it independently. All native scale buffers contain
encoded bytes; choosing a format does not quantize or convert the buffer.

| Matrix operand (A or B) | Scale dtype (independent for A and B) |
| --- | --- |
| FP8 / BF8 / FP6 / BF6 | E8M0 |
| FP4 | E8M0, E5M3, or E4M3 |

Here FP8 and FP6 each include both exponent/mantissa formats. E8M0 has selector
0, E5M3 selector 1, and E4M3 selector 2. The reference decoder interprets E5M3
as an unsigned eight-bit scale format with bias 15, distinct from E5M2.
A and B may use different scale formats when only one operand is FP4.
FP4 x FP4 requires matching scale formats. Both SCALE and SCALE16 use these
rules, giving 43 combinations per mode and 86 scaled atoms in the catalog.
The filter follows [LLVM's supported-combination validation](https://github.com/llvm/llvm-project/commit/7597c0c938b364306cbee21dd03835ff5b699c74),
including on older compilers that accept unsupported combinations without a diagnostic.
Standard MXFP formats use E8M0 scales over K32 blocks; E4M3/E5M3
and K16 variants are broader block-scaled contracts.
The `i8` scale spelling remains an alias for E8M0 for existing callers.

```python
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec, build_block_scaled_gemm,
)

spec = BlockScaledGemmSpec(
    name="mixed_fp6_fp4", M=32, N=48, K=256,
    dtype_a="fp6e2m3", dtype_b="fp4",
    matrix_path="wmma_scale16", block_k=16,
    scale_dtype="e8m0", scale_dtype_a="e8m0", scale_dtype_b="e4m3",
)
kernel = build_block_scaled_gemm(spec)
```

The atom catalog owns the scale formats and K-group size. Query
`target.mma.op_for_shape(family="wmma_scaled", ..., scales=("e8m0", "e4m3", 16))`
and pass the returned `op_id` to the ordinary `IRBuilder.mma` / `rocke_b_mma`
APIs. Python and native lowering derive both hardware selectors from that
contract. Existing E8M0 calls keep their operation IDs and emitted code.
Nondefault scale types are included in generated kernel names. The builder
accepts the existing `fp8e4m3` scale alias for E4M3; E5M2 matrix spellings are
not aliases for the unsigned E5M3 scale format.

## Evidence and planned experiment

The public [machine-readable ISA](https://gpuopen.com/download/machine-readable-isa/latest/)
2026-08-06 snapshot's `amdgpu_isa_cdna5.xml` defines SCALE/SCALE16 operand widths
and block sizes, but does not enumerate E5M3 scale decoding or the complete
scale legality table. LLVM's `WMMA::MatrixScaleFmt` defines the selector names
and values used here. The linked LLVM commit adds the supported-combination
predicate to both backend lowering and the assembler. The implementation uses
that contract for catalog admission and independently validates it before lowering.
Compiler acceptance alone does not establish numerical behavior.

Existing mixed-format fixtures cover bounded finite values and compare against
an independent host calculation. They do not establish arbitrary-input rounding,
scale NaN behavior, or performance. The [scale-format discrimination experiment](SCALE_FORMAT_EXPERIMENT.md)
describes comparisons of identical raw scale bytes against literal predictions.
The fixed-byte numeric tests sweep the three selectors on FP4 with a non-FP4
E8M0 partner, and keep E8M0 controls for FP8/BF8/FP6/BF6 operands.

```sh
ROCKE_LLVM_FLAVOR=llvm23 ROCKE_BACKEND=both ROCKE_CPP_STRICT=1 \
ROCKE_REQUIRE_GFX1250=1 python -m pytest -q -s \
    tests/instances/test_gfx1250_scale_formats_numeric.py
```

Use a matching gfx1250 device, one release-matched COMGR/HIP stack, NumPy/ml_dtypes, and a freshly
built C++ extension. COMGR checks Python/C++ LLVM identity; HIP fixtures compile
Python-generated HIP. Choosing a scale format does not quantize or convert input
buffers. The rocKE emission flavor is `llvm23`; the consuming compiler may be
LLVM 23 or 24, as in ROCm 10.0 and 10.1 respectively.
