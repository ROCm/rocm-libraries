---
name: f64
aliases: [f64, fp64, double, float64]
family: float
bits: [64]
variants:
  - {arch: all, encoding: "IEEE 754 binary64"}
status: on develop
related: [platform/python/rocke/core/ir.py, platform/python/rocke/core/lower_llvm.py]
---

# f64

IEEE 754 binary64. Constants are in the [Quick values](./README.md#quick-values)
table; shared rules are in the [README](./README.md#shared-rules). Which ops
and kernel families accept f64 is in
[dtype_support.md](../reference/dtype_support.md) (section 2.5).

## Conversions

| builder op | lowers to | rounding | special values |
|---|---|---|---|
| `cvt_f32_to_f64` | `fpext float to double` | exact | NaN, +-inf, -0 and subnormals kept |
| `cvt_f64_to_f32` | `fptrunc double to float` | RNE | NaN kept; beyond the f32 range gives +-inf; below half the smallest f32 subnormal gives +-0 |

Sources: [core/ir.py:3344-3364](../../python/rocke/core/ir.py),
[core/lower_llvm.py:2953-2964](../../python/rocke/core/lower_llvm.py).
`cast_to_f32` and `cast_f32_to` refuse f64, so a double is never narrowed by
accident; there is no direct f64 <-> f16 or bf16 conversion.

## Arithmetic and matrix operands

f64 arithmetic is IEEE, with subnormals kept ([README](./README.md#the-mode-register)).
The f64 MFMA accumulates in f64 and follows the
[matrix-core rules](./README.md#subnormals).

## Policy

IEEE 754 throughout: RNE, overflow to +-inf, NaN kept, subnormals kept.

## Tests

| rule | test |
|---|---|
| each conversion is one LLVM instruction; the hub casts refuse f64 | [tests/core/test_f64_conversions.py](../../tests/core/test_f64_conversions.py) |
| f64 arithmetic and conversions are exact on edge values, on the GPU | same file (`test_f64_arithmetic_and_conversions_are_exact_on_edge_values_on_gpu`) |
| f64 MFMA and GEMM results | [tests/numeric/test_gfx942_f64_mma_numeric.py](../../tests/numeric/test_gfx942_f64_mma_numeric.py), [tests/numeric/test_f64_gemm_numeric.py](../../tests/numeric/test_f64_gemm_numeric.py) |

## How to decide

- **Narrowing to f32**: `cvt_f64_to_f32`; values beyond the f32 range become
  +-inf.
- **Narrowing to f16 or bf16**: go through f32. This rounds twice, which can
  differ from a single rounding at ties; if that matters, say so in the
  kernel's contract.
