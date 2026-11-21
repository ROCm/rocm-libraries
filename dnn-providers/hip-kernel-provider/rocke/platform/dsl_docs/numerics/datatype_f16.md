---
name: f16
aliases: [f16, fp16, half, float16]
family: float
bits: [16]
variants:
  - {arch: all, encoding: "IEEE 754 binary16"}
status: on develop
related: [platform/python/rocke/core/ir.py, platform/python/rocke/core/lower_llvm.py]
---

# f16

IEEE 754 binary16, a storage and matrix-operand type. Constants are in the
[Quick values](./README.md#quick-values) table; shared rules are in the
[README](./README.md#shared-rules).

## Conversions

| builder op | lowers to | rounding | NaN | +-inf | f32 above 65504 |
|---|---|---|---|---|---|
| `cast_f32_to(v, F16)`, `trunc_f32_to_f16` | `fptrunc float to half` (`v_cvt_f16_f32`) | RNE | NaN | +-inf | +-inf (from 65520, half a step above 65504) |
| `cast_to_f32` | `fpext half to float` | exact | NaN | +-inf | - |

Sources: [core/ir.py:3239-3325](../../python/rocke/core/ir.py),
[core/lower_llvm.py:2929-2951](../../python/rocke/core/lower_llvm.py).

The conversion does not saturate: rocKE leaves MODE.FP16_OVFL clear
([README](./README.md#the-mode-register)), so f16 overflow, in a conversion or
in f16 arithmetic, gives +-inf as IEEE 754 requires. f16 subnormals are kept
(`.amdhsa_float_denorm_mode_16_64 3`), so f32 values below 2^-14 become f16
subnormals and those below half of 2^-24 become +-0.

There is no direct f16 <-> bf16 conversion; go through f32. Each step that
narrows rounds once, and widening is exact, so the route rounds only once.
i4 values dequantize to f16 exactly
([datatype_int4.md](./datatype_int4.md)).

## Matrix operands

f16 matrix inputs follow the [matrix-core subnormal rules](./README.md#subnormals)
and accumulate as the [README](./README.md#accumulation) describes.

## Policy

rocKE follows IEEE 754 for f16: RNE, overflow to +-inf, NaN kept, subnormals
kept. There is no saturating f16 conversion. A kernel that must not produce
+-inf from a large f32 value clamps to +-max ([Quick values](./README.md#quick-values))
first, keeping NaN with a select if NaN must survive
([README](./README.md#nan-and-infinity-in-arithmetic)).

## Tests

No device test pins the f16 conversion's special-value results yet; the
behaviour above comes from the compiled instruction (`v_cvt_f16_f32`) and the
descriptor fields. Host references use numpy's native `float16`, which rounds
the same way.

## How to decide

- **Storing an f32 result as f16**: `cast_f32_to(v, F16)`. Values above
  65504 become +-inf; clamp first only if the consumer cannot take infinity.
- **Mixing f16 and bf16**: widen both to f32.
