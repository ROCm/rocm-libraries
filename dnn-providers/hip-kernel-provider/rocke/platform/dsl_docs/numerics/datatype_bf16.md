---
name: bf16
aliases: [bf16, bfloat16]
family: float
bits: [16]
variants:
  - {arch: all, encoding: "bfloat16 (upper half of IEEE binary32)"}
status: on develop
related: [platform/python/rocke/core/lower_llvm.py, platform/python/rocke/numeric/bf16.py]
---

# bf16

bfloat16: the upper 16 bits of an IEEE binary32, so it has f32's exponent
range with 7 stored mantissa bits. Constants are in the
[Quick values](./README.md#quick-values) table; shared rules are in the
[README](./README.md#shared-rules).

## Conversions

| builder op | lowers to | rounding | NaN | +-inf | overflow |
|---|---|---|---|---|---|
| `cast_f32_to(v, BF16)`, `trunc_f32_to_bf16` | `fptrunc float to bfloat` | RNE | quiet NaN | +-inf | +-inf for f32 bit patterns from `0x7f7f8000` up |
| `cast_to_f32` | `fpext bfloat to float` | exact (shift left 16) | NaN | +-inf | - |

Sources: the builders in [core/ir.py](../../python/rocke/core/ir.py) and
their `_op_arith_*` lowerings in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py); the
overflow threshold is the one the `_split_f32_to_bf16` docstring states in
[core/lower_llvm.py](../../python/rocke/core/lower_llvm.py).

What `fptrunc` becomes depends on the target:

- **gfx942** has no f32 -> bf16 instruction. The compiler emits an integer
  sequence: add `0x7fff` plus the lowest kept bit (round to nearest even),
  and for NaN take the upper half with the quiet bit (`0x400000`) set instead,
  so a NaN cannot round into an infinity.
- **gfx950** uses `v_cvt_pk_bf16_f32`.

Both round finite values to nearest even (the instruction sequences were read
from a rocKE kernel compiled with the ROCm 7.1 `llc` for both targets; see the
[README](./README.md#the-mode-register) for the recipe). How the gfx950
instruction treats NaN payloads and f32 subnormal inputs has not been
verified on hardware. A guard-free variant that skips the NaN check exists only as a kernel
lever in attention backward, with NaN results unspecified
([attention_bwd.md](../instances/attention_bwd.md)).

## Host codec

[numeric/bf16.py](../../python/rocke/numeric/bf16.py) is the torch-free host
encoder: RNE by adding the tie bias, NaN forced quiet with its sign and the
payload bits that fit, matching `ml_dtypes.bfloat16` for every non-NaN value.
Use it, not a hand-written shift, for reference inputs.

## Policy

IEEE-style: RNE, overflow to +-inf, NaN kept (quieted), subnormals kept.
Truncating instead of rounding is not an accepted conversion in rocKE: it
biases every value toward zero.

## Known differences from other libraries

`ml_dtypes` canonicalises every NaN to `0x7fc0` / `0xffc0`; rocKE's host
codec keeps the payload bits that fit
([numeric/bf16.py](../../python/rocke/numeric/bf16.py)).

## Tests

| rule | test |
|---|---|
| host codec: RNE, carry into the exponent, NaN stays NaN, infinity and signed zero kept, matches ml_dtypes | [tests/numeric/test_bf16_codec.py](../../tests/numeric/test_bf16_codec.py) |
| device conversion special values | none yet |

## How to decide

- **Storing f32 as bf16**: `cast_f32_to(v, BF16)`; never mask or shift the
  bits yourself.
- **Building bf16 reference data on the host**: `numeric.bf16.f32_to_bf16`.
- **bf16 used to emulate higher precision** (the XF32 BF16x3 split): see
  [datatype_tf32_xf32.md](./datatype_tf32_xf32.md); that path caps values
  instead of overflowing.
