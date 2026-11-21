---
name: tf32
aliases: [tf32, xf32]
family: float
bits: [32]
variants:
  - {arch: gfx942, encoding: "native XF32 matrix input"}
  - {arch: gfx950, encoding: "emulated by three bf16 MFMAs"}
status: on develop
related: [platform/python/rocke/core/ir.py, platform/python/rocke/core/tf32.py, platform/python/rocke/core/lower_llvm.py]
---

# tf32 / xf32

A float with f32's sign and 8 exponent bits but only 10 mantissa bits,
carried in a 32-bit word whose low 13 bits the matrix instruction ignores.
`tf32` and `xf32` name the same format. It is only a matrix-operand format:
generic tf32 arithmetic is refused. Constants are in the
[Quick values](./README.md#quick-values) table.

## Preparing operands

| builder op | what it does | rounding | special values |
|---|---|---|---|
| `cvt_f32_to_tf32(v)` | rounds the f32 to 10 mantissa bits in integer arithmetic and zeroes the low 13 bits | RNE | +-inf and signed zero kept; NaN quieted, upper payload kept; finite values that round past the tf32 max become +-inf |
| `bitcast(v, TF32)` | wraps the f32 bits unchanged | none here; the instruction ignores the low bits | as the instruction treats them, see below |

Source: [core/ir.py:3269-3310](../../python/rocke/core/ir.py).

**Raw bits are truncated, not rounded.** On gfx942, both native XF32 shapes
accepted raw f32 payloads, and in the measured corpus their results matched
masking the low 13 bits; a signaling NaN whose payload was only in those bits
came out as infinity. Prepare with `cvt_f32_to_tf32` when the contract is
RNE inputs or NaN must stay NaN. The evidence and corpus are recorded in the
[gfx942 TF32 numerics example](../../python/rocke/examples/gfx942/tf32_numerics/README.md).

## Per target

- **gfx942**: native `V_MFMA_F32_*_XF32`; f32 accumulator; subnormal inputs
  as in the [matrix-core rules](./README.md#subnormals).
- **gfx950**: no XF32 instruction. The catalog's two tf32 atoms split each
  operand into two bf16 parts (a rounded leading part and the remainder) and
  run three bf16 MFMAs on them. The split caps the leading part at the largest
  finite bf16, so that rounding near the f32 max cannot create
  `inf * 0 = NaN` in the cross terms; an infinity or NaN in an operand still
  makes every output it feeds non-finite
  ([core/lower_llvm.py:4059-4091](../../python/rocke/core/lower_llvm.py)).
  Magnitudes at or below 2^-134 split to zero, so the smallest subnormals are
  lost. The precision contract is in
  [xf32_bf16x3_emulation.md](../architecture/xf32_bf16x3_emulation.md).
- **Other targets**: the atoms are refused with the catalog's error
  ([core/tf32.py](../../python/rocke/core/tf32.py)).

## Policy

Inputs: RNE when prepared with `cvt_f32_to_tf32`, the instruction's
truncation when passed raw. Accumulation and results: f32, IEEE.

## Tests

| rule | test |
|---|---|
| RNE preparation at ties, carries and special values | [tests/core/test_tf32.py](../../tests/core/test_tf32.py) (`test_rne_reference_edges`) |
| the numeric probe's checks reject a bad corpus | [tests/core/test_tf32_numerics_validation.py](../../tests/core/test_tf32_numerics_validation.py) |
| native gfx942 results | [tests/numeric/test_gfx942_tf32_numeric.py](../../tests/numeric/test_gfx942_tf32_numeric.py) |
| BF16x3 split, term order and accuracy | [tests/core/test_xf32_bf16x3_numerics.py](../../tests/core/test_xf32_bf16x3_numerics.py) |

## How to decide

- **Want RNE tf32 inputs**: `cvt_f32_to_tf32`.
- **Want the instruction's own behaviour** (match another library that feeds
  raw f32): `bitcast`, and accept truncation and the NaN-to-infinity case.
- **Need the same answer on gfx942 and gfx950**: the emulation is not
  bit-identical to native XF32; compare with a tolerance.
