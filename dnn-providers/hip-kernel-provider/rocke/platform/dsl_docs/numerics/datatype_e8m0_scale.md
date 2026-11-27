---
name: e8m0
aliases: [e8m0, fp8_e8m0, e5m3]
family: float
bits: [8]
variants:
  - {arch: gfx950, encoding: "OCP MX E8M0 (scaled MFMA operand)"}
  - {arch: all, encoding: "OCP MX E8M0 (software decode)"}
status: on develop; e5m3 recognized only
related: [platform/python/rocke/helpers/mx_scale.py, platform/python/rocke/instances/common/mx_gemm.py, platform/python/rocke/core/ir.py]
---

# e8m0 block scale (and e5m3)

E8M0 is the OCP MX shared scale: one byte per block of 32 elements (fp8,
bf8, fp6 or fp4), an unsigned biased exponent with no sign and no mantissa.
Constants are in the [Quick values](./README.md#quick-values) table.

## Encoding

Code `c` from 0 to 254 is exactly `2^(c - 127)`; code 255 is NaN (OCP MX
specification v1.0). There is no zero, no negative scale and no infinity.
Code 0 is `2^-127`, an f32 **subnormal**: decoding it needs subnormals kept,
which rocKE does by default ([README](./README.md#subnormals)). Code 127 is
1.0.

`e5m3` is a different 8-bit scale format that the dtype table recognizes
([core/dtypes.py:86-121](../../python/rocke/core/dtypes.py)); its numerics are
not documented here yet.

## Decoding and applying scales

| helper | what it does | special values |
|---|---|---|
| `decode_mx_scale_e8m0(b, byte)` | reads the byte unsigned, `exp2(c - 127)`, then selects NaN for 255 | exact for every code, 0 included; 255 gives NaN |
| `add_mx_scaled_dot(...)` | `acc + ldexp(dot, ea + eb - 254)` per element: one rounding, no product of decoded scales | NaN if either code is 255; a term overflows to +-inf only if the scaled dot product itself is beyond the f32 range |
| `largest_safe_mx_scale_code(a_fmt, b_fmt, products)` | the largest code both scales may have so that a scaled MFMA's partial sums stay at most `2^127` | uses each element format's max ([Quick values](./README.md#quick-values)) |
| `cvt_scalef32_pk_f32_fp8x4` / `_bf8x4` (gfx950) | fp8 / bf8 decode times a scale | reads only the exponent bits of the f32 scale, so it applies an E8M0 scale, not an arbitrary f32 |

Sources: [helpers/mx_scale.py:98-223](../../python/rocke/helpers/mx_scale.py),
[core/ir.py:3443-3500](../../python/rocke/core/ir.py).

**Why the safe code exists.** The gfx950 scaled MFMA multiplies every product
by `2^(ea + eb - 254)` before summing. With large codes two products of
opposite sign can each overflow, and `inf - inf` is NaN although the true sum
is an ordinary f32. `mx_gemm` tracks the largest code a wave saw and, above
the safe code, recomputes with `add_mx_scaled_dot`
(`build_mx_gemm` in [instances/common/mx_gemm.py](../../python/rocke/instances/common/mx_gemm.py)).
The bound depends on the element format's max, which for fp8 depends on the
target ([Quick values](./README.md#quick-values)).

## Conversion to fp4 and fp6 with a scale

The CDNA4 ISA (section 7.3) describes the scaled conversions to fp4 / fp6 as
saturating (out of range gives +-max, below the smallest subnormal gives 0),
unlike the fp8 conversions; this has not been measured on hardware. rocKE has no builder for them yet
([datatype_fp6_fp4.md](./datatype_fp6_fp4.md)).

## Policy

A NaN scale makes its whole block NaN. rocKE does not produce E8M0 scales
on the device today; a host that computes them chooses the exponent so the
block's max element fits the element format.

## Tests

| rule | test |
|---|---|
| all 256 codes decode to the specification value on the GPU; the byte is read unsigned in both engines | [tests/core/test_mx_scale_decode.py](../../tests/core/test_mx_scale_decode.py) |
| `add_mx_scaled_dot` uses `ldexp` and selects NaN; the safe code keeps sums a binade below overflow | same file |
| MX GEMM special values, codes at the safe bound, per-row and per-column scaling | [tests/instances/test_mx_gemm_scale_range.py](../../tests/instances/test_mx_gemm_scale_range.py) |

## How to decide

- **Turning a scale byte into a multiplier**: `decode_mx_scale_e8m0`; never
  `2 ** (byte - 127)` on a signed byte.
- **Applying two scales to a dot product**: `add_mx_scaled_dot`, not the
  product of two decoded scales, which can overflow or underflow on its own.
- **Feeding scales to a scaled MFMA**: check them against
  `largest_safe_mx_scale_code` for the element formats of the target.
- **An arbitrary f32 scale**: decode, then `fmul`; the `scalef32` conversions
  do not apply one.
