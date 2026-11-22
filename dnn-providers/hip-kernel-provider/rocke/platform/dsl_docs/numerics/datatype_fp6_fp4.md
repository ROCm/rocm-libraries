---
name: fp6_fp4
aliases: [fp6e2m3, fp6, fp6_e2m3, fp6e3m2, bf6, fp6_e3m2, fp4e2m1, fp4, fp4_e2m1]
family: float
bits: [6, 4]
variants:
  - {arch: gfx950, encoding: "OCP MX fp6 e2m3, fp6 e3m2, fp4 e2m1 (f8f6f4 MFMA operands)"}
status: on develop; decode only, no f32 to fp6 or fp4 conversion
related: [platform/python/rocke/helpers/i4_dequant.py, platform/python/rocke/helpers/mx_scale.py]
---

# fp6 and fp4 (OCP MX)

Three OCP MX element formats, always used with an [E8M0 block
scale](./datatype_e8m0_scale.md):

| rocKE name | aliases | format |
|---|---|---|
| `fp6e2m3` | `fp6` | 6 bits, 2 exponent, 3 mantissa |
| `fp6e3m2` | `bf6` | 6 bits, 3 exponent, 2 mantissa |
| `fp4e2m1` | `fp4` | 4 bits, 2 exponent, 1 mantissa |

Constants are in the [Quick values](./README.md#quick-values) table. How they
are packed into bytes and words is in
[dtype_storage_packing.md](../architecture/dtype_storage_packing.md).

## Encoding

Sign, exponent, mantissa; exponent field 0 is zero or subnormal. None of the
three has an infinity or a NaN, and each has -0 (OCP MX specification v1.0).
Every code is a finite number, so a NaN or an infinity can only enter an MX
block through its scale (code 255) or through overflow in the f32
accumulation.

## Conversions

- **Decode (host-checked tables)**: rocKE has no fp4 decoder.
  The 6-bit table `_FP6_CODEBOOK`
  ([helpers/i4_dequant.py:163-181](../../python/rocke/helpers/i4_dequant.py))
  is **e3m2** (bf6) and correct for every
  code, signed zeros and unflushed subnormals included; there is no e2m3
  table. The entry point `unpack_fp6_bytes_to_quad_f32` has width and naming
  defects recorded in [dtype_support.md](../reference/dtype_support.md)
  (section 2.5); do not use it.
- **Encode**: rocKE has no f32 -> fp6 or fp4 conversion. The CDNA4 ISA
  describes its scaled conversions as saturating (see
  [e8m0](./datatype_e8m0_scale.md)); when a
  builder is added it should document that it saturates and how it treats
  NaN, which these formats cannot hold.
- **Matrix cores**: the gfx950 `f8f6f4` MFMAs read these formats directly,
  with or without E8M0 scales, and accumulate in f32.

## Policy

Decoding is exact. Until an encoder exists there is no rounding policy to
state; quantized fp6 / fp4 data comes from the host.

## Tests

| rule | test |
|---|---|
| the fp6 e3m2 table matches the specification for every code; signed zeros; subnormals not flushed; all codes on the GPU | [tests/core/test_fp6_decode.py](../../tests/core/test_fp6_decode.py) |
| gfx950 MX GEMMs with fp4 and fp6 operands | [tests/instances/test_gfx950_mxfp4_numeric.py](../../tests/instances/test_gfx950_mxfp4_numeric.py), [tests/instances/test_gfx950_mxfp6_numeric.py](../../tests/instances/test_gfx950_mxfp6_numeric.py) |

## How to decide

- **Need fp6 e2m3 values on the device**: there is no decoder; add one with
  its own table test rather than reusing the e3m2 table.
- **Need to produce fp4 / fp6**: quantize on the host for now.
