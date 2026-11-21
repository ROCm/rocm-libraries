---
name: i8
aliases: [i8, int8, u8, uint8, i16, i32, int32, i64, int64]
family: integer
bits: [8, 16, 32, 64]
variants:
  - {arch: all, encoding: "two's complement (i8, i16, i32, i64); unsigned (u8)"}
status: on develop
related: [platform/python/rocke/helpers/quant.py, platform/python/rocke/core/ir.py, platform/python/rocke/core/lower_llvm.py]
---

# i8, u8 and the wider integers

`i8` is the quantized integer type; `u8` is its unsigned twin; `i32` is the
integer matrix accumulator; `i16` and `i64` are index and address types.
Ranges are in the [Quick values](./README.md#integer-formats) table.

## f32 -> i8 quantization

`quantize_scalar_f32(b, x, *, inv_scale, qdtype="i8")`
([helpers/quant.py:167-232](../../python/rocke/helpers/quant.py)):

```text
scaled  = x * inv_scale
clamped = clamp_f32(scaled, -127, 127)    # NaN -> -127
q       = cvt_f32_to_i8_sat(clamped)      # RNE, then [-128, 127]
```

| input | result |
|---|---|
| NaN | -127 (the clamp's lower bound; i8 has no NaN) |
| +-inf, finite beyond +-127 | +-127 |
| finite in range | round to nearest, ties to even |

The range is the symmetric -127 .. 127 of CK Tile's SmoothQuant, not the
type's -128, so `-q` never overflows. The bound comes from `quant_max_abs("i8")`
(pending: `quant_max_abs("i8", arch=...)`, still 127).

### The raw conversion is defined only in range

`cvt_f32_to_i8_sat` and `cvt_pk_i8_f32x4` lower to `llvm.rint`, `fptosi` to
i32, then `smax` / `smin` to -128 .. 127
([core/lower_llvm.py:3224-3323](../../python/rocke/core/lower_llvm.py)).
LLVM's `fptosi` is **poison** for NaN and for values that do not fit in i32
(LLVM Language Reference), so the "saturation" only covers values within the
i32 range. Clamp in f32 first, as `quantize_scalar_f32` does; `pack_quant_chunk_f32`
([helpers/io.py:285-322](../../python/rocke/helpers/io.py)) leaves that clamp to
its caller.

## u8

`cvt_f32_to_u8_sat` lowers to `rint`, `maxnum(r, 0)`, `minnum(r, 255)`,
`fptoui` ([core/lower_llvm.py:3325-3360](../../python/rocke/core/lower_llvm.py)):
NaN gives 0 (the `maxnum` rule), `-inf` 0, `+inf` 255, RNE in between. It is
defined for every input because the clamp is in f32. `uitofp_f32` is exact.

## Integer arithmetic

- Signed `add`, `sub` and `mul` carry LLVM's `nsw` flag: a signed overflow is
  poison, as in C. Do not rely on wrap-around for signed values
  ([core/lower_llvm.py:2614-2639](../../python/rocke/core/lower_llvm.py)).
- Unsigned (`u8`) arithmetic wraps modulo 256 and uses the unsigned forms of
  division, remainder and compares ([dtype_support.md, section 1](../reference/dtype_support.md#1-the-canonical-dtype-set)).
- Integer matrix accumulation wraps on overflow. rocKE passes `clamp = false`
  to the integer WMMA ([core/isa/matrix_calls.py:271-272](../../python/rocke/core/isa/matrix_calls.py)).
  For the CDNA `V_MFMA_I32_*_I8`, the CDNA4 ISA ties the clamp to
  MODE.FP16_OVFL: "If set, I32 Result is clamped to +/-MAX on
  overflow/underflow, otherwise the carry out bits are dropped". The bit is
  clear in rocKE kernels, and measured on gfx942 an overflowing i32
  accumulator wraps (with the bit set it saturates, one reason rocKE does not
  set it, see the [README](./README.md#the-mode-register)).

## Policy

i8 quantization is symmetric, saturating, RNE, with NaN mapped to -127 (the
clamp's lower bound), through `quantize_scalar_f32`. u8 saturates with NaN to
0. Integer arithmetic is two's complement; signed overflow is undefined and
must be avoided, unsigned wraps, the matrix accumulator wraps.

## Dequantization

`dequantize_scalar_to_f32(b, q, *, scale)`: sign-extend to i32, `sitofp`
(exact for every i8), multiply by the forward scale
([helpers/quant.py:235-274](../../python/rocke/helpers/quant.py)).

## Tests

| rule | test |
|---|---|
| u8 ops lower to their unsigned forms; GPU results match numpy `uint8` | [tests/core/test_u8_arith.py](../../tests/core/test_u8_arith.py) |
| SmoothQuant and RMSNorm-quant emission in both engines | [tests/instances/parity/smoothquant_emit.py](../../tests/instances/parity/smoothquant_emit.py), [tests/instances/parity/add_rmsnorm2d_rdquant_emit.py](../../tests/instances/parity/add_rmsnorm2d_rdquant_emit.py) |
| **pending**: the i8 path clamps without a NaN select and NaN gives -127 on the GPU | `tests/core/test_dtypes.py` on the pending branch |

## How to decide

- **Quantizing to i8**: `quantize_scalar_f32(..., qdtype="i8")`; never call
  `cvt_f32_to_i8_sat` on values that can be NaN or huge.
- **Need -128**: the helpers never produce it; use the raw conversion on
  clamped data and say so in the kernel's contract.
- **Integer overflow**: widen before it can happen; signed overflow is
  undefined, and the matrix accumulator wraps.
