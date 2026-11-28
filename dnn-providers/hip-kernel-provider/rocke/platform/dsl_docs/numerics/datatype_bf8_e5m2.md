---
name: bf8e5m2
aliases: [bf8e5m2, bf8, fp8_e5m2, fp8e5m2, fp8_e5m2_fnuz, float8_e5m2, float8_e5m2fnuz]
family: float
bits: [8]
variants:
  - {arch: gfx942, encoding: "e5m2 FNUZ"}
  - {arch: gfx950, encoding: "OCP e5m2"}
  - {arch: gfx1201, encoding: "OCP e5m2"}
  - {arch: gfx12-generic, encoding: "OCP e5m2"}
status: on develop
related: [platform/python/rocke/helpers/quant.py, platform/python/rocke/core/ir.py, platform/python/rocke/numeric/fp8.py]
---

# bf8 e5m2 (`bf8e5m2`)

An 8-bit float with 5 exponent and 2 mantissa bits: more range, less
precision than [fp8e4m3](./datatype_fp8_e4m3.md). Its constants are in the
[Quick values](./README.md#quick-values) table. Everything not listed here is
as in the fp8e4m3 file, which is the home of the shared 8-bit rules: which
target uses which encoding, the conversion builders, the quantize and
dequantize helpers, MMA operands and how to decide.

## Encoding

`s.eeeee.mm`. Exponent field 0 is zero or subnormal.

- **OCP e5m2** (`float8_e5m2`): bias 15 and IEEE-like special values: the
  all-ones exponent holds +-inf (mantissa 0) and NaN (mantissa not 0). `0x80`
  is -0.
- **e5m2 FNUZ** (`float8_e5m2fnuz`): bias 16, no infinity, no -0. `0x80` is
  the only NaN, and the all-ones exponent holds ordinary numbers, which is why
  the largest value is the same 57344 as OCP.

Which target uses which: [per-target table](./README.md#encoding-per-target-fp8-and-bf8).

## Conversions

The builders mirror fp8e4m3: `cvt_f32_to_bf8`, `cvt_pk_bf8_f32x4`,
`cvt_bf8_to_f32`, `cvt_pk_f32_bf8x4` and, on gfx950 only,
`cvt_scalef32_pk_f32_bf8x4` ([core/ir.py](../../python/rocke/core/ir.py)).
Decoding is exact. The f32 -> bf8 conversion does not saturate (MODE.FP16_OVFL
is 0):

| input | gfx950 (OCP) | gfx942 (FNUZ) |
|---|---|---|
| NaN | NaN | NaN (`0x80`) |
| +-inf | +-inf | NaN |
| finite past 57344 after rounding (from 61440: the tie rounds up, because 57344's mantissa is odd) | +-inf | NaN |

The OCP column is OFP8 Table 3 (non-saturating e5m2) and the ISA table
(CDNA4 ISA section 7.3), and gfx950 measures it
([README](./README.md#the-mode-register)): NaN gives NaN, +-inf gives +-inf,
61440 and 1e30 give +inf, 57344 stays 57344 and +-10000 gives +-10240.
Negative finite overflow was not measured. FNUZ has no infinity:
the CDNA3 ISA's format table (section 7.2, Table 30) gives `0x80` as the BF8
"INF" code, the same code as NaN, so the gfx942 conversion gives NaN where the
conversion table lists +-inf. Measured on gfx942: +-inf and values from 61440
give `0x80`; 57344 to 61439 give `0x7f`. The `cvt_f32_to_bf8` docstring
states the same ([core/ir.py](../../python/rocke/core/ir.py)).

## Policy

Same as [fp8e4m3](./datatype_fp8_e4m3.md#policy): NaN stays NaN, +-inf and
finite overflow give +-max, which is 57344 in both encodings, RNE, and FNUZ
never produces `0x80` for a zero. The
[packed routes](./datatype_fp8_e4m3.md#packed-routes) apply no clamp here
either.
rocKE's policy therefore differs from OCP's non-saturating e5m2, where
infinity survives: a quantized bf8 tensor from rocKE holds no infinities.

## Known differences from other libraries

| library (public source) | difference from rocKE's policy |
|---|---|
| HIP `amd_hip_fp8.h` with `__HIP_SATFINITE`; Composable Kernel `include/ck/utility/amd_ck_fp8.hpp` | OCP e5m2: +-inf stays +-inf; FNUZ: +-inf gives NaN |
| PyTorch `.to(float8_e5m2)` (`torch/headeronly/util/Float8_e5m2.h`); ml_dtypes (`ml_dtypes/include/float8.h`) | non-saturating: +-inf and overflow give +-inf |
| PyTorch `.to(float8_e5m2fnuz)` (`Float8_e5m2fnuz.h`) | non-saturating: +-inf and overflow give NaN |

The NaN rows of the [fp8e4m3 table](./datatype_fp8_e4m3.md#known-differences-from-other-libraries)
(vLLM, Triton `tl.clamp`) apply here too.

## Tests

| rule | test |
|---|---|
| OCP e5m2 decode table: infinities and NaNs at the all-ones exponent; OCP and FNUZ tables match ml_dtypes | [tests/numeric/test_fp8_codec.py](../../tests/numeric/test_fp8_codec.py) |
| bare conversion on the GPU (OCP gives `0x7c` / `0xfc` for overflow and +-inf); `quantize_scalar_f32` saturates and keeps NaN | [tests/core/test_dtypes.py](../../tests/core/test_dtypes.py) (`test_fp8_conversion_does_not_saturate_on_gpu`, `test_quantize_scalar_f32_saturates_and_keeps_nan_on_gpu`) |

## How to decide

As for [fp8e4m3](./datatype_fp8_e4m3.md#how-to-decide). One addition: a
consumer that expects OCP e5m2 infinities (for example a framework tensor
compared with `isinf`) will not see them in rocKE's quantized output.
