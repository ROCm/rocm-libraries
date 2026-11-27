---
name: fp8e4m3
aliases: [fp8e4m3, fp8, e4m3, fp8_e4m3, fp8_e4m3_fnuz, float8_e4m3fn, float8_e4m3fnuz]
family: float
bits: [8]
variants:
  - {arch: gfx942, encoding: "e4m3 FNUZ"}
  - {arch: gfx950, encoding: "OCP e4m3"}
  - {arch: gfx1201, encoding: "OCP e4m3"}
  - {arch: gfx12-generic, encoding: "OCP e4m3"}
status: on develop
related: [platform/python/rocke/helpers/quant.py, platform/python/rocke/core/ir.py, platform/python/rocke/helpers/mx_scale.py, platform/python/rocke/instances/common/mx_gemm.py, platform/python/rocke/numeric/fp8.py]
---

# fp8 e4m3 (`fp8e4m3`)

An 8-bit float with 4 exponent and 3 mantissa bits, used for quantized
activations, weights and K/V caches. Its constants are in the
[Quick values](./README.md#quick-values) table (two rows: OCP and FNUZ). This
file is also the home of the rules shared by both 8-bit floats; the
[bf8e5m2 file](./datatype_bf8_e5m2.md) lists only where e5m2 differs.

## Encoding

`s.eeee.mmm`. Exponent field 0 is zero or subnormal; every other field is
normal. Two encodings use this layout:

- **OCP e4m3** (PyTorch and ml_dtypes `float8_e4m3fn`): bias 7. There is no
  infinity; the all-ones pattern `S.1111.111` is NaN, so the top exponent
  still holds seven finite values per sign. `0x80` is -0.
- **e4m3 FNUZ** (`float8_e4m3fnuz`): bias 8, one higher, so a code that is
  finite in both encodings is worth half as much in FNUZ. There is no infinity
  and no -0: `0x80`, the pattern that would be -0, is the only NaN. All other
  codes are finite, so the largest is `0x7f`.

## Which target uses which

One IR type, `fp8e4m3`, carries both; the kernel's target decides what a byte
means ([per-target table](./README.md#encoding-per-target-fp8-and-bf8)). Read
under the other encoding, a finite code is off by a factor of two and the
special codes differ (`0x80` is NaN in FNUZ but -0 in OCP; `0x7f` is NaN in
OCP but the FNUZ max), with no error. Nothing in the IR type records which
encoding a buffer holds.

- The per-target table has one home, `fp8_element_format(dtype, arch)` in
  [helpers/fp8_encoding.py](../../python/rocke/helpers/fp8_encoding.py), which
  `quant_max_abs` and `mx_gemm_element_format` read.
- The attention kernels guard the encoding themselves: `FmhaFwdFp8Spec` refuses
  OCP-quantized K/V on gfx942 unless `fp8_fnuz=True`
  ([fmha_fwd_fp8.py:92-151](../../../library/kernels/common/fmha_fwd_fp8.py)),
  and unified attention refuses a `fp8_fnuz` flag that does not match the
  target ([attention_unified.py:508-523](../../../library/kernels/common/attention_unified.py)).

## Conversions

| builder op | lowers to | rounding | out-of-range result |
|---|---|---|---|
| `cvt_f32_to_fp8`, `cvt_pk_fp8_f32x4` | `llvm.amdgcn.cvt.pk.fp8.f32` (`v_cvt_pk_fp8_f32`) | RNE | non-saturating, see below |
| `cvt_fp8_to_f32`, `cvt_pk_f32_fp8x4` | `llvm.amdgcn.cvt.f32.fp8`, `llvm.amdgcn.cvt.pk.f32.fp8` | exact | NaN codes give NaN |
| `cvt_scalef32_pk_f32_fp8x4` (gfx950 only) | `llvm.amdgcn.cvt.scalef32.pk.f32.fp8` | exact decode, then scale | uses only the exponent of the f32 scale |

Sources: builders [core/ir.py:3366-3540](../../python/rocke/core/ir.py),
lowerings [core/lower_llvm.py:2983-3185](../../python/rocke/core/lower_llvm.py).

**The f32 -> fp8 conversion does not saturate.** MODE.FP16_OVFL is clear in
rocKE kernels ([README](./README.md#the-mode-register)), and with it clear
the conversion gives (CDNA3 ISA section 7.2, CDNA4 ISA section 7.3):

| input | result |
|---|---|
| NaN | NaN |
| +-inf | NaN |
| finite past max after rounding: above 464 on OCP (464 itself rounds down to 448, whose mantissa is even); from 248 on FNUZ (248 rounds up, because 240's mantissa is odd) | NaN |

That is OFP8's non-saturating mode for e4m3. Measured on gfx942: NaN, +-inf
and finite values from 248 (either sign) all give `0x80`; 247.9 gives `0x7f`.
Measured on gfx950 ([README](./README.md#the-mode-register)): NaN, +-inf,
10000 and -10000, 57344, 61440 and 1e30 give NaN; 447, 449 and 464 give 448
(464 is the tie and rounds down); 247 gives 240 and 248 gives 256. Finite
inputs between 464 and 10000 were not measured. The `cvt_f32_to_fp8`
docstring states the same rule ([core/ir.py:3521-3540](../../python/rocke/core/ir.py)).

**Decoding is exact**: every fp8 value, subnormals included, is an f32 value,
and rocKE keeps subnormals ([README](./README.md#subnormals)). On FNUZ, `0x80`
decodes to NaN; on OCP, `0x80` is -0 and `0x7f` / `0xff` are NaN.

**The scaled decode is not a multiply by an f32.** `v_cvt_scalef32_pk_f32_fp8`
reads the scale's exponent bits only; for an arbitrary f32 scale, decode with
`cvt_pk_f32_fp8x4` and multiply
([core/ir.py:3443-3500](../../python/rocke/core/ir.py)).

## Policy

rocKE's policy for f32 -> fp8 quantization (`quantize_scalar_f32` and
`quantize_prescaled_f32` in [helpers/quant.py](../../python/rocke/helpers/quant.py)):

| input | result |
|---|---|
| NaN | NaN (the encoding's NaN code) |
| +-inf | +-max of the target's encoding |
| finite above max | +-max |
| finite below half the smallest subnormal | +-0 by RNE; FNUZ gives `0x00` for both signs |
| bound | 240 on gfx942 (FNUZ), 448 on gfx950 and gfx1250 (OCP) |

This is the saturating mode of every standard that defines one: OFP8 Table 3
(saturating), IEEE P3109 `SatFinite` (and `SatPropagate`, which saturates
infinity for a format without infinities, as FNUZ is), and the saturating
row of the ISA's conversion table. rocKE reaches it in software: clamp the
f32 value to +-max, but select the unclamped value for NaN, then convert.
Setting FP16_OVFL instead would not give it: measured on gfx942, +-inf still
converts to NaN with the bit set ([README](./README.md#the-mode-register)).

Both helpers take the target (`arch`) and read its bound from
`quant_max_abs`.

### Packed routes

These routes apply no clamp, so they give the bare conversion's results for
+-inf, NaN and anything the scale does not bound: both fp8 / bf8 paths of
`pack_quant_chunk_local_f32` in
[helpers/quant.py](../../python/rocke/helpers/quant.py), the 4-wide packed one
and the per-element one for 2-wide chunks, and `pack_quant_chunk_f32`
([helpers/io.py:285-322](../../python/rocke/helpers/io.py)). The caller's
scale must keep every finite value within the target's bound; the quantizing
instances build it as `amax / quant_max_abs`.

## Quantize and dequantize helpers

- `quantize_scalar_f32(b, x, *, inv_scale, qdtype, arch)`: `x * inv_scale`,
  clamp, convert. Takes the **inverse** scale.
- `quantize_prescaled_f32(b, scaled, *, qdtype, arch)`: the same clamp and
  conversion for a value the caller has already scaled.
- `dequantize_scalar_to_f32(b, q, *, scale)`: exact decode, then `* scale`.
  Takes the **forward** scale, so `dequant(quant(x, 1/s), s)` is `x` up to the
  fp8 rounding error.
- `quant_max_abs(qdtype, *, arch)`: the clamp bound of the target's
  encoding. Use it; never write a bound into a kernel.
- Host side (numpy): [numeric/fp8.py](../../python/rocke/numeric/fp8.py)
  builds the OCP and FNUZ decode tables from the formats and offers exact
  encoders and `fp8e4m3_round` / `fp8e4m3fnuz_round` (RNE, saturating, reject
  NaN).

**Dynamic scaling and infinities.** With `scale = amax / bound`, a +-inf in
the row makes `amax` infinite and `inv_scale` 0; `inf * 0` is NaN, so that
element becomes NaN and the finite ones 0, whatever the conversion policy.
The +-inf rule matters for static scales and for `x * inv_scale` overflowing
f32.

## MMA operands

The fp8 MFMA and WMMA instructions read the bytes in the target's encoding;
`core/mma_operand.py` only reinterprets operand bits, it never converts them
([core/mma_operand.py:1-20](../../python/rocke/core/mma_operand.py)). So a
buffer quantized for one encoding must not feed another target's matrix
cores. Mixed atoms (`fp8_bf8`, `bf8_fp8`) read each operand in its own
format, and accumulate in f32.

## Known differences from other libraries

A parity test against one of these references on NaN, infinity or
out-of-range inputs must expect the difference. Sources are the public
repositories named.

| library (public source) | difference from rocKE's policy |
|---|---|
| HIP `amd_hip_fp8.h` with `__HIP_SATFINITE`; Composable Kernel `include/ck/utility/amd_ck_fp8.hpp` and `type_convert.hpp`; CK Tile `include/ck_tile/core/numeric/float8.hpp` (device path); MIOpen `src/kernels/fp8_dev.hpp`; hipBLASLt `tensilelite` `Components/PackData.py` | +-inf gives NaN for e4m3, OCP and FNUZ: they skip the clamp for non-finite inputs |
| vLLM `csrc/quantization/w8a8/fp8/common.cuh` (`scaled_fp8_conversion`) | NaN gives +max (`fminf` returns the non-NaN operand) |
| Triton AMD backend `tl.clamp` (`third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp`), used by vLLM and SGLang Triton quantizers | NaN gives -max |
| vLLM `csrc/quantization/utils.cuh`; SGLang `HIP_FP8_E4M3_FNUZ_MAX` | bound 224 instead of 240 on FNUZ. vLLM's stated reason is "accuracy issues when running dynamic quantization"; a likely mechanism is that OCP checkpoints reinterpreted as FNUZ top out at 224. It is a scale target, not the format's max. |
| PyTorch `.to(float8_e4m3fnuz)` (`torch/headeronly/util/Float8_e4m3fnuz.h`); ml_dtypes `astype` (`ml_dtypes/include/float8.h`) | non-saturating: +-inf and overflow give NaN |
| PyTorch `.to(float8_e4m3fn)` (2.12 and newer, `Float8_e4m3fn.h`); torchao `to_fp8_saturated` (`torchao/float8/float8_utils.py`) | same as rocKE |

## Tests

| rule | test |
|---|---|
| OCP and FNUZ decode tables match the formats and ml_dtypes; OCP NaN only at `0x7f` / `0xff` | [tests/numeric/test_fp8_codec.py](../../tests/numeric/test_fp8_codec.py) |
| the gfx950-only scaled decode is refused elsewhere | [tests/core/test_fp8_scale_conversion_gate.py](../../tests/core/test_fp8_scale_conversion_gate.py) |
| the per-target largest values bound the MX scale codes | [tests/core/test_mx_scale_decode.py](../../tests/core/test_mx_scale_decode.py) (`test_safe_code_for_the_mfma_formats`) |
| bare conversion does not saturate on the GPU; `quantize_scalar_f32` saturates to the target bound and keeps NaN. They test 0.4 and 0.6 of a step above max, not the exact tie. | [tests/core/test_dtypes.py](../../tests/core/test_dtypes.py) (`test_fp8_conversion_does_not_saturate_on_gpu`, `test_quantize_scalar_f32_saturates_and_keeps_nan_on_gpu`) |

## How to decide

- **Quantizing f32 to fp8 in a kernel**: call `quantize_scalar_f32`. Do not
  call `cvt_f32_to_fp8` on values that can exceed the bound, and do not clamp
  with `clamp_f32` alone if NaN must survive.
- **Need the bound**: ask `quant_max_abs(qdtype, arch=arch)`. It depends on
  the target.
- **Using a packed route**: make sure the scale bounds every finite value, and
  decide what +-inf and NaN inputs should give; the route itself does not
  clamp.
- **Reading fp8 bytes from outside the kernel** (a checkpoint, a framework
  tensor): check which encoding produced them. On gfx942 they must be FNUZ;
  OCP bytes need converting first.
- **Matching another library's bytes**: see the table above; do not change
  the bound or the NaN rule to match. Matching vLLM's 224 on gfx942 would need
  a separate scale-target setting, which has not been decided.
- **Writing a host reference**: decode with `numeric/fp8.py` or ml_dtypes as
  an independent oracle, and pick the table of the kernel's target (OCP or
  FNUZ).
