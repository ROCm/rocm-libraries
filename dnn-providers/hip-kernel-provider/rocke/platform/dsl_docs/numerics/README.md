# Numerics: how rocKE treats each data type

This folder is the one home for how rocKE handles the numbers of each data
type: encodings, special values (NaN, infinity, signed zero, subnormals),
conversions and their rounding and saturation, and rocKE's policy for NaN,
infinity and overflow. Use it to answer "what happens to this value?" before
writing or reviewing a conversion, a quantizer, a reduction or a mask.

Other documents keep their own concerns and point here for numerics:
which layer supports a dtype is in
[dtype_support.md](../reference/dtype_support.md); how a packed dtype is
stored is in [dtype_storage_packing.md](../architecture/dtype_storage_packing.md);
test tolerances are in the
[optimization runbook, section 1.4](../optimization/optimization_runbook.md#14-dtypes-and-numerics).

**Anchors.** A link into the code names the function, class or constant it
points at, never a line number, so it stays true as the file changes.
Behaviour that exists only on an open branch is marked **pending** and listed
once, in [Pending changes](#pending-changes).

## How to use this folder

1. Find the dtype name as it is written in the code (IR type, spec string or
   alias) in the [dispatch table](#dispatch-table) and open that one file.
2. Constants (largest value, smallest normal, NaN codes) live in
   [Quick values](#quick-values). A per-type file quotes a constant only where
   the number is the rule itself (a clamp bound, an overflow threshold), next
   to a link here.
3. Rules that hold for every type (rounding mode, saturation, NaN and
   infinity propagation, subnormals, the MODE register) are in
   [Shared rules](#shared-rules). The per-type files say only where a type
   differs.

Each per-type file starts with YAML front matter (`name`, `aliases`,
`family`, `bits`, `variants`, `status`, `related`). It describes the type and
its conversions first, then always has a **Policy**, a **Tests** and a
**How to decide** section (in that order, so `#policy`, `#tests` and
`#how-to-decide` work in every file), with "Known differences from other
libraries" before Tests where there are any. The dispatch table is the union of the
`aliases` lists, so the two must agree.

### Terms

- **FNUZ**: "finite, no negative zero". The 8-bit float encodings used by
  gfx942: no infinity, and the bit pattern of -0 (`0x80`) is the only NaN.
- **OCP**: the Open Compute Project formats (OFP8 for e4m3 and e5m2, MX for
  fp6, fp4 and e8m0), used by gfx950 and the gfx12 targets.
- **max** or **bound**: the largest finite magnitude of the encoding the
  target uses ([Saturating and non-saturating conversion](#saturating-and-non-saturating-conversion)).
- **RNE**: round to nearest, ties to even.
- **ulp**: unit in the last place, the gap between adjacent values.
- **binade**: the values that share one exponent, `[2^e, 2^(e+1))`.
- **poison**: LLVM's "undefined value". Code that uses it may compute
  anything; a conversion that returns poison is not a defined result.

## Dispatch table

Spellings are those accepted by `normalize_dtype`
(`_DTYPE_ALIASES` in [core/dtypes.py](../../python/rocke/core/dtypes.py)), the IR type names
(the `Type` constants such as `F32` and `I8` in [core/ir.py](../../python/rocke/core/ir.py)), the quantization aliases
(`_QUANT_TYPE_ALIAS` in [helpers/quant.py](../../python/rocke/helpers/quant.py)), the SDPA
dispatch dtypes and hipDNN names
(`Dtype` in [library/dispatch/sdpa/common.py](../../../library/dispatch/sdpa/common.py),
`HIPDNN_DTYPES` in [library/dispatch/sdpa/hipdnn.py](../../../library/dispatch/sdpa/hipdnn.py)),
and the PyTorch / ml_dtypes names these docs use. Look names up
case-insensitively here (the Python constants are the upper-case names, such
as `FP8E4M3`), but note that only `normalize_dtype` ignores case: the
quantization helpers match their aliases exactly.

| dtype name or alias | file |
|---|---|
| `f32`, `fp32`, `float`, `float32` | [datatype_f32.md](./datatype_f32.md) |
| `f16`, `fp16`, `half`, `float16` | [datatype_f16.md](./datatype_f16.md) |
| `bf16`, `bfloat16` | [datatype_bf16.md](./datatype_bf16.md) |
| `f64`, `fp64`, `double`, `float64` | [datatype_f64.md](./datatype_f64.md) |
| `tf32`, `xf32` | [datatype_tf32_xf32.md](./datatype_tf32_xf32.md) |
| `fp8e4m3`, `fp8`, `e4m3`, `fp8_e4m3`, `fp8_e4m3_fnuz`, `float8_e4m3fn`, `float8_e4m3fnuz` | [datatype_fp8_e4m3.md](./datatype_fp8_e4m3.md) |
| `bf8e5m2`, `bf8`, `fp8_e5m2`, `fp8e5m2`, `fp8_e5m2_fnuz`, `float8_e5m2`, `float8_e5m2fnuz` | [datatype_bf8_e5m2.md](./datatype_bf8_e5m2.md) |
| `fp6e2m3`, `fp6`, `fp6_e2m3`, `fp6e3m2`, `bf6`, `fp6_e3m2`, `fp4e2m1`, `fp4`, `fp4_e2m1` | [datatype_fp6_fp4.md](./datatype_fp6_fp4.md) |
| `e8m0`, `fp8_e8m0`, `e5m3` (e5m3 not covered yet) | [datatype_e8m0_scale.md](./datatype_e8m0_scale.md) |
| `i8`, `int8`, `u8`, `uint8`, `i16`, `i32`, `int32`, `i64`, `int64` | [datatype_int8_uint8.md](./datatype_int8_uint8.md) |
| `i4`, `int4`, `u4`, `uint4` | [datatype_int4.md](./datatype_int4.md) |
| `i1`, `bool`, `boolean` | [datatype_i1_bool.md](./datatype_i1_bool.md) |

`e4m3` is an alias of `fp8e4m3`, but `e5m3` is a different 8-bit format, not
an alias of `bf8e5m2` (`dtype_info` in [core/dtypes.py](../../python/rocke/core/dtypes.py)).

## Quick values

The one home for these constants. "max" is the largest finite magnitude;
"min normal" and "min subnormal" are the smallest positive normal and
subnormal values. Codes are the stored bit patterns in hex, sign bit
included.

### Floating-point formats

| format | bits | exp / mant | bias | max | min normal | min subnormal | NaN codes | infinity codes | -0 |
|---|---|---|---|---|---|---|---|---|---|
| f64 (IEEE binary64) | 64 | 11 / 52 | 1023 | 1.7976931348623157e308 | 2^-1022 | 2^-1074 | exp all ones, mantissa != 0 | exp all ones, mantissa 0 | yes |
| f32 (IEEE binary32) | 32 | 8 / 23 | 127 | 3.4028235e38 (`0x7f7fffff`) | 2^-126 | 2^-149 | exp all ones, mantissa != 0 | `0x7f800000`, `0xff800000` | yes |
| tf32 (in an f32 word) | 32 | 8 / 10 used | 127 | (2 - 2^-10) * 2^127 = 3.4011621e38 | 2^-126 | 2^-136 | as f32 | as f32 | yes |
| f16 (IEEE binary16) | 16 | 5 / 10 | 15 | 65504 (`0x7bff`) | 2^-14 | 2^-24 | exp all ones, mantissa != 0 | `0x7c00`, `0xfc00` | yes |
| bf16 | 16 | 8 / 7 | 127 | 3.3895314e38 (`0x7f7f`) | 2^-126 | 2^-133 | exp all ones, mantissa != 0 (quiet: `0x7fc0`) | `0x7f80`, `0xff80` | yes |
| fp8 e4m3, OCP (`e4m3fn`) | 8 | 4 / 3 | 7 | 448 (`0x7e`) | 2^-6 | 2^-9 | `0x7f`, `0xff` | none | yes (`0x80`) |
| fp8 e4m3, FNUZ (`e4m3fnuz`) | 8 | 4 / 3 | 8 | 240 (`0x7f`) | 2^-7 | 2^-10 | `0x80` only | none | no |
| bf8 e5m2, OCP | 8 | 5 / 2 | 15 | 57344 (`0x7b`) | 2^-14 | 2^-16 | `0x7d`-`0x7f`, `0xfd`-`0xff` | `0x7c`, `0xfc` | yes (`0x80`) |
| bf8 e5m2, FNUZ (`e5m2fnuz`) | 8 | 5 / 2 | 16 | 57344 (`0x7f`) | 2^-15 | 2^-17 | `0x80` only | none | no |
| fp6 e2m3 (OCP MX) | 6 | 2 / 3 | 1 | 7.5 | 1.0 | 0.125 | none | none | yes |
| fp6 e3m2, "bf6" (OCP MX) | 6 | 3 / 2 | 3 | 28.0 | 0.25 | 0.0625 | none | none | yes |
| fp4 e2m1 (OCP MX) | 4 | 2 / 1 | 1 | 6.0 | 1.0 | 0.5 | none | none | yes |
| e8m0 scale (OCP MX) | 8 | 8 / 0, unsigned | 127 | 2^127 | 2^-127 | none | `0xff` | none | no zero, no sign |

Sources: IEEE 754-2019 for f64, f32 and f16; the OCP 8-bit Floating Point
Specification (OFP8) rev 1.0 for OCP e4m3 and e5m2; the OCP Microscaling
Formats (MX) v1.0 specification for fp6, fp4 and e8m0; for FNUZ, the CDNA3
ISA (section 7.2, Table 30), which lists bias 8 and 16, NaN `0x80` and the
max values, and IEEE P3109's signed finite formats. The FNUZ idea, one NaN
codeword in place of -0, comes from Noune et al. (arXiv:2206.02915). The table
matches `ml_dtypes.finfo` (max, min normal, min subnormal) for every 8-, 6-
and 4-bit row, except that ml_dtypes reports e8m0's smallest value as its
smallest subnormal too; it also matches the rocKE host decode tables in
[numeric/fp8.py](../../python/rocke/numeric/fp8.py) and
[helpers/i4_dequant.py](../../python/rocke/helpers/i4_dequant.py).

### Integer formats

| format | bits | range | rocKE quantization range |
|---|---|---|---|
| i1 | 1 | 0, 1 (predicate) | - |
| i4 (packed only) | 4 | -8 .. 7 | no quantizer |
| u4 (packed only) | 4 | 0 .. 15 | no quantizer |
| i8 | 8 | -128 .. 127 | -127 .. 127 (symmetric) |
| u8 | 8 | 0 .. 255 | 0 .. 255 (`cvt_f32_to_u8_sat`) |
| i16, i32, i64 | 16, 32, 64 | two's complement | - |

### Encoding per target (fp8 and bf8)

The same byte means different values on different targets. The matrix cores
and the conversion instructions of a target read and write one family.

| target | fp8e4m3 | bf8e5m2 | in rocKE's table |
|---|---|---|---|
| gfx942 | e4m3 FNUZ | e5m2 FNUZ | yes |
| gfx950 | OCP e4m3 | OCP e5m2 | yes |
| gfx1250 | OCP e4m3 | OCP e5m2 | yes |
| gfx1201, gfx12-generic | OCP e4m3 | OCP e5m2 | no: refused |

rocKE's table is `fp8_element_format(dtype, arch)` in
[helpers/fp8_encoding.py](../../python/rocke/helpers/fp8_encoding.py) (C++
twin: `cpp/helpers/fp8_encoding.cpp`), the one home that `quant_max_abs` and
`mx_gemm_element_format` read; it refuses any target it does not list. The
gfx12 rows come from the public ROCm precision-support page
(https://rocm.docs.amd.com/en/latest/reference/precision-support.html), which
maps the FNUZ HIP types to gfx942 only and the OCP types to gfx950 and gfx12.

### rocKE's f32 -> narrow policy at a glance

What `quantize_scalar_f32` (or the plain cast for 16-bit types) gives for
each class of input. Details and the reasons are in each type's file.

| destination | NaN | +-inf | finite above max | rounding |
|---|---|---|---|---|
| f16, bf16 (`cast_f32_to`) | NaN | +-inf | +-inf | RNE |
| fp8e4m3, bf8e5m2 (`quantize_scalar_f32`) | NaN | +-max of the target encoding | +-max | RNE |
| i8 (`quantize_scalar_f32`) | -127 | +-127 | +-127 | RNE |
| u8 (`cvt_f32_to_u8_sat`) | 0 | 0, 255 | 0, 255 | RNE |

## Shared rules

### Rounding

Every conversion rocKE emits rounds to nearest, ties to even (RNE), unless its
file says otherwise. The kernel descriptor leaves the MODE round fields at 0
(`.amdhsa_float_round_mode_32 0`, `.amdhsa_float_round_mode_16_64 0`; the
CDNA3 ISA, section 3.5, defines 0 as "nearest even"). Stochastic rounding is
not implemented.

### Saturating and non-saturating conversion

A narrowing conversion meets three input classes the destination may not
hold: NaN, +-inf, and finite values that round past the destination's max.

- **Saturating**: NaN gives NaN; +-inf and finite overflow give +-max. This
  is OFP8 "saturating mode" (Table 3) and IEEE P3109 `SatFinite`.
- **Non-saturating**: NaN gives NaN; +-inf and overflow give +-inf where the
  destination has infinities, NaN where it does not (OFP8 Table 3).

"max" or "bound" always means the largest finite magnitude of the encoding
the **target** uses (see [Quick values](#quick-values)), so it depends on the
target for fp8e4m3 (240 on gfx942, 448 on gfx950). The one exception is i8,
whose rocKE bound is the symmetric 127.

### The MODE register

The hardware decides some of this per wave through the MODE register. rocKE
emits no `denormal-fp-math` or other floating-point function attribute
(the function attributes `finalize` writes in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py)), so the
compiler's defaults apply. Compiling a rocKE kernel (llvm20 flavor, ROCm 7.1
`llc`) for gfx942 and gfx950 gives the values below, and the MODE register
read at kernel entry on gfx942 agrees (`0x3f0`):

| kernel descriptor field | value | meaning (CDNA3 ISA section 3.5; CDNA4 ISA) |
|---|---|---|
| `.amdhsa_float_denorm_mode_32` | 3 | f32 subnormal inputs and outputs are kept |
| `.amdhsa_float_denorm_mode_16_64` | 3 | f16 and f64 subnormals are kept |
| `.amdhsa_ieee_mode` | 1 | min/max quiet signaling NaNs per IEEE 754-2008 |
| `.amdhsa_dx10_clamp` | 1 | the instruction clamp modifier sends NaN to 0 |
| `.amdhsa_fp16_overflow` | 0 | MODE.FP16_OVFL is clear |

**FP16_OVFL is clear in every rocKE kernel.** LLVM's code generator never
sets this descriptor bit (`COMPUTE_PGM_RSRC1` bit 26): only the assembler
directive `.amdhsa_fp16_overflow` can, there is no IR attribute for it
(llvm-project `SIProgramInfo.cpp`, `getComputePGMRSrc1Reg`), and rocKE never
writes MODE on gfx942 or gfx950: its only `s_setreg` builder is refused there
(`_op_tile_s_setreg` in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py)). So the
f32 -> fp8 / bf8 conversion instructions are **non-saturating**, f16 overflow
gives +-inf, and rocKE saturates in software, with a clamp before the
conversion.

**Why rocKE does not set it.** The ISA says the bit makes the fp8 / bf8
conversions saturate, +-inf included (CDNA3 ISA section 7.2, CDNA4 ISA
section 7.3). Measured on gfx942 with the bit set, finite overflow saturates
but +-inf still becomes NaN (`0x80`), so the bit cannot give rocKE's policy
on its own. It is also wave-wide: with it set, gfx942 measured f16 overflow,
`v_rcp_f16(0)` and out-of-range f32 -> f16 conversions giving 65504 instead of
+-inf, and the i8 MFMA's i32 accumulator saturating instead of wrapping; LLVM
does not model the bit, so constant-folded code would disagree with the
hardware. rocKE keeps the software clamp.

**Measured on gfx950 with the bit clear** (the state every rocKE kernel runs
in): the raw f32 -> fp8 / bf8 conversions (`v_cvt_pk_fp8_f32`,
`v_cvt_pk_bf8_f32`) and develop's `quantize_scalar_f32` clamp, on `develop`
at [63e29d36559](https://github.com/ROCm/rocm-libraries/commit/63e29d365593a4f0a52a1f44b4136242195d89ad).
The results are in the [fp8e4m3](./datatype_fp8_e4m3.md#conversions) and
[bf8e5m2](./datatype_bf8_e5m2.md#conversions) files. gfx950 with the bit set
has not been measured.

To re-check the kernel descriptor values in the table above: lower a kernel
with `lower_kernel_to_llvm(kernel, arch=...)`, compile the `.ll` with `llc
-mtriple=amdgcn-amd-amdhsa -mcpu=<gfx>` and read the `.amdhsa_*` lines.

### NaN and infinity in arithmetic

rocKE emits no fast-math flags (`nnan`, `ninf`, `contract`, `afn`) on IR
arithmetic, so the compiler may not assume NaN or infinity away and does not
fuse a multiply and an add on its own. Arithmetic follows IEEE 754: NaN
propagates, `inf - inf` and `0 * inf` are NaN.

Operations that do **not** propagate NaN:

| operation | NaN behaviour | source |
|---|---|---|
| `fmax(a, b)`, `fmin(a, b)` | `llvm.maxnum` / `llvm.minnum`: if one operand is a quiet NaN the other is returned (results of arithmetic are always quiet NaNs; the backend quiets inputs first) | `_op_arith_fmax`, `_op_arith_fmin` in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py) |
| `fmax3`, `fmin3` | two nested `maxnum` / `minnum` | `_op_arith_fmax3`, `_op_arith_fmin3` in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py) |
| `clamp_f32(v, lo, hi)` | `fmin(hi, fmax(lo, v))`: a NaN `v` becomes `lo` | `clamp_f32` in [core/ir.py](../../python/rocke/core/ir.py) |
| `fcmp` with an ordered predicate (`olt`, `ole`, `ogt`, `oge`, `oeq`, `one`) | false when either operand is NaN | `fcmp` in [core/ir.py](../../python/rocke/core/ir.py) |

So a reduction built on `fmax` drops NaN, and a clamp turns NaN into the lower
bound. To keep a NaN, test it explicitly and select:
`b.select(b.fcmp("uno", x, x), x, clamped)`. `fcmp("uno", x, x)` is true
exactly when `x` is NaN; `fcmp("ord", x, x)` is its complement.

### Subnormals

Subnormals are kept by default (descriptor above). This is the one home for
the matrix-core rules (CDNA3 ISA section 7.3, CDNA4 ISA section 7.4): f32 MFMA
inputs honour the MODE denormal flags; the C input and the result, every input
narrower than 32 bits (f16, bf16, fp8, bf8) and the f64 MFMA (which also
rounds to nearest even regardless of MODE) do not flush; on gfx942 the XF32
inputs do not flush either (CDNA3 only). `exp2_fast` (`v_exp_f32`) flushes
subnormals ([f32](./datatype_f32.md)), and the gfx950 tf32 emulation loses
the smallest values ([tf32](./datatype_tf32_xf32.md)). The f32 atomic
add rocKE emits carries `!amdgpu.ignore.denormal.mode`
(`_op_memref_global_atomic_add` in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py)), which
allows the hardware atomic to flush subnormals.

### Signed zero

IEEE formats keep -0. The FNUZ fp8 and bf8 encodings have no -0: its bit
pattern is their NaN, so a conversion to FNUZ must never produce `0x80` for a
zero. The scale format e8m0 has no zero at all.

### Accumulation

The matrix atoms rocKE's kernel families use accumulate f16, bf16, fp8, bf8,
fp6, fp4 and tf32 products in f32 and integer products in i32; reductions and
softmax run in f32. The catalog also declares WMMA atoms that accumulate in
f16 or bf16 (gfx11 and gfx12 targets); no family uses them yet, and their
results follow the f16 / bf16 rules. Narrow types are storage and matrix-operand formats: widen to f32,
compute, and narrow once at the end. The tolerances that follow from this are
in the [runbook](../optimization/optimization_runbook.md#14-dtypes-and-numerics).

## Pending changes

The single list of numerics behaviour that is decided but not yet on
`develop`. When one lands, update the type's file and delete its row here.

None at present.

## Standards cited

- IEEE 754-2019, Standard for Floating-Point Arithmetic.
- OCP 8-bit Floating Point Specification (OFP8), revision 1.0, 2023-12-01,
  opencompute.org. Table 3 defines saturating and non-saturating conversion.
- OCP Microscaling Formats (MX) Specification, version 1.0, opencompute.org.
- IEEE P3109 interim report (github.com/P3109/Public): `SatFinite`,
  `SatPropagate`, `SatNone`; FNUZ-style signed finite formats.
- Noune et al., "8-bit Numerical Formats for Deep Neural Networks",
  arXiv:2206.02915: the single NaN codeword in place of -0.
- AMD CDNA3 Instruction Set Architecture reference guide (gfx942) and AMD
  CDNA4 Instruction Set Architecture reference guide (gfx950), amd.com: MODE
  register (section 3.5), FP8 and BF8 formats and conversions (CDNA3 7.2,
  CDNA4 7.3), matrix-core subnormal handling (CDNA3 7.3, CDNA4 7.4).
- LLVM Language Reference (llvm.org/docs/LangRef.html): `fptrunc`, `fptosi`
  and `sitofp`, `llvm.maxnum` / `llvm.minnum`, `llvm.exp2`; the AMDGPU kernel
  descriptor (llvm.org/docs/AMDGPUUsage.html).

## Related

- [Glossary](../reference/glossary.md): FP8, BF8, BF16 entries.
- [Quantization primitives](../primitives/quantization.md): the quantization
  API and recipes.
- [XF32 BF16x3 emulation](../architecture/xf32_bf16x3_emulation.md).
