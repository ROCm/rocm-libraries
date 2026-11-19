# Quantization Primitives

`rocke` exposes a complete FP8 / BF8 / INT8 quantization stack used by the unified-attention FP8 K/V path and by quantized GEMM epilogues. This page documents what is implemented today and how to compose it. Source: `core/ir.py` (`cvt_*`, `clamp_f32`), `helpers/quant.py`, `helpers/io.py`.

## Supported Element Types

| IR `Type` | Bits | Range (approx.) | Notes |
|------------|-----:|-----------------------|----------------------------------------|
| `FP8E4M3` | 8 | +/- 240 (gfx942, FNUZ), +/- 448 (OCP) | Round-to-nearest-even, no saturation |
| `BF8E5M2` | 8 | +/- 57344 | 5 exponent bits; less precision |
| `I8` | 8 | -128..127 | Saturating cast from f32 |

The lookup helpers in `helpers/quant.py`:

```text
QDType # Literal["i8", "fp8e4m3", "bf8e5m2"] (not an enum)
quant_ir_type(qdtype) -> Type # "fp8e4m3" -> FP8E4M3, etc.
ir_to_qdtype(ir_type) # inverse
quant_max_abs(qdtype, *, arch) -> float # clamp bound of qdtype on arch
```

`QDType` is a string-literal type, so values are plain strings:

```python
from rocke.helpers import quant_max_abs, quant_ir_type
quant_max_abs("fp8e4m3", arch="gfx942") # 240.0 (e4m3 FNUZ)
quant_max_abs("fp8e4m3", arch="gfx950") # 448.0 (OCP e4m3)
quant_max_abs("i8", arch="gfx950") # 127.0
quant_ir_type("fp8e4m3") # FP8E4M3 (the IR Type)
```

The fp8 / bf8 bound is the largest finite value of the encoding the target
converts to, `fp8_element_format(dtype, arch)` in `helpers/fp8_encoding.py`: the
FNUZ encodings on gfx942, the OCP ones on gfx950 and gfx1250. A target with no
known encoding is refused with a `ValueError`.

## Per-Element Conversion Ops

All conversions are single-element. Vector forms are composed from these.

```text
cvt_fp8_to_f32(v) # llvm.amdgcn.cvt.f32.fp8 (e4m3 -> f32)
cvt_bf8_to_f32(v) # llvm.amdgcn.cvt.f32.bf8 (e5m2 -> f32)
cvt_f32_to_fp8(v) # llvm.amdgcn.cvt.pk.fp8.f32 (low-byte extract)
cvt_f32_to_bf8(v) # llvm.amdgcn.cvt.pk.bf8.f32 (low-byte extract)
cvt_f32_to_i8_sat(v) # round-to-nearest-even + saturate to int8 [-128, 127]
clamp_f32(v, lo, hi) # fmin(hi, fmax(lo, v)) -> v_med3_f32
```

Rounding semantics for all `cvt_f32_to_*`: round-to-nearest-even. The fp8 / bf8 conversions do not saturate: a value that rounds past the encoding's largest finite value, and +-inf and NaN, become NaN (bf8 on OCP targets: +-inf). Clamp to `quant_max_abs` first for a saturating conversion.

## Quantization Recipes

Per-tensor symmetric quantization to `fp8e4m3`:

```python
from rocke.core.ir import F32, IRBuilder
from rocke.helpers import quantize_scalar_f32

b: IRBuilder = ...
scale = b.param("scale", F32)

# f32 input value `v`, quantized for the kernel's target `arch`:
fp8 = quantize_scalar_f32(b, v, inv_scale=b.rcp(scale), qdtype="fp8e4m3", arch=arch)
```

`quantize_scalar_f32` multiplies by the inverse scale, clamps to
`quant_max_abs(qdtype, arch=arch)` and converts. Finite values and +-inf
saturate to the bound; NaN stays NaN (the encoding's NaN code). A bare
`clamp_f32` would turn NaN into the lower bound, which is why the helper
selects the unclamped value for NaN.

Per-tensor dequantization from `fp8e4m3`:

```python
f32 = b.cvt_fp8_to_f32(fp8)
scaled = b.fmul(f32, scale)
```

Saturating int8 quantization for SmoothQuant-style epilogues:

```python
scaled = b.fmul(v, b.rcp(scale)) # /scale
i8 = b.cvt_f32_to_i8_sat(scaled) # round-to-nearest-even + saturate
```

The `helpers/quant.py` module wraps these into:

```text
quantize_scalar_f32(b, x_f32, *, inv_scale, qdtype, arch) -> Value
quantize_prescaled_f32(b, scaled_f32, *, qdtype, arch) -> Value
dequantize_scalar_to_f32(b, x_q, *, scale) -> Value
```

`quantize_scalar_f32` takes the **inverse** scale (`inv_scale = 1 /
scale`); pre-computing the reciprocal once per row amortises one
`v_rcp_f32` over the whole row, which is the same trick CK Tile's
SmoothQuant pipeline uses. `dequantize_scalar_to_f32` takes the
**forward** scale so the two are symmetric: `dequant(quant(x, inv_s),
s) == x` modulo quantisation error. `quantize_prescaled_f32` is the same clamp
and conversion for a value the kernel has already scaled some other way,
for example `x / scale`.

Use the helpers in instance code rather than re-deriving the clamp +
rounding sequence; every quant kernel
(`smoothquant`, `moe_smoothquant`, `add_rmsnorm2d_rdquant`) calls
`quantize_scalar_f32(...)` and the per-row scale plumbing is identical
across all three.

## Vector / LDS Patterns

The conversion ops are single-element, but the vector ops in `IRBuilder` and the higher-level I/O helpers compose them naturally:

```text
helpers/io.py:
 load_vec(b, ptr, idx, dtype, n)
 store_vec(b, ptr, idx, value, dtype, n)
```

For FP8 K/V cache in unified attention:

```text
# 8-byte vector load of fp8 from paged KV cache:
v_i32 = b.buffer_load_vN_f16(rsrc, off_bytes, c0, dwords=2) # 8 fp8 bytes -> 4 halves
 # actually <2 x i32>
# bitcast to <8 x fp8>, then convert per-lane to f32, scale, cast back to query dtype.
```

`attention_tiled_*` does this through `helpers/io.py` helpers; see `instances/attention.md` for the full path.

## Numerical Behavior

For fp8 with f32 accumulation on attention-like reductions:

- `max_abs` versus f32 reference is dominated by fp8 quantization error, not accumulation order.
- Expected `max_abs` is ~1 ULP of the fp8 dtype, roughly the dtype's smallest representable step at the output magnitude.
- Significantly larger errors indicate structural issues (missing clamp, wrong scale, missing rescale in online softmax, accumulator reset bug).

For int8 GEMM:

- Round-to-nearest-even on `cvt_f32_to_i8_sat` matches PyTorch's `quantize_per_tensor` semantics.
- Saturation is hardware-clamped; do not also software-clamp.

## When To Quantize Where

- Inputs (Q, K, V, A, B): keep storage low-precision, dequantize in-flight to f32 (or to the MFMA input dtype) on load.
- Accumulators: always f32. Do not accumulate in fp8/bf8/int8 — the dynamic range is too small.
- Outputs (epilogue): quantize from f32 accumulator to the target output dtype. Use `clamp_f32` + `cvt_f32_to_*` rather than a software pack.

## Limitations

- The scalar `cvt_*` ops above are single-element, but packed `v_cvt_pk_*` codepaths are now exposed: `cvt_pk_f32_fp8x4` / `cvt_pk_f32_bf8x4` (decode `<4 x fp8/bf8> -> <4 x f32>`) and `cvt_pk_fp8_f32x4` / `cvt_pk_bf8_f32x4` / `cvt_pk_i8_f32x4` (encode `<4 x f32> -> <4 x quantized>`), each covering all 4 bytes of the packed instruction. Use the packed forms in dequant/quant loops that process 4-or-more contiguous elements per lane; reach into raw LLVM IR or extend `core/ir.py` only for widths these don't cover.
- `cvt_f32_to_i8_sat` is a true saturating cast. Non-saturating int8 truncation is not exposed.
- bf8e5m2 is supported in the type system and conversions; full-pipeline support is documented in attention but not yet uniform across every instance.
- OCP MX (FP4 / FP6) MFMA ops and the scaled-conversion ops are implemented: `IRBuilder.mfma_scale_f32_16x16x128_f8f6f4` and `IRBuilder.mfma_f32_16x16x128_fp4fp6fp8` (`a_format` and `b_format` select the fp4 or fp6 format), plus the scaled FP8/BF8 conversions `cvt_scalef32_pk_f32_fp8x4` / `cvt_scalef32_pk_f32_bf8x4` (and the reverse pack ops) in `core/ir.py`. Standalone single-element `cvt_fp4_to_f32` / `cvt_fp6_to_f32` ops are still not exposed — fp4 / fp6 values are consumed through the MX MFMA path with per-warp E8M0 scales, not a per-element cast.

## See Also

- `instances/smoothquant.py` — per-row dynamic quantisation .
- `instances/moe_smoothquant.py` — per-expert variant; threads
 `topk_ids` through an SGPR-pinned indirect lookup for SmScale.
- `instances/add_rmsnorm2d_rdquant.py` — fused `(a + b)` -> RMSNorm
 -> quantise; two block-LDS reductions, one for `sum(x^2)` and one
 for `max(|x * gamma|)`.
- `attention_unified.py` and `attention_tiled_*` for FP8 K/V cache integration (in progress).
- `runbook_compliance.md` §10 for the compiler-flag and metadata coverage.
- `gpu-op-optimization-runbook` §1.4 (Dtypes And Numerics) for tolerance policy.
