# Quantization Primitives

`ck_dsl` exposes a complete FP8 / BF8 / INT8 quantization stack used by the unified-attention FP8 K/V path and by quantized GEMM epilogues. This page documents what is implemented today and how to compose it. Source: `core/ir.py` (`cvt_*`, `clamp_f32`), `helpers/quant.py`, `helpers/io.py`.

## Supported Element Types

| IR `Type` | Bits | Range (approx.) | Notes |
|------------|-----:|-----------------------|----------------------------------------|
| `FP8E4M3` | 8 | +/- 448 | Round-to-nearest-even, hardware clamp |
| `BF8E5M2` | 8 | +/- 57344 | 5 exponent bits; less precision |
| `I8` | 8 | -128..127 | Saturating cast from f32 |

The lookup helpers in `helpers/quant.py`:

```text
QDType # Literal["i8", "fp8e4m3", "bf8e5m2"] (not an enum)
QUANT_MAX_ABS # dict: { "i8": 127.0, "fp8e4m3": 448.0, "bf8e5m2": 57344.0 }
quant_ir_type(qdtype) -> Type # "fp8e4m3" -> FP8E4M3, etc.
ir_to_qdtype(ir_type) # inverse
quant_max_abs(qdtype) -> float
```

`QDType` is a string-literal type, so values are plain strings:

```python
from ck_dsl.helpers import quant_max_abs, quant_ir_type
quant_max_abs("fp8e4m3") # 448.0
quant_ir_type("fp8e4m3") # FP8E4M3 (the IR Type)
```

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

Rounding semantics for all `cvt_f32_to_*`: round-to-nearest-even. Out-of-range f32 inputs to fp8/bf8 are hardware-clamped to the dtype's representable range.

## Quantization Recipes

Per-tensor symmetric quantization to `fp8e4m3`:

```python
from ck_dsl.core.ir import F32, IRBuilder
from ck_dsl.helpers import quant_max_abs

b: IRBuilder = ...
scale = b.param("scale", F32)
clamp_hi = b.const_f32(quant_max_abs("fp8e4m3")) # 448.0
clamp_lo = b.fneg(clamp_hi)

# f32 input value `v`:
scaled = b.fmul(v, b.rcp(scale)) # /scale via reciprocal
clamped = b.clamp_f32(scaled, clamp_lo, clamp_hi)
fp8 = b.cvt_f32_to_fp8(clamped)
```

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
quantize_scalar_f32(b, x_f32, *, inv_scale, qdtype) -> Value
dequantize_scalar_to_f32(b, x_q, *, scale) -> Value
```

`quantize_scalar_f32` takes the **inverse** scale (`inv_scale = 1 /
scale`); pre-computing the reciprocal once per row amortises one
`v_rcp_f32` over the whole row, which is the same trick CK Tile's
SmoothQuant pipeline uses. `dequantize_scalar_to_f32` takes the
**forward** scale so the two are symmetric: `dequant(quant(x, inv_s),
s) == x` modulo quantisation error.

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

- The conversion ops are single-element. Packed 2-element `v_cvt_pk_*` codepaths exist on the hardware but are not exposed at the DSL level today; the helpers extract the low byte. If you need the high byte from the same `cvt.pk` instruction (e.g. to convert two f32 in one instruction), reach into raw LLVM IR or extend `core/ir.py`.
- `cvt_f32_to_i8_sat` is a true saturating cast. Non-saturating int8 truncation is not exposed.
- bf8e5m2 is supported in the type system and conversions; full-pipeline support is documented in attention but not yet uniform across every instance.
- FP4 / FP6 / scaled MFMA conversions are not implemented in this DSL today; the runbook lists them as natural extensions.

## See Also

- `instances/smoothquant.py` — per-row dynamic quantisation .
- `instances/moe_smoothquant.py` — per-expert variant; threads
 `topk_ids` through an SGPR-pinned indirect lookup for SmScale.
- `instances/add_rmsnorm2d_rdquant.py` — fused `(a + b)` -> RMSNorm
 -> quantise; two block-LDS reductions, one for `sum(x^2)` and one
 for `max(|x * gamma|)`.
- `attention_unified.py` and `attention_tiled_*` for FP8 K/V cache integration (in progress).
- `RUNBOOK_COMPLIANCE.md` §10 for the compiler-flag and metadata coverage.
- `/workspace/OPTIMIZATION_RUNBOOK (1).md` §1.4 (Dtypes And Numerics) for tolerance policy.
