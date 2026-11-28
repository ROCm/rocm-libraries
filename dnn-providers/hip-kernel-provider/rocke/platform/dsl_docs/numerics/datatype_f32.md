---
name: f32
aliases: [f32, fp32, float, float32]
family: float
bits: [32]
variants:
  - {arch: all, encoding: "IEEE 754 binary32"}
status: on develop
related: [platform/python/rocke/core/ir.py, platform/python/rocke/core/lower_llvm.py, library/kernels/common/_lse_store.py, library/kernels/common/_sdpa_padding_mask.py]
---

# f32

IEEE 754 binary32. Constants are in the [Quick values](./README.md#quick-values)
table. f32 is rocKE's computing type: every narrower float converts through
it, matrix instructions accumulate in it, and reductions and softmax run in
it. The rules that hold for every type (round to nearest even, subnormals
kept, no fast-math flags, `fmax` / `clamp_f32` dropping NaN, the MODE
register) are in the [README](./README.md#shared-rules); this file covers what
is specific to f32 arithmetic.

## Math operations and their special values

| builder op | lowers to | accuracy | special values |
|---|---|---|---|
| `exp2` | `llvm.exp2.f32` | the compiler's lowering | libm `exp2` semantics: `exp2(-inf) = +0`, `exp2(+inf) = +inf`, NaN gives NaN |
| `exp2_fast` | `llvm.amdgcn.exp2.f32` (`v_exp_f32`) | 1 ulp; "denormals are flushed" (ISA) | `exp(-inf) = 0`, `exp(+inf) = +inf` (CDNA3 ISA, V_EXP_F32) |
| `rcp` | `fdiv float 1.0, x` | IEEE division | `1/+-0 = +-inf`, `1/+-inf = +-0` |
| `rcp_fast` | `llvm.amdgcn.rcp.f32` (`v_rcp_f32`) | about 1 ulp | per the ISA's V_RCP_F32 table |
| `rsqrt` | `llvm.amdgcn.rsq.f32` (`v_rsq_f32`) | about 1 ulp | per the ISA's V_RSQ_F32 table |
| `log2`, `sqrt` | `llvm.log2.f32`, `llvm.sqrt.f32` | the compiler's | libm semantics: `log2(+0) = -inf`, negative input gives NaN |
| `tanh` | `1 - 2 / (exp2(2 x log2 e) + 1)` with `rcp_fast`, and OCML's polynomial below 0.625 | about 1 f32 epsilon absolute on [-20, 20] | `tanh(+-inf) = +-1`; NaN gives NaN |

Sources: the `_op_math_*` lowerings in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py);
the `tanh` bound is the one pinned by
[tests/core/test_tanh_lowering.py](../../tests/core/test_tanh_lowering.py).

## Conversions from f32

The narrowing conversions are documented in the destination type's file:
[f16](./datatype_f16.md), [bf16](./datatype_bf16.md),
[tf32](./datatype_tf32_xf32.md), [fp8e4m3](./datatype_fp8_e4m3.md),
[bf8e5m2](./datatype_bf8_e5m2.md), [i8 and u8](./datatype_int8_uint8.md),
[f64](./datatype_f64.md) (widening, exact).

## Masking and softmax with infinities

Attention kernels drive masked scores to a "minus infinity" and must then
keep `exp2` and the running max well defined. Two conventions exist in the
code; know which one a kernel uses before changing its mask.

| convention | where | what it needs |
|---|---|---|
| finite sentinel `-1e30` | for example the warp-level FMHA body (`fmha_warp_fwd_inner_body` in [_fmha_warp_body.py](../../../library/kernels/common/_fmha_warp_body.py)), the dense kernels, the split-KV decode, the 4-warp GQA bodies | a fully masked row has max `-1e30`, so every `exp2(score - max)` is `exp2(0) = 1`: the epilogue must test whether the max ever rose above the sentinel and write `O = 0` (the module docstring of [_sdpa_padding_mask.py](../../../library/kernels/common/_sdpa_padding_mask.py)). A large softmax scale can push a visible score below `-1e30` or past the f32 max, which is why the dense kernels bound the scale (`check_dense_scale` in [attention_dense_spec.py](../../../library/kernels/common/attention_dense_spec.py)). |
| true `-inf` | unified attention segment kernels (`build_unified_attention_2d` and `build_unified_attention_3d` in [attention_unified.py](../../../library/kernels/common/attention_unified.py)) | `exp2(-inf - -inf)` is NaN, so the exponent reference is `select(fcmp("ogt", m, -inf), m, 0)` while the raw max is carried; a segment that saw no key stores max `-inf` and sum 0 (`build_unified_attention_reduce` in [attention_unified.py](../../../library/kernels/common/attention_unified.py)) |

The output contract for a row with no visible key, in both conventions:
`O = 0` and log-sum-exp `-inf`. A row whose sum is NaN keeps NaN: the
"has mass" predicate is `fcmp one OR uno` rather than a negated `oeq`
(`lse_row_has_mass` in [_lse_store.py](../../../library/kernels/common/_lse_store.py)), so a NaN
input row is not hidden as an empty one.

## Integer conversions

`uitofp_f32` (u8) is exact. `sitofp_f32` (i32) is exact for magnitudes up to
2^24, which covers every i8, i16 and u8 value; larger i32 values round to
nearest even. f32 -> integer is in
[datatype_int8_uint8.md](./datatype_int8_uint8.md): LLVM's `fptosi` is poison
for NaN and for values that do not fit, so rocKE clamps first.

## Policy

IEEE 754 binary32 arithmetic with the shared rules of the
[README](./README.md#shared-rules). Special values propagate except through
the operations listed there; a kernel that must keep or reject NaN says so and
selects on `fcmp("uno", x, x)`. Masked attention rows follow the output
contract above.

## Tests

| rule | test |
|---|---|
| `ldexp` emits one `v_ldexp_f32` and matches numpy | [tests/core/test_ldexp.py](../../tests/core/test_ldexp.py) |
| `tanh` accuracy bound | [tests/core/test_tanh_lowering.py](../../tests/core/test_tanh_lowering.py) |
| masked tiles carry the raw running max; the keyless guard gives exact zeros | [library/tests/test_attention_unified_masked_tile_ir.py](../../../library/tests/test_attention_unified_masked_tile_ir.py) |
| a NaN query row keeps a NaN LSE | [library/tests/test_attention_unified_lse_numeric.py](../../../library/tests/test_attention_unified_lse_numeric.py) (`test_split_kv_lse_nan_query_row_stays_nan`) |
| dead rows in the backward pass; removing the dead-row select gives NaN | [library/tests/test_attention_bwd_numeric.py](../../../library/tests/test_attention_bwd_numeric.py), [library/tests/test_attention_bwd_mutations.py](../../../library/tests/test_attention_bwd_mutations.py) |

## How to decide

- **Max or min that must see a NaN**: see
  [NaN and infinity in arithmetic](./README.md#nan-and-infinity-in-arithmetic).
- **Masking a score**: follow the kernel's existing convention (table above).
  With `-inf`, guard the exponent reference; with `-1e30`, guard the epilogue.
  Never mix the two in one kernel.
- **Dividing by a sum that can be 0**: decide what an empty row means (rocKE's
  attention writes 0) and select on it; `x * rcp(0)` is `inf` or NaN.
- **Exactness**: `exp2_fast`, `rcp_fast` and `rsqrt` are approximations; use
  `exp2`, `rcp` and `sqrt` where a test compares bits.
