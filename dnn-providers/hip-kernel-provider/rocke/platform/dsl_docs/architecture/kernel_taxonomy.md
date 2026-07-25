# Kernel taxonomy: which primitive does each kernel use, and why

> Matmul-shaped kernels use the exact gfx target's MFMA or WMMA operation where
> their current builder supports it. Some attention variants still use a
> warp-distributed scalar body.
> Non-matmul ops use VALU, cross-lane reductions, atomics, or data movement
> according to their actual computation.

This page is an index grounded in the current builders. The owning validator
and builder remain the source of truth when a spec can select more than one
body.

Path notation in this page uses `<project_root>` for the rocKE component root,
`<platform_root>` for `<project_root>/platform`, and `<library_root>` for
`<project_root>/library`.

## What matrix MMA is for

Matrix MMA operations compute tiled multiply-accumulates. For example, the
gfx942/gfx950 catalogs provide `mfma_f32_16x16x16_f16`, while the gfx1250
catalog provides its own WMMA operations. The checked-in catalogs cover
supported f16, bf16, fp8e4m3, and bf8e5m2 input combinations with f32
accumulation; operation shape, layout, and availability are exact-gfx facts.
Wavefront mode is a separate compile-time target capability: gfx942/gfx950 admit
wave64 only; gfx1250 admits wave32 only; and gfx1151, gfx11-generic, and gfx1201
default to wave32 while permitting wave64. `ArchTarget` records one validated mode
per exact target. Another rocKE mode needs matching backend, operation-layout,
geometry, and validator support. Wave mode must agree with the operation layout but
does not select MFMA versus WMMA.

MFMA and WMMA are **matmul primitives**. They implement:
* GEMM (`C += A @ B`)
* Convolution (rewritten as implicit-GEMM)
* Attention (`scores = Q @ K^T`, `out = P @ V`)
* MoE per-expert GEMMs

It does **not** accelerate:
* Element-wise ops (`y = silu(x)`, `z = x + y`, `q = round(x * s)`)
* Reductions (`max(x[0..N])`, `sum(x[0..N])`)
* Norm layers (reduce + scale + add)
* Softmax outside attention (reduce + exponential + normalize)
* Pooling (windowed reduce)
* Scatter / gather (`y[idx] = x` or `y = x[idx]`)
* Histogram / scan / sort
* Quantization (cast + saturate)

Representing non-matmul work as a matrix product (for example, a reduction as
`[1, N] @ [N, 1]`) materializes matrix-shaped work and output ownership that
the scalar result does not need. Use the primitive that matches the operation's
actual shape.

## Kernel-by-kernel taxonomy

The columns are:

* **Shape** -- matmul / reduce / elementwise / data movement / mixed.
* **Primitive** -- MFMA / WMMA / VALU / cross-lane or LDS reduction /
  global data movement / atomic / hybrid.
* **Status** -- whether the row describes the current checked-in body.

### Matmul-shaped kernels (use a supported matrix path)

The exact primitive is target- and builder-specific: supported common builders
select MFMA or WMMA through the exact gfx target's catalog, while MFMA-only
families reject targets that do not provide the required MFMA operation. Wave
width is configured separately at compile time and is not a matrix-family
control.

| Kernel | Shape | Primitive | Status |
|---|---|---|---|
| `gemm_universal` | matmul | target-selected MFMA or WMMA | ✓ |
| `batched_gemm` | matmul (batched) | MFMA or WMMA (via universal) | ✓ |
| `grouped_gemm` | matmul (per-group) | MFMA or WMMA (via universal) | ✓ |
| `flatmm` | matmul (small-decode) | MFMA or WMMA (via universal) | ✓ |
| `gemm_multi_d` | matmul + variadic D | MFMA or WMMA (via universal) | ✓ |
| `gemm_multi_abd` | matmul (one A and one B + optional D; multi-A/B planned) | MFMA or WMMA (via universal) | ✓ (v1 subset) |
| `mfma_gemm` | matmul (16x16 atom) | MFMA direct | ✓ |
| `streamk_gemm` | matmul (atomic split-K) | MFMA + atomic f32 | ✓ |
| `block_scale_gemm` | matmul (FP8/BF8 + scale) | MFMA + explicit per-group scale | ✓ (`abquant` subset) |
| `mx_gemm` | matmul (MX shared exponent) | MFMA + E8M0 decode/scale | ✓ (FP8/BF8 subset) |
| `batched_contraction` | matmul (N-D) | MFMA or WMMA (via universal) | ✓ |
| `conv_implicit_gemm` | conv = matmul | target-selected MFMA or restricted WMMA | ✓ |
| `conv_direct_grouped` | conv (small-channel) | MFMA 4x4x4 atom | ✓ |
| `fused_moe` per-expert | matmul (per-expert) | MFMA or WMMA (via universal) | ✓ |
| `attention_tiled_2d` | attention (paged) | gfx942/gfx950 MFMA or gfx1250 WMMA QK + PV | ✓ |
| `attention_tiled_3d` | attention (split-KV) | gfx942/gfx950 MFMA or gfx1250 WMMA QK + PV | ✓ |
| `fmha_mfma` | attention | target-selected MFMA or WMMA QK + PV (historical family name) | ✓ |
| `fmha_varlen` | attention (varlen) | current MFMA QK + PV body | ✓ |
| `fmha_head_grouping` | attention (GQA / MQA) | current MFMA QK + PV body | ✓ |
| `fmha_paged_prefill` | attention (paged) | spec-selectable MFMA or warp-distributed body | ✓ |
| `fmha_splitkv_decode` | attention (split-KV) | warp-distributed scalar segment + reduction | ✓ |
| `fmha_fwd_fp8` | attention (fp8 K/V) | dequant + f16 MFMA QK/PV | ✓ (f16 activation/output contract) |
| `fmha_bwd` | dQ/dK/dV attention backward | warp-distributed scalar + global atomics | ✓ |
| `sage_attention` | attention + per-block scale | MFMA for aligned fp16/fp8 modes; warp fallback otherwise | ✓ |
| `jenga_sparse_attention` | attention (block-sparse) | MFMA + LDS-staged mask predicate | ✓ |
| `vsa_sparse_attention` | attention (LUT-sparse) | MFMA + LDS-staged LUT bitmap | ✓ |

The shared target-aware forward body is
`<platform_root>/python/rocke/helpers/mfma_attention.py::mfma_attention_fwd_inner_body`;
the warp fallback is
`<library_root>/kernels/common/_fmha_warp_body.py`. Each owning validator
decides which body and target-catalog operation its current spec can select.
The current `fmha_varlen`, `fmha_head_grouping`, and paged-prefill gfx9 MFMA paths
compile for required wave64 geometry; that constraint is not why they use MFMA.

### Non-matmul kernels (correctly NOT MFMA)

These kernels have **no matmul** in their inner loop; using matrix MMA would
require adding matrix-shaped work that the operation does not need.

| Kernel | Shape | Primitive | Why not matrix MMA |
|---|---|---|---|
| `layernorm2d` | reduce + scale | VALU + Welford LDS reduction | reduce, not matmul |
| `rmsnorm2d` | reduce + scale | VALU + warp/LDS reduction | reduce, not matmul |
| `add_rmsnorm2d_rdquant` | reduce + scale + quant | VALU + paired LDS reductions | reduce + cast |
| `smoothquant` | row-reduce + per-row cast | VALU + LDS block-max | row-reduce + cast |
| `moe_smoothquant` | per-expert smoothquant | VALU + LDS block-max | row-reduce + cast |
| `reduce` | reduce (axis sum/max/min/mean/prod) | VALU + target-wave shuffle and cross-wave LDS, with full-LDS fallback | pure reduce |
| `pooling` | windowed reduce | VALU + descriptor-driven buffer loads | small-window reduce |
| `elementwise` | unary / binary / swiglu | VALU SIMD | pure pointwise |
| `permute_nd` | rank-N transpose | descriptor-driven scalar/vector global load + store | pure data motion |
| `transpose` | 2D transpose | LDS-staged/coalesced data movement | pure data motion |
| `batched_transpose` | batched 2D transpose | LDS-staged/coalesced data movement | pure data motion |
| `img2col` | NHWC → unfold matrix | descriptor-driven vector load or scalar gather + global store | data prep (matmul follows in `conv_implicit_gemm`) |
| `topk_softmax` | tournament reduce over K | VALU + wave-XOR argmax when the block fits one target wave; LDS reduction otherwise | tournament reduce |
| `moe_sorting` | histogram + scan + scatter | LDS/global atomics + wave Kogge-Stone or LDS Hillis-Steele scan + scatter | mixed reduce / data movement |
| `moe_gather` | gather by token-expert id | indexed global load + store | pure gather |
| `moe_silu_mul` | elementwise activation | VALU SIMD | pure pointwise |
| `moe_topk_weighted_reduce` | weighted sum across experts | VALU + atomic | reduce + scatter |
| `fmha_appendkv` | cache scatter + optional rotary | global load/store + optional VALU | data motion + pointwise transform |

Using MFMA or WMMA for the current inner loops would not match their operation
shape. For example, expressing a scalar layer-norm reduction as a matrix product would
execute an MFMA for every K tile while materializing a full output tile to
obtain one scalar, followed by a horizontal reduction across that tile. The
extra matrix work and data rearrangement do not match the reduction shape.

## Choosing a primitive

Do not infer the active primitive from the family name alone. Paged prefill and
Sage attention select between matrix and warp-distributed bodies, while
split-KV decode and backward currently use warp-distributed scalar bodies.
Read the owning validator and build function, then confirm the emitted IR/ISA.
Resolve `ArchTarget` for the exact gfx target and select through its
`MmaCatalog`. Resolve a target-supported wavefront mode and validate it against the
operation layout. Do not offer wave32 on gfx942/gfx950 or wave64 on gfx1250; wave
width does not choose the matrix instruction family or establish target support.

If a new kernel lands and someone wants to know "should this use a matrix
instruction?", the rule is:

1. Does the inner loop compute `C[i, j] += sum_k A[i, k] * B[k, j]`?
   * If yes → select a matching `MmaOp` from the exact gfx target's catalog and
     use its layout maps through the owning builder
     (`mfma_gemm_inner.mfma_k_loop` or `mfma_attention_fwd_inner_body` for
     current matrix-oriented families).
2. Is the inner loop a reduction `acc = op(acc, x[i])`?
   * If yes → choose by reduction scope and combiner: a wave-local shuffle,
     an LDS block reduction from `helpers/reduction.py`, or a hybrid wave/LDS
     reduction. Welford and paired reductions require their matching combiners;
     `helpers/attention.warp_xor_reduce_*` is one wave-local implementation,
     not the universal reduction path.
3. Is the inner loop a pure pointwise transform?
   * If yes → straight VALU; no extra helper needed.
4. Is the inner loop a scatter / gather with no compute?
   * If yes → the appropriate global/buffer load and store operations;
     `helpers/persistent.py` if the work is irregular.

Anything else is a custom kernel; consult
[`../optimization/optimization_runbook.md`](../optimization/optimization_runbook.md).
