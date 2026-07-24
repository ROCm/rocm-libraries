# Kernel taxonomy: which primitive does each kernel use, and why

> Matmul-shaped kernels use MFMA or WMMA where their current builder supports
> it. Some attention variants still use a warp-distributed scalar body.
> Non-matmul ops use VALU, cross-lane reductions, atomics, or data movement
> according to their actual computation.

This page is an index grounded in the current builders. The owning validator
and builder remain the source of truth when a spec can select more than one
body.

## What MFMA is for

The wave64 `mfma_f32_16x16x16_f16` instruction and related atoms compute a tiled matrix
multiply-accumulate in one instruction. The checked-in catalogs cover the
supported f16, bf16, fp8e4m3, and bf8e5m2 input families with f32
accumulation; exact availability is target-specific.

MFMA is a **matmul primitive**. It accelerates:
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

Trying to "fake" non-matmul work as a matmul (e.g. reduce as
`[1, N] @ [N, 1]`) wastes 99% of the MFMA FLOPs and is slower
than the natural VALU + cross-lane reduction path. **MFMA is not
a hammer; it's a saw.**

## Kernel-by-kernel taxonomy

The columns are:

* **Shape** -- matmul / reduce / elementwise / data movement / mixed.
* **Primitive** -- MFMA / WMMA / VALU / cross-lane or LDS reduction /
  global data movement / atomic / hybrid.
* **Status** -- whether the row describes the current checked-in body.

### Matmul-shaped kernels (use a supported matrix path)

The exact primitive is target- and builder-specific: supported common builders
select MFMA or WMMA through the target catalog, while MFMA-only families reject
WMMA targets. Wave width is not a user-selectable matrix-family control.

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
| `attention_tiled_2d` | attention (paged) | architecture-specific MFMA or WMMA QK + PV | ✓ |
| `attention_tiled_3d` | attention (split-KV) | architecture-specific MFMA or WMMA QK + PV | ✓ |
| `fmha_mfma` | attention | MFMA QK + PV | ✓ |
| `fmha_varlen` | attention (varlen) | MFMA QK + PV | ✓ |
| `fmha_head_grouping` | attention (GQA / MQA) | MFMA QK + PV | ✓ |
| `fmha_paged_prefill` | attention (paged) | spec-selectable MFMA or warp-distributed body | ✓ |
| `fmha_splitkv_decode` | attention (split-KV) | warp-distributed scalar segment + reduction | ✓ |
| `fmha_fwd_fp8` | attention (fp8 K/V) | dequant + f16 MFMA QK/PV | ✓ (f16 activation/output contract) |
| `fmha_bwd` | dQ/dK/dV attention backward | warp-distributed scalar + global atomics | ✓ |
| `sage_attention` | attention + per-block scale | MFMA for aligned fp16/fp8 modes; warp fallback otherwise | ✓ |
| `jenga_sparse_attention` | attention (block-sparse) | MFMA + LDS-staged mask predicate | ✓ |
| `vsa_sparse_attention` | attention (LUT-sparse) | MFMA + LDS-staged LUT bitmap | ✓ |

The shared MFMA forward body is
`python/rocke/helpers/mfma_attention.py::mfma_attention_fwd_inner_body`; the
warp fallback is `rocke/library/kernels/common/_fmha_warp_body.py`. Each family
validator decides which body its current spec can select.

### Non-matmul kernels (correctly NOT MFMA)

These kernels have **no matmul** in their inner loop; using MFMA
would require faking a matmul shape that wastes 99% of the
intrinsic's FLOPs.

| Kernel | Shape | Primitive | Why not matrix MMA |
|---|---|---|---|
| `layernorm2d` | reduce + scale | VALU + Welford LDS reduction | reduce, not matmul |
| `rmsnorm2d` | reduce + scale | VALU + warp/LDS reduction | reduce, not matmul |
| `add_rmsnorm2d_rdquant` | reduce + scale + quant | VALU + paired LDS reductions | reduce + cast |
| `smoothquant` | row-reduce + per-row cast | VALU + LDS block-max | row-reduce + cast |
| `moe_smoothquant` | per-expert smoothquant | VALU + LDS block-max | row-reduce + cast |
| `reduce` | reduce (axis sum/max/min/mean/prod) | VALU + warp shuffle + cross-warp LDS | pure reduce |
| `pooling` | windowed reduce | VALU + tile-window | small-window reduce |
| `elementwise` | unary / binary / swiglu | VALU SIMD | pure pointwise |
| `permute_nd` | rank-N transpose | direct scalar/vector global gather + store | pure data motion |
| `transpose` | 2D transpose | LDS-staged/coalesced data movement | pure data motion |
| `batched_transpose` | batched 2D transpose | LDS-staged/coalesced data movement | pure data motion |
| `img2col` | NHWC → unfold matrix | gather + scatter | data prep (matmul follows in `conv_implicit_gemm`) |
| `topk_softmax` | tournament reduce over K | VALU + cross-lane shuffle | tournament reduce |
| `moe_sorting` | histogram + scan + scatter | atomic + scan + scatter | mixed reduce / data movement |
| `moe_gather` | gather by token-expert id | indexed global load + store | pure gather |
| `moe_silu_mul` | elementwise activation | VALU SIMD | pure pointwise |
| `moe_topk_weighted_reduce` | weighted sum across experts | VALU + atomic | reduce + scatter |
| `fmha_appendkv` | cache scatter + optional rotary | global load/store + optional VALU | data motion + pointwise transform |

Trying to use MFMA or WMMA for any of these would be a categorical mistake. For
example: expressing a scalar layer-norm reduction as a matrix product would
execute an MFMA for every K tile while materializing a full output tile to
obtain one scalar, followed by a horizontal reduction across that tile. The
extra matrix work and data rearrangement do not match the reduction shape.

## Choosing a primitive

Do not infer the active primitive from the family name alone. Paged prefill and
Sage attention select between matrix and warp-distributed bodies, while
split-KV decode and backward currently use warp-distributed scalar bodies.
Read the owning validator and build function, then confirm the emitted IR/ISA.

If a new kernel lands and someone wants to know "should this use a matrix
instruction?", the rule is:

1. Does the inner loop compute `C[i, j] += sum_k A[i, k] * B[k, j]`?
   * If yes → the target-supported MFMA/WMMA path via the relevant catalog and
     builder (`mfma_gemm_inner.mfma_k_loop` or
     `mfma_attention_fwd_inner_body` for current MFMA-oriented families).
2. Is the inner loop a reduction `acc = op(acc, x[i])`?
   * If yes → `helpers/attention.warp_xor_reduce_*` family.
3. Is the inner loop a pure pointwise transform?
   * If yes → straight VALU; no extra helper needed.
4. Is the inner loop a scatter / gather with no compute?
   * If yes → the appropriate global/buffer load and store operations;
     `helpers/persistent.py` if the work is irregular.

Anything else is a custom kernel; consult
[`../optimization/optimization_runbook.md`](../optimization/optimization_runbook.md).
