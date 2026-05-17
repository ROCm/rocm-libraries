# Kernel taxonomy: which primitive does each kernel use, and why

> Every matmul-shaped op uses MFMA. Non-matmul ops use the right
> hardware primitive for their shape -- VALU for elementwise,
> `ds_bpermute` for cross-lane reductions, async DMA for pure
> scatter / gather. The whole point of GPUs is using the right
> SIMD primitive for the work.

This file is the **source of truth** for which kernel uses which
primitive, and why. The gremlin auditor walks this list whenever
the question "shouldn't this be MFMA?" comes up.

## What MFMA is for

CDNA3's `mfma_f32_16x16x16_f16` (and siblings) computes a 16×16×16
matrix multiply-accumulate **in one instruction**: 4 cycles, 256
MAC operations, 64 FLOPs/cycle/lane. The intrinsic family covers
f16 / bf16 / fp8e4m3 / bf8e5m2 inputs and f32 accumulator output.

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

* **Shape** -- matmul / reduce / elementwise / DMA / mixed.
* **Primitive** -- MFMA / VALU / `ds_bpermute` (warp reduce) /
  async DMA / atomic / hybrid.
* **Status** -- on the right primitive (`✓`) or follow-on (`→`).

### Matmul-shaped kernels (must use MFMA)

| Kernel | Shape | Primitive | Status |
|---|---|---|---|
| `gemm_universal` | matmul | MFMA | ✓ |
| `batched_gemm` | matmul (batched) | MFMA (via universal) | ✓ |
| `grouped_gemm` | matmul (per-group) | MFMA (via universal) | ✓ |
| `flatmm` | matmul (small-decode) | MFMA (via universal) | ✓ |
| `gemm_multi_d` | matmul + variadic D | MFMA (via universal) | ✓ |
| `gemm_multi_abd` | matmul (multi-A/B) | MFMA (via universal) | ✓ |
| `mfma_gemm` | matmul (16x16 atom) | MFMA direct | ✓ |
| `streamk_gemm` | matmul (atomic split-K) | MFMA + atomic f32 | ✓ |
| `block_scale_gemm` | matmul (fp8 + scale) | MFMA fp8 + per-group scale | ✓ |
| `mx_gemm` | matmul (MX shared exp) | MFMA fp8 + E8M0 decode | ✓ |
| `batched_contraction` | matmul (N-D) | MFMA (via universal) | ✓ |
| `conv_implicit_gemm` | conv = matmul | MFMA (via universal) | ✓ |
| `conv_direct_grouped` | conv (small-channel) | MFMA 4x4x4 atom | ✓ |
| `fused_moe` per-expert | matmul (per-expert) | MFMA (via universal) | ✓ |
| `attention_tiled_2d` | attention (paged) | MFMA QK + PV | ✓ |
| `attention_tiled_3d` | attention (split-KV) | MFMA QK + PV | ✓ |
| `fmha_mfma` | attention | MFMA QK + PV | ✓ |
| `fmha_varlen` | attention (varlen) | warp-scalar (MFMA pending) | → |
| `fmha_head_grouping` | attention (GQA / MQA) | warp-scalar (MFMA pending) | → |
| `fmha_paged_prefill` | attention (paged) | warp-scalar (MFMA pending) | → |
| `fmha_splitkv_decode` | attention (split-KV) | warp-scalar (MFMA pending) | → |
| `fmha_fwd_fp8` | attention (fp8 K/V) | warp-scalar (MFMA pending) | → |
| `fmha_bwd` | 3 matmuls (dQ/dK/dV) | warp-scalar (MFMA pending) | → |
| `sage_attention` | attention + per-block scale | warp-scalar (MFMA pending) | → |
| `jenga_sparse_attention` | attention (block-sparse) | warp-scalar (MFMA pending) | → |
| `vsa_sparse_attention` | attention (LUT-sparse) | warp-scalar (MFMA pending) | → |

The `→` entries are matmul-shaped but currently use the warp-
distributed scalar inner body. The MFMA hoist drops in via the
shared `helpers/mfma_attention.py::mfma_attention_fwd_inner_body`
helper -- the variant kernels keep their I/O layer (cu_seqlens,
block_table, segment range, fp8 dequant, codebook lookup, mask
predicate) and swap the inner body. This work is staged after the
`fmha_mfma` reference path proved out the helper.

### Non-matmul kernels (correctly NOT MFMA)

These kernels have **no matmul** in their inner loop; using MFMA
would require faking a matmul shape that wastes 99% of the
intrinsic's FLOPs.

| Kernel | Shape | Primitive | Why not MFMA |
|---|---|---|---|
| `layernorm2d` | reduce + scale | VALU + `ds_bpermute` warp-sum | reduce, not matmul |
| `rmsnorm2d` | reduce + scale | VALU + warp-sum | reduce, not matmul |
| `add_rmsnorm2d_rdquant` | reduce + scale + quant | VALU + warp-sum | reduce + cast |
| `smoothquant` | row-reduce + per-row cast | VALU + warp-max | row-reduce + cast |
| `moe_smoothquant` | per-expert smoothquant | VALU + warp-max | row-reduce + cast |
| `reduce` | reduce (axis sum/max/min/mean) | `ds_bpermute` butterfly | pure reduce |
| `pooling` | windowed reduce | VALU + tile-window | small-window reduce |
| `elementwise` | unary / binary / swiglu | VALU SIMD | pure pointwise |
| `permute_nd` | rank-N transpose | LDS-coalesced DMA | pure data motion |
| `transpose` | 2D transpose | LDS-coalesced DMA | pure data motion |
| `batched_transpose` | batched 2D transpose | LDS-coalesced DMA | pure data motion |
| `img2col` | NHWC → unfold matrix | gather + scatter | data prep (matmul follows in `conv_implicit_gemm`) |
| `topk_softmax` | tournament reduce over K | VALU + cross-lane shuffle | tournament reduce |
| `moe_sorting` | histogram + scan + scatter | atomic + scan + scatter | mixed reduce / DMA |
| `moe_gather` | gather by token-expert id | indexed DMA | pure gather |
| `moe_silu_mul` | elementwise activation | VALU SIMD | pure pointwise |
| `moe_topk_weighted_reduce` | weighted sum across experts | VALU + atomic | reduce + scatter |
| `fmha_appendkv` | scatter into KV cache | DMA scatter | pure data motion |

Trying to MFMA any of these would be a categorical mistake. For
example: a layer norm reduce over `N=4096` elements has 4096
adds (one per element); writing it as `[1, 4096] @ [4096, 1] = [1, 1]`
on the 16×16×16 atom would execute 256 MFMA atoms (each doing
256 useless MACs at 16-element granularity, then a horizontal
reduce across the atom output -- which itself isn't a matmul).
Net throughput would be **lower** than the warp-shuffle reduce.

## Where MFMA improvements still apply (and where they don't)

The remaining MFMA work in the codebase is the FMHA variant
migration listed above. Every other kernel either:

* Already uses MFMA via the shared helpers, or
* Uses the primitive matched to its shape (reduce / DMA / VALU).

If a new kernel lands and someone wants to know "should this be
MFMA?", the rule is:

1. Does the inner loop compute `C[i, j] += sum_k A[i, k] * B[k, j]`?
   * If yes → MFMA via `mfma_gemm_inner.mfma_k_loop` (or
     `mfma_attention_fwd_inner_body` for the attention case).
2. Is the inner loop a reduction `acc = op(acc, x[i])`?
   * If yes → `helpers/attention.warp_xor_reduce_*` family.
3. Is the inner loop a pure pointwise transform?
   * If yes → straight VALU; no extra helper needed.
4. Is the inner loop a scatter / gather with no compute?
   * If yes → `buffer_load` / `buffer_store` async DMA;
     `helpers/persistent.py` if the work is irregular.

Anything else is a custom kernel; consult `dsl_docs/optimization/
optimization_playbook.md` for the pattern catalog.
