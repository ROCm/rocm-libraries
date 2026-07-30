# Kernel Taxonomy

This page inventories the current kernel builders and the compute primitive used
by each body. It is grounded in two current source locations:

- [`P`](../../python/rocke/instances) = `platform/python/rocke/instances`
- [`L`](../../../library/kernels) = `library/kernels`

These aliases describe the current source layout only. They do not define
different authoring rules or separate kinds of kernels.

## Matrix Operations By Exact Target

The checked-in
[`arch_specs.json`](../../python/rocke/core/arch/data/arch_specs.json) file is
the source of truth for the `MmaOp` entries available to each exact gfx target.
The catalog currently contains the following operation IDs.

- **`gfx90a` (wave64)**
  - f16: `mfma_f32_16x16x16_f16`, `mfma_f32_32x32x8_f16`
  - bf16: `mfma_f32_16x16x16_bf16`, `mfma_f32_32x32x8_bf16`
- **`gfx942` (wave64)**
  - f32: `mfma_f32_16x16x4_f32`, `mfma_f32_32x32x2_f32`
  - f16: `mfma_f32_16x16x16_f16`, `mfma_f32_32x32x8_f16`
  - bf16: `mfma_f32_16x16x16_bf16`, `mfma_f32_32x32x8_bf16`
  - fp8: `mfma_f32_16x16x32_fp8`, `mfma_f32_32x32x16_fp8`
  - bf8: `mfma_f32_16x16x32_bf8`, `mfma_f32_32x32x16_bf8`
- **`gfx950` (wave64)**
  - f32: `mfma_f32_16x16x4_f32`, `mfma_f32_32x32x2_f32`
  - f16: `mfma_f32_16x16x16_f16`, `mfma_f32_16x16x32_f16`,
    `mfma_f32_32x32x8_f16`, `mfma_f32_32x32x16_f16`
  - bf16: `mfma_f32_16x16x16_bf16`, `mfma_f32_16x16x32_bf16`,
    `mfma_f32_32x32x8_bf16`, `mfma_f32_32x32x16_bf16`
  - fp8: `mfma_f32_16x16x32_fp8`, `mfma_f32_32x32x16_fp8`
  - bf8: `mfma_f32_16x16x32_bf8`, `mfma_f32_32x32x16_bf8`
  - fp4: `mfma_f32_16x16x128_fp4`
  - fp6: `mfma_f32_16x16x96_fp6`
- **`gfx1151` and `gfx11-generic` (wave32)**
  - f16: `wmma_f32_16x16x16_f16`
  - bf16: `wmma_f32_16x16x16_bf16`
  - integer: `wmma_i32_16x16x16_iu4`, `wmma_i32_16x16x16_iu8`
- **`gfx1201` (wave32)**
  - f16: `wmma_gfx12_f32_16x16x16_f16`
  - bf16: `wmma_gfx12_f32_16x16x16_bf16`
- **`gfx1250` (wave32)**
  - f16: `wmma_gfx1250_f32_16x16x32_f16`
  - bf16: `wmma_gfx1250_f32_16x16x32_bf16`
  - fp8/bf8 combinations: `wmma_gfx1250_f32_16x16x64_fp8_fp8`,
    `wmma_gfx1250_f32_16x16x64_fp8_bf8`,
    `wmma_gfx1250_f32_16x16x64_bf8_fp8`,
    `wmma_gfx1250_f32_16x16x64_bf8_bf8`

Catalog presence, builder support, and wave geometry are separate facts. A
builder must select an operation accepted by its validator and use the layout
maps attached to that `MmaOp`. A catalog row does not imply that every builder
supports every operation in that row.

## Inventory Method

The tables below were produced by scanning both `P` and `L` for kernel-building
entry points. A row groups private helper builders or exact-target
implementations only when they implement the same public kernel family. The
source column names every module in that group so the grouping remains
auditable. Registry wrappers are listed with the implementation they select.

Application-integration modules are summarized by operation rather than by an
application or model name. This keeps the taxonomy about reusable computation
and exact gfx targets.

## GEMM, Convolution, And Fused Matrix Kernels

| Builder or family | Current source | Shape | Primitive or body | Target selection |
|---|---|---|---|---|
| `build_universal_gemm` | [P/common/gemm_universal.py](../../python/rocke/instances/common/gemm_universal.py) | GEMM | catalog-selected matrix MMA | owning validator |
| `build_batched_gemm`, `build_persistent_batched_gemm` | [P/common/batched_gemm.py](../../python/rocke/instances/common/batched_gemm.py) | batched GEMM | universal or persistent catalog-selected matrix MMA | owning validator |
| `build_grouped_gemm`, `build_grouped_gemm_single_launch` | [P/common/grouped_gemm.py](../../python/rocke/instances/common/grouped_gemm.py) | grouped GEMM | universal per-group or single-launch matrix MMA | owning validator |
| `build_flatmm` | [P/common/flatmm.py](../../python/rocke/instances/common/flatmm.py) | small-M GEMM | catalog-selected matrix MMA | owning validator |
| `build_gemm_multi_d` | [P/common/gemm_multi_d.py](../../python/rocke/instances/common/gemm_multi_d.py) | GEMM plus D operands | universal matrix MMA plus fused epilogue | owning validator |
| `build_gemm_multi_abd` | [P/common/gemm_multi_abd.py](../../python/rocke/instances/common/gemm_multi_abd.py) | GEMM with current A/B/D contract | universal matrix MMA plus fused epilogue | owning validator |
| `build_batched_contraction` | [P/common/batched_contraction.py](../../python/rocke/instances/common/batched_contraction.py) | batched N-D contraction | universal matrix MMA | owning validator |
| `build_wsp3_gemm` | [P/common/gemm_wsp3.py](../../python/rocke/instances/common/gemm_wsp3.py) | GEMM | universal GEMM body with WSP3 schedule | owning validator |
| `build_mfma_gemm` | [P/common/mfma_gemm.py](../../python/rocke/instances/common/mfma_gemm.py) | GEMM | direct MFMA body | validator requires its selected MFMA operation |
| `build_streamk_gemm`, `build_streamk_gemm_block_tile` | [P/common/streamk_gemm.py](../../python/rocke/instances/common/streamk_gemm.py) | split-K GEMM | MFMA plus workspace or atomic reduction | owning validator |
| `build_block_scale_gemm` | [P/common/block_scale_gemm.py](../../python/rocke/instances/common/block_scale_gemm.py) | scaled low-precision GEMM | MFMA plus explicit scale handling | owning validator |
| `build_mx_gemm` | [P/common/mx_gemm.py](../../python/rocke/instances/common/mx_gemm.py) | shared-exponent GEMM | target-catalog MFMA plus exponent decode/scale | owning validator |
| `build_matmul_nbits`, `build_decode_gemv_matmul_nbits`, `build_large_n_matmul_nbits`, `build_large_n_opt_matmul_nbits` | [P/common/matmul_nbits.py](../../python/rocke/instances/common/matmul_nbits.py), [P/common/_matmul_nbits_decode_gemv.py](../../python/rocke/instances/common/_matmul_nbits_decode_gemv.py), [P/common/_matmul_nbits_large_n.py](../../python/rocke/instances/common/_matmul_nbits_large_n.py), [P/common/_matmul_nbits_large_n_opt.py](../../python/rocke/instances/common/_matmul_nbits_large_n_opt.py) | quantized matmul or decode GEMV | family-selected dequantization plus matrix or vector body | family validator |
| `build_implicit_gemm_conv` | [P/common/conv_implicit_gemm.py](../../python/rocke/instances/common/conv_implicit_gemm.py) | convolution as GEMM | catalog-selected matrix MMA | owning validator |
| `build_direct_conv_16c` | [P/common/conv_direct_grouped.py](../../python/rocke/instances/common/conv_direct_grouped.py) | grouped direct convolution | `mfma_f32_16x16x16_f16`, or `mfma_f32_16x16x32_f16` with K folding | owning validator |
| `build_direct_conv_4c` | [P/common/conv_direct_grouped.py](../../python/rocke/instances/common/conv_direct_grouped.py) | small-channel grouped convolution | direct `mfma_f32_4x4x4_f16` intrinsic outside `MmaCatalog` | validated exact gfx target |
| `build_deep_fused_conv_pool` | [P/common/deep_fused_conv_pool.py](../../python/rocke/instances/common/deep_fused_conv_pool.py), [P/gfx1151/deep_fused_conv_pool.py](../../python/rocke/instances/gfx1151/deep_fused_conv_pool.py), [P/gfx1201/deep_fused_conv_pool.py](../../python/rocke/instances/gfx1201/deep_fused_conv_pool.py), [P/gfx950/deep_fused_conv_pool.py](../../python/rocke/instances/gfx950/deep_fused_conv_pool.py) | convolution plus pooling | target-specific matrix MMA plus VALU/window reduction | exact implementation selected by target |
| `build_moe_gate_up_silu_gemm`, `build_moe_interleaved_gate_up_silu_gemm`, `build_moe_down_reduce_gemm`, `build_moe_down_silu_reduce_gemm` | [P/common/moe_gemm_fused.py](../../python/rocke/instances/common/moe_gemm_fused.py) | expert GEMM plus activation or reduction | catalog-selected matrix MMA plus fused VALU/reduction | owning validator |
| `build_moe_fused_mega_gemm`, `build_moe_fused_mega_gemm_fp8`, `build_moe_fused_mega_wmma` | [P/common/moe_fused_mega.py](../../python/rocke/instances/common/moe_fused_mega.py), [P/common/moe_fused_mega_fp8.py](../../python/rocke/instances/common/moe_fused_mega_fp8.py), [P/gfx1250/fused_moe_mega_wmma.py](../../python/rocke/instances/gfx1250/fused_moe_mega_wmma.py) | persistent expert GEMM pipeline | target-selected matrix MMA plus fused routing/activation | owning validator or exact-target module |
| `build_wmma_gemm` | [P/gfx1151/wmma_gemm.py](../../python/rocke/instances/gfx1151/wmma_gemm.py), [P/gfx1201/wmma_gemm.py](../../python/rocke/instances/gfx1201/wmma_gemm.py), [P/gfx1250/wmma_gemm.py](../../python/rocke/instances/gfx1250/wmma_gemm.py) | exact-target GEMM | target-catalog WMMA | module's exact gfx target |
| `build_wmma_gemm_int8`, `build_wmma_gemm_iu8`, `build_wmma_gemm_iu8_dequant` | [P/gfx1151/wmma_gemm_int8.py](../../python/rocke/instances/gfx1151/wmma_gemm_int8.py), [P/gfx1151/wmma_gemm_iu8.py](../../python/rocke/instances/gfx1151/wmma_gemm_iu8.py), [P/gfx1151/wmma_gemm_iu8_dequant.py](../../python/rocke/instances/gfx1151/wmma_gemm_iu8_dequant.py) | integer GEMM, optionally dequantized | target-catalog integer WMMA plus optional conversion | `gfx1151` |
| `build_block_scaled_gemm` | [P/gfx1250/block_scaled_gemm.py](../../python/rocke/instances/gfx1250/block_scaled_gemm.py) | block-scaled low-precision GEMM | target-catalog WMMA plus scale handling | `gfx1250` |

## Reduction, Pointwise, And Data-Movement Kernels

| Builder or family | Current source | Shape | Primitive or body |
|---|---|---|---|
| `build_elementwise` | [P/common/elementwise.py](../../python/rocke/instances/common/elementwise.py) | unary or binary pointwise | VALU |
| `build_layernorm2d` | [P/common/layernorm2d.py](../../python/rocke/instances/common/layernorm2d.py) | reduction plus scale/shift | VALU plus Welford LDS reduction |
| `build_rmsnorm2d` | [P/common/rmsnorm2d.py](../../python/rocke/instances/common/rmsnorm2d.py) | reduction plus scale | VALU plus wave/LDS reduction |
| `build_add_rmsnorm2d_bf16` | [P/common/add_rmsnorm2d_bf16.py](../../python/rocke/instances/common/add_rmsnorm2d_bf16.py) | add plus reduction and scale | VALU plus block reduction |
| `build_add_rmsnorm2d_rdquant` | [P/common/add_rmsnorm2d_rdquant.py](../../python/rocke/instances/common/add_rmsnorm2d_rdquant.py) | add plus reduction, scale, and quantization | VALU plus paired LDS reductions and conversion |
| `build_smoothquant` | [P/common/smoothquant.py](../../python/rocke/instances/common/smoothquant.py) | row reduction plus quantization | VALU plus LDS block maximum |
| `build_moe_smoothquant` | [P/common/moe_smoothquant.py](../../python/rocke/instances/common/moe_smoothquant.py) | per-expert row reduction plus quantization | VALU plus LDS block maximum |
| `build_reduce2d` | [P/common/reduce.py](../../python/rocke/instances/common/reduce.py) | row reduction | VALU plus wave/LDS reduction selected by scope |
| `build_pooling2d` | [P/common/pooling.py](../../python/rocke/instances/common/pooling.py) | windowed reduction | descriptor-driven loads plus VALU reduction |
| `build_transpose2d` | [P/common/transpose.py](../../python/rocke/instances/common/transpose.py) | 2D transpose | LDS-staged data movement |
| `build_batched_transpose2d` | [P/common/batched_transpose.py](../../python/rocke/instances/common/batched_transpose.py) | batched 2D transpose | LDS-staged data movement |
| `build_transpose_bc` | [P/common/transpose_bc.py](../../python/rocke/instances/common/transpose_bc.py) | B/C dimension transpose | LDS-staged data movement |
| `build_permute` | [P/common/permute_nd.py](../../python/rocke/instances/common/permute_nd.py) | rank-N permutation | descriptor-driven global load/store |
| `build_img2col` | [P/common/img2col.py](../../python/rocke/instances/common/img2col.py) | convolution-to-matrix transform | descriptor-driven global load/store |
| `build_topk_softmax` | [P/common/topk_softmax.py](../../python/rocke/instances/common/topk_softmax.py) | selection plus softmax reduction | VALU plus wave/LDS tournament reduction |
| `build_moe_sort_histogram`, `build_moe_sort_scan`, `build_moe_sort_scatter`, `build_moe_sort_persistent` | [P/common/moe_sorting.py](../../python/rocke/instances/common/moe_sorting.py) | histogram, scan, scatter, or persistent combination | LDS/global atomics, scan, and data movement |
| `build_moe_gather`, `build_moe_silu_mul`, `build_moe_silu_mul_packed`, `build_moe_static_scatter_gather`, `build_moe_topk_weighted_reduce` | [P/common/fused_moe.py](../../python/rocke/instances/common/fused_moe.py) | gather, pointwise activation, scatter/gather, or weighted reduction | indexed loads/stores, VALU, and atomics as selected by the stage |
| five target-specific application-integration builders | [`P/gfx1250/`](../../python/rocke/instances/gfx1250) | embedding, sampling, normalization/rotation, or cache-update operations | target-specific VALU, reduction, and global load/store bodies |

## Attention Kernels

| Builder or family | Current source | Shape | Primitive or body | Target selection |
|---|---|---|---|---|
| `build_unified_attention_2d`, `build_unified_attention_3d`, `build_unified_attention_reduce` | [L/common/attention_unified.py](../../../library/kernels/common/attention_unified.py) | paged attention and split reduction | warp-distributed scalar/VALU body | owning validator |
| `build_unified_attention_2d_tiled`, `build_unified_attention_3d_tiled`, `build_unified_attention_reduce_tiled` | [L/gfx942/attention_tiled_2d.py](../../../library/kernels/gfx942/attention_tiled_2d.py), [L/gfx942/attention_tiled_3d.py](../../../library/kernels/gfx942/attention_tiled_3d.py), [L/gfx950/attention_tiled_2d.py](../../../library/kernels/gfx950/attention_tiled_2d.py), [L/gfx950/attention_tiled_3d.py](../../../library/kernels/gfx950/attention_tiled_3d.py), [L/gfx1250/attention_tiled_2d.py](../../../library/kernels/gfx1250/attention_tiled_2d.py), [L/gfx1250/attention_tiled_3d.py](../../../library/kernels/gfx1250/attention_tiled_3d.py) | QK/PV attention plus optional split reduction | exact-target matrix MMA plus VALU softmax/reduction | module's exact gfx target |
| `build_unified_attention_2d_fastkv_register_p` | [L/gfx950/attention_tiled_2d_fastkv_regp.py](../../../library/kernels/gfx950/attention_tiled_2d_fastkv_regp.py) | tiled attention with register-resident probability fragments | MFMA plus VALU softmax | `gfx950` |
| `build_attention_dense` | [L/gfx950/attention_dense.py](../../../library/kernels/gfx950/attention_dense.py) | dense attention | target-specific MFMA plus VALU softmax | `gfx950` |
| `build_wmma_fmha_fwd` | [L/gfx1151/wmma_fmha_fwd.py](../../../library/kernels/gfx1151/wmma_fmha_fwd.py) | forward attention | target-catalog WMMA QK/PV plus VALU softmax | `gfx1151` or validator-accepted compatible target |
| `build_wmma_attention_fwd` | [L/gfx1250/wmma_attention_fwd.py](../../../library/kernels/gfx1250/wmma_attention_fwd.py) | forward attention | target-catalog WMMA QK/PV plus VALU softmax | `gfx1250` |
| `build_fmha_fwd_mfma` | [L/common/fmha_mfma.py](../../../library/kernels/common/fmha_mfma.py) | forward attention | validator-selected matrix MMA QK/PV plus VALU softmax | owning validator |
| `build_fmha_fwd_varlen` | [L/common/fmha_varlen.py](../../../library/kernels/common/fmha_varlen.py) | variable-length forward attention | current matrix MMA QK/PV body | owning validator |
| `build_fmha_fwd_head_grouping` | [L/common/fmha_head_grouping.py](../../../library/kernels/common/fmha_head_grouping.py) | grouped-head forward attention | current matrix MMA QK/PV body | owning validator |
| `build_fmha_fwd_paged_prefill` | [L/common/fmha_paged_prefill.py](../../../library/kernels/common/fmha_paged_prefill.py) | paged forward attention | spec-selected matrix or warp-distributed body | owning validator |
| `build_fmha_fwd_splitkv_decode_segment`, `build_fmha_fwd_splitkv_decode_reduce` | [L/common/fmha_splitkv_decode.py](../../../library/kernels/common/fmha_splitkv_decode.py) | split-KV attention plus reduction | warp-distributed scalar segment and reduction bodies | owning validator |
| `build_fmha_fwd_fp8` | [L/common/fmha_fwd_fp8.py](../../../library/kernels/common/fmha_fwd_fp8.py) | low-precision K/V forward attention | dequantization plus f16 matrix MMA QK/PV | owning validator |
| `build_fmha_bwd` | [L/common/fmha_bwd.py](../../../library/kernels/common/fmha_bwd.py) | attention backward | warp-distributed scalar body plus global atomics | owning validator |
| `build_sage_attention` | [L/common/sage_attention.py](../../../library/kernels/common/sage_attention.py) | scaled attention | aligned matrix body with warp fallback | owning validator |
| `build_jenga_sparse_attention`, `build_vsa_sparse_attention` | [L/common/sparse_attention.py](../../../library/kernels/common/sparse_attention.py) | block- or lookup-sparse attention | matrix MMA plus predicate/LUT handling | owning validator |
| `build_fmha_fwd_appendkv` | [L/common/fmha_appendkv.py](../../../library/kernels/common/fmha_appendkv.py) | cache update plus optional rotary transform | global load/store plus optional VALU | owning validator |

## Choosing A Primitive

Do not infer a primitive from a family or module name. Read the selected build
function and validator, then confirm the emitted IR or ISA when necessary.

1. For an inner loop of the form
   `C[i, j] += sum_k A[i, k] * B[k, j]`, select a matching `MmaOp` from the
   exact target catalog and use its operand/output layouts.
2. For a reduction, choose a wave-local shuffle, an LDS block reduction, or a
   hybrid wave/LDS reduction according to the scope and combiner. Welford and
   paired reductions require their matching combiners.
3. For a pointwise transform, use VALU operations.
4. For data movement, use the global/buffer load and store path that matches the
   addressing and bounds contract.

For optimization workflow guidance, see
[`../optimization/optimization_runbook.md`](../optimization/optimization_runbook.md).
