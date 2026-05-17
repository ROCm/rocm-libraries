# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parametric instance builders for the CK DSL.

Each module here knows how to take a *dispatcher-grade* config schema
(the same `(TileConfig, TraitConfig)` shape that
`dispatcher/codegen/default_config.json` and
`dispatcher/codegen/preselected_kernels.py` consume) and emit a full
Python IR kernel ready for `ck_dsl.core.lower_llvm` and
`ck_dsl.runtime.comgr` to compile in-process.

Instance builders decouple three concerns so a sweep driver can mix
and match:

 - the **tile geometry** (block/warp/MFMA-atom shape)
 - the **pipeline** (`mem` / `compv3` / `compv4`)
 - the **epilogue** (`default` direct-store / `cshuffle` LDS-staged)

This mirrors how CK Tile's templates do it; we keep the same names so
benchmarks against CK's generated kernels can refer to the same config.

Builders shipped today:

GEMM family
 - `build_universal_gemm` : the dispatcher-grade GEMM
 (`gemm_universal.UniversalGemmSpec`).
 Covers `tile_m/n/k in [16..256]`, warp
 grids `1x1..4x4`, MFMA atoms
 `16x16x{16,32}`, `32x32x{8,16}` fp16.
 Supports `batched=True` for per-block
 (block_id_z) batch indexing.
 - `build_batched_gemm` : CK Tile `16_batched_gemm` counterpart;
 same kernel as `build_universal_gemm`
 with `batched=True`, plus a clean
 `BatchedGemmSpec` / signature surface.
 - `build_grouped_gemm` + : CK Tile `17_grouped_gemm` counterpart.
 `GroupedGemmLauncher` The launcher re-uses the universal GEMM
 and issues one launch per group; a
 single-launch variant is on the
 follow-up list.

Convolution family
 - `build_implicit_gemm_conv` : NHWC × KRSC -> NHWK implicit-GEMM conv.
 - `build_direct_conv_16c` : `cpg=kpg=16` grouped direct conv with
 K=32 folding.
 - `build_direct_conv_4c` : `cpg=kpg=4` grouped direct conv via
 `mfma_f32_4x4x4_f16`.

Attention
 - `build_unified_attention_2d` : paged-attention 2D scalar.
 - `build_unified_attention_2d_tiled` : production 2D tiled kernel
 (MFMA + async LDS + cshuffle).
 - `build_unified_attention_3d_tiled` : split-KV 3D tiled segment kernel.
 - `build_unified_attention_reduce_tiled` : split-KV reduce kernel.

CK Tile small-op counterparts (Tier 1)
 - `build_elementwise` : CK Tile `21_elementwise` counterpart.
 Unary (copy/neg/abs/relu/silu/gelu_tanh
 /exp2) and binary (add/sub/mul/max/min)
 ops over contiguous f16/bf16 buffers.
 - `build_layernorm2d` : CK Tile `02_layernorm2d` forward, with
 optional save_mean_invstd.
 - `build_rmsnorm2d` : CK Tile `10_rmsnorm2d` forward.
 - `build_reduce2d` : CK Tile `05_reduce` row-wise sum / max /
 mean.
 - `build_transpose2d` : CK Tile `37_transpose` / `35_batched_
 transpose` block transpose with
 LDS-staged bank-padded layout.

Each builder ships with a matching `_signature(spec)` and `_grid(...)`
helper for use with :class:`ck_dsl.runtime.launcher.KernelLauncher`.
End-to-end parity vs torch reference for all of these is exercised by
:mod:`ck_dsl.examples.ck_tile_parity`; the GEMM/attention parity drivers
live in :mod:`ck_dsl.examples.attention.parity_unified_attention` and
:mod:`ck_dsl.examples.bake_off_implicit_gemm`.
"""

from .conv_direct_grouped import (  # noqa: F401
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    build_direct_conv_4c,
    build_direct_conv_16c,
)
from .conv_implicit_gemm import (  # noqa: F401
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    make_a_descriptor,
    make_b_descriptor,
    make_d_descriptor,
)
from .gemm_universal import (  # noqa: F401
    DataSpec,
    Epilogue,
    Pipeline,
    Scheduler,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    all_dispatcher_configs,
    build_universal_gemm,
    is_valid_spec,
)
from .attention_unified import (  # noqa: F401
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    attention_3d_workspace_nbytes,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
    run_unified_attention_torch,
    supports_native_unified_attention,
    supports_native_unified_attention_tiled,
    supports_native_unified_attention_3d_tiled,
)
from .attention_tiled_2d import (  # noqa: F401
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
    supports_tiled_2d,
)
from .attention_tiled_3d import (  # noqa: F401
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
    supports_tiled_3d,
)
from .elementwise import (  # noqa: F401
    BinaryOp,
    DType as ElementwiseDType,
    ElementwiseSpec,
    UnaryOp,
    build_elementwise,
    elementwise_grid,
    elementwise_signature,
    is_valid_spec as is_valid_elementwise_spec,
)
from .layernorm2d import (  # noqa: F401
    LayerNorm2DSpec,
    build_layernorm2d,
    is_valid_spec as is_valid_layernorm2d_spec,
    layernorm2d_grid,
    layernorm2d_signature,
)
from .rmsnorm2d import (  # noqa: F401
    RMSNorm2DSpec,
    build_rmsnorm2d,
    is_valid_spec as is_valid_rmsnorm2d_spec,
    rmsnorm2d_grid,
    rmsnorm2d_signature,
)
from .reduce import (  # noqa: F401
    Reduce2DSpec,
    ReduceOp,
    build_reduce2d,
    is_valid_spec as is_valid_reduce2d_spec,
    reduce2d_grid,
    reduce2d_signature,
)
from .transpose import (  # noqa: F401
    Transpose2DSpec,
    build_transpose2d,
    is_valid_spec as is_valid_transpose2d_spec,
    transpose2d_grid,
    transpose2d_signature,
)
from .batched_transpose import (  # noqa: F401
    BatchedTranspose2DSpec,
    batched_transpose2d_grid,
    batched_transpose2d_signature,
    build_batched_transpose2d,
    is_valid_spec as is_valid_batched_transpose2d_spec,
)
from .img2col import (  # noqa: F401
    Img2ColSpec,
    build_img2col,
    img2col_grid,
    img2col_signature,
    is_valid_spec as is_valid_img2col_spec,
)
from .pooling import (  # noqa: F401
    Pooling2DSpec,
    PoolingProblem,
    PoolOp,
    build_pooling2d,
    is_valid_spec as is_valid_pooling2d_spec,
    pooling2d_grid,
    pooling2d_signature,
)
from .permute_nd import (  # noqa: F401
    MAX_RANK as PERMUTE_MAX_RANK,
    PermuteSpec,
    build_permute,
    is_valid_spec as is_valid_permute_spec,
    permute_grid,
    permute_signature,
)
from .gemm_multi_d import (  # noqa: F401
    DOp as GemmMultiDOp,
    GemmMultiDSpec,
    MAX_D as GEMM_MULTI_D_MAX,
    build_gemm_multi_d,
    gemm_multi_d_grid,
    gemm_multi_d_signature,
    is_valid_spec as is_valid_gemm_multi_d_spec,
)
from .gemm_multi_abd import (  # noqa: F401
    AOperand as GemmMultiAbdAOperand,
    BOperand as GemmMultiAbdBOperand,
    DOperand as GemmMultiAbdDOperand,
    GemmMultiAbdSpec,
    MAX_A as GEMM_MULTI_ABD_MAX_A,
    MAX_B as GEMM_MULTI_ABD_MAX_B,
    build_gemm_multi_abd,
    gemm_multi_abd_grid,
    gemm_multi_abd_signature,
    is_valid_spec as is_valid_gemm_multi_abd_spec,
)
from .smoothquant import (  # noqa: F401
    SmoothQuantSpec,
    build_smoothquant,
    is_valid_spec as is_valid_smoothquant_spec,
    smoothquant_grid,
    smoothquant_signature,
)
from .moe_smoothquant import (  # noqa: F401
    MoeSmoothQuantSpec,
    build_moe_smoothquant,
    is_valid_spec as is_valid_moe_smoothquant_spec,
    moe_smoothquant_grid,
    moe_smoothquant_signature,
)
from .add_rmsnorm2d_rdquant import (  # noqa: F401
    AddRmsnorm2DRdquantSpec,
    add_rmsnorm2d_rdquant_grid,
    add_rmsnorm2d_rdquant_signature,
    build_add_rmsnorm2d_rdquant,
    is_valid_spec as is_valid_add_rmsnorm2d_rdquant_spec,
)
from .topk_softmax import (  # noqa: F401
    TopkSoftmaxSpec,
    build_topk_softmax,
    is_valid_spec as is_valid_topk_softmax_spec,
    topk_softmax_grid,
    topk_softmax_signature,
)
from .moe_sorting import (  # noqa: F401
    MoeSortingSpec,
    build_moe_sort_histogram,
    build_moe_sort_scan,
    build_moe_sort_scatter,
    is_valid_spec as is_valid_moe_sorting_spec,
    moe_sort_histogram_grid,
    moe_sort_histogram_signature,
    moe_sort_scan_grid,
    moe_sort_scan_signature,
    moe_sort_scatter_grid,
    moe_sort_scatter_signature,
    moe_sorting_workspace_bytes,
)
from .fused_moe import (  # noqa: F401
    FusedMoeLauncher,
    FusedMoeSpec,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_topk_weighted_reduce,
    is_valid_spec as is_valid_fused_moe_spec,
    moe_fused_workspace_bytes,
    moe_gather_grid,
    moe_gather_signature,
    moe_silu_mul_grid,
    moe_silu_mul_signature,
    moe_topk_weighted_reduce_grid,
    moe_topk_weighted_reduce_signature,
)
from .flatmm import (  # noqa: F401
    FlatMMSpec,
    build_flatmm,
    flatmm_grid,
    flatmm_signature,
    is_valid_spec as is_valid_flatmm_spec,
)
from .batched_contraction import (  # noqa: F401
    BatchedContractionSpec,
    batched_contraction_grid,
    batched_contraction_signature,
    build_batched_contraction,
    flatten_batch_strides,
    is_valid_spec as is_valid_batched_contraction_spec,
)
from .streamk_gemm import (  # noqa: F401
    StreamKGemmSpec,
    StreamKReductionStrategy,
    build_streamk_gemm,
    is_valid_spec as is_valid_streamk_gemm_spec,
    streamk_gemm_grid,
    streamk_gemm_signature,
    streamk_gemm_workspace_bytes,
)
from .mfma_gemm import (  # noqa: F401
    MfmaGemmSpec,
    build_mfma_gemm,
    is_valid_spec as is_valid_mfma_gemm_spec,
    mfma_gemm_grid,
    mfma_gemm_signature,
)
from .fmha_mfma import (  # noqa: F401
    FmhaMfmaSpec,
    build_fmha_fwd_mfma,
    fmha_fwd_mfma_grid,
    fmha_fwd_mfma_signature,
    is_valid_spec as is_valid_fmha_mfma_spec,
)
from .block_scale_gemm import (  # noqa: F401
    BlockScaleGemmSpec,
    MantissaDType,
    QuantMode,
    block_scale_gemm_grid,
    block_scale_gemm_signature,
    build_block_scale_gemm,
    is_valid_spec as is_valid_block_scale_gemm_spec,
)
from .mx_gemm import (  # noqa: F401
    MxGemmSpec,
    MxMantissaDType,
    build_mx_gemm,
    is_valid_spec as is_valid_mx_gemm_spec,
    mx_gemm_grid,
    mx_gemm_signature,
)
from .batched_gemm import (  # noqa: F401
    BatchedGemmSpec,
    batched_gemm_grid,
    batched_gemm_signature,
    build_batched_gemm,
    is_valid_spec as is_valid_batched_gemm_spec,
)
from .grouped_gemm import (  # noqa: F401
    GroupedGemmLauncher,
    GroupedGemmProblem,
    GroupedGemmSpec,
    build_grouped_gemm,
    grouped_gemm_problems,
    grouped_gemm_signature,
)
from ._fmha_common import (  # noqa: F401
    FmhaCommonSpec,
    FmhaMaskMode,
    FmhaShape,
    validate_common_spec as validate_fmha_common_spec,
)
from .fmha_varlen import (  # noqa: F401
    FmhaFwdVarlenSpec,
    build_fmha_fwd_varlen,
    fmha_fwd_varlen_grid,
    fmha_fwd_varlen_signature,
    is_valid_spec as is_valid_fmha_fwd_varlen_spec,
)
from .fmha_appendkv import (  # noqa: F401
    FmhaAppendKvSpec,
    build_fmha_fwd_appendkv,
    fmha_appendkv_grid,
    fmha_appendkv_signature,
    is_valid_spec as is_valid_fmha_appendkv_spec,
)
from .fmha_paged_prefill import (  # noqa: F401
    FmhaFwdPagedPrefillSpec,
    build_fmha_fwd_paged_prefill,
    fmha_fwd_paged_prefill_grid,
    fmha_fwd_paged_prefill_signature,
    is_valid_spec as is_valid_fmha_fwd_paged_prefill_spec,
)
from .fmha_splitkv_decode import (  # noqa: F401
    FmhaFwdSplitKvDecodeSpec,
    build_fmha_fwd_splitkv_decode_reduce,
    build_fmha_fwd_splitkv_decode_segment,
    fmha_fwd_splitkv_decode_reduce_grid,
    fmha_fwd_splitkv_decode_reduce_signature,
    fmha_fwd_splitkv_decode_segment_grid,
    fmha_fwd_splitkv_decode_segment_signature,
    is_valid_spec as is_valid_fmha_fwd_splitkv_decode_spec,
)
from .fmha_head_grouping import (  # noqa: F401
    FmhaFwdHeadGroupingSpec,
    build_fmha_fwd_head_grouping,
    fmha_fwd_head_grouping_grid,
    fmha_fwd_head_grouping_signature,
    is_valid_spec as is_valid_fmha_fwd_head_grouping_spec,
)
from .fmha_bwd import (  # noqa: F401
    FmhaBwdSpec,
    build_fmha_bwd,
    fmha_bwd_grid,
    fmha_bwd_signature,
    is_valid_spec as is_valid_fmha_bwd_spec,
)
from .fmha_fwd_fp8 import (  # noqa: F401
    FmhaFwdFp8Spec,
    build_fmha_fwd_fp8,
    fmha_fwd_fp8_grid,
    fmha_fwd_fp8_signature,
    is_valid_spec as is_valid_fmha_fwd_fp8_spec,
)
from .sage_attention import (  # noqa: F401
    SageAttentionSpec,
    SageQuantMode,
    build_sage_attention,
    is_valid_spec as is_valid_sage_attention_spec,
    sage_attention_grid,
    sage_attention_signature,
)
from .sparse_attention import (  # noqa: F401
    JengaSparseSpec,
    VsaSparseSpec,
    build_jenga_sparse_attention,
    build_vsa_sparse_attention,
    is_valid_jenga_spec,
    is_valid_vsa_spec,
    jenga_sparse_attention_grid,
    jenga_sparse_attention_signature,
    vsa_sparse_attention_grid,
    vsa_sparse_attention_signature,
)
