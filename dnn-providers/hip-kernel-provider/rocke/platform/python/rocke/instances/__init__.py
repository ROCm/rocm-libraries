# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# ruff: noqa: E402 -- curated registry: re-exports are grouped after section comments

"""Parametric instance builders for the CK DSL.

Each module here knows how to take a *dispatcher-grade* config schema
(the same `(TileConfig, TraitConfig)` shape that
`dispatcher/codegen/default_config.json` and
`dispatcher/codegen/preselected_kernels.py` consume) and emit a full
Python IR kernel ready for `rocke.core.lower_llvm` and
`rocke.runtime.comgr` to compile in-process.

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
 - `build_implicit_gemm_conv` : NHWC × KYXC -> NHWK implicit-GEMM conv.
 - `build_direct_conv_16c` : `cpg=kpg=16` grouped direct conv with
 K=32 folding.
 - `build_direct_conv_4c` : `cpg=kpg=4` grouped direct conv via
 `mfma_f32_4x4x4_f16`.


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
helper for use with :class:`rocke.runtime.launcher.KernelLauncher`.
End-to-end parity vs torch reference for all of these is exercised by
:mod:`rocke.examples.common.ck_tile_parity`; the GEMM parity driver lives in
:mod:`rocke.examples.common.bake_off_implicit_gemm`.
"""

from .common.conv_direct_grouped import (  # noqa: F401
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    build_direct_conv_4c,
    build_direct_conv_16c,
)
from .common.conv_implicit_gemm import (  # noqa: F401
    ConvAccumulatorEpilogue,
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    make_a_descriptor,
    make_b_descriptor,
    make_d_descriptor,
)
from .common.conv_implicit_gemm_wgrad import (  # noqa: F401
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
    is_valid_wgrad_spec,
    make_dy_descriptor,
    make_dw_descriptor,
    make_x_wgrad_descriptor,
)
from .common.gemm_universal import (  # noqa: F401
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
from .gfx950.deep_fused_conv_pool import (  # noqa: F401
    FusedConvPoolProblem,
    Gfx950DeepFusedConvPoolSpec,
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    deep_fused_conv_pool_signature,
    is_valid_spec as is_valid_deep_fused_conv_pool_spec,
)
from .common.elementwise import (  # noqa: F401
    BinaryOp,
    DType as ElementwiseDType,
    ElementwiseSpec,
    UnaryOp,
    build_elementwise,
    elementwise_grid,
    elementwise_signature,
    is_valid_spec as is_valid_elementwise_spec,
)
from .common.layernorm2d import (  # noqa: F401
    LayerNorm2DSpec,
    build_layernorm2d,
    is_valid_spec as is_valid_layernorm2d_spec,
    layernorm2d_grid,
    layernorm2d_signature,
)
from .common.rmsnorm2d import (  # noqa: F401
    RMSNorm2DSpec,
    build_rmsnorm2d,
    is_valid_spec as is_valid_rmsnorm2d_spec,
    rmsnorm2d_grid,
    rmsnorm2d_signature,
)
from .common.reduce import (  # noqa: F401
    Reduce2DSpec,
    ReduceOp,
    build_reduce2d,
    is_valid_spec as is_valid_reduce2d_spec,
    reduce2d_grid,
    reduce2d_signature,
)
from .common.transpose import (  # noqa: F401
    Transpose2DSpec,
    build_transpose2d,
    is_valid_spec as is_valid_transpose2d_spec,
    transpose2d_grid,
    transpose2d_signature,
)
from .common.transpose_bc import (  # noqa: F401
    TransposeBcSpec,
    build_transpose_bc,
    is_valid_spec as is_valid_transpose_bc_spec,
    transpose_bc_grid,
    transpose_bc_signature,
)
from .common.batched_transpose import (  # noqa: F401
    BatchedTranspose2DSpec,
    batched_transpose2d_grid,
    batched_transpose2d_signature,
    build_batched_transpose2d,
    is_valid_spec as is_valid_batched_transpose2d_spec,
)
from .common.img2col import (  # noqa: F401
    Img2ColSpec,
    build_img2col,
    img2col_grid,
    img2col_signature,
    is_valid_spec as is_valid_img2col_spec,
)
from .common.pooling import (  # noqa: F401
    Pooling2DSpec,
    PoolingProblem,
    PoolOp,
    build_pooling2d,
    is_valid_spec as is_valid_pooling2d_spec,
    pooling2d_grid,
    pooling2d_signature,
)
from .common.permute_nd import (  # noqa: F401
    MAX_RANK as PERMUTE_MAX_RANK,
    PermuteSpec,
    build_permute,
    is_valid_spec as is_valid_permute_spec,
    permute_grid,
    permute_signature,
)
from .common.gemm_multi_d import (  # noqa: F401
    DOp as GemmMultiDOp,
    GemmMultiDSpec,
    MAX_D as GEMM_MULTI_D_MAX,
    build_gemm_multi_d,
    gemm_multi_d_grid,
    gemm_multi_d_signature,
    is_valid_spec as is_valid_gemm_multi_d_spec,
)
from .common.gemm_multi_abd import (  # noqa: F401
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
from .common.smoothquant import (  # noqa: F401
    SmoothQuantSpec,
    build_smoothquant,
    is_valid_spec as is_valid_smoothquant_spec,
    smoothquant_grid,
    smoothquant_signature,
)
from .common.moe_smoothquant import (  # noqa: F401
    MoeSmoothQuantSpec,
    build_moe_smoothquant,
    is_valid_spec as is_valid_moe_smoothquant_spec,
    moe_smoothquant_grid,
    moe_smoothquant_signature,
)
from .common.add_rmsnorm2d_rdquant import (  # noqa: F401
    AddRmsnorm2DRdquantSpec,
    add_rmsnorm2d_rdquant_grid,
    add_rmsnorm2d_rdquant_signature,
    build_add_rmsnorm2d_rdquant,
    is_valid_spec as is_valid_add_rmsnorm2d_rdquant_spec,
)
from .common.topk_softmax import (  # noqa: F401
    TopkSoftmaxSpec,
    build_topk_softmax,
    is_valid_spec as is_valid_topk_softmax_spec,
    topk_softmax_grid,
    topk_softmax_signature,
)
from .common.moe_sorting import (  # noqa: F401
    MoeSortingLauncher,
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
from .common.fused_moe import (  # noqa: F401
    FusedMoeLauncher,
    FusedMoeSpec,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_silu_mul_packed,
    build_moe_static_scatter_gather,
    build_moe_topk_weighted_reduce,
    is_valid_spec as is_valid_fused_moe_spec,
    moe_fused_workspace_bytes,
    moe_gather_grid,
    moe_gather_signature,
    moe_silu_mul_grid,
    moe_silu_mul_packed_grid,
    moe_silu_mul_packed_signature,
    moe_silu_mul_signature,
    moe_static_scatter_gather_grid,
    moe_static_scatter_gather_signature,
    moe_topk_weighted_reduce_grid,
    moe_topk_weighted_reduce_signature,
)
from .common.fused_moe_e2e import (  # noqa: F401
    FusedMoeForward,
    FusedMoeForwardSpec,
)
from .common.moe_gemm_fused import (  # noqa: F401
    FusedDownReduceGemmSpec,
    FusedGateUpSiluGemmSpec,
    FusedInterleavedGateUpSiluGemmSpec,
    build_moe_down_reduce_gemm,
    build_moe_gate_up_silu_gemm,
    build_moe_interleaved_gate_up_silu_gemm,
    moe_down_reduce_gemm_grid,
    moe_down_reduce_gemm_signature,
    moe_gate_up_silu_gemm_grid,
    moe_gate_up_silu_gemm_signature,
    moe_interleaved_gate_up_silu_gemm_grid,
    moe_interleaved_gate_up_silu_gemm_signature,
)
from .common.flatmm import (  # noqa: F401
    FlatMMSpec,
    build_flatmm,
    flatmm_grid,
    flatmm_signature,
    is_valid_spec as is_valid_flatmm_spec,
)
from .common.batched_contraction import (  # noqa: F401
    BatchedContractionSpec,
    batched_contraction_grid,
    batched_contraction_signature,
    build_batched_contraction,
    flatten_batch_strides,
    is_valid_spec as is_valid_batched_contraction_spec,
)
from .common.streamk_gemm import (  # noqa: F401
    StreamKGemmSpec,
    StreamKReductionStrategy,
    build_streamk_gemm,
    is_valid_spec as is_valid_streamk_gemm_spec,
    streamk_gemm_grid,
    streamk_gemm_signature,
    streamk_gemm_workspace_bytes,
)
from .common.mfma_gemm import (  # noqa: F401
    MfmaGemmSpec,
    build_mfma_gemm,
    is_valid_spec as is_valid_mfma_gemm_spec,
    mfma_gemm_grid,
    mfma_gemm_signature,
)
from .common.block_scale_gemm import (  # noqa: F401
    BlockScaleGemmSpec,
    MantissaDType,
    QuantMode,
    block_scale_gemm_grid,
    block_scale_gemm_signature,
    build_block_scale_gemm,
    is_valid_spec as is_valid_block_scale_gemm_spec,
)
from .common.matmul_nbits import (  # noqa: F401
    MatMulNBitsFamily,
    MatMulNBitsSpec,
    build_matmul_nbits,
    dequant_i4_weights as dequant_matmul_nbits_weights,
    is_valid_spec as is_valid_matmul_nbits_spec,
    matmul_nbits_grid,
    matmul_nbits_outer_tiles,
    matmul_nbits_reference,
    matmul_nbits_signature,
    pack_i4_weights_for_matmul_nbits,
)
from .common.mx_gemm import (  # noqa: F401
    MxGemmSpec,
    MxMantissaDType,
    build_mx_gemm,
    is_valid_spec as is_valid_mx_gemm_spec,
    mx_gemm_grid,
    mx_gemm_signature,
)
from .common.batched_gemm import (  # noqa: F401
    BatchedGemmSpec,
    batched_gemm_grid,
    batched_gemm_signature,
    build_batched_gemm,
    is_valid_spec as is_valid_batched_gemm_spec,
)
from .common.grouped_gemm import (  # noqa: F401
    GroupedGemmLauncher,
    GroupedGemmProblem,
    GroupedGemmSpec,
    build_grouped_gemm,
    grouped_gemm_problems,
    grouped_gemm_signature,
)
from .common.gemm_policy import (  # noqa: F401
    GemmPipelinePolicy,
    ValidationResult,
)
