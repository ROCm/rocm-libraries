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
  - `build_universal_gemm`     : the dispatcher-grade GEMM
                                  (`gemm_universal.UniversalGemmSpec`).
                                  Covers `tile_m/n/k in [16..256]`, warp
                                  grids `1x1..4x4`, MFMA atoms
                                  `16x16x{16,32}`, `32x32x{8,16}` fp16.
                                  Supports `batched=True` for per-block
                                  (block_id_z) batch indexing.
  - `build_batched_gemm`       : CK Tile `16_batched_gemm` counterpart;
                                  same kernel as `build_universal_gemm`
                                  with `batched=True`, plus a clean
                                  `BatchedGemmSpec` / signature surface.
  - `build_grouped_gemm` +      : CK Tile `17_grouped_gemm` counterpart.
    `GroupedGemmLauncher`        The launcher re-uses the universal GEMM
                                  and issues one launch per group; a
                                  single-launch variant is on the
                                  follow-up list.

Convolution family
  - `build_implicit_gemm_conv` : NHWC × KRSC -> NHWK implicit-GEMM conv.
  - `build_direct_conv_16c`    : `cpg=kpg=16` grouped direct conv with
                                  K=32 folding.
  - `build_direct_conv_4c`     : `cpg=kpg=4` grouped direct conv via
                                  `mfma_f32_4x4x4_f16`.

Attention
  - `build_unified_attention_2d`        : paged-attention 2D scalar.
  - `build_unified_attention_2d_tiled`  : production 2D tiled kernel
                                          (MFMA + async LDS + cshuffle).
  - `build_unified_attention_3d_tiled`  : split-KV 3D tiled segment kernel.
  - `build_unified_attention_reduce_tiled` : split-KV reduce kernel.

CK Tile small-op counterparts (Tier 1)
  - `build_elementwise`        : CK Tile `21_elementwise` counterpart.
                                  Unary (copy/neg/abs/relu/silu/gelu_tanh
                                  /exp2) and binary (add/sub/mul/max/min)
                                  ops over contiguous f16/bf16 buffers.
  - `build_layernorm2d`        : CK Tile `02_layernorm2d` forward, with
                                  optional save_mean_invstd.
  - `build_rmsnorm2d`          : CK Tile `10_rmsnorm2d` forward.
  - `build_reduce2d`           : CK Tile `05_reduce` row-wise sum / max /
                                  mean.
  - `build_transpose2d`        : CK Tile `37_transpose` / `35_batched_
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
