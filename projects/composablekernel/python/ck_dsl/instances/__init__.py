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

  - `build_universal_gemm`   : the dispatcher-grade GEMM
                                (`gemm_universal.UniversalGemmSpec`).
                                Covers `tile_m/n/k in [16..256]`, warp
                                grids `1x1..4x4`, MFMA atoms
                                `16x16x{16,32}`, `32x32x{8,16}` fp16.
  - `build_implicit_gemm_conv`: NHWC × KRSC -> NHWK implicit-GEMM conv
                                (`conv_implicit_gemm.ImplicitGemmConvSpec`).
                                Uses the coord-transform DAG for the A
                                tile's `(m, k) -> NHWC offset` mapping.
  - `build_direct_conv_16c`   : `cpg=kpg=16` grouped direct conv
                                (`conv_direct_grouped.DirectConv16cSpec`)
                                with K=32 folding and a wide direct
                                vector epilogue.
  - `build_direct_conv_4c`    : `cpg=kpg=4` grouped direct conv via
                                `mfma_f32_4x4x4_f16` (16 groups per
                                wave).
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
