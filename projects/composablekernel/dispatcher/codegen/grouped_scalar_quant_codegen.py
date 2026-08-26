#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared code generator for the grouped scalar-quant GEMM ops.

``rowcolquant`` (per-row scale on A, per-column scale on B) and ``tensorquant``
(one scalar scale for each of A and B) are the same generator.  Name-normalized,
their two 627-line source files differed in six lines -- all prose -- and the
headers they emit differ in one comment, one ``QuantType`` enumerator and the
kernel-name prefix.  Everything downstream of that -- the tile shape, the
partitioner, the traits, the pipeline selection, the ``launch`` overloads, the
config sweep and the CLI -- was identical, so a fix to any of it had to be made
twice or it was made once and silently forgotten.

The three genuine differences are carried in :class:`ScalarQuantOp`.  Each op
keeps its own module so that ``--output-dir`` invocations, the ``_default_config``
/ ``_build_specs`` imports the tests use, and the ``DO NOT EDIT -- regenerate
via`` line in every emitted header all keep naming the right script.

Emitted bytes are unchanged; see ``tests/test_codegen_byte_identity.py``, which
runs the pre-refactor generator out of git and diffs the two output trees.
"""

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional

from codegen_common import (
    ROWCOL_TENSOR_QUANT_BASE_PIPELINE_MAP,
    ROWCOL_TENSOR_QUANT_DEFAULT_TILE,
    ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS,
    ROWCOL_TENSOR_QUANT_EPILOGUE_MAP,
    ROWCOL_TENSOR_QUANT_PIPELINE_MAP,
    ROWCOL_TENSOR_QUANT_SUPPORTED_LAYOUTS,
    TileConfig,
    generate_kernels_generic,
    run_codegen_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# The per-op delta -- the whole of it
# =============================================================================


@dataclass(frozen=True)
class ScalarQuantOp:
    """Everything that distinguishes rowcolquant from tensorquant.

    ``args_scale_stride`` is deliberately absent: the two ctypes bridges pass 0
    and 1 respectively, and the kernel reads neither -- ``QuantType::RowColQuant``
    builds its AQ view with a literal ``make_tuple(1, 0)`` and
    ``QuantType::TensorQuant`` has no AQ view at all
    (``gemm_quant_kernel.hpp``).  The field is inert for both ops, so the 0-vs-1
    divergence carries no meaning and is not part of this spec.
    """

    op_name: str          # "rowcolquant"
    display_name: str     # "RowColQuant" -- appears in the emitted header comment
    op_label: str         # "GroupedRowColQuant" -- log label / CLI label
    quant_type: str       # the emitted ck_tile::QuantType enumerator
    codegen_script: str   # the script named in the emitted "DO NOT EDIT" line
    description: str      # argparse description
    aq_bq_note: str       # documents the op's AQ/BQ layout choice
    make_kernel_name: Callable[..., str]


# =============================================================================
# Dtype variant definitions
# =============================================================================

SCALAR_QUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "ck_a":   "ck_tile::fp8_t",
        "ck_b":   "ck_tile::fp8_t",
        "ck_c":   "ck_tile::half_t",
        "ck_aq":  "float",
        "ck_bq":  "float",
        "ck_acc": "float",
    },
    "bf8": {
        "ck_a":   "ck_tile::bf8_t",
        "ck_b":   "ck_tile::bf8_t",
        "ck_c":   "ck_tile::half_t",
        "ck_aq":  "float",
        "ck_bq":  "float",
        "ck_acc": "float",
    },
}

SCALAR_QUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

SCALAR_QUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
    "default":   "ck_tile::GemmPipelineScheduler::Default",
}

# RowColQuant currently supports only the CompV3 pipeline with a CShuffle epilogue.
# These maps make the config keys load-bearing: the emitted C++ is interpolated from
# them, and _build_specs rejects any key that is absent. Without that, a config
# naming an unsupported pipeline would produce a header *named* for it while
# containing a CompV3 kernel -- silently mislabelled for any name-keyed autotuner.
SCALAR_QUANT_PIPELINE_MAP      = dict(ROWCOL_TENSOR_QUANT_PIPELINE_MAP)
SCALAR_QUANT_BASE_PIPELINE_MAP = dict(ROWCOL_TENSOR_QUANT_BASE_PIPELINE_MAP)
SCALAR_QUANT_EPILOGUE_MAP      = dict(ROWCOL_TENSOR_QUANT_EPILOGUE_MAP)
SCALAR_QUANT_SUPPORTED_LAYOUTS = ROWCOL_TENSOR_QUANT_SUPPORTED_LAYOUTS


# =============================================================================
# Configuration dataclasses
# =============================================================================


# Verbatim redeclaration of codegen_common.TileConfig; aliased so the tile
# validity rule cannot drift between the generators that share it.
ScalarQuantTileConfig = TileConfig


@dataclass
class ScalarQuantKernelSpec:
    """Complete specification for one scalar-quant grouped kernel."""

    dtype: str          # "fp8" or "bf8"
    layout: str         # "rcr"
    pipeline: str       # "compv3"
    epilogue: str       # "cshuffle"
    scheduler: str      # "intrawave"
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool
    tile: ScalarQuantTileConfig
    block_size: int = 256
    k_block_per_cu: int = 1

    # Bound by each op's thin subclass; not a dataclass field.
    op: ClassVar[ScalarQuantOp] = None

    @property
    def name(self) -> str:
        t = self.tile
        return self.op.make_kernel_name(
            dtype=self.dtype,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            persistent=self.persistent,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
        )


# =============================================================================
# Header generator
# =============================================================================


class ScalarQuantKernelHeaderGenerator:
    """Emits the .hpp kernel specialization header for one ScalarQuantKernelSpec."""

    # Bound by each op's thin subclass.
    op: ScalarQuantOp = None

    def generate(self, spec: ScalarQuantKernelSpec) -> str:
        op = self.op
        variant = SCALAR_QUANT_VARIANTS[spec.dtype]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a   = variant["ck_a"]
        ck_b   = variant["ck_b"]
        ck_c   = variant["ck_c"]
        ck_aq  = variant["ck_aq"]
        ck_bq  = variant["ck_bq"]
        ck_acc = variant["ck_acc"]

        layout_a_ck  = SCALAR_QUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck  = SCALAR_QUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck  = SCALAR_QUANT_LAYOUT_TO_CK[spec.layout[2]]
        # Both ops emit RowMajor AQ / ColumnMajor BQ; what that means differs:
        # op.aq_bq_note carries the op's own reading of it.
        log.debug("%s AQ/BQ layout: %s", op.display_name, op.aq_bq_note)
        layout_aq_ck = SCALAR_QUANT_LAYOUT_TO_CK["r"]
        layout_bq_ck = SCALAR_QUANT_LAYOUT_TO_CK["c"]

        scheduler_ck = SCALAR_QUANT_SCHEDULER_TO_CK[spec.scheduler]

        # Interpolated rather than hardwired, so the kernel name and the emitted code
        # cannot disagree. _build_specs guarantees these keys exist.
        pipeline_ck      = SCALAR_QUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = SCALAR_QUANT_BASE_PIPELINE_MAP[spec.pipeline]
        epilogue_ck      = SCALAR_QUANT_EPILOGUE_MAP[spec.epilogue]

        pad_m      = str(spec.pad_m).lower()
        pad_n      = str(spec.pad_n).lower()
        pad_k      = str(spec.pad_k).lower()
        persistent = str(spec.persistent).lower()

        grid_size_expr = (
            "Kernel::MaxOccupancyGridSize(stream)"
            if spec.persistent
            else "Kernel::GridSize(gemm_descs)"
        )

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated {op.display_name} Grouped GEMM kernel header.
// DO NOT EDIT — regenerate via {op.codegen_script}
#pragma once

#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using AQDataType  = {ck_aq};
using BQDataType  = {ck_bq};
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};
using BQLayout = {layout_bq_ck};

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using AQDataType  = {ns}::AQDataType;
    using BQDataType  = {ns}::BQDataType;
    using AccDataType = {ns}::AccDataType;

    static constexpr ck_tile::index_t TileM          = {t.tile_m};
    static constexpr ck_tile::index_t TileN          = {t.tile_n};
    static constexpr ck_tile::index_t TileK          = {t.tile_k};
    static constexpr ck_tile::index_t WarpPerBlock_M = {t.warp_m};
    static constexpr ck_tile::index_t WarpPerBlock_N = {t.warp_n};
    static constexpr ck_tile::index_t WarpPerBlock_K = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM      = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN      = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK      = {t.warp_tile_k};
    // Informational only: the launch below uses Kernel::BlockSize(), which the
    // pipeline derives from the warp counts. Changing the `block_size` config key
    // changes this constant but not the launch geometry.
    static constexpr ck_tile::index_t BlockSize       = {spec.block_size};
    static constexpr int               kBlockPerCu    = {spec.k_block_per_cu};

    static constexpr bool kPadM               = {pad_m};
    static constexpr bool kPadN               = {pad_n};
    static constexpr bool kPadK               = {pad_k};
    static constexpr bool TransposeC          = false;
    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr bool APreshuffleQuant    = false;
    static constexpr bool BPreshuffleQuant    = false;
    static constexpr bool PreshuffleB         = false;
    static constexpr bool UsePersistentKernel = {persistent};

    // TileGemmShape's trailing template parameters are PermuteA_ / PermuteB_. This
    // bridge does not use preshuffled operands, so both are false.
    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        PermuteA, PermuteB>;

    // GemmSpatiallyLocalTilePartitioner groups workgroups to improve cache reuse; the
    // two integers are GroupNum (number of big groups) and M01 (groups in the M dim
    // within a spatially local WGP). The values below are the gfx94x-tuned defaults
    // used by the tile_engine instance builder and the 17_grouped_gemm examples, where
    // they appear as TileParitionerGroupNum / TileParitionerM01.
    static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
    static constexpr ck_tile::index_t TilePartitionerM01      = 4;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        TileShape, TilePartitionerGroupNum, TilePartitionerM01>;

    using GemmQuantTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant,
        BPreshuffleQuant,
        PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        {op.quant_type},
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC,
        DoubleSmemBuffer,
        UsePersistentKernel>;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, {ns}::ALayout, {ns}::BLayout, {ns}::CLayout>;
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblem>;

    // preprocess runs once before every kernel invocation, including each iteration of
    // the timing loop (see ck_tile::launch_kernel_time_mask). Callers must use it to
    // re-zero C whenever k_batch > 1: split-K selects the atomic_add epilogue, so a
    // C that is zeroed only once ends up holding the sum over cold_niters + nrepeat
    // launches. The overload below supplies a no-op and is safe only for k_batch == 1,
    // where the epilogue is `set` and repeated launches are idempotent.
    template <typename PreprocessFunc>
    static float launch(const std::vector<ck_tile::QuantGroupedGemmHostArgs>& gemm_descs,
                        const ck_tile::stream_config& stream,
                        void* kargs_ptr,
                        PreprocessFunc preprocess)
    {{
        constexpr auto scheduler = {scheduler_ck};

        const ck_tile::index_t k_grain = gemm_descs[0].k_batch * TileShape::kK;
        const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * TileShape::kK;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{{0}};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using QuantGemmProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType, BDataType, AccDataType, AccDataType,
                TileShape, GemmQuantTraits,
                TransposeC, BDataType,
                scheduler,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = {pipeline_ck}<QuantGemmProblem>;

            using GemmEpilogue = {epilogue_ck}<
                ck_tile::CShuffleEpilogueProblem<
                    ADataType, BDataType, ck_tile::tuple<>,
                    AccDataType, CDataType, ck_tile::tuple<>,
                    {ns}::CLayout, ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    WarpPerBlock_M, WarpPerBlock_N,
                    WarpTileM, WarpTileN, WarpTileK,
                    QuantGemmProblem::TransposeC>>;

            using Kernel = ck_tile::QuantGroupedGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                GemmQuantTraits::kQuantType>;

            auto kargs = Kernel::MakeKargs(gemm_descs);
            if(!Kernel::IsSupportedArgument(kargs)) {{
                return -1.0f;
            }}

            const dim3 grids  = {grid_size_expr};
            const dim3 blocks = Kernel::BlockSize();

            HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                                kargs.data(),
                                                kargs.size() * sizeof(ck_tile::QuantGemmTransKernelArg),
                                                hipMemcpyHostToDevice,
                                                stream.stream_id_));

            constexpr int kBlockPerCu_ = kBlockPerCu;
            return ave_time = ck_tile::launch_kernel_time_mask(
                stream,
                preprocess,
                ck_tile::make_kernel<kBlockPerCu_>(
                    Kernel{{}},
                    grids,
                    blocks,
                    0,
                    ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                    gemm_descs.size()));
        }};

        return ave_time = BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}

    // Convenience overload for k_batch == 1. See the note on preprocess above.
    static float launch(const std::vector<ck_tile::QuantGroupedGemmHostArgs>& gemm_descs,
                        const ck_tile::stream_config& stream,
                        void* kargs_ptr)
    {{
        return launch(gemm_descs, stream, kargs_ptr, []() {{}});
    }}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using SelectedKernel = {ns}::{struct};
constexpr const char* KERNEL_NAME = {ns}::KERNEL_NAME;
using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using AQDataType  = {ck_aq};
using BQDataType  = {ck_bq};
using AccDataType = {ck_acc};
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config sweep
# =============================================================================


def default_config(op: ScalarQuantOp) -> dict:
    # Traits and tile come from codegen_common so this default and the runtime
    # default_{fp8,bf8}_config() in the op's utils module cannot drift.
    return {
        "dtypes": ["fp8", "bf8"],
        "layouts": list(SCALAR_QUANT_SUPPORTED_LAYOUTS),
        **ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS,
        "tile_configs": [dict(ROWCOL_TENSOR_QUANT_DEFAULT_TILE)],
    }


def build_specs(op: ScalarQuantOp, spec_cls, config: dict) -> List[ScalarQuantKernelSpec]:
    specs = []
    defaults   = ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS
    pipeline   = config.get("pipeline", defaults["pipeline"])
    epilogue   = config.get("epilogue", defaults["epilogue"])
    scheduler  = config.get("scheduler", defaults["scheduler"])
    pad_m      = config.get("pad_m", defaults["pad_m"])
    pad_n      = config.get("pad_n", defaults["pad_n"])
    pad_k      = config.get("pad_k", defaults["pad_k"])
    persistent = config.get("persistent", defaults["persistent"])
    block_size     = config.get("block_size", defaults["block_size"])
    k_block_per_cu = config.get("k_block_per_cu", defaults["k_block_per_cu"])

    # Reject unsupported pipeline/epilogue/scheduler up front rather than emitting a
    # header whose name advertises something the generated code does not implement.
    if pipeline not in SCALAR_QUANT_PIPELINE_MAP:
        log.warning(
            "Unsupported pipeline '%s' (supported: %s) — no kernels generated",
            pipeline, ", ".join(sorted(SCALAR_QUANT_PIPELINE_MAP)),
        )
        return []
    if epilogue not in SCALAR_QUANT_EPILOGUE_MAP:
        log.warning(
            "Unsupported epilogue '%s' (supported: %s) — no kernels generated",
            epilogue, ", ".join(sorted(SCALAR_QUANT_EPILOGUE_MAP)),
        )
        return []
    if scheduler not in SCALAR_QUANT_SCHEDULER_TO_CK:
        log.warning(
            "Unsupported scheduler '%s' (supported: %s) — no kernels generated",
            scheduler, ", ".join(sorted(SCALAR_QUANT_SCHEDULER_TO_CK)),
        )
        return []

    for dtype, layout, tile_dict in itertools.product(
        config.get("dtypes", ["fp8"]),
        config.get("layouts", list(SCALAR_QUANT_SUPPORTED_LAYOUTS)),
        config.get("tile_configs", []),
    ):
        if dtype not in SCALAR_QUANT_VARIANTS:
            log.warning("Unknown dtype %s — skipping", dtype)
            continue

        # A non-rcr layout would flip BLayout in the generated header while the ctypes
        # bridge kept requiring stride_A == K, stride_B == K, stride_C == N, so the
        # kernel would build but every call would be rejected at runtime.
        if layout not in SCALAR_QUANT_SUPPORTED_LAYOUTS:
            log.warning(
                "Unsupported layout '%s' (supported: %s) — skipping",
                layout, ", ".join(SCALAR_QUANT_SUPPORTED_LAYOUTS),
            )
            continue

        tile = ScalarQuantTileConfig(
            tile_m=tile_dict["tile_m"],
            tile_n=tile_dict["tile_n"],
            tile_k=tile_dict["tile_k"],
            warp_m=tile_dict["warp_m"],
            warp_n=tile_dict["warp_n"],
            warp_k=tile_dict["warp_k"],
            warp_tile_m=tile_dict["warp_tile_m"],
            warp_tile_n=tile_dict["warp_tile_n"],
            warp_tile_k=tile_dict["warp_tile_k"],
        )
        if not tile.is_valid():
            log.debug("Invalid tile config %s — skipping", tile)
            continue

        specs.append(spec_cls(
            dtype=dtype,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            persistent=persistent,
            tile=tile,
            block_size=block_size,
            k_block_per_cu=k_block_per_cu,
        ))

    return specs


