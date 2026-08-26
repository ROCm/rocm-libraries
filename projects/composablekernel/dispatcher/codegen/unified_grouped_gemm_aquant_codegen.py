#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm AQuant Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_aquant_ctypes_lib.cpp

Covers fp8, bf8, fp8i4 and bf8i4 dtype variants with configurable QuantGroupShape.
APreshuffleQuant is a trait flag on AQuantGemmPipelineAgBgCrCompV3 (not a separate pipeline).

Naming convention (byte-exact with AQuantKernelConfig.name in grouped_gemm_aquant_utils.py):
    grouped_gemm_aquant_{variant_key}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    aqg{gM}x{gN}x{gK}[_preshuffleaq]

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp
    example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped_preshufflequant.cpp
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp  (GemmConfigQuantDecodeInterwave)
"""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from codegen_common import (
    TileConfig,
    aquant_effective_epilogue,
    generate_kernels_generic,
    make_aquant_kernel_name,
    run_codegen_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry: (variant_key, ADataType, BDataType, CDataType, QDataType=AQDataType)
# Matches example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped*.cpp
# =============================================================================

AQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "dtype_a": "fp8",
        "dtype_b": "fp8",
        "dtype_c": "half",
        "dtype_q": "float",    # AQ scale type
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",       # AQDataType
        "ck_acc": "float",
    },
    "bf8": {
        "dtype_a": "bf8",
        "dtype_b": "bf8",
        "dtype_c": "half",
        "dtype_q": "float",
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    # fp8i4/bf8i4: A is pk_int4 -- AQuant scales the A operand, so the i4 operand
    # is A and the AQ scale tensor is what makes it meaningful.  B is the 8-bit
    # float operand.  Old-TE spells this
    #   GemmQuantTypeConfig<pk_int4_t, fp8_t, half_t, fp8_t>   (A, B, C, Q)
    # in gemm_aquant_quantgrouped.cpp:37-56, and the non-grouped bridge
    # (unified_gemm_aquant_codegen.AQUANT_VARIANTS) agrees.
    #
    # The reverse assignment (A=fp8, B=pk_int4) does not merely underperform: it
    # is rejected at compile time by
    # block_universal_gemm_as_aquant_bs_cr.hpp:102, whose supported-combination
    # static_assert requires BDataType in {fp8_t, bf8_t}.
    "fp8i4": {
        "dtype_a": "pk_int4",
        "dtype_b": "fp8",
        "dtype_c": "half",
        "dtype_q": "fp8",
        "ck_a": "ck_tile::pk_int4_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::fp8_t",
        "ck_acc": "float",
    },
    "bf8i4": {
        "dtype_a": "pk_int4",
        "dtype_b": "bf8",
        "dtype_c": "half",
        "dtype_q": "bf8",
        "ck_a": "ck_tile::pk_int4_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::bf8_t",
        "ck_acc": "float",
    },
}

# Standard rcr layout: RowMajor A, ColMajor B, RowMajor C
AQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# AQuant pipeline selection:
# - non-preshuffle (APreshuffleQuant=false) → AQuantGemmPipelineAgBgCrMem / BaseGemmPipelineAgBgCrMem
# - preshuffle     (APreshuffleQuant=true)  → AQuantGemmPipelineAgBgCrCompV3 / BaseGemmPipelineAgBgCrCompV3
# APreshuffleQuant is a traits flag that also switches the pipeline class.
AQUANT_PIPELINE_MAP = {
    "mem":    "ck_tile::AQuantGemmPipelineAgBgCrMem",    # non-preshuffle
    "compv3": "ck_tile::AQuantGemmPipelineAgBgCrCompV3", # preshuffle (APreshuffleQuant=true)
}

AQUANT_BASE_PIPELINE_MAP = {
    "mem":    "ck_tile::BaseGemmPipelineAgBgCrMem",
    "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
}

AQUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


# =============================================================================
# Configuration dataclasses
# =============================================================================


# Verbatim redeclaration of codegen_common.TileConfig; aliased so the tile
# validity rule cannot drift between the generators that share it.
AQuantTileConfig = TileConfig


@dataclass
class AQuantKernelSpec:
    """Complete specification for one AQuant kernel."""

    variant_key: str          # "fp8", "bf8", "fp8i4", "bf8i4"
    layout: str               # "rcr"
    pipeline: str             # "mem" (non-preshuffle) or "compv3" (preshuffle)
    epilogue: str             # "cshuffle" or "permute_n" (effective epilogue overrides)
    scheduler: str            # "intrawave"
    tile: AQuantTileConfig
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    preshuffle_aq: bool = False   # APreshuffleQuant trait flag
    double_smem_buffer: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    block_size: int = 256
    k_block_per_cu: int = 1
    transpose_c: bool = False

    @property
    def name(self) -> str:
        t = self.tile
        return make_aquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            quant_group_m=self.quant_group_m,
            quant_group_n=self.quant_group_n,
            quant_group_k=self.quant_group_k,
            preshuffle_aq=self.preshuffle_aq,
        )


# =============================================================================
# Header generator
# =============================================================================


class AQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one AQuantKernelSpec."""

    def generate(self, spec: AQuantKernelSpec) -> str:
        variant = AQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_q = variant["ck_q"]     # AQDataType
        ck_acc = variant["ck_acc"]

        layout_a_ck = AQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = AQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = AQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # AQ is RowMajor: scale is stored [ceil(M/gM), ceil(K/gK)]
        layout_aq_ck = AQUANT_LAYOUT_TO_CK["r"]
        # BQ layout placeholder (unused for AQuant-only)
        layout_bq_ck = layout_b_ck

        pipeline_ck = AQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = AQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = AQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        preshuffle_aq = str(spec.preshuffle_aq).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()
        transpose_c = str(spec.transpose_c).lower()

        # Determine epilogue — same PermuteN logic as BQuant (B-side tile geometry governs)
        use_permute_n_epilogue = (
            aquant_effective_epilogue(t.tile_n, t.warp_n, t.warp_tile_n, spec.quant_group_n)
            == "permute_n"
        )

        if use_permute_n_epilogue:
            epilogue_block = f"""\
            using GemmEpilogue = ck_tile::PermuteNEpilogue<
                ck_tile::PermuteNEpilogueProblem<
                    typename PipelineProblem::AComputeDataType,
                    typename PipelineProblem::BComputeDataType,
                    ck_tile::tuple<>,
                    AccDataType,
                    CDataType,
                    ck_tile::tuple<>,
                    {ns}::CLayout,
                    ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    WarpM, WarpN,
                    WarpTileM, WarpTileN, WarpTileK,
                    TransposeC,
                    false,
                    1>>;"""
        else:
            epilogue_block = f"""\
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<
                    typename PipelineProblem::AComputeDataType,
                    typename PipelineProblem::BComputeDataType,
                    ck_tile::tuple<>,
                    AccDataType,
                    CDataType,
                    ck_tile::tuple<>,
                    {ns}::CLayout,
                    ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    WarpM, WarpN,
                    WarpTileM, WarpTileN, WarpTileK,
                    TransposeC>>;"""

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated AQuantGrouped GEMM kernel header.
// DO NOT EDIT — regenerate via unified_grouped_gemm_aquant_codegen.py
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/epilogue.hpp"

namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using QDataType   = {ck_q};   // AQDataType (A-side scale)
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};
using BQLayout = {layout_bq_ck};

// AQuantGroupSize: scale is [ceil(M/gM), ceil(K/gK)] in RowMajor
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    {spec.quant_group_m}, {spec.quant_group_n}, {spec.quant_group_k}>>;

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using QDataType   = {ns}::QDataType;
    using AccDataType = {ns}::AccDataType;

    static constexpr ck_tile::index_t TileM      = {t.tile_m};
    static constexpr ck_tile::index_t TileN      = {t.tile_n};
    static constexpr ck_tile::index_t TileK      = {t.tile_k};
    static constexpr ck_tile::index_t WarpM      = {t.warp_m};
    static constexpr ck_tile::index_t WarpN      = {t.warp_n};
    static constexpr ck_tile::index_t WarpK      = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM  = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN  = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK  = {t.warp_tile_k};
    static constexpr ck_tile::index_t BlockSize  = {spec.block_size};
    static constexpr int               kBlockPerCu = {spec.k_block_per_cu};

    static constexpr bool kPadM           = {pad_m};
    static constexpr bool kPadN           = {pad_n};
    static constexpr bool kPadK           = {pad_k};
    static constexpr bool APreshuffleQuant = {preshuffle_aq};
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB     = false;
    static constexpr bool TransposeC      = {transpose_c};
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpM, WarpN, WarpK>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<TileShape>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::AQuantGrouped,
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC, DoubleSmemBuffer>;

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& s)
    {{
        const ck_tile::index_t K_split =
            (args.k_batch == 1)
                ? ck_tile::integer_least_multiple(args.K, TileK)
                : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, TileK);

        const ck_tile::index_t num_loop  = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop          = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](auto has_hot_loop_, auto tail_number_) {{
            using PipelineProblem = ck_tile::GemmAQuantPipelineProblem<
                ADataType,
                QDataType,     // AQDataType (A-side scale type)
                BDataType,
                AccDataType,   // 4th arg is AccDataType, matching run_gemm_quant_example.inc
                TileShape,
                GemmTraits,
                QuantGroupSize,
                TransposeC,
                ADataType,     // AComputeDataType — A activations compute type
                {scheduler_ck},
                has_hot_loop_.value,
                tail_number_.value>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::AQuantGrouped>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                return -1.0f;

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
            return ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
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
using QDataType   = {ck_q};
using AccDataType = {ck_acc};
using QuantGroupSize = {ns}::QuantGroupSize;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    """Default sweep config matching GemmConfigQuantDecodeInterwave tile defaults (non-preshuffle).

    Uses the Mem pipeline (AQuantGemmPipelineAgBgCrMem) which is correct for
    non-preshuffle AQuant kernels.  For APreshuffleQuant=true kernels, set
    pipeline="compv3" and preshuffle_aq=true in the config.
    """
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "mem",        # non-preshuffle AQuant uses Mem pipeline
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            # GemmConfigQuantDecodeInterwave<fp8_t>: M=16, N=64, K=256/sizeof(fp8_t)=256
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 16},
        ],
        "quant_groups": [
            {"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": True,
        "block_size": 256,
        "k_block_per_cu": 1,
        "double_smem_buffer": False,
        "preshuffle_aq": False,
        "transpose_c": False,
    }


def _build_specs(config: dict) -> List[AQuantKernelSpec]:
    specs = []
    pipeline   = config.get("pipeline", "compv3")
    epilogue   = config.get("epilogue", "cshuffle")
    scheduler  = config.get("scheduler", "intrawave")
    pad_m      = config.get("pad_m", False)
    pad_n      = config.get("pad_n", False)
    pad_k      = config.get("pad_k", True)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)
    preshuffle_aq      = config.get("preshuffle_aq", False)
    transpose_c        = config.get("transpose_c", False)

    for variant_key, layout, tile_dict, qg in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        config.get("quant_groups", [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
    ):
        if variant_key not in AQUANT_VARIANTS:
            log.warning("Unknown variant_key %s — skipping", variant_key)
            continue
        # Derive the correct pipeline from preshuffle_aq if not explicitly set
        effective_pipeline = pipeline
        if effective_pipeline not in AQUANT_PIPELINE_MAP:
            log.warning("Unsupported pipeline %s — skipping", effective_pipeline)
            continue
        # Guard: compv3 requires preshuffle_aq=true; mem requires preshuffle_aq=false
        if effective_pipeline == "compv3" and not preshuffle_aq:
            log.warning("pipeline=compv3 requires preshuffle_aq=true — skipping")
            continue
        if effective_pipeline == "mem" and preshuffle_aq:
            log.warning("pipeline=mem does not support preshuffle_aq=true — skipping")
            continue

        tile = AQuantTileConfig(
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

        specs.append(AQuantKernelSpec(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile=tile,
            quant_group_m=qg.get("quant_group_m", 1),
            quant_group_n=qg.get("quant_group_n", 1),
            quant_group_k=qg.get("quant_group_k", 128),
            preshuffle_aq=preshuffle_aq,
            double_smem_buffer=double_smem_buffer,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            block_size=block_size,
            k_block_per_cu=k_block_per_cu,
            transpose_c=transpose_c,
        ))

    return specs

def generate_kernels(
    output_dir: Path,
    config: Optional[dict] = None,
    parallel: bool = True,
) -> List[Path]:
    """Generate all GroupedAQuant kernel headers into output_dir.

    Returns list of generated .hpp paths.
    """
    return generate_kernels_generic(
        op_label="GroupedAQuant",
        generator=AQuantKernelHeaderGenerator(),
        specs=_build_specs(config or _default_config()),
        output_dir=output_dir,
        parallel=parallel,
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    return run_codegen_cli(
        description="AQuantGrouped GEMM kernel header generator",
        op_label="GroupedAQuant",
        make_generator=AQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
