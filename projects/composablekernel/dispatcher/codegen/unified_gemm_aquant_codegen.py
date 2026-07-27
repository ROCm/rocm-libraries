#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
AQuant GEMM Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking ck_tile::QuantGemmHostArgs — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_aquant_ctypes_lib.cpp

Naming convention (byte-exact with AQuantKernelConfig.name in gemm_aquant_utils.py):
    gemm_aquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {PadM}_{PadN}_{PadK}_{APreshuffle}_{tile}

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp
    tile_engine/ops/gemm/gemm_instance_builder.py  (gemm_aquant branch)
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures

from codegen_common import make_aquant_kernel_name, aquant_effective_epilogue, BQUANT_DTYPE_MAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry: (variant_key, ADataType, BDataType, CDataType, AQDataType, AccDataType)
# Matches tile_engine/ops/gemm/gemm_instance_builder.py for gemm_aquant
# AQDataType is always float (float32) per gemm_instance_builder.py line 662
# =============================================================================

AQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "dtype_a": "fp8",
        "dtype_b": "fp8",
        "dtype_c": "half",
        "dtype_aq": "float",
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_aq": "float",
        "ck_acc": "float",
    },
    "bf8": {
        "dtype_a": "bf8",
        "dtype_b": "bf8",
        "dtype_c": "half",
        "dtype_aq": "float",
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_aq": "float",
        "ck_acc": "float",
    },
    "fp8i4": {
        "dtype_a": "fp8",
        "dtype_b": "pk_int4",
        "dtype_c": "half",
        "dtype_aq": "float",
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::pk_int4_t",
        "ck_c": "ck_tile::half_t",
        "ck_aq": "float",
        "ck_acc": "float",
    },
    "bf8i4": {
        "dtype_a": "bf8",
        "dtype_b": "pk_int4",
        "dtype_c": "half",
        "dtype_aq": "float",
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::pk_int4_t",
        "ck_c": "ck_tile::half_t",
        "ck_aq": "float",
        "ck_acc": "float",
    },
}

AQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# Pipeline map for AQuant kernels.
# "mem"    -> AQuantGemmPipelineAgBgCrMem   (slower, lower-occupancy, fallback)
# "compv3" -> AQuantGemmPipelineAgBgCrCompV3 (default for dispatcher bridge)
AQUANT_PIPELINE_MAP = {
    "mem":    "ck_tile::AQuantGemmPipelineAgBgCrMem",
    "compv3": "ck_tile::AQuantGemmPipelineAgBgCrCompV3",
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

@dataclass
class AQuantTileConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    def is_valid(self) -> bool:
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            return False
        return (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
        )


@dataclass
class AQuantKernelSpec:
    """Complete specification for one AQuant kernel."""

    variant_key: str          # "fp8", "bf8", "fp8i4", "bf8i4"
    layout: str               # "rcr", "rrr", "crr", "ccr"
    pipeline: str             # "compv3" or "mem"
    epilogue: str             # always "cshuffle"
    scheduler: str            # "intrawave" or "interwave"
    tile: AQuantTileConfig
    quant_group_k: int = 128
    preshuffle_quant: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    block_size: int = 256
    k_block_per_cu: int = 1

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
            quant_group_k=self.quant_group_k,
            preshuffle_quant=self.preshuffle_quant,
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

        ck_a   = variant["ck_a"]
        ck_b   = variant["ck_b"]
        ck_c   = variant["ck_c"]
        ck_aq  = variant["ck_aq"]
        ck_acc = variant["ck_acc"]

        layout_a_ck  = AQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck  = AQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck  = AQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # AQ is always RowMajor per gemm_instance_builder.py line 685
        layout_aq_ck = AQUANT_LAYOUT_TO_CK["r"]

        pipeline_ck      = AQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = AQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck     = AQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m            = str(spec.pad_m).lower()
        pad_n            = str(spec.pad_n).lower()
        pad_k            = str(spec.pad_k).lower()
        preshuffle_quant = str(spec.preshuffle_quant).lower()

        # AQuant always uses CShuffleEpilogue (TiledMMAPermuteN=false)
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
// Auto-generated AQuant GEMM kernel header.
// DO NOT EDIT — regenerate via unified_gemm_aquant_codegen.py
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
using AQDataType  = {ck_aq};
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};
// BQLayout placeholder — not used for AQuant (bq_ptr=nullptr at runtime)
using BQLayout = AQLayout;

using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, {spec.quant_group_k}>>;

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using AQDataType  = {ns}::AQDataType;
    using AccDataType = {ns}::AccDataType;

    static constexpr ck_tile::index_t TileM     = {t.tile_m};
    static constexpr ck_tile::index_t TileN     = {t.tile_n};
    static constexpr ck_tile::index_t TileK     = {t.tile_k};
    static constexpr ck_tile::index_t WarpM     = {t.warp_m};
    static constexpr ck_tile::index_t WarpN     = {t.warp_n};
    static constexpr ck_tile::index_t WarpK     = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK = {t.warp_tile_k};
    static constexpr ck_tile::index_t BlockSize  = {spec.block_size};
    static constexpr int               kBlockPerCu = {spec.k_block_per_cu};
    static constexpr ck_tile::index_t GroupSizeK = {spec.quant_group_k};

    static constexpr bool kPadM            = {pad_m};
    static constexpr bool kPadN            = {pad_n};
    static constexpr bool kPadK            = {pad_k};
    static constexpr bool APreshuffleQuant = {preshuffle_quant};
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB      = false;
    static constexpr bool TransposeC       = false;

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpM, WarpN, WarpK>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::AQuantGrouped,
        {ns}::AQLayout>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits, BDataType>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblem>;

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& s)
    {{
        const ck_tile::index_t K_split = ck_tile::integer_least_multiple(args.K, TileShape::kK);
        const ck_tile::index_t num_loop  = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop          = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using PipelineProblem = ck_tile::GemmAQuantPipelineProblem<
                ADataType,
                AQDataType,
                BDataType,
                AccDataType,
                TileShape,
                GemmTraits,
                QuantGroupSize,
                TransposeC,
                BDataType,
                {scheduler_ck},
                has_hot_loop_v,
                tail_number_v>;

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
using AQDataType  = {ck_aq};
using AccDataType = {ck_acc};
using QuantGroupSize = {ns}::QuantGroupSize;
constexpr ck_tile::index_t GroupSizeK = {ns}::{struct}::GroupSizeK;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            # GemmConfigQuantDecodeInterwave<fp8_t>: default tile for aquant decode path
            # WarpTileK=128 for fp8/bf8: get_k_warp_tile<fp8_t, 16>() = 128 on gfx950
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 128},
        ],
        "quant_group_k": 128,
        "pad_m": False,
        "pad_n": False,
        "pad_k": True,
        "block_size": 256,
        "k_block_per_cu": 1,
        "preshuffle_quant": False,
    }


def _build_specs(config: dict) -> List[AQuantKernelSpec]:
    specs = []
    pipeline         = config.get("pipeline", "compv3")
    epilogue         = config.get("epilogue", "cshuffle")
    scheduler        = config.get("scheduler", "intrawave")
    pad_m            = config.get("pad_m", False)
    pad_n            = config.get("pad_n", False)
    pad_k            = config.get("pad_k", True)
    block_size       = config.get("block_size", 256)
    k_block_per_cu   = config.get("k_block_per_cu", 1)
    quant_group_k    = config.get("quant_group_k", 128)
    preshuffle_quant = config.get("preshuffle_quant", False)

    # Support list of preshuffle_quant values for sweep
    if isinstance(preshuffle_quant, bool):
        preshuffle_vals = [preshuffle_quant]
    else:
        preshuffle_vals = list(preshuffle_quant)

    for variant_key, layout, tile_dict, pq in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        preshuffle_vals,
    ):
        if variant_key not in AQUANT_VARIANTS:
            log.warning("Unknown variant_key %s — skipping", variant_key)
            continue
        if pipeline not in AQUANT_PIPELINE_MAP:
            log.warning("Unsupported pipeline %s — skipping", pipeline)
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
            quant_group_k=quant_group_k,
            preshuffle_quant=pq,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            block_size=block_size,
            k_block_per_cu=k_block_per_cu,
        ))

    return specs


# =============================================================================
# Generation entry point
# =============================================================================


def generate_kernels(
    output_dir: Path,
    config: Optional[dict] = None,
    parallel: bool = True,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No kernel specs produced from config — check variant_keys and tile_configs")
        return []

    log.info("Generating %d AQuant kernel headers into %s", len(specs), output_dir)

    gen = AQuantKernelHeaderGenerator()
    generated: List[Path] = []

    def _generate_one(spec: AQuantKernelSpec) -> Path:
        header = gen.generate(spec)
        out_path = output_dir / f"{spec.name}.hpp"
        out_path.write_text(header)
        log.info("  wrote %s", out_path.name)
        return out_path

    if parallel and len(specs) > 1:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {ex.submit(_generate_one, s): s for s in specs}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    generated.append(fut.result())
                except Exception as e:
                    log.error("Failed generating %s: %s", futures[fut].name, e)
    else:
        for spec in specs:
            try:
                generated.append(_generate_one(spec))
            except Exception as e:
                log.error("Failed generating %s: %s", spec.name, e)

    log.info("Generated %d / %d headers", len(generated), len(specs))
    return generated


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AQuant GEMM kernel header generator"
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write generated .hpp files")
    parser.add_argument("--config", type=Path,
                        help="JSON config file (defaults to built-in sweep)")
    parser.add_argument("--config-json", type=str,
                        help="Inline JSON config string")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel generation")
    parser.add_argument("--list-names", action="store_true",
                        help="Print kernel names that would be generated and exit")
    args = parser.parse_args()

    cfg: Optional[dict] = None
    if args.config_json:
        try:
            cfg = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            log.error("Invalid --config-json: %s", e)
            return 1
    elif args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    if args.list_names:
        specs = _build_specs(cfg or _default_config())
        for s in specs:
            print(s.name)
        return 0

    paths = generate_kernels(
        output_dir=args.output_dir,
        config=cfg,
        parallel=not args.no_parallel,
    )
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
