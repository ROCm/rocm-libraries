#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
BQuant GEMM Code Generator (block_scale_gemm operator, gemm_bquant_* naming)

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking ck_tile::QuantGemmHostArgs — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_bquant_ctypes_lib.cpp

Naming convention (byte-exact with GemmBQuantKernelConfig.name in gemm_bquant_utils.py):
    gemm_bquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {PadM}_{PadN}_{PadK}_{BPreshuffleQuant}_{tile}

Note: This generates "gemm_bquant_*" names matching the block_scale_gemm tile engine operator.
The existing unified_grouped_gemm_bquant_codegen.py generates "grouped_gemm_bquant_*" names.

Reference:
    tile_engine/ops/gemm/block_scale_gemm/gemm_bquant/gemm_bquant_instance_builder.py
    tile_engine/ops/gemm/gemm_instance_builder.py  (gemm_bquant branch)
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures

from codegen_common import make_gemm_bquant_kernel_name, BQUANT_DTYPE_MAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# BQDataType is always float32 per gemm_instance_builder.py line 671
# Variants: fp8, bf8 (same as block_scale_gemm gemm_bquant_instance_builder.py)
# =============================================================================

GEMM_BQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "ck_a":   "ck_tile::fp8_t",
        "ck_b":   "ck_tile::fp8_t",
        "ck_c":   "ck_tile::half_t",
        "ck_bq":  "float",
        "ck_acc": "float",
    },
    "bf8": {
        "ck_a":   "ck_tile::bf8_t",
        "ck_b":   "ck_tile::bf8_t",
        "ck_c":   "ck_tile::half_t",
        "ck_bq":  "float",
        "ck_acc": "float",
    },
}

GEMM_BQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# Only compv3 pipeline is supported for gemm_bquant in block_scale_gemm
GEMM_BQUANT_PIPELINE_MAP = {
    "compv3": "ck_tile::BQuantGemmPipelineAgBgCrCompV3",
}

GEMM_BQUANT_BASE_PIPELINE_MAP = {
    "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
}

GEMM_BQUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


@dataclass
class GemmBQuantTileConfig:
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
class GemmBQuantKernelSpec:
    """Complete specification for one gemm_bquant (block_scale_gemm) kernel."""

    variant_key: str       # "fp8", "bf8"
    layout: str            # "rcr", "rrr", "ccr", "crr"
    pipeline: str          # "compv3"
    epilogue: str          # "default" or "cshuffle"
    scheduler: str         # "intrawave"
    tile: GemmBQuantTileConfig
    quant_group_k: int = 128
    preshuffle_bquant: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False    # Note: block_scale_gemm example config uses pad_k=False by default
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_gemm_bquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            preshuffle_bquant=self.preshuffle_bquant,
        )


class GemmBQuantKernelHeaderGenerator:
    def generate(self, spec: GemmBQuantKernelSpec) -> str:
        variant = GEMM_BQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a   = variant["ck_a"]
        ck_b   = variant["ck_b"]
        ck_c   = variant["ck_c"]
        ck_bq  = variant["ck_bq"]
        ck_acc = variant["ck_acc"]

        layout_a_ck  = GEMM_BQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck  = GEMM_BQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck  = GEMM_BQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # BQ layout follows B layout (same convention as gemm_instance_builder)
        layout_bq_ck = layout_b_ck
        layout_aq_ck = layout_a_ck  # placeholder (unused for BQuant-only)

        pipeline_ck      = GEMM_BQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = GEMM_BQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck     = GEMM_BQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m            = str(spec.pad_m).lower()
        pad_n            = str(spec.pad_n).lower()
        pad_k            = str(spec.pad_k).lower()
        preshuffle_bquant = str(spec.preshuffle_bquant).lower()

        # Epilogue block: "default" uses DefaultGemm2DEpilogue; "cshuffle" uses CShuffleEpilogue
        if spec.epilogue == "cshuffle":
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
        else:
            # "default" — DefaultGemm2DEpilogue
            epilogue_block = f"""\
            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                typename PipelineProblem::AComputeDataType,
                typename PipelineProblem::BComputeDataType,
                ck_tile::tuple<>,
                AccDataType,
                CDataType,
                ck_tile::tuple<>,
                {ns}::CLayout,
                ck_tile::element_wise::PassThrough,
                TileM, TileN,
                kPadM, kPadN,
                WarpTileM, WarpTileN, WarpTileK,
                TransposeC>;

            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated BQuant GEMM kernel header (block_scale_gemm operator, gemm_bquant_* naming).
// DO NOT EDIT — regenerate via unified_gemm_bquant_codegen.py
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
using BQDataType  = {ck_bq};
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using BQLayout = {layout_bq_ck};
using AQLayout = {layout_aq_ck};  // placeholder — unused for BQuant-only

using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, {spec.quant_group_k}>>;

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using BQDataType  = {ns}::BQDataType;
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
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = {preshuffle_bquant};
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
        ck_tile::QuantType::BQuantGrouped,
        {ns}::BQLayout>;

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

            using PipelineProblem = ck_tile::GemmBQuantPipelineProblem<
                ADataType,
                BDataType,
                BQDataType,
                AccDataType,
                TileShape,
                GemmTraits,
                QuantGroupSize,
                ADataType,   // ComputeDataType
                {scheduler_ck},
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::BQuantGrouped>;

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
using BQDataType  = {ck_bq};
using AccDataType = {ck_acc};
using QuantGroupSize = {ns}::QuantGroupSize;
constexpr ck_tile::index_t GroupSizeK = {ns}::{struct}::GroupSizeK;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


def _default_config() -> dict:
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "default",
        "scheduler": "intrawave",
        "tile_configs": [
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 128},
        ],
        "quant_group_k": 128,
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "block_size": 256,
        "k_block_per_cu": 1,
        "preshuffle_bquant": False,
    }


def _build_specs(config: dict) -> List[GemmBQuantKernelSpec]:
    specs = []
    pipeline         = config.get("pipeline", "compv3")
    epilogue         = config.get("epilogue", "default")
    scheduler        = config.get("scheduler", "intrawave")
    pad_m            = config.get("pad_m", False)
    pad_n            = config.get("pad_n", False)
    pad_k            = config.get("pad_k", False)
    block_size       = config.get("block_size", 256)
    k_block_per_cu   = config.get("k_block_per_cu", 1)
    quant_group_k    = config.get("quant_group_k", 128)
    preshuffle_bquant = config.get("preshuffle_bquant", False)

    pq_vals = preshuffle_bquant if isinstance(preshuffle_bquant, list) else [preshuffle_bquant]

    for variant_key, layout, tile_dict, pq in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        pq_vals,
    ):
        if variant_key not in GEMM_BQUANT_VARIANTS:
            log.warning("Unknown variant_key %s — skipping", variant_key)
            continue
        if pipeline not in GEMM_BQUANT_PIPELINE_MAP:
            log.warning("Unsupported pipeline %s — skipping", pipeline)
            continue

        tile = GemmBQuantTileConfig(
            tile_m=tile_dict["tile_m"], tile_n=tile_dict["tile_n"], tile_k=tile_dict["tile_k"],
            warp_m=tile_dict["warp_m"], warp_n=tile_dict["warp_n"], warp_k=tile_dict["warp_k"],
            warp_tile_m=tile_dict["warp_tile_m"], warp_tile_n=tile_dict["warp_tile_n"],
            warp_tile_k=tile_dict["warp_tile_k"],
        )
        if not tile.is_valid():
            continue

        specs.append(GemmBQuantKernelSpec(
            variant_key=variant_key, layout=layout,
            pipeline=pipeline, epilogue=epilogue, scheduler=scheduler,
            tile=tile, quant_group_k=quant_group_k,
            preshuffle_bquant=pq,
            pad_m=pad_m, pad_n=pad_n, pad_k=pad_k,
            block_size=block_size, k_block_per_cu=k_block_per_cu,
        ))

    return specs


def generate_kernels(output_dir: Path, config: Optional[dict] = None, parallel: bool = True) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No gemm_bquant kernel specs produced")
        return []

    log.info("Generating %d gemm_bquant kernel headers into %s", len(specs), output_dir)

    gen = GemmBQuantKernelHeaderGenerator()
    generated: List[Path] = []

    def _generate_one(spec):
        out_path = output_dir / f"{spec.name}.hpp"
        out_path.write_text(gen.generate(spec))
        return out_path

    if parallel and len(specs) > 1:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {ex.submit(_generate_one, s): s for s in specs}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    generated.append(fut.result())
                except Exception as e:
                    log.error("Failed: %s", e)
    else:
        for spec in specs:
            try:
                generated.append(_generate_one(spec))
            except Exception as e:
                log.error("Failed: %s", e)

    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="BQuant GEMM (block_scale_gemm) kernel header generator")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-json", type=str)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--list-names", action="store_true")
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
        for s in _build_specs(cfg or _default_config()):
            print(s.name)
        return 0

    paths = generate_kernels(output_dir=args.output_dir, config=cfg, parallel=not args.no_parallel)
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
