#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
TensorQuant GEMM Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.

Naming convention (byte-exact with TensorQuantKernelConfig.name in gemm_tensorquant_utils.py):
    gemm_tensor_quant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {PadM}_{PadN}_{PadK}_{Persistent}_{tile}

Persistent is always False. Note the prefix is "gemm_tensor_quant" (with underscore).

Reference:
    tile_engine/ops/gemm/block_scale_gemm/gemm_tensor_quant/gemm_tensor_quant_instance_builder.py

Key differences from RowColQuant:
  - TileGemmQuantTraits has NO UsePersistentKernel (no extra args)
  - Base pipeline: BaseWeightPreshufflePipelineAGmemBGmemCRegV2 (not BaseGemmPipelineAgBgCrCompV3)
  - GemmRowColTensorQuantPipelineProblem uses TransposeC=false, ComputeDataType=void
  - Uses GemmTile1DPartitioner
  - Field names: M_Tile, N_Tile, K_Tile, M_Warp, N_Warp, K_Warp, M_Warp_Tile, N_Warp_Tile, K_Warp_Tile
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures

from codegen_common import make_tensorquant_kernel_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


TENSORQUANT_DTYPE_TO_CK = {
    "fp8": "ck_tile::fp8_t",
    "bf8": "ck_tile::bf8_t",
}


@dataclass
class TensorQuantTileConfig:
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
class TensorQuantKernelSpec:
    variant_key: str       # "fp8", "bf8"
    layout: str            # "rcr" (only layout supported)
    pipeline: str          # "compv3"
    epilogue: str          # "cshuffle"
    scheduler: str         # "intrawave"
    tile: TensorQuantTileConfig
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_tensorquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
        )


class TensorQuantKernelHeaderGenerator:
    def generate(self, spec: TensorQuantKernelSpec) -> str:
        t = spec.tile
        ck_dtype = TENSORQUANT_DTYPE_TO_CK[spec.variant_key]
        struct = "SelectedKernel"

        pad_m = "true" if spec.pad_m else "false"
        pad_n = "true" if spec.pad_n else "false"
        pad_k = "true" if spec.pad_k else "false"

        # Scheduler string for CK: capitalize first letter
        scheduler_ck = f"ck_tile::GemmPipelineScheduler::{spec.scheduler.capitalize()}"

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated TensorQuant GEMM kernel header.
// DO NOT EDIT — regenerate via unified_gemm_tensorquant_codegen.py
#pragma once

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/epilogue.hpp"

using ADataType   = {ck_dtype};
using BDataType   = {ck_dtype};
using AQDataType  = float;
using BQDataType  = float;
using AccDataType = float;
using CDataType   = ck_tile::half_t;

// TensorQuant: rcr layout (A=RowMajor, B=ColMajor, C=RowMajor)
using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

constexpr const char* KERNEL_NAME = "{spec.name}";

struct {struct} {{
    static constexpr auto Scheduler = {scheduler_ck};

    static constexpr bool kPadM            = {pad_m};
    static constexpr bool kPadN            = {pad_n};
    static constexpr bool kPadK            = {pad_k};
    static constexpr bool TransposeC       = false;
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB      = false;
    static constexpr bool DoubleSmemBuffer = false;

    // Field names match gemm_tensor_quant_instance_builder.py
    static constexpr ck_tile::index_t M_Tile      = {t.tile_m};
    static constexpr ck_tile::index_t N_Tile      = {t.tile_n};
    static constexpr ck_tile::index_t K_Tile      = {t.tile_k};
    static constexpr ck_tile::index_t M_Warp      = {t.warp_m};
    static constexpr ck_tile::index_t N_Warp      = {t.warp_n};
    static constexpr ck_tile::index_t K_Warp      = {t.warp_k};
    static constexpr ck_tile::index_t M_Warp_Tile = {t.warp_tile_m};
    static constexpr ck_tile::index_t N_Warp_Tile = {t.warp_tile_n};
    static constexpr ck_tile::index_t K_Warp_Tile = {t.warp_tile_k};

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& stream)
    {{
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
            ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
            ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

        // TensorQuant TileGemmQuantTraits has NO UsePersistentKernel extra args
        using GemmTraits = ck_tile::TileGemmQuantTraits<
            kPadM, kPadN, kPadK,
            APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
            ALayout, BLayout, CLayout,
            ck_tile::QuantType::TensorQuant,
            ck_tile::tensor_layout::gemm::RowMajor,   // AQLayout placeholder
            ck_tile::tensor_layout::gemm::ColumnMajor, // BQLayout placeholder
            TransposeC, DoubleSmemBuffer>;

        // Base pipeline for hot loop detection — TensorQuant uses BaseWeightPreshuffle
        using BaseProblem = ck_tile::GemmPipelineProblemBase<
            ADataType, BDataType, AccDataType, GemmShape, GemmTraits, void>;

        using BaseGemmPipeline = ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<BaseProblem>;

        const ck_tile::index_t k_split = ck_tile::integer_least_multiple(args.K, K_Tile);
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const auto tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType, BDataType, AccDataType, AccDataType,
                GemmShape, GemmTraits,
                false,   // TransposeC
                void,    // ComputeDataType = void for TensorQuant
                Scheduler,
                has_hot_loop_v, tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<
                    typename PipelineProblem::AComputeDataType,
                    typename PipelineProblem::BComputeDataType,
                    ck_tile::tuple<>,
                    AccDataType, CDataType, ck_tile::tuple<>,
                    CLayout,
                    ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                    M_Warp, N_Warp,
                    M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,
                    false>>;  // TransposeC

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::TensorQuant>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                throw std::runtime_error("Arguments not supported for TensorQuant kernel");

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            constexpr int kBlockPerCu = {spec.k_block_per_cu};
            return ck_tile::launch_kernel(
                stream, ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(run, has_hot_loop, tail_num);
    }}
}};

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
// All types and SelectedKernel are already at global scope (not in a namespace)
// so no additional aliases are needed.
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


def _default_config() -> dict:
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 128},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": True,
        "k_block_per_cu": 1,
    }


def _build_specs(config: dict) -> List[TensorQuantKernelSpec]:
    specs = []
    pipeline     = config.get("pipeline", "compv3")
    epilogue     = config.get("epilogue", "cshuffle")
    scheduler    = config.get("scheduler", "intrawave")
    pad_m        = config.get("pad_m", False)
    pad_n        = config.get("pad_n", False)
    pad_k        = config.get("pad_k", True)
    k_block_per_cu = config.get("k_block_per_cu", 1)

    for variant_key, layout, tile_dict in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
    ):
        if variant_key not in TENSORQUANT_DTYPE_TO_CK:
            log.warning("Unknown variant_key %s — skipping", variant_key)
            continue

        tile = TensorQuantTileConfig(
            tile_m=tile_dict["tile_m"], tile_n=tile_dict["tile_n"], tile_k=tile_dict["tile_k"],
            warp_m=tile_dict["warp_m"], warp_n=tile_dict["warp_n"], warp_k=tile_dict["warp_k"],
            warp_tile_m=tile_dict["warp_tile_m"], warp_tile_n=tile_dict["warp_tile_n"],
            warp_tile_k=tile_dict["warp_tile_k"],
        )
        if not tile.is_valid():
            continue

        specs.append(TensorQuantKernelSpec(
            variant_key=variant_key, layout=layout,
            pipeline=pipeline, epilogue=epilogue, scheduler=scheduler,
            tile=tile, pad_m=pad_m, pad_n=pad_n, pad_k=pad_k,
            k_block_per_cu=k_block_per_cu,
        ))

    return specs


def generate_kernels(output_dir: Path, config: Optional[dict] = None, parallel: bool = True) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No TensorQuant kernel specs produced")
        return []

    gen = TensorQuantKernelHeaderGenerator()
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
    parser = argparse.ArgumentParser(description="TensorQuant GEMM kernel header generator")
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
