#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
RowColQuant GEMM Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.

Naming convention (byte-exact with RowColQuantKernelConfig.name in gemm_rowcolquant_utils.py):
    gemm_rowcolquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {PadM}_{PadN}_{PadK}_{Persistent}_{tile}

Persistent is always False (RowColQuant forces persistent=False in trait generation).

Reference:
    tile_engine/ops/gemm/block_scale_gemm/gemm_rowcolquant/gemm_rowcolquant_instance_builder.py
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures

from codegen_common import make_rowcolquant_kernel_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


ROWCOLQUANT_DTYPE_TO_CK = {
    "fp8": "ck_tile::fp8_t",
    "bf8": "ck_tile::bf8_t",
}


@dataclass
class RowColQuantTileConfig:
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
class RowColQuantKernelSpec:
    variant_key: str       # "fp8", "bf8"
    layout: str            # "rcr" (only layout supported)
    pipeline: str          # "compv3"
    epilogue: str          # "cshuffle"
    scheduler: str         # "intrawave"
    tile: RowColQuantTileConfig
    persistent: bool = False  # always False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_rowcolquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            persistent=self.persistent,
        )


class RowColQuantKernelHeaderGenerator:
    def generate(self, spec: RowColQuantKernelSpec) -> str:
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_dtype = ROWCOLQUANT_DTYPE_TO_CK[spec.variant_key]

        pad_m     = str(spec.pad_m).lower()
        pad_n     = str(spec.pad_n).lower()
        pad_k     = str(spec.pad_k).lower()
        persistent = str(spec.persistent).lower()

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated RowColQuant GEMM kernel header.
// DO NOT EDIT — regenerate via unified_gemm_rowcolquant_codegen.py
#pragma once

#include <stdexcept>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_dtype};
using BDataType   = {ck_dtype};
using AQDataType  = float;
using BQDataType  = float;
using AccDataType = float;
using CDataType   = ck_tile::half_t;

// RowColQuant always uses RowMajor A, ColMajor B (rcr layout)
using ALayout  = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout  = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout  = ck_tile::tensor_layout::gemm::RowMajor;
using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using AQDataType  = {ns}::AQDataType;
    using BQDataType  = {ns}::BQDataType;
    using AccDataType = {ns}::AccDataType;

    static constexpr ck_tile::index_t kBlockPerCu    = {spec.k_block_per_cu};
    static constexpr ck_tile::index_t TileM          = {t.tile_m};
    static constexpr ck_tile::index_t TileN          = {t.tile_n};
    static constexpr ck_tile::index_t TileK          = {t.tile_k};
    static constexpr ck_tile::index_t WarpPerBlock_M = {t.warp_m};
    static constexpr ck_tile::index_t WarpPerBlock_N = {t.warp_n};
    static constexpr ck_tile::index_t WarpPerBlock_K = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM      = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN      = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK      = {t.warp_tile_k};

    static constexpr bool kPadM            = {pad_m};
    static constexpr bool kPadN            = {pad_n};
    static constexpr bool kPadK            = {pad_k};
    static constexpr bool TransposeC       = false;
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB      = false;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool UsePersistentKernel = {persistent};

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& stream)
    {{
        using ComputeDataType = ADataType;

        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<TileM, TileN, TileK>,
            ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
            ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

        // RowColQuant TileGemmQuantTraits has extra UsePersistentKernel + 16 args
        using GemmTraits = ck_tile::TileGemmQuantTraits<
            kPadM, kPadN, kPadK,
            APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
            {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
            ck_tile::QuantType::RowColQuant,
            {ns}::AQLayout, {ns}::BQLayout,
            TransposeC, DoubleSmemBuffer,
            UsePersistentKernel, 16>;

        using BasePipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
            ADataType, BDataType, AccDataType, AccDataType,
            GemmShape, GemmTraits, TransposeC, ComputeDataType,
            ck_tile::GemmPipelineScheduler::Intrawave>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<BasePipelineProblem>;

        const ck_tile::index_t k_split = ck_tile::integer_least_multiple(args.K, TileK);
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType, BDataType, AccDataType, AccDataType,
                GemmShape, GemmTraits, TransposeC, ComputeDataType,
                ck_tile::GemmPipelineScheduler::Intrawave,
                has_hot_loop_v, tail_number_v>;

            using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<
                    ADataType, BDataType, ck_tile::tuple<>,
                    AccDataType, CDataType, ck_tile::tuple<>,
                    {ns}::CLayout,
                    ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
                    WarpPerBlock_M, WarpPerBlock_N,
                    WarpTileM, WarpTileN, WarpTileK,
                    TransposeC>>;

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::RowColQuant>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                throw std::runtime_error("Arguments not supported for RowColQuant kernel");

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
            return ck_tile::launch_kernel(
                stream, ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(run, has_hot_loop, tail_num);
    }}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using SelectedKernel = {ns}::{struct};
constexpr const char* KERNEL_NAME = {ns}::KERNEL_NAME;
using ADataType   = {ck_dtype};
using BDataType   = {ck_dtype};
using CDataType   = ck_tile::half_t;
using AQDataType  = float;
using BQDataType  = float;
using AccDataType = float;
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


def _build_specs(config: dict) -> List[RowColQuantKernelSpec]:
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
        if variant_key not in ROWCOLQUANT_DTYPE_TO_CK:
            log.warning("Unknown variant_key %s — skipping", variant_key)
            continue

        tile = RowColQuantTileConfig(
            tile_m=tile_dict["tile_m"], tile_n=tile_dict["tile_n"], tile_k=tile_dict["tile_k"],
            warp_m=tile_dict["warp_m"], warp_n=tile_dict["warp_n"], warp_k=tile_dict["warp_k"],
            warp_tile_m=tile_dict["warp_tile_m"], warp_tile_n=tile_dict["warp_tile_n"],
            warp_tile_k=tile_dict["warp_tile_k"],
        )
        if not tile.is_valid():
            continue

        specs.append(RowColQuantKernelSpec(
            variant_key=variant_key, layout=layout,
            pipeline=pipeline, epilogue=epilogue, scheduler=scheduler,
            tile=tile, persistent=False,  # always False
            pad_m=pad_m, pad_n=pad_n, pad_k=pad_k,
            k_block_per_cu=k_block_per_cu,
        ))

    return specs


def generate_kernels(output_dir: Path, config: Optional[dict] = None, parallel: bool = True) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No RowColQuant kernel specs produced")
        return []

    gen = RowColQuantKernelHeaderGenerator()
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
    parser = argparse.ArgumentParser(description="RowColQuant GEMM kernel header generator")
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
