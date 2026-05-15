#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Pooling-backward kernel instance builder for tile_engine.

Generates C++ kernel headers for max-pool backward operations with specific
block-size / vector-size combinations and trait combinations
(pooling_dim x has_overlap).

Usage:
    --list_kernels:   List valid kernel configurations
    --gen_single:     Generate a single kernel header
    --gen_individual: Generate all kernel headers
"""

import argparse
import concurrent.futures
import itertools
import json
import logging
import multiprocessing
import os
from pathlib import Path

from pooling_bwd_validation_utils import (
    get_dtype_string,
    is_tile_config_valid,
    is_trait_combination_valid,
)

logger = logging.getLogger(__name__)


class PoolingBwdKernelBuilder:
    def __init__(self, working_path, datatype, config_json=None):
        self.working_path = Path(working_path)
        self.datatype = datatype
        self.config_json = config_json

        self.working_path.mkdir(parents=True, exist_ok=True)

        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()

    def _get_default_config(self):
        return {
            "tile_config": {
                "block_size": {"values": [256]},
                "vector_size": {"values": [1, 2, 4]},
            },
            "trait_config": {
                "pooling_dim": {"values": ["2d", "3d"]},
                "has_overlap": {"values": [True, False]},
            },
        }

    def _get_tile_configs(self, fast_mode=False):
        if "tile_config" not in self.config:
            return []

        tile_config = self.config["tile_config"]
        block_size_values = tile_config.get("block_size", {}).get("values", [256])
        vector_size_values = tile_config.get("vector_size", {}).get("values", [1, 2, 4])

        configs = []
        for block_size in block_size_values:
            for vector_size in vector_size_values:
                if is_tile_config_valid(
                    block_size, vector_size, self.datatype, fast_mode=fast_mode
                ):
                    configs.append(
                        {"block_size": block_size, "vector_size": vector_size}
                    )
        return configs

    def _generate_trait_combinations(self):
        if "trait_config" not in self.config:
            return [("2d", True), ("2d", False), ("3d", True), ("3d", False)]

        trait_config = self.config["trait_config"]
        pooling_dims = trait_config.get("pooling_dim", {}).get("values", ["2d", "3d"])
        has_overlaps = trait_config.get("has_overlap", {}).get("values", [True, False])

        combinations = []
        for combo in itertools.product(pooling_dims, has_overlaps):
            pooling_dim, has_overlap = combo
            if is_trait_combination_valid(pooling_dim, has_overlap):
                combinations.append(combo)
        return combinations

    def _get_dtype_string(self):
        return get_dtype_string(self.datatype)

    def _generate_kernel_instance(self, tile_config, trait_combo):
        pooling_dim, has_overlap = trait_combo

        kernel_name = (
            f"pool_bwd_{self.datatype}_{pooling_dim}_"
            f"{'overlap' if has_overlap else 'nooverlap'}"
        )

        tile_str = f"{tile_config['block_size']}x{tile_config['vector_size']}"
        kernel_name += f"_{tile_str}"

        din_type = self._get_dtype_string()
        dout_type = din_type
        compute_type = "float"
        index_type = "ck_tile::index_t"

        has_overlap_str = "true" if has_overlap else "false"

        if pooling_dim == "2d":
            tensor_shape_type = (
                "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t, "
                "ck_tile::index_t, ck_tile::index_t>"
            )
            window_shape_type = "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t>"
            window_rank = 2
        else:
            tensor_shape_type = (
                "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t, "
                "ck_tile::index_t, ck_tile::index_t, ck_tile::index_t>"
            )
            window_shape_type = (
                "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t>"
            )
            window_rank = 3

        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/ops/pooling_bwd.hpp"

using DInDataType = {din_type};
using DOutDataType = {dout_type};
using ComputeDataType = {compute_type};
using IndexDataType = {index_type};

using TensorShape = {tensor_shape_type};
using WindowShape = {window_shape_type};

constexpr const char* KERNEL_NAME = "{kernel_name}";
constexpr int POOLING_BWD_DIM = {window_rank};

struct SelectedKernel {{
    static constexpr ck_tile::index_t kBlockSize  = {tile_config["block_size"]};
    static constexpr ck_tile::index_t kVectorSize = {tile_config["vector_size"]};
    static constexpr bool kHasOverlap = {has_overlap_str};

    using BwdShape = ck_tile::PoolBwdShape<kBlockSize, kVectorSize>;
    using BwdProblem = ck_tile::PoolBwdProblem<DOutDataType,
                                                IndexDataType,
                                                DInDataType,
                                                BwdShape,
                                                kHasOverlap>;
    using BwdKernel  = ck_tile::PoolBwdKernel<BwdProblem>;
    using CastKernel = ck_tile::PoolBwdCastKernel<float,
                                                  DInDataType,
                                                  BwdKernel::kBlockSize,
                                                  BwdKernel::kVectorSize>;

    static float launch(ck_tile::PoolBwdHostArgs& args,
                        const ck_tile::stream_config& stream) {{
        constexpr ck_tile::index_t kBlockPerCu = 1;
        const ck_tile::index_t kBlockSizeRt   = BwdKernel::BlockSize();

        if(!BwdKernel::IsSupportedArgument(args)) {{
            throw std::runtime_error(
                std::string("Unsupported arguments for pooling bwd kernel: ") + KERNEL_NAME);
        }}

        auto kernel_args = BwdKernel::MakeKernelArgs(args);

        const ck_tile::index_t kGridSize = BwdKernel::CalculateGridSize(stream, args.dout_length);

        if(stream.log_level_ > 0) {{
            std::cout << "Launching pooling-bwd kernel: " << KERNEL_NAME << "\\n"
                      << "  grid_size: " << kGridSize << ", block_size: " << kBlockSizeRt
                      << std::endl;
        }}

        if constexpr(BwdProblem::kNeedFp32Workspace) {{
            auto memset_workspace = [p_workspace = args.p_workspace,
                                     din_length  = args.din_length](
                                        const ck_tile::stream_config& s) {{
                HIP_CHECK_ERROR(hipMemsetAsync(p_workspace,
                                               0,
                                               static_cast<std::size_t>(din_length) *
                                                   sizeof(float),
                                               s.stream_id_));
            }};

            ck_tile::PoolBwdCastHostArgs cast_host_args{{args.p_workspace,
                                                        args.p_din,
                                                        args.din_length}};
            if(!CastKernel::IsSupportedArgument(cast_host_args)) {{
                throw std::runtime_error(
                    std::string("Unsupported arguments for pooling bwd cast kernel: ") +
                    KERNEL_NAME);
            }}
            auto cast_kargs = CastKernel::MakeKernelArgs(cast_host_args);
            const ck_tile::index_t cast_grid_size =
                CastKernel::CalculateGridSize(stream, args.din_length);

            return ck_tile::launch_kernel(
                stream,
                memset_workspace,
                ck_tile::make_kernel<kBlockPerCu>(
                    BwdKernel{{}}, kGridSize, kBlockSizeRt, 0, kernel_args),
                ck_tile::make_kernel<kBlockPerCu>(
                    CastKernel{{}}, cast_grid_size, CastKernel::BlockSize(), 0, cast_kargs));
        }} else {{
            auto memset_din = [p_din      = args.p_din,
                               din_length = args.din_length](
                                  const ck_tile::stream_config& s) {{
                HIP_CHECK_ERROR(hipMemsetAsync(p_din,
                                               0,
                                               static_cast<std::size_t>(din_length) *
                                                   sizeof(DInDataType),
                                               s.stream_id_));
            }};

            return ck_tile::launch_kernel(
                stream,
                memset_din,
                ck_tile::make_kernel<kBlockPerCu>(
                    BwdKernel{{}}, kGridSize, kBlockSizeRt, 0, kernel_args));
        }}
    }}
}};
"""
        return kernel_name, instance_code

    def write_kernel_list(self):
        tile_configs = self._get_tile_configs(fast_mode=False)
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                pooling_dim, has_overlap = trait_combo

                kernel_name = (
                    f"pool_bwd_{self.datatype}_{pooling_dim}_"
                    f"{'overlap' if has_overlap else 'nooverlap'}"
                )

                tile_str = f"{tile_config['block_size']}x{tile_config['vector_size']}"

                kernel_name += f"_{tile_str}"

                trait_str = f"{pooling_dim}_{'overlap' if has_overlap else 'nooverlap'}"

                kernel_list.append(
                    {
                        "name": kernel_name,
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                        "tile_str": tile_str,
                        "trait_str": trait_str,
                    }
                )

        with open(self.working_path / "pool_bwd_kernel_count.txt", "w") as f:
            f.write(str(len(kernel_list)))

        with open(self.working_path / "pool_bwd_kernel_list.txt", "w") as f:
            for kernel in kernel_list:
                f.write(
                    f"{kernel['name']}|{kernel['tile_str']}|{kernel['trait_str']}\n"
                )

        print(f"Listed {len(kernel_list)} kernel configurations")

    def generate_individual(self, num_workers=None):
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                work_items.append(
                    (tile_config, trait_combo, self.working_path, self.datatype)
                )

        print(
            f"Generating {len(work_items)} individual kernel files "
            f"using {num_workers} workers..."
        )

        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }

            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 10 == 0 or completed == len(work_items):
                    print(
                        f"  Progress: {completed}/{len(work_items)} kernels generated"
                    )

                try:
                    result = future.result()
                    if result:
                        kernel_list.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"Kernel generation failed for {item}: {exc}")

        kernel_list.sort(key=lambda x: x[0])
        print(
            f"Generated {len(kernel_list)} individual kernel files "
            f"in {self.working_path}"
        )

    def run(self, num_workers=None):
        self.generate_individual(num_workers)


def _generate_single_kernel_individual(work_item):
    tile_config, trait_combo, working_path, datatype = work_item

    builder = PoolingBwdKernelBuilder(working_path, datatype)

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        header_file = working_path / f"pooling_bwd_single_{kernel_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Pooling backward kernel instance builder for tile_engine"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "bf16", "fp32"],
        help="Data type (must stay in sync with SUPPORTED_DATATYPES "
        "in pooling_bwd_validation_utils.py)",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument(
        "--num_workers", type=int, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--gen_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument(
        "--tile_config", help="Tile configuration string for single generation"
    )
    parser.add_argument(
        "--trait_combo", help="Trait combination string for single generation"
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )

    args = parser.parse_args()

    builder = PoolingBwdKernelBuilder(
        args.working_path, args.datatype, args.config_json
    )

    if args.list_kernels:
        builder.write_kernel_list()
    elif args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        tile_parts = args.tile_config.split("x")
        tile_config = {
            "block_size": int(tile_parts[0]),
            "vector_size": int(tile_parts[1]),
        }

        trait_parts = args.trait_combo.split("_")
        pooling_dim = trait_parts[0]
        has_overlap = trait_parts[1] == "overlap"
        trait_combo = (pooling_dim, has_overlap)

        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        header_file = builder.working_path / f"pooling_bwd_single_{kernel_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

    elif args.gen_individual:
        builder.run(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
