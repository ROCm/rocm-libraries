# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
from pathlib import Path


def _import_gemm_kernel_builder():
    current_dir = Path(__file__).resolve().parent
    module_path = current_dir.parent / "gemm_instance_builder.py"

    spec = importlib.util.spec_from_file_location("gemm_instance_builder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load GemmKernelBuilder from {module_path}")

    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)
    return gemm_builder_module.GemmKernelBuilder


GemmKernelBuilder = _import_gemm_kernel_builder()


class BatchedGemmKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ):
        super().__init__(
            "batched_gemm", working_path, gpu_target, datatype, layout, config_json
        )

    @staticmethod
    def _bool_from_str(value):
        return str(value).lower() in ["1", "true", "yes"]

    def list_kernels(self):
        self._list_kernels()

    def generate_single(self, kernel_name, tile_config_str, trait_combo_str):
        tile_parts = tile_config_str.split("_")
        tile_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")

        tile_config = {
            "tile_m": int(tile_dims[0]),
            "tile_n": int(tile_dims[1]),
            "tile_k": int(tile_dims[2]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "warp_k": int(warp_dims[2]),
            "warp_tile_m": int(warp_tile_dims[0]),
            "warp_tile_n": int(warp_tile_dims[1]),
            "warp_tile_k": int(warp_tile_dims[2]),
        }

        trait_parts = trait_combo_str.split("_")
        trait_combo = (
            trait_parts[0],
            trait_parts[1],
            trait_parts[2],
            self._bool_from_str(trait_parts[3]),
            self._bool_from_str(trait_parts[4]),
            self._bool_from_str(trait_parts[5]),
            self._bool_from_str(trait_parts[6]),
        )

        generated_name, _ = self._generate_kernel_instance(tile_config, trait_combo)
        if kernel_name and kernel_name != generated_name:
            raise ValueError(
                f"Kernel name mismatch: expected {kernel_name}, generated {generated_name}"
            )

    def populate_kernel_header(self, kernel_name):
        return f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <stdexcept>
#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
"""

    def populate_launch(
        self,
        scheduler_type_map,
        scheduler,
        pipeline_impl_map,
        pipeline,
        epilogue,
        k_block_per_cu,
        persistent,
    ):
        del persistent

        instance_code = f"""

    // Launch function
    static float launch(const ck_tile::BatchedGemmHostArgs& args, const ck_tile::stream_config& stream) {{
        constexpr auto scheduler = {scheduler_type_map.get(scheduler)};

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
            ADataType,
            BDataType,
            AccDataType,
            TileShape,
            ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                            ALayout, BLayout, CLayout, TransposeC,
                                            UseStructuredSparsity, UsePersistentKernel,
                                            NumWaveGroups, Preshuffle>,
            scheduler>;

        using GemmPipeline = {pipeline_impl_map.get(pipeline)}<UniversalGemmProblem>;
"""

        instance_code += self.populate_epilogue(epilogue)

        instance_code += f"""

        // Kernel type
        using GemmKernel = ck_tile::BatchedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        // Kernel arguments
        auto kargs = GemmKernel::MakeKernelArgs(args);

        if(!GemmKernel::IsSupportedArgument(kargs)) {{
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
        }}

        // Get grid and block sizes
        const dim3 grids = GemmKernel::GridSize(args.M, args.N, args.k_batch, args.batch_count);
        const dim3 blocks = GemmKernel::BlockSize();

        if(stream.log_level_ > 0) {{
            std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\\n'
                      << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
        }}

        // Launch kernel
        constexpr int kBlockPerCu = {k_block_per_cu};
        float ave_time = ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));

        return ave_time;
    }}
}};
"""

        return instance_code


def main():
    parser = argparse.ArgumentParser(description="Batched GEMM tile engine instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--gpu_target",
        required=True,
        help="GPU target architecture",
    )
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "fp8", "bf16", "bf8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", required=True, help="Configuration JSON file")
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument(
        "--tile_config", help="Tile configuration string for single generation"
    )
    parser.add_argument(
        "--trait_combo", help="Trait combination string for single generation"
    )

    args = parser.parse_args()

    layout_parts = args.layout.lower()
    assert len(layout_parts) == 3, (
        f"Invalid layout string: {args.layout} (must be 3 characters like 'rcr' where r stands for row major and c stands for column major)"
    )
    assert layout_parts[0] in ["r", "c"] and layout_parts[1] in ["r", "c"], (
        f"Invalid matrix_a layout : {layout_parts[0]} or matrix_b layout: {layout_parts[1]} (matrix_a and matrix_b must be either 'r' for row major or 'c' for column major)"
    )
    assert layout_parts[2] == "r", (
        f"Invalid matrix_c layout: {layout_parts[2]} (must be 'r' only as currently we are supporting only row major)"
    )

    builder = BatchedGemmKernelBuilder(
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.config_json,
    )

    if args.list_kernels:
        builder.list_kernels()
    elif args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error("--gen_single requires --kernel_name, --tile_config, and --trait_combo")
        builder.generate_single(args.kernel_name, args.tile_config, args.trait_combo)
    else:
        parser.error("Must specify one of: --list_kernels or --gen_single")


if __name__ == "__main__":
    main()
