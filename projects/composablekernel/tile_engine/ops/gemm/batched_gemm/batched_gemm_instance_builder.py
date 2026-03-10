# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path


def _import_validation_utils():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    spec = importlib.util.spec_from_file_location(
        "validation_utils",
        os.path.join(parent_dir, "gemm_validation_utils.py"),
    )
    validation_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_utils)
    return validation_utils


_validation_utils = _import_validation_utils()
is_tile_config_valid = _validation_utils.is_tile_config_valid
is_trait_combination_valid = _validation_utils.is_trait_combination_valid
get_abc_layouts = _validation_utils.get_abc_layouts
get_dtype_string = _validation_utils.get_dtype_string


class BatchedGemmKernelBuilder:
    def __init__(
        self,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ):
        self.kernel_name_prefix = "batched_gemm"
        self.working_path = Path(working_path)
        self.gpu_target = gpu_target
        self.datatype = datatype
        self.layout = layout

        self.working_path.mkdir(parents=True, exist_ok=True)

        with open(config_json, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    @staticmethod
    def _generate_values(min_val, max_val, step):
        values = []
        val = min_val
        while val <= max_val:
            values.append(val)
            val += step
        return values

    @staticmethod
    def _bool_to_cpp(value):
        return "true" if value else "false"

    @staticmethod
    def _bool_from_str(value):
        return str(value).lower() in ["1", "true", "yes"]

    @staticmethod
    def _tile_to_string(tile_config):
        tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
        tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
        tile_str += (
            f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"
        )
        return tile_str

    @staticmethod
    def _trait_to_string(trait_combo):
        return f"{trait_combo[0]}_{trait_combo[1]}_{trait_combo[2]}_" + "_".join(
            str(x) for x in trait_combo[3:]
        )

    def _parse_tile_config_values(self):
        tile_config = self.config["tile_config"]

        def parse_field(name):
            field = tile_config[name]
            if field.get("values") is not None:
                return field.get("values")
            return self._generate_values(field.get("min"), field.get("max"), field.get("step"))

        return {
            "tile_m": parse_field("tile_m"),
            "tile_n": parse_field("tile_n"),
            "tile_k": parse_field("tile_k"),
            "warp_m": parse_field("warp_m"),
            "warp_n": parse_field("warp_n"),
            "warp_k": parse_field("warp_k"),
            "warp_tile_m": parse_field("warp_tile_m"),
            "warp_tile_n": parse_field("warp_tile_n"),
            "warp_tile_k": parse_field("warp_tile_k"),
        }

    def _get_all_tile_configs(self):
        values = self._parse_tile_config_values()

        configs = []
        for tile_m in values["tile_m"]:
            for tile_n in values["tile_n"]:
                for tile_k in values["tile_k"]:
                    for warp_m in values["warp_m"]:
                        for warp_n in values["warp_n"]:
                            for warp_k in values["warp_k"]:
                                for warp_tile_m in values["warp_tile_m"]:
                                    for warp_tile_n in values["warp_tile_n"]:
                                        for warp_tile_k in values["warp_tile_k"]:
                                            configs.append(
                                                {
                                                    "tile_m": tile_m,
                                                    "tile_n": tile_n,
                                                    "tile_k": tile_k,
                                                    "warp_m": warp_m,
                                                    "warp_n": warp_n,
                                                    "warp_k": warp_k,
                                                    "warp_tile_m": warp_tile_m,
                                                    "warp_tile_n": warp_tile_n,
                                                    "warp_tile_k": warp_tile_k,
                                                }
                                            )
        return configs

    def _generate_trait_combinations(self):
        trait_config = self.config["trait_config"]

        pipelines = trait_config.get("pipeline").get("values")
        epilogues = trait_config.get("epilogue").get("values")
        schedulers = trait_config.get("scheduler").get("values")
        pad_m_values = trait_config.get("pad_m").get("values")
        pad_n_values = trait_config.get("pad_n").get("values")
        pad_k_values = trait_config.get("pad_k").get("values")
        persistent_values = trait_config.get("persistent").get("values")

        all_combinations = list(
            itertools.product(
                pipelines,
                epilogues,
                schedulers,
                pad_m_values,
                pad_n_values,
                pad_k_values,
                persistent_values,
            )
        )

        combinations = []
        for combo in all_combinations:
            pipeline, epilogue, scheduler = combo[:3]
            if is_trait_combination_valid(pipeline, epilogue, scheduler):
                combinations.append(combo)
        return combinations

    def _is_tile_valid_for_pipeline(self, tile_config, pipeline):
        a_datatype = self.datatype
        b_datatype = self.datatype
        c_datatype = self.datatype
        if self.datatype in ["fp8", "bf8"]:
            c_datatype = "fp16"

        return is_tile_config_valid(
            tile_config["tile_m"],
            tile_config["tile_n"],
            tile_config["tile_k"],
            tile_config["warp_m"],
            tile_config["warp_n"],
            tile_config["warp_k"],
            tile_config["warp_tile_m"],
            tile_config["warp_tile_n"],
            tile_config["warp_tile_k"],
            a_datatype,
            b_datatype,
            c_datatype,
            pipeline,
            self.layout,
            self.gpu_target,
        )

    def _kernel_name(self, tile_config, trait_combo):
        tile_str = self._tile_to_string(tile_config)
        (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent) = trait_combo

        return (
            f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_"
            f"{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_"
            f"{str(persistent).capitalize()}_{tile_str}"
        )

    def list_kernels(self):
        tile_configs = self._get_all_tile_configs()
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                pipeline = trait_combo[0]
                if not self._is_tile_valid_for_pipeline(tile_config, pipeline):
                    continue

                kernel_list.append(
                    {
                        "name": self._kernel_name(tile_config, trait_combo),
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                    }
                )

        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_count.txt", "w", encoding="utf-8"
        ) as f:
            f.write(str(len(kernel_list)))

        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_list.txt", "w", encoding="utf-8"
        ) as f:
            for kernel in kernel_list:
                tile_str = self._tile_to_string(kernel["tile_config"])
                trait_str = self._trait_to_string(kernel["trait_combo"])
                f.write(f"{kernel['name']}|{tile_str}|{trait_str}\n")

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _epilogue_code(self, epilogue):
        if epilogue == "cshuffle":
            return """
        using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
            ADataType,
            BDataType,
            ck_tile::tuple<>,
            AccDataType,
            CDataType,
            ck_tile::tuple<>,
            CLayout,
            ck_tile::element_wise::PassThrough,
            TileM,
            TileN,
            WarpPerBlock_M,
            WarpPerBlock_N,
            WarpTileM,
            WarpTileN,
            WarpTileK,
            TransposeC,
            NumWaveGroups>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
"""

        return """
        using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
            ADataType,
            BDataType,
            ck_tile::tuple<>,
            AccDataType,
            CDataType,
            ck_tile::tuple<>,
            CLayout,
            ck_tile::element_wise::PassThrough,
            TileM,
            TileN,
            kPadM,
            kPadN,
            WarpTileM,
            WarpTileN,
            WarpTileK,
            TransposeC>;
        using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
"""

    def _generate_kernel_instance(self, kernel_name, tile_config, trait_combo):
        (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent) = trait_combo

        pipeline_impl_map = {
            "mem": "ck_tile::GemmPipelineAgBgCrMem",
            "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
            "compv4": "ck_tile::GemmPipelineAgBgCrCompV4",
        }
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        if pipeline not in pipeline_impl_map:
            raise ValueError(f"Unsupported pipeline for batched_gemm: {pipeline}")

        a_layout, b_layout, c_layout = get_abc_layouts(self.layout)

        c_dtype = self.datatype
        if self.datatype in ["fp8", "bf8"]:
            c_dtype = "fp16"

        k_block_per_cu = self.config.get("k_block_per_cu", 1)

        instance_code = f"""// Generated kernel instance for {kernel_name}
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

using ADataType = {get_dtype_string(self.datatype)};
using BDataType = {get_dtype_string(self.datatype)};
using AccDataType = float;
using CDataType = {get_dtype_string(c_dtype)};

using ALayout = {a_layout};
using BLayout = {b_layout};
using CLayout = {c_layout};

constexpr const char* KERNEL_NAME = "{kernel_name}";

struct SelectedKernel
{{
    static constexpr ck_tile::index_t TileM = {tile_config['tile_m']};
    static constexpr ck_tile::index_t TileN = {tile_config['tile_n']};
    static constexpr ck_tile::index_t TileK = {tile_config['tile_k']};

    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config['warp_m']};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config['warp_n']};
    static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config['warp_k']};

    static constexpr ck_tile::index_t WarpTileM = {tile_config['warp_tile_m']};
    static constexpr ck_tile::index_t WarpTileN = {tile_config['warp_tile_n']};
    static constexpr ck_tile::index_t WarpTileK = {tile_config['warp_tile_k']};

    static constexpr bool kPadM = {self._bool_to_cpp(pad_m)};
    static constexpr bool kPadN = {self._bool_to_cpp(pad_n)};
    static constexpr bool kPadK = {self._bool_to_cpp(pad_k)};
    static constexpr bool UsePersistentKernel = {self._bool_to_cpp(persistent)};

    static constexpr bool DoubleSmemBuffer = {self._bool_to_cpp(pipeline == 'compv4')};
    static constexpr bool TransposeC = false;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool Preshuffle = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        false,
        false>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    static float launch(const ck_tile::BatchedGemmHostArgs& args, const ck_tile::stream_config& stream)
    {{
        constexpr auto scheduler = {scheduler_type_map[scheduler]};

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
            ADataType,
            BDataType,
            AccDataType,
            TileShape,
            ck_tile::TileGemmUniversalTraits<kPadM,
                                            kPadN,
                                            kPadK,
                                            DoubleSmemBuffer,
                                            ALayout,
                                            BLayout,
                                            CLayout,
                                            TransposeC,
                                            UseStructuredSparsity,
                                            UsePersistentKernel,
                                            NumWaveGroups,
                                            Preshuffle>,
            scheduler>;

        using GemmPipeline = {pipeline_impl_map[pipeline]}<UniversalGemmProblem>;
"""

        instance_code += self._epilogue_code(epilogue)

        instance_code += f"""
        using GemmKernel = ck_tile::BatchedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = GemmKernel::MakeKernelArgs(args);

        if(!GemmKernel::IsSupportedArgument(kargs))
        {{
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
        }}

        const dim3 grids = GemmKernel::GridSize(args.M, args.N, args.k_batch, args.batch_count);
        const dim3 blocks = GemmKernel::BlockSize();

        if(stream.log_level_ > 0)
        {{
            std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\\n'
                      << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
        }}

        constexpr int kBlockPerCu = {k_block_per_cu};
        return ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
    }}
}};
"""

        simplified_name = kernel_name
        if simplified_name.startswith(f"{self.kernel_name_prefix}_"):
            simplified_name = simplified_name[len(self.kernel_name_prefix) + 1 :]

        header_file = self.working_path / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

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

        self._generate_kernel_instance(kernel_name, tile_config, trait_combo)


def main():
    parser = argparse.ArgumentParser(description="Batched GEMM tile engine instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "bf16", "fp8", "bf8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", required=True, help="Configuration JSON file")

    parser.add_argument("--list_kernels", action="store_true", help="List kernel configurations")
    parser.add_argument("--gen_single", action="store_true", help="Generate one kernel header")
    parser.add_argument("--kernel_name", help="Kernel name for --gen_single")
    parser.add_argument("--tile_config", help="Tile config string for --gen_single")
    parser.add_argument("--trait_combo", help="Trait combo string for --gen_single")

    args = parser.parse_args()

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
