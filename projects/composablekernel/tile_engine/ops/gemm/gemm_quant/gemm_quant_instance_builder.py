#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
import os


def _import_gemm_builder_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gemm_builder_module = _import_gemm_builder_module()
GemmKernelBuilder = gemm_builder_module.GemmKernelBuilder
get_dtype_string = gemm_builder_module.get_dtype_string
get_abc_layouts = gemm_builder_module.get_abc_layouts


class GemmQuantKernelBuilder(GemmKernelBuilder):
    PROFILE_MAP = {
        "aquantdecode": {
            "quant_mode": "AQuantGrouped",
            "scheduler": "interwave",
            "validation_pipeline": "mem",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": False,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "aquantprefill": {
            "quant_mode": "AQuantGrouped",
            "scheduler": "intrawave",
            "validation_pipeline": "mem",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": False,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "aquantpreshufflequant": {
            "quant_mode": "AQuantGrouped",
            "scheduler": "intrawave",
            "validation_pipeline": "compv3",
            "apreshuffle_quant": True,
            "bpreshuffle_quant": False,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "bquantdecode": {
            "quant_mode": "BQuantGrouped",
            "scheduler": "intrawave",
            "validation_pipeline": "compv3",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": False,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "bquantprefill": {
            "quant_mode": "BQuantGrouped",
            "scheduler": "intrawave",
            "validation_pipeline": "compv3",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": False,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "bquantpreshufflequant": {
            "quant_mode": "BQuantGrouped",
            "scheduler": "intrawave",
            "validation_pipeline": "compv3",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": True,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "bquantpreshufflequantprefill": {
            "quant_mode": "BQuantGrouped",
            "scheduler": "intrawave",
            "validation_pipeline": "compv3",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": True,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "rowcol": {
            "quant_mode": "RowColQuant",
            "scheduler": "intrawave",
            "validation_pipeline": "compv3",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": False,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
        "tensor": {
            "quant_mode": "TensorQuant",
            "scheduler": "intrawave",
            "validation_pipeline": "compv3",
            "apreshuffle_quant": False,
            "bpreshuffle_quant": False,
            "preshuffle_b": False,
            "double_smem_buffer": False,
        },
    }

    def _generate_trait_combinations(self):
        trait_config = self.config["trait_config"]
        quant_config = self.config["quant_config"]

        profiles = trait_config.get("profile").get("values")
        epilogues = trait_config.get("epilogue").get("values")
        pad_m_values = trait_config.get("pad_m").get("values")
        pad_n_values = trait_config.get("pad_n").get("values")
        pad_k_values = trait_config.get("pad_k").get("values")
        persistent_values = trait_config.get("persistent").get("values")
        a_group_values = quant_config.get("a_group").get("values")
        b_group_values = quant_config.get("b_group").get("values")

        combinations = []
        for profile in profiles:
            for epilogue in epilogues:
                for pad_m in pad_m_values:
                    for pad_n in pad_n_values:
                        for pad_k in pad_k_values:
                            for persistent in persistent_values:
                                for a_group in a_group_values:
                                    for b_group in b_group_values:
                                        trait_combo = {
                                            "profile": profile,
                                            "epilogue": epilogue,
                                            "scheduler": self.PROFILE_MAP[profile][
                                                "scheduler"
                                            ],
                                            "pad_m": pad_m,
                                            "pad_n": pad_n,
                                            "pad_k": pad_k,
                                            "persistent": persistent,
                                            "a_group": a_group,
                                            "b_group": b_group,
                                        }
                                        if self._is_supported_trait_combo(trait_combo):
                                            combinations.append(trait_combo)
        return combinations

    def _is_supported_trait_combo(self, trait_combo):
        profile = trait_combo["profile"]
        epilogue = trait_combo["epilogue"]

        if epilogue != "cshuffle":
            return False

        if trait_combo["persistent"]:
            return False

        if trait_combo["pad_m"] or trait_combo["pad_n"]:
            return False

        if trait_combo["pad_k"] is not True:
            return False

        if self.layout != "rcr":
            return False

        if profile.startswith("aquant"):
            return (
                trait_combo["a_group"] == "1x1x128"
                and trait_combo["b_group"] == "1x1x1"
            )

        if profile.startswith("bquant"):
            return (
                trait_combo["a_group"] == "1x1x1"
                and trait_combo["b_group"]
                in ["1x1x128", "1x8x128", "1x32x128", "1x64x128", "1x128x128"]
            )

        return trait_combo["a_group"] == "1x1x1" and trait_combo["b_group"] == "1x1x1"

    def _parse_quant_group(self, group_name):
        return [int(dim) for dim in group_name.split("x")]

    def _is_quant_configuration_valid(self, tile_config, trait_combo):
        profile = trait_combo["profile"]
        a_group_m, _, a_group_k = self._parse_quant_group(trait_combo["a_group"])
        b_group_m, _, b_group_k = self._parse_quant_group(trait_combo["b_group"])

        if tile_config["tile_m"] % a_group_m != 0 or tile_config["tile_k"] % a_group_k != 0:
            return False

        if tile_config["tile_m"] % b_group_m != 0 or tile_config["tile_k"] % b_group_k != 0:
            return False

        if profile.startswith("bquant") and b_group_k % tile_config["warp_tile_k"] != 0:
            return False

        return True

    def _is_supported_configuration(self, tile_config, trait_combo):
        profile_spec = self.PROFILE_MAP[trait_combo["profile"]]

        if not self._is_quant_configuration_valid(tile_config, trait_combo):
            return False

        return self._validate_tile_config(
            tile_config["tile_m"],
            tile_config["tile_n"],
            tile_config["tile_k"],
            tile_config["warp_m"],
            tile_config["warp_n"],
            tile_config["warp_k"],
            tile_config["warp_tile_m"],
            tile_config["warp_tile_n"],
            tile_config["warp_tile_k"],
            profile_spec["validation_pipeline"],
        )

    def _build_trait_string(self, trait_combo):
        parts = []
        for key in [
            "profile",
            "epilogue",
            "scheduler",
            "pad_m",
            "pad_n",
            "pad_k",
            "persistent",
            "a_group",
            "b_group",
        ]:
            parts.append(f"{key}={trait_combo[key]}")
        return "__".join(parts)

    def _parse_trait_string(self, trait_string):
        trait_combo = {}
        for part in trait_string.split("__"):
            key, value = part.split("=", 1)
            if key in ["pad_m", "pad_n", "pad_k", "persistent"]:
                trait_combo[key] = value == "True"
            else:
                trait_combo[key] = value
        return trait_combo

    def _build_kernel_name(self, tile_config, trait_combo):
        profile = trait_combo["profile"]
        epilogue = trait_combo["epilogue"]
        scheduler = trait_combo["scheduler"]
        tile_str = self._format_tile_config(tile_config)

        return (
            f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_{profile}_{epilogue}_"
            f"{scheduler}_{str(trait_combo['pad_m']).capitalize()}_"
            f"{str(trait_combo['pad_n']).capitalize()}_"
            f"{str(trait_combo['pad_k']).capitalize()}_"
            f"{str(trait_combo['persistent']).capitalize()}_{trait_combo['a_group']}_"
            f"{trait_combo['b_group']}_{tile_str}"
        )

    def _list_kernels(self):
        tile_configs = self._iter_tile_config_candidates()
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                if not self._is_supported_configuration(tile_config, trait_combo):
                    continue

                kernel_list.append(
                    {
                        "name": self._build_kernel_name(tile_config, trait_combo),
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                    }
                )

        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_count.txt", "w"
        ) as f:
            f.write(str(len(kernel_list)))

        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_list.txt", "w"
        ) as f:
            for kernel in kernel_list:
                f.write(
                    f"{kernel['name']}|{self._format_tile_config(kernel['tile_config'])}|"
                    f"{self._build_trait_string(kernel['trait_combo'])}\n"
                )

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _group_to_type(self, group_string):
        m, n, k = [int(dim) for dim in group_string.split("x")]
        return f"ck_tile::QuantGroupShape<ck_tile::sequence<{m}, {n}, {k}>>"

    def _generate_kernel_instance(self, tile_config, trait_combo):
        if not self._is_supported_configuration(tile_config, trait_combo):
            raise ValueError("Unsupported gemm_quant configuration")

        profile = trait_combo["profile"]
        profile_spec = self.PROFILE_MAP[profile]
        kernel_name = self._build_kernel_name(tile_config, trait_combo)
        k_block_per_cu = self.config.get("k_block_per_cu", 1)

        a_layout, b_layout, c_layout = get_abc_layouts(self.layout)
        aq_layout = "ck_tile::tensor_layout::gemm::RowMajor"
        bq_layout = "ck_tile::tensor_layout::gemm::ColumnMajor"

        quant_mode = profile_spec["quant_mode"]
        q_dtype = "float"

        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

using ADataType = {get_dtype_string(self.datatype)};
using BDataType = {get_dtype_string(self.datatype)};
using AQDataType = {q_dtype};
using BQDataType = {q_dtype};
using AccDataType = float;
using CDataType = ck_tile::half_t;

using ALayout = {a_layout};
using AQLayout = {aq_layout};
using BLayout = {b_layout};
using BQLayout = {bq_layout};
using CLayout = {c_layout};

using AQuantGroupSize = {self._group_to_type(trait_combo["a_group"])};
using BQuantGroupSize = {self._group_to_type(trait_combo["b_group"])};

constexpr const char* KERNEL_NAME = "{kernel_name}";
constexpr const char* QUANT_MODE_NAME = "{quant_mode}";
constexpr const char* QUANT_PROFILE_NAME = "{profile}";
constexpr const char* AQ_GROUP_NAME = "{trait_combo["a_group"]}";
constexpr const char* BQ_GROUP_NAME = "{trait_combo["b_group"]}";

struct SelectedKernel {{
    static constexpr auto QuantMode = ck_tile::QuantType::{quant_mode};
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::{trait_combo["scheduler"].capitalize()};

    static constexpr bool kPadM = {"true" if trait_combo["pad_m"] else "false"};
    static constexpr bool kPadN = {"true" if trait_combo["pad_n"] else "false"};
    static constexpr bool kPadK = {"true" if trait_combo["pad_k"] else "false"};
    static constexpr bool TransposeC = false;
    static constexpr bool APreshuffleQuant = {"true" if profile_spec["apreshuffle_quant"] else "false"};
    static constexpr bool BPreshuffleQuant = {"true" if profile_spec["bpreshuffle_quant"] else "false"};
    static constexpr bool PreshuffleB = {"true" if profile_spec["preshuffle_b"] else "false"};
    static constexpr bool DoubleSmemBuffer = {"true" if profile_spec["double_smem_buffer"] else "false"};

    static constexpr ck_tile::index_t M_Tile = {tile_config["tile_m"]};
    static constexpr ck_tile::index_t N_Tile = {tile_config["tile_n"]};
    static constexpr ck_tile::index_t K_Tile = {tile_config["tile_k"]};
    static constexpr ck_tile::index_t M_Warp = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t N_Warp = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t K_Warp = {tile_config["warp_k"]};
    static constexpr ck_tile::index_t M_Warp_Tile = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t N_Warp_Tile = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t K_Warp_Tile = {tile_config["warp_tile_k"]};

    static float launch(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& stream)
    {{
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
            ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
            ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;
        using GemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
                                                        kPadN,
                                                        kPadK,
                                                        APreshuffleQuant,
                                                        BPreshuffleQuant,
                                                        PreshuffleB,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        QuantMode,
                                                        AQLayout,
                                                        BQLayout,
                                                        TransposeC,
                                                        DoubleSmemBuffer>;

        using BaseProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             GemmShape,
                                                             GemmTraits,
                                                             void>;

        using BaseGemmPipeline = std::conditional_t<
            QuantMode == ck_tile::QuantType::AQuantGrouped && APreshuffleQuant,
            ck_tile::BaseGemmPipelineAgBgCrCompV3<BaseProblem>,
            std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped,
                               ck_tile::BaseGemmPipelineAgBgCrMem<BaseProblem>,
                               ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<BaseProblem>>>;

        const ck_tile::index_t k_split = ck_tile::integer_least_multiple(args.K, K_Tile);
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const auto tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;
            constexpr auto b_cast_policy = ck_tile::CastPolicy::BeforeLDSWrite;

            using PipelineProblem = std::conditional_t<
                QuantMode == ck_tile::QuantType::RowColQuant || QuantMode == ck_tile::QuantType::TensorQuant,
                ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              AccDataType,
                                                              GemmShape,
                                                              GemmTraits,
                                                              false,
                                                              void,
                                                              Scheduler,
                                                              has_hot_loop_v,
                                                              tail_number_v>,
                std::conditional_t<
                    QuantMode == ck_tile::QuantType::AQuantGrouped,
                    ck_tile::GemmAQuantPipelineProblem<ADataType,
                                                       AQDataType,
                                                       BDataType,
                                                       AccDataType,
                                                       GemmShape,
                                                       GemmTraits,
                                                       AQuantGroupSize,
                                                       false,
                                                       void,
                                                       Scheduler,
                                                       has_hot_loop_v,
                                                       tail_number_v>,
                    ck_tile::GemmBQuantPipelineProblem<ADataType,
                                                       BDataType,
                                                       BQDataType,
                                                       AccDataType,
                                                       GemmShape,
                                                       GemmTraits,
                                                       BQuantGroupSize,
                                                       void,
                                                       Scheduler,
                                                       has_hot_loop_v,
                                                       tail_number_v,
                                                       b_cast_policy>>>;

            using AQuantPipeline = std::conditional_t<APreshuffleQuant,
                                                      ck_tile::AQuantGemmPipelineAgBgCrCompV3<PipelineProblem>,
                                                      ck_tile::AQuantGemmPipelineAgBgCrMem<PipelineProblem>>;
            using BQuantPipeline = ck_tile::BQuantGemmPipelineAgBgCrCompV3<PipelineProblem>;
            using GemmPipeline = std::conditional_t<
                QuantMode == ck_tile::QuantType::RowColQuant || QuantMode == ck_tile::QuantType::TensorQuant,
                ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>,
                std::conditional_t<QuantMode == ck_tile::QuantType::AQuantGrouped,
                                   AQuantPipeline,
                                   BQuantPipeline>>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<typename PipelineProblem::ComputeDataType,
                                                 typename PipelineProblem::ComputeDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 M_Warp,
                                                 N_Warp,
                                                 M_Warp_Tile,
                                                 N_Warp_Tile,
                                                 K_Warp_Tile,
                                                 false>>;

            using Kernel = ck_tile::QuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
            {{
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm_quant!");
            }}

            const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            constexpr int kBlockPerCu = {k_block_per_cu};
            return ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(run, has_hot_loop, tail_num);
    }}
}};
"""

        simplified_name = kernel_name[len(self.kernel_name_prefix) + 1 :]
        header_file = (
            self.working_path / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")
        return kernel_name, instance_code


def main():
    parser = argparse.ArgumentParser(description="GEMM Quant kernel instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp8", "bf8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", required=True, help="Configuration JSON file")
    parser.add_argument("--gen_single", action="store_true", help="Generate a single kernel file")
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument("--tile_config", help="Tile configuration string for single generation")
    parser.add_argument("--trait_combo", help="Trait combination string for single generation")
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )

    args = parser.parse_args()

    builder = GemmQuantKernelBuilder(
        "gemm_quant",
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.config_json,
    )

    if args.list_kernels:
        builder._list_kernels()
        return

    if args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        tile_parts = args.tile_config.split("_")
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
        trait_combo = builder._parse_trait_string(args.trait_combo)
        builder._generate_kernel_instance(tile_config, trait_combo)
        return

    parser.error("Must specify either --list_kernels or --gen_single")


if __name__ == "__main__":
    main()
