// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "flatmm_common.hpp"
#include "flatmm_profiler.hpp"

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
        .insert("n", "4096", "The value for n dimension. Default is 4096.")
        .insert("k", "2048", "The value for k dimension. Default is 2048.")
        .insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
        .insert("stride_b", "0", "The stride value for tensor B. Default is 0.")
        .insert("stride_c", "0", "The stride value for tensor C. Default is 0.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("verify",
                "2",
                "Set to 0 for no validation, 1 for CPU validation, or 2 for GPU validation.")
        .insert(
            "log", "false", "Whether to output kernel instance information. Values: true or false.")
        .insert("warmup", "50", "The number of warmup iterations. Default is 50.")
        .insert("repeat", "100", "The number of benchmark iterations. Default is 100.")
        .insert("timer",
                "true",
                "Whether to use the GPU timer. Values: true or false. Default is true.")
        .insert(
            "init", "0", "Initialization method. 0=random, 1=linear, 2=constant(1). Default is 0.")
        .insert("flush_cache",
                "true",
                "Whether to flush cache between iterations. Values: true or false.")
        .insert("rotating_count", "1000", "Number of rotating-cache iterations.")
        .insert(
            "metric", "0", "Metric to select the best kernel. 0=latency, 1=tflops, 2=bandwidth.")
        .insert("csv_filename", "", "The filename stem for CSV benchmark output. Default is empty.")
        .insert("json_output",
                "false",
                "Whether to output results in JSON format only. Values: true or false.");

    const bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    const std::string dtype_a   = DataTypeTraits<ADataType>::name;
    const std::string dtype_b   = DataTypeTraits<BDataType>::name;
    const std::string dtype_acc = DataTypeTraits<AccDataType>::name;
    const std::string dtype_c   = DataTypeTraits<CDataType>::name;

    const std::string layout_a = ALayout::name;
    const std::string layout_b = BLayout::name;
    const std::string layout_c = CLayout::name;

    FlatmmProblem flatmm_problem{arg_parser.get_int("split_k"),
                                 arg_parser.get_int("m"),
                                 arg_parser.get_int("n"),
                                 arg_parser.get_int("k"),
                                 arg_parser.get_int("stride_a"),
                                 arg_parser.get_int("stride_b"),
                                 arg_parser.get_int("stride_c"),
                                 dtype_a,
                                 dtype_b,
                                 dtype_acc,
                                 dtype_c,
                                 layout_a,
                                 layout_b,
                                 layout_c};

    Setting setting{arg_parser.get_int("warmup"),
                    arg_parser.get_int("repeat"),
                    arg_parser.get_bool("timer"),
                    arg_parser.get_int("verify"),
                    arg_parser.get_int("init"),
                    arg_parser.get_bool("log"),
                    arg_parser.get_str("csv_filename"),
                    arg_parser.get_bool("flush_cache"),
                    arg_parser.get_int("rotating_count"),
                    arg_parser.get_bool("json_output")};

    auto& profiler = FlatmmProfiler::instance(setting);

    try
    {
        const std::tuple<int, int, int> warp_tile_dims = std::make_tuple(
            SelectedKernel::WarpTileM, SelectedKernel::WarpTileN, SelectedKernel::WarpTileK);
        const std::tuple<int, int, int> tile_dims =
            std::make_tuple(SelectedKernel::TileM, SelectedKernel::TileN, SelectedKernel::TileK);
        const std::tuple<int, int, int> warp_dims = std::make_tuple(SelectedKernel::WarpPerBlock_M,
                                                                    SelectedKernel::WarpPerBlock_N,
                                                                    SelectedKernel::WarpPerBlock_K);

        KernelConfig config{tile_dims, warp_dims, warp_tile_dims, SelectedKernel::PermuteN};

        auto kernel_func = [](const ck_tile::FlatmmHostArgs<>& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        profiler.benchmark(flatmm_problem, kernel_func, config);
        profiler.select_best_instance(static_cast<Metric>(arg_parser.get_int("metric")));
    }
    catch(const std::exception& error)
    {
        std::cerr << "Benchmark failed: " << error.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv);
        if(!result)
        {
            return EXIT_FAILURE;
        }

        benchmark_single(parser);
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
