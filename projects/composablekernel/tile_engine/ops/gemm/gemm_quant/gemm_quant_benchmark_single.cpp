// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "gemm_quant_benchmark.hpp"
#include "gemm_quant_profiler.hpp"

void benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    GemmQuantProblem gemm_problem{};
    gemm_problem.split_k_ = arg_parser.get_int("split_k");
    gemm_problem.m_ = arg_parser.get_int("m");
    gemm_problem.n_ = arg_parser.get_int("n");
    gemm_problem.k_ = arg_parser.get_int("k");
    gemm_problem.stride_a_ = arg_parser.get_int("stride_a");
    gemm_problem.stride_b_ = arg_parser.get_int("stride_b");
    gemm_problem.stride_c_ = arg_parser.get_int("stride_c");

    const int stride_q = arg_parser.get_int("stride_q");
    gemm_problem.stride_aq_ = arg_parser.get_int("stride_aq");
    gemm_problem.stride_bq_ = arg_parser.get_int("stride_bq");
    if(gemm_problem.stride_aq_ == 0)
    {
        gemm_problem.stride_aq_ = stride_q;
    }
    if(gemm_problem.stride_bq_ == 0)
    {
        gemm_problem.stride_bq_ = stride_q;
    }

    gemm_problem.qk_a_ = 0;
    gemm_problem.qk_b_ = 0;
    gemm_problem.dtype_a_ = ck_tile::DataTypeTraits<ADataType>::name;
    gemm_problem.dtype_q_ = ck_tile::DataTypeTraits<AQDataType>::name;
    gemm_problem.dtype_b_ = ck_tile::DataTypeTraits<BDataType>::name;
    gemm_problem.dtype_acc_ = ck_tile::DataTypeTraits<AccDataType>::name;
    gemm_problem.dtype_c_ = ck_tile::DataTypeTraits<CDataType>::name;
    gemm_problem.layout_a_ = ALayout::name;
    gemm_problem.layout_aq_ = AQLayout::name;
    gemm_problem.layout_b_ = BLayout::name;
    gemm_problem.layout_bq_ = BQLayout::name;
    gemm_problem.layout_c_ = CLayout::name;
    gemm_problem.quant_mode_ = QUANT_MODE_NAME;
    gemm_problem.quant_profile_ = QUANT_PROFILE_NAME;
    gemm_problem.aq_group_ = AQ_GROUP_NAME;
    gemm_problem.bq_group_ = BQ_GROUP_NAME;
    gemm_problem.structured_sparsity_ = false;

    Settings setting{arg_parser.get_int("warmup"),
                     arg_parser.get_int("repeat"),
                     arg_parser.get_bool("timer"),
                     arg_parser.get_int("verify"),
                     arg_parser.get_int("init"),
                     arg_parser.get_bool("log"),
                     arg_parser.get_str("csv_filename"),
                     arg_parser.get_bool("flush_cache"),
                     arg_parser.get_int("rotating_count"),
                     arg_parser.get_bool("json_output")};

    auto& profiler = GemmQuantProfiler::BaseGemm::instance(setting);

    try
    {
        auto kernel_func = [](const ck_tile::QuantGemmHostArgs& args,
                              const ck_tile::stream_config& stream) {
            return SelectedKernel::launch(args, stream);
        };

        profiler.benchmark(gemm_problem, kernel_func);
        profiler.select_best_instance(static_cast<Metric>(arg_parser.get_int("metric")));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv, 1, add_quant_benchmark_args);
        if(!result)
        {
            return EXIT_FAILURE;
        }

        benchmark_single(parser);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
