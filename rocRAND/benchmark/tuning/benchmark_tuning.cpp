// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "benchmark_utils.hpp"
#include "benchmarked_generators.hpp"
#include "cmdparser.hpp"
#include "rng/xorwow.hpp"

int main(int argc, char** argv)
{
    constexpr std::size_t default_bytes  = 1024 * 1024 * 512;
    constexpr double      default_lambda = 10;

    benchmark::Initialize(&argc, argv);
    cli::Parser parser(argc, argv);
    parser.set_optional<std::size_t>("bytes",
                                     "bytes",
                                     default_bytes,
                                     "number of bytes to generate");
    parser.set_optional<double>("lambda",
                                "lambda",
                                default_lambda,
                                "lambda value to be used in the Poisson distribution");
    parser.run_and_exit_if_error();

    const benchmark_config config{
        parser.get<std::size_t>("bytes"),
        parser.get<double>("lambda"),
    };

    benchmark::AddCustomContext("bytes", std::to_string(config.bytes));
    benchmark::AddCustomContext("lambda", std::to_string(config.lambda));

    add_common_benchmark_rocrand_info();

    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::lfsr113_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::mrg31k3p_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::mrg32k3a_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::mt19937_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<benchmark_tuning::mtgp32_generator_template>(
        benchmarks,
        config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::philox4x32_10_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::threefry2x32_20_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::threefry2x64_20_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::threefry4x32_20_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<
        benchmark_tuning::threefry4x64_20_generator_template>(benchmarks, config);
    benchmark_tuning::add_all_benchmarks_for_generator<benchmark_tuning::xorwow_generator_template>(
        benchmarks,
        config);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }
    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
}
