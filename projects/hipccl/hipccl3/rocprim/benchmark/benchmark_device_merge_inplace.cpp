// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

#include "../common/utils_data_generation.hpp"

#include <benchmark/benchmark.h>

#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <rocprim/device/device_merge_inplace.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits_interface.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <random>
#include <sstream>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

#ifndef DEFAULT_BYTES
const size_t DEFAULT_BYTES = 1024 * 1024 * 32 * 4;
#endif

template<typename T, T increment>
struct random_monotonic_iterator
{
    const unsigned int seed;
    random_monotonic_iterator(unsigned int seed) : seed(seed) {}

    using difference_type = std::ptrdiff_t;
    using value_type      = T;

    // not all integral types are valid for int distribution
    using dist_value_type
        = std::conditional_t<rocprim::is_integral<T>::value
                                 && !common::is_valid_for_int_distribution<T>::value,
                             int,
                             T>;

    using dist_type = std::conditional_t<rocprim::is_integral<T>::value,
                                         common::uniform_int_distribution<dist_value_type>,
                                         std::uniform_real_distribution<T>>;

    std::mt19937 engine{seed};
    dist_type    dist{dist_value_type{0}, dist_value_type{increment}};

    dist_value_type value = dist_value_type{0};

    int operator*() const
    {
        return limit_cast<value_type>(value);
    }

    random_monotonic_iterator& operator++()
    {
        // prefix
        value += dist(engine);
        return *this;
    }

    random_monotonic_iterator operator++(int)
    {
        // postfix
        random_monotonic_iterator retval{*this};
        value += dist(engine);
        return retval;
    }
};

template<class ValueT>
struct inplace_runner
{
    using value_type = ValueT;
    using compare_op_type =
        typename std::conditional<std::is_same<value_type, rocprim::half>::value,
                                  half_less,
                                  rocprim::less<value_type>>::type;

    value_type* d_data;
    size_t      left_size;
    size_t      right_size;
    hipStream_t stream;

    void*  d_temporary_storage     = nullptr;
    size_t temporary_storage_bytes = 0;

    compare_op_type compare_op{};

    inplace_runner(value_type* data, size_t left_size, size_t right_size, hipStream_t stream)
        : d_data(data), left_size(left_size), right_size(right_size), stream(stream)
    {}

    size_t prepare()
    {
        HIP_CHECK(rocprim::merge_inplace(d_temporary_storage,
                                         temporary_storage_bytes,
                                         d_data,
                                         left_size,
                                         right_size,
                                         compare_op,
                                         stream));
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        return temporary_storage_bytes;
    }

    void run()
    {
        HIP_CHECK(rocprim::merge_inplace(d_temporary_storage,
                                         temporary_storage_bytes,
                                         d_data,
                                         left_size,
                                         right_size,
                                         compare_op,
                                         stream));
    }

    void clean()
    {
        HIP_CHECK(hipFree(d_temporary_storage));
    }
};

template<class ValueT, class RunnerT>
void run_merge_inplace_benchmarks(benchmark::State&   state,
                                  size_t              size_a,
                                  size_t              size_b,
                                  const managed_seed& seed,
                                  hipStream_t         stream)
{
    using value_type  = ValueT;
    using runner_type = RunnerT;

    size_t                  total_size = size_a + size_b;
    std::vector<value_type> h_data(total_size);

    auto gen_a_it = random_monotonic_iterator<value_type, 4>{seed.get_0()};
    auto gen_b_it = random_monotonic_iterator<value_type, 4>{seed.get_1()};

    // generate left array
    for(size_t i = 0; i < size_a; ++i)
    {
        h_data[i] = static_cast<value_type>(*(gen_a_it++));
    }

    // generate right array
    for(size_t i = 0; i < size_b; ++i)
    {
        h_data[size_a + i] = static_cast<value_type>(*(gen_b_it++));
    }

    size_t num_bytes = total_size * sizeof(value_type);

    value_type* d_data;

    HIP_CHECK(hipMalloc(&d_data, num_bytes));

    runner_type runner{d_data, size_a, size_b, stream};

    size_t temp_storage_size = runner.prepare();

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        HIP_CHECK(hipMemcpy(d_data, h_data.data(), num_bytes, hipMemcpyHostToDevice));

        HIP_CHECK(hipEventRecord(start, stream));
        runner.run();
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);

        std::stringstream label;
        label << "temp_storage=" << temp_storage_size;

        state.SetLabel(label.str());
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * num_bytes);
    state.SetItemsProcessed(state.iterations() * total_size);

    HIP_CHECK(hipFree(d_data));
    runner.clean();
}

#define CREATE_MERGE_INPLACE_BENCHMARK(Value)                                         \
    benchmark::RegisterBenchmark(                                                     \
        bench_naming::format_name("{lvl:device,algo:merge_inplace,value_type:" #Value \
                                  ",cfg:default_config}")                             \
            .c_str(),                                                                 \
        [=](benchmark::State& state) {                                                \
            run_merge_inplace_benchmarks<Value, inplace_runner<Value>>(state,         \
                                                                       size,          \
                                                                       size,          \
                                                                       seed,          \
                                                                       stream);       \
        })

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = hipStreamDefault; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {
        CREATE_MERGE_INPLACE_BENCHMARK(int8_t),
        CREATE_MERGE_INPLACE_BENCHMARK(int16_t),
        CREATE_MERGE_INPLACE_BENCHMARK(int32_t),
        CREATE_MERGE_INPLACE_BENCHMARK(int64_t),
        CREATE_MERGE_INPLACE_BENCHMARK(rocprim::int128_t),
    };

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
