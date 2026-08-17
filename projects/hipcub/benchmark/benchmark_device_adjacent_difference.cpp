// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// CUB's implementation of DeviceRunLengthEncode has unused parameters,
// disable the warning because all warnings are threated as errors:

#include "common_benchmark_header.hpp"

#include <benchmark/benchmark.h>

#include "cmdparser.hpp"

#include <hipcub/device/device_adjacent_difference.hpp>

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{

#ifndef DEFAULT_N
constexpr std::size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

constexpr unsigned int batch_size  = 10;
constexpr unsigned int warmup_size = 5;

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::true_type /*copy*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeftCopy(temporary_storage,
                                                                storage_size,
                                                                input,
                                                                output,
                                                                std::forward<Args>(args)...);
}

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::true_type /*copy*/,
                                  void* const    temporary_storage,
                                  std::size_t&   storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRightCopy(temporary_storage,
                                                                 storage_size,
                                                                 input,
                                                                 output,
                                                                 std::forward<Args>(args)...);
}

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::false_type /*copy*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeft(temporary_storage,
                                                            storage_size,
                                                            input,
                                                            std::forward<Args>(args)...);
}

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::false_type /*copy*/,
                                  void* const   temporary_storage,
                                  std::size_t&  storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRight(temporary_storage,
                                                             storage_size,
                                                             input,
                                                             std::forward<Args>(args)...);
}

template<typename T, bool left, bool copy>
void run_benchmark(benchmark::State& state, const std::size_t size, const hipStream_t stream)
{
    using output_type = T;

    // Generate data
    const std::vector<T> input = benchmark_utils::get_random_data<T>(size, 1, 100);

    T*           d_input;
    output_type* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
    HIP_CHECK(
        hipMemcpy(d_input, input.data(), input.size() * sizeof(input[0]), hipMemcpyHostToDevice));

    if(copy)
    {
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(output_type)));
    }

    static constexpr std::integral_constant<bool, left> left_tag;
    static constexpr std::integral_constant<bool, copy> copy_tag;

    // Allocate temporary storage
    std::size_t temp_storage_size{};
    void*       d_temp_storage = nullptr;

    const auto launch = [&]
    {
        return dispatch_adjacent_difference(left_tag,
                                            copy_tag,
                                            d_temp_storage,
                                            temp_storage_size,
                                            d_input,
                                            d_output,
                                            size,
                                            hipcub::Sum{},
                                            stream);
    };
    HIP_CHECK(launch());
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(launch());
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Run
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(launch());
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    if(copy)
    {
        HIP_CHECK(hipFree(d_output));
    }
    HIP_CHECK(hipFree(d_temp_storage));
}

} // namespace

using namespace std::string_literals;

#define CREATE_BENCHMARK(T, left, copy)                                             \
    benchmark::RegisterBenchmark(std::string("device_adjacent_difference"           \
                                             "<data_type:" #T ">."                  \
                                             "sub_algorithm_name:subtract_"         \
                                             + std::string(left ? "left" : "right") \
                                             + std::string(copy ? "_copy" : ""))    \
                                     .c_str(),                                      \
                                 &run_benchmark<T, left, copy>,                     \
                                 size,                                              \
                                 stream)

// clang-format off
#define CREATE_BENCHMARKS(T)           \
    CREATE_BENCHMARK(T, true,  false), \
    CREATE_BENCHMARK(T, true,  true),  \
    CREATE_BENCHMARK(T, false, false), \
    CREATE_BENCHMARK(T, false, true)
// clang-format on

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");

    // HIP
    const hipStream_t stream = 0; // default
    hipDeviceProp_t   devProp;
    int               device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    std::cout << "benchmark_device_adjacent_difference" << std::endl;
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    using custom_float2  = benchmark_utils::custom_type<float, float>;
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    // Add benchmarks
    const std::vector<benchmark::internal::Benchmark*> benchmarks = {
        CREATE_BENCHMARKS(int),
        CREATE_BENCHMARKS(std::int64_t),

        CREATE_BENCHMARKS(uint8_t),

        CREATE_BENCHMARKS(float),
        CREATE_BENCHMARKS(double),

        CREATE_BENCHMARKS(custom_float2),
        CREATE_BENCHMARKS(custom_double2),
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
