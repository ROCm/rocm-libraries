// MIT License
//
// Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_benchmark_header.hpp"
#include "hipcub/config.hpp"

// HIP API
#include <hipcub/device/device_reduce.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

template<class T, class OutputT, class ReduceKernel>
void run_benchmark(benchmark::State& state,
                   size_t            size,
                   const hipStream_t stream,
                   ReduceKernel      reduce)
{
    std::vector<T> input = benchmark_utils::get_random_data<T>(size, T(0), T(1000));

    T*       d_input;
    OutputT* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, sizeof(OutputT)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // Allocate temporary storage memory
    size_t temp_storage_size_bytes = 0;
    void*  d_temp_storage          = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK(reduce(d_temp_storage, temp_storage_size_bytes, d_input, d_output, size, stream));
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
    HIP_CHECK(hipDeviceSynchronize());
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(reduce(d_temp_storage, temp_storage_size_bytes, d_input, d_output, size, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                reduce(d_temp_storage, temp_storage_size_bytes, d_input, d_output, size, stream));
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
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}

template<typename T, typename Op>
struct Benchmark;

template<typename T>
struct Benchmark<T, hipcub::Sum>
{
    static void run(benchmark::State& state, size_t size, const hipStream_t stream)
    {
        hipError_t (*ptr_to_sum)(void*, size_t&, T*, T*, int, hipStream_t)
            = &hipcub::DeviceReduce::Sum;
        run_benchmark<T, T>(state, size, stream, ptr_to_sum);
    }
};

template<typename T>
struct Benchmark<T, hipcub::Min>
{
    static void run(benchmark::State& state, size_t size, const hipStream_t stream)
    {
        hipError_t (*ptr_to_min)(void*, size_t&, T*, T*, int, hipStream_t)
            = &hipcub::DeviceReduce::Min;
        run_benchmark<T, T>(state, size, stream, ptr_to_min);
    }
};

template<typename T>
struct Benchmark<T, hipcub::ArgMin>
{
    using Difference = int;
    using Iterator   = typename hipcub::ArgIndexInputIterator<T*, Difference>;
    using KeyValue   = typename Iterator::value_type;

    static void run(benchmark::State& state, size_t size, const hipStream_t stream)
    {
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
        hipError_t (*ptr_to_argmin)(void*, size_t&, T*, KeyValue*, int, hipStream_t)
        = &hipcub::DeviceReduce::ArgMin;
        HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
        run_benchmark<T, KeyValue>(state, size, stream, ptr_to_argmin);
    }
};

#define CREATE_BENCHMARK(T, REDUCE_OP)                                                \
    benchmark::RegisterBenchmark(std::string("device_reduce"                          \
                                             "<data_type:" #T ",op:" #REDUCE_OP ">.") \
                                     .c_str(),                                        \
                                 &Benchmark<T, REDUCE_OP>::run,                       \
                                 size,                                                \
                                 stream)

#define CREATE_BENCHMARKS(REDUCE_OP)                                             \
    CREATE_BENCHMARK(int, REDUCE_OP), CREATE_BENCHMARK(long long, REDUCE_OP),    \
        CREATE_BENCHMARK(float, REDUCE_OP), CREATE_BENCHMARK(double, REDUCE_OP), \
        CREATE_BENCHMARK(int8_t, REDUCE_OP)

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

    std::cout << "benchmark_device_reduce" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    using custom_double2 = benchmark_utils::custom_type<double, double>;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {
        CREATE_BENCHMARKS(hipcub::Sum),
        CREATE_BENCHMARK(custom_double2, hipcub::Sum),
        CREATE_BENCHMARKS(hipcub::Min),
#ifdef HIPCUB_ROCPRIM_API
        CREATE_BENCHMARK(custom_double2, hipcub::Min),
#endif
        CREATE_BENCHMARKS(hipcub::ArgMin),
#ifdef HIPCUB_ROCPRIM_API
        CREATE_BENCHMARK(custom_double2, hipcub::ArgMin),
#endif
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
