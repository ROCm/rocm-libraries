// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// CUB's implementation of single_pass_scan_operators has maybe uninitialized parameters,
// disable the warning because all warnings are threated as errors:

#include "common_benchmark_header.hpp"

// HIP API
#include <hipcub/device/device_for.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

const unsigned int batch_size  = 10;
const unsigned int warmup_size = 5;

template<class T>
struct op_t
{
    unsigned int* d_count;

    HIPCUB_DEVICE
    void          operator()(T i)
    {
        // The data is non zero so atomic will never be activated.
        if(i == 0)
        {
            atomicAdd(d_count, 1);
        }
    }
};

template<class Value>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    using T = Value;

    // Generate data
    std::vector<T> values_input(size, 4);

    T* d_input;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, values_input.data(), size * sizeof(T), hipMemcpyHostToDevice));

    unsigned int* d_count;
    HIP_CHECK(hipMalloc(&d_count, sizeof(T)));
    HIP_CHECK(hipMemset(d_count, 0, sizeof(T)));
    op_t<T> device_op{d_count};

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(hipcub::DeviceFor::ForEach(d_input, d_input + size, device_op, stream));
    }
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipcub::DeviceFor::ForEach(d_input, d_input + size, device_op, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_count));
    HIP_CHECK(hipFree(d_input));
}

#define CREATE_BENCHMARK(Value)                                     \
    benchmark::RegisterBenchmark(("for_each<Datatype:" #Value ">"), \
                                 &run_benchmark<Value>,             \
                                 stream,                            \
                                 size)

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

    std::cout << "benchmark_device_reduce_by_key" << std::endl;

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
        CREATE_BENCHMARK(float),
        CREATE_BENCHMARK(double),
        CREATE_BENCHMARK(custom_double2),
        CREATE_BENCHMARK(int8_t),
        CREATE_BENCHMARK(float),
        CREATE_BENCHMARK(double),
        CREATE_BENCHMARK(long long),
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
