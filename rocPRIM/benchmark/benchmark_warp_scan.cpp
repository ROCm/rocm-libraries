// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// CmdParser
#include "cmdparser.hpp"

#include "../common/utils_custom_type.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>
// HIP API
#include <hip/hip_runtime.h>
// rocPRIM
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_scan.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

#ifndef DEFAULT_BYTES
const size_t DEFAULT_BYTES = 1024 * 1024 * 32 * 4;
#endif

enum class scan_type
{
    inclusive_scan,
    exclusive_scan,
    broadcast
};

template<class Runner, class T, unsigned int WarpSize, unsigned int Trials>
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void kernel(const T* input, T* output, const T init)
{
    Runner::template run<T, WarpSize, Trials>(input, output, init);
}

struct inclusive_scan
{
    template<typename T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i     = blockIdx.x * blockDim.x + threadIdx.x;
        auto               value = input[i];

        using wscan_t = rocprim::warp_scan<T, WarpSize>;
        __shared__ typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wscan_t().inclusive_scan(value, value, storage);
        }

        output[i] = value;
    }
};

struct exclusive_scan
{
    template<typename T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        const unsigned int i     = blockIdx.x * blockDim.x + threadIdx.x;
        auto               value = input[i];

        using wscan_t = rocprim::warp_scan<T, WarpSize>;
        __shared__ typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wscan_t().exclusive_scan(value, value, init, storage);
        }

        output[i] = value;
    }
};

struct broadcast
{
    template<typename T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static void run(const T* input, T* output, const T init)
    {
        (void)init;
        const unsigned int i        = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int warp_id  = i / WarpSize;
        const unsigned int src_lane = warp_id % WarpSize;
        auto               value    = input[i];

        using wscan_t = rocprim::warp_scan<T, WarpSize>;
        __shared__ typename wscan_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            value = wscan_t().broadcast(value, src_lane, storage);
        }

        output[i] = value;
    }
};

template<typename T,
         unsigned int BlockSize,
         unsigned int WarpSize,
         class Type,
         unsigned int Trials = 100>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t bytes)
{
    // Calculate the number of elements
    size_t size = bytes / sizeof(T);

    // Make sure size is a multiple of BlockSize
    size = BlockSize * ((size + BlockSize - 1) / BlockSize);
    // Allocate and fill memory
    std::vector<T> input(size, (T)1);
    T*             d_input;
    T*             d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Type, T, WarpSize, Trials>),
                           dim3(size / BlockSize),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_input,
                           d_output,
                           input[0]);
        HIP_CHECK(hipGetLastError());

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK_IMPL(T, BS, WS, OP)                                                   \
    benchmark::RegisterBenchmark(                                                              \
        bench_naming::format_name("{lvl:warp,algo:scan,key_type:" #T ",subalgo:" + method_name \
                                  + ",ws:" #WS ",cfg:{bs:" #BS "}}")                           \
            .c_str(),                                                                          \
        run_benchmark<T, BS, WS, OP>,                                                          \
        stream,                                                                                \
        bytes)

#define CREATE_BENCHMARK(T, BS, WS) CREATE_BENCHMARK_IMPL(T, BS, WS, Benchmark)

#define BENCHMARK_TYPE(type)                                              \
    CREATE_BENCHMARK(type, 64, 64), CREATE_BENCHMARK(type, 128, 64),      \
        CREATE_BENCHMARK(type, 256, 64), CREATE_BENCHMARK(type, 256, 32), \
        CREATE_BENCHMARK(type, 256, 16), CREATE_BENCHMARK(type, 63, 63),  \
        CREATE_BENCHMARK(type, 62, 31), CREATE_BENCHMARK(type, 60, 15)

#define BENCHMARK_TYPE_P2(type)                                           \
    CREATE_BENCHMARK(type, 64, 64), CREATE_BENCHMARK(type, 128, 64),      \
        CREATE_BENCHMARK(type, 256, 64), CREATE_BENCHMARK(type, 256, 32), \
        CREATE_BENCHMARK(type, 256, 16)

template<typename Benchmark>
auto add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string&                            method_name,
                    hipStream_t                                   stream,
                    size_t                                        bytes)
    -> std::enable_if_t<std::is_same<Benchmark, inclusive_scan>::value
                        || std::is_same<Benchmark, exclusive_scan>::value>
{
    using custom_double2    = common::custom_type<double, double>;
    using custom_int_double = common::custom_type<int, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks
        = {BENCHMARK_TYPE(int),
           BENCHMARK_TYPE(float),
           BENCHMARK_TYPE(double),
           BENCHMARK_TYPE(int8_t),
           BENCHMARK_TYPE(uint8_t),
           BENCHMARK_TYPE(rocprim::half),
           BENCHMARK_TYPE(custom_double2),
           BENCHMARK_TYPE(custom_int_double),
           BENCHMARK_TYPE(rocprim::int128_t),
           BENCHMARK_TYPE(rocprim::uint128_t)};
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
}

template<typename Benchmark>
auto add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string&                            method_name,
                    hipStream_t                                   stream,
                    size_t bytes) -> std::enable_if_t<std::is_same<Benchmark, broadcast>::value>
{
    using custom_double2    = common::custom_type<double, double>;
    using custom_int_double = common::custom_type<int, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks
        = {BENCHMARK_TYPE_P2(int),
           BENCHMARK_TYPE_P2(float),
           BENCHMARK_TYPE_P2(double),
           BENCHMARK_TYPE_P2(int8_t),
           BENCHMARK_TYPE_P2(uint8_t),
           BENCHMARK_TYPE_P2(rocprim::half),
           BENCHMARK_TYPE_P2(custom_double2),
           BENCHMARK_TYPE_P2(custom_int_double),
           BENCHMARK_TYPE_P2(rocprim::int128_t),
           BENCHMARK_TYPE_P2(rocprim::uint128_t)};
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t bytes  = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("bytes", std::to_string(bytes));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<inclusive_scan>(benchmarks, "inclusive_scan", stream, bytes); //inclusive
    add_benchmarks<exclusive_scan>(benchmarks, "exclusive_scan", stream, bytes); //exclusive
    add_benchmarks<broadcast>(benchmarks, "broadcast", stream, bytes); //broadcast

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
