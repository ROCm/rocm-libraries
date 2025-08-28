// MIT License
//
// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// HIP API
#include <hipcub/warp/warp_scan.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

enum class scan_type
{
    inclusive_scan,
    exclusive_scan,
    broadcast
};

template<class Runner, class T, unsigned int BlockSize, unsigned int WarpSize, unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void kernel(const T* input, T* output, const T init)
{
    Runner::template run<T, WarpSize, Trials>(input, output, init);
}

struct inclusive_scan
{
    template<class T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static auto run(const T* input, T* output, const T init)
        -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {
        (void)init;

        const unsigned int i     = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        auto               value = input[i];

        using wscan_t = hipcub::WarpScan<T, WarpSize>;
        __shared__ typename wscan_t::TempStorage storage;
        auto                                     scan_op = hipcub::Sum();
#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            wscan_t(storage).InclusiveScan(value, value, scan_op);
        }

        output[i] = value;
    }

    template<class T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static auto run(const T* /*input*/, T* /*output*/, const T /*init*/)
        -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {}
};

struct exclusive_scan
{
    template<class T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static auto run(const T* input, T* output, const T init)
        -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {
        const unsigned int i     = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        auto               value = input[i];

        using wscan_t = hipcub::WarpScan<T, WarpSize>;
        __shared__ typename wscan_t::TempStorage storage;
        auto                                     scan_op = hipcub::Sum();
#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            wscan_t(storage).ExclusiveScan(value, value, init, scan_op);
        }

        output[i] = value;
    }
    template<class T, unsigned int WarpSize, unsigned int Trials>
        __device__
    static auto run(const T* /*input*/, T* /*output*/, const T /*init*/)
        -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>>
    {}
};

struct broadcast
{
    template<class T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static auto run(const T* input, T* output, const T init)
        -> std::enable_if_t<(benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>
                             && benchmark_utils::is_power_of_two(WarpSize))>
    {
        (void)init;

        const unsigned int i        = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int warp_id  = i / WarpSize;
        const unsigned int src_lane = warp_id % WarpSize;
        auto               value    = input[i];

        using wscan_t = hipcub::WarpScan<T, WarpSize>;
        __shared__ typename wscan_t::TempStorage storage;
#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            value = wscan_t(storage).Broadcast(value, src_lane);
        }

        output[i] = value;
    }

    template<class T, unsigned int WarpSize, unsigned int Trials>
    __device__
    static auto run(const T* /*input*/, T* /*output*/, const T /*init*/)
        -> std::enable_if_t<!(benchmark_utils::device_test_enabled_for_warp_size_v<WarpSize>
                              && benchmark_utils::is_power_of_two(WarpSize))>
    {}
};

template<class Benchmark,
         class T,
         unsigned int BlockSize,
         unsigned int WarpSize,
         unsigned int Trials = 100>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t size)
{
    // Make sure size is a multiple of BlockSize
    size = BlockSize * ((size + BlockSize - 1) / BlockSize);
    // Allocate and fill memory
    std::vector<T> input(size, 1.0f);
    T*             d_input;
    T*             d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, WarpSize, Trials>),
                           dim3(size / BlockSize),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_input,
                           d_output,
                           input[0]);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * Trials);
    state.SetItemsProcessed(state.iterations() * size * Trials);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK_IMPL(T, BS, WS, OP)                                              \
    benchmark::RegisterBenchmark(std::string("warp_scan<data_type:" #T ",block_size:" #BS \
                                             ",warp_size:" #WS ">.sub_algorithm_name:"    \
                                             + method_name)                               \
                                     .c_str(),                                            \
                                 &run_benchmark<OP, T, BS, WS>,                           \
                                 stream,                                                  \
                                 size)

#define CREATE_BENCHMARK(T, BS, WS) CREATE_BENCHMARK_IMPL(T, BS, WS, Benchmark)

// clang-format off
#if HIPCUB_WARP_THREADS_MACRO == 32
    #define BENCHMARK_TYPE(type)         \
        CREATE_BENCHMARK(type, 60, 15),  \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 62, 31),  \
        CREATE_BENCHMARK(type, 256, 32)
#else
    #define BENCHMARK_TYPE(type)         \
        CREATE_BENCHMARK(type, 60, 15),  \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 62, 31),  \
        CREATE_BENCHMARK(type, 256, 32), \
        CREATE_BENCHMARK(type, 63, 63),  \
        CREATE_BENCHMARK(type, 64, 64),  \
        CREATE_BENCHMARK(type, 128, 64), \
        CREATE_BENCHMARK(type, 256, 64)
#endif

#if HIPCUB_WARP_THREADS_MACRO == 32
    #define BENCHMARK_TYPE_P2(type)      \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 256, 32)
#else
    #define BENCHMARK_TYPE_P2(type)      \
        CREATE_BENCHMARK(type, 256, 16), \
        CREATE_BENCHMARK(type, 256, 32), \
        CREATE_BENCHMARK(type, 64, 64),  \
        CREATE_BENCHMARK(type, 128, 64), \
        CREATE_BENCHMARK(type, 256, 64)
#endif
// clang-format on

template<typename Benchmark>
auto add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string&                            method_name,
                    hipStream_t                                   stream,
                    size_t                                        size)
    -> std::enable_if_t<std::is_same<Benchmark, inclusive_scan>::value
                        || std::is_same<Benchmark, exclusive_scan>::value>
{
    using custom_double2    = benchmark_utils::custom_type<double, double>;
    using custom_int_double = benchmark_utils::custom_type<int, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks
        = {BENCHMARK_TYPE(int),
           BENCHMARK_TYPE(float),
           BENCHMARK_TYPE(double),
           BENCHMARK_TYPE(int8_t),
           BENCHMARK_TYPE(custom_double2),
           BENCHMARK_TYPE(custom_int_double)};
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
}

template<typename Benchmark>
auto add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string&                            method_name,
                    hipStream_t                                   stream,
                    size_t size) -> std::enable_if_t<std::is_same<Benchmark, broadcast>::value>
{
    using custom_double2    = benchmark_utils::custom_type<double, double>;
    using custom_int_double = benchmark_utils::custom_type<int, double>;

    std::vector<benchmark::internal::Benchmark*> new_benchmarks
        = {BENCHMARK_TYPE_P2(int),
           BENCHMARK_TYPE_P2(float),
           BENCHMARK_TYPE_P2(double),
           BENCHMARK_TYPE_P2(int8_t),
           BENCHMARK_TYPE_P2(custom_double2),
           BENCHMARK_TYPE_P2(custom_int_double)};
    benchmarks.insert(benchmarks.end(), new_benchmarks.begin(), new_benchmarks.end());
}

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

    std::cout << "benchmark_warp_scan" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<inclusive_scan>(benchmarks, "inclusive_scan", stream, size);
    add_benchmarks<exclusive_scan>(benchmarks, "exclusive_scan", stream, size);
    add_benchmarks<broadcast>(benchmarks, "broadcast", stream, size);

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
