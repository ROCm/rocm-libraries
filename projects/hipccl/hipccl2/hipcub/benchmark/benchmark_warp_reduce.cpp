// MIT License
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_benchmark_header.hpp"

// HIP API
#include "hipcub/warp/warp_reduce.hpp"


#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<
    class T,
    unsigned int WarpSize,
    unsigned int Trials
>
__global__
__launch_bounds__(64, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void warp_reduce_kernel(const T * d_input, T * d_output)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = d_input[i];

    using wreduce_t = hipcub::WarpReduce<T, WarpSize>;
    __shared__ typename wreduce_t::TempStorage storage;
    auto reduce_op = hipcub::Sum();
    #pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        value = wreduce_t(storage).Reduce(value, reduce_op);
    }

    d_output[i] = value;
}

template<
    class T,
    class Flag,
    unsigned int WarpSize,
    unsigned int Trials
>
__global__
__launch_bounds__(64, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void segmented_warp_reduce_kernel(const T* d_input, Flag* d_flags, T* d_output)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = d_input[i];
    auto flag = d_flags[i];

    using wreduce_t = hipcub::WarpReduce<T, WarpSize>;
    __shared__ typename wreduce_t::TempStorage storage;
    #pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        value = wreduce_t(storage).HeadSegmentedSum(value, flag);
    }

    d_output[i] = value;
}

template<
    bool Segmented,
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int Trials,
    class T,
    class Flag
>
inline
auto execute_warp_reduce_kernel(T* input, T* output, Flag* /* flags */,
                                size_t size, hipStream_t stream)
    -> typename std::enable_if<!Segmented>::type
{
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(warp_reduce_kernel<T, WarpSize, Trials>),
        dim3(size/BlockSize), dim3(BlockSize), 0, stream,
        input, output
    );
    HIP_CHECK(hipPeekAtLastError());
}

template<
    bool Segmented,
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int Trials,
    class T,
    class Flag
>
inline
auto execute_warp_reduce_kernel(T* input, T* output, Flag* flags,
                                size_t size, hipStream_t stream)
    -> typename std::enable_if<Segmented>::type
{
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(segmented_warp_reduce_kernel<T, Flag, WarpSize, Trials>),
        dim3(size/BlockSize), dim3(BlockSize), 0, stream,
        input, flags, output
    );
    HIP_CHECK(hipPeekAtLastError());
}

template<
    bool Segmented,
    class T,
    unsigned int WarpSize,
    unsigned int BlockSize,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    using flag_type = unsigned char;

    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);

    std::vector<T> input = benchmark_utils::get_random_data<T>(size, T(0), T(10));
    std::vector<flag_type> flags = benchmark_utils::get_random_data<flag_type>(size, 0, 1);
    T * d_input;
    flag_type * d_flags;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_flags, size * sizeof(flag_type)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_flags, flags.data(),
            size * sizeof(flag_type),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        execute_warp_reduce_kernel<Segmented, WarpSize, BlockSize, Trials>(
            d_input, d_output, d_flags, size, stream
        );
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_flags));
}

#define CREATE_BENCHMARK(T, WS, BS) \
benchmark::RegisterBenchmark( \
    (std::string("warp_reduce<" #T ", " #WS ", " #BS ">.") + name).c_str(), \
    run_benchmark<Segmented, T, WS, BS>, \
    stream, size \
)


// If warp size limit is 16
#define BENCHMARK_TYPE_WS16(type) \
    CREATE_BENCHMARK(type, 15, 32), \
    CREATE_BENCHMARK(type, 16, 32)


// If warp size limit is 32
#define BENCHMARK_TYPE_WS32(type) \
    BENCHMARK_TYPE_WS16(type), \
    CREATE_BENCHMARK(type, 31, 32), \
    CREATE_BENCHMARK(type, 32, 32), \
    CREATE_BENCHMARK(type, 32, 64)


// If warp size limit is 64
#define BENCHMARK_TYPE_WS64(type) \
    BENCHMARK_TYPE_WS32(type), \
    CREATE_BENCHMARK(type, 37, 64), \
    CREATE_BENCHMARK(type, 61, 64), \
    CREATE_BENCHMARK(type, 64, 64)


template<bool Segmented>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    std::vector<benchmark::internal::Benchmark*> bs =
    {
#if HIPCUB_WARP_THREADS_MACRO == 16
        BENCHMARK_TYPE_WS16(int),
        BENCHMARK_TYPE_WS16(float),
        BENCHMARK_TYPE_WS16(double),
        BENCHMARK_TYPE_WS16(int8_t),
        BENCHMARK_TYPE_WS16(uint8_t)
#elif HIPCUB_WARP_THREADS_MACRO == 32
        BENCHMARK_TYPE_WS32(int),
        BENCHMARK_TYPE_WS32(float),
        BENCHMARK_TYPE_WS32(double),
        BENCHMARK_TYPE_WS32(int8_t),
        BENCHMARK_TYPE_WS32(uint8_t)
#else
        BENCHMARK_TYPE_WS64(int),
        BENCHMARK_TYPE_WS64(float),
        BENCHMARK_TYPE_WS64(double),
        BENCHMARK_TYPE_WS64(int8_t),
        BENCHMARK_TYPE_WS64(uint8_t)
#endif
    };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<false>("reduce", benchmarks, stream, size);
    add_benchmarks<true>("segmented_reduce", benchmarks, stream, size);

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
