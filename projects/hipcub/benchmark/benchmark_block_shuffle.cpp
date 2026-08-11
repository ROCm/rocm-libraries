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

#include "common_benchmark_header.hpp"

#include <hipcub/block/block_shuffle.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<class Runner,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(input, output);
}

struct offset
{
    template<class T,
             unsigned int BlockSize,
             unsigned int /* ItemsPerThread */,
             unsigned int Trials>
    __device__ static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T value = input[tid];

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Offset(value, value, 1);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        output[tid] = value;
    }

    static constexpr bool uses_ipt = false;
};

struct rotate
{
    template<class T,
             unsigned int BlockSize,
             unsigned int /* ItemsPerThread */,
             unsigned int Trials>
    __device__ static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T value = input[tid];

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Rotate(value, value, 1);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        output[tid] = value;
    }

    static constexpr bool uses_ipt = false;
};

struct up
{
    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__ static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values[i] = input[ItemsPerThread * tid + i];
        }

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Up(values, values);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[ItemsPerThread * tid + i] = values[i];
        }
    }

    static constexpr bool uses_ipt = true;
};

struct down
{
    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__ static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values[i] = input[ItemsPerThread * tid + i];
        }

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Down(values, values);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[ItemsPerThread * tid + i] = values[i];
        }
    }

    static constexpr bool uses_ipt = true;
};

template<class Benchmark,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread = 1,
         unsigned int Trials         = 100>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    std::vector<T> input(size, T(1));
    T*             d_input;
    T*             d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>),
                           dim3(size / items_per_block),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_input,
                           d_output);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK_IPT(BS, IPT)                                                   \
    benchmark::RegisterBenchmark(                                                       \
        ("block_shuffle<data_type:" + type_name                                         \
         + ",block_size:" #BS ",items_per_thread:" #IPT ">.sub_algorithm_name:" + name) \
            .c_str(),                                                                   \
        &run_benchmark<Benchmark, T, BS, IPT>,                                          \
        stream,                                                                         \
        size)

#define CREATE_BENCHMARK(BS)                                                           \
    benchmark::RegisterBenchmark(("block_shuffle<data_type:" + type_name               \
                                  + ",block_size:" #BS ">.sub_algorithm_name:" + name) \
                                     .c_str(),                                         \
                                 &run_benchmark<Benchmark, T, BS>,                     \
                                 stream,                                               \
                                 size)

template<class Benchmark, class T, std::enable_if_t<Benchmark::uses_ipt, bool> = true>
void add_benchmarks_type(const std::string&                            name,
                         std::vector<benchmark::internal::Benchmark*>& benchmarks,
                         hipStream_t                                   stream,
                         size_t                                        size,
                         const std::string&                            type_name)
{
    std::vector<benchmark::internal::Benchmark*> bs = {
        CREATE_BENCHMARK_IPT(256, 1),
        CREATE_BENCHMARK_IPT(256, 3),
        CREATE_BENCHMARK_IPT(256, 4),
        CREATE_BENCHMARK_IPT(256, 8),
        CREATE_BENCHMARK_IPT(256, 16),
        CREATE_BENCHMARK_IPT(256, 32),
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

template<class Benchmark, class T, std::enable_if_t<!Benchmark::uses_ipt, bool> = true>
void add_benchmarks_type(const std::string&                            name,
                         std::vector<benchmark::internal::Benchmark*>& benchmarks,
                         hipStream_t                                   stream,
                         size_t                                        size,
                         const std::string&                            type_name)
{
    std::vector<benchmark::internal::Benchmark*> bs = {
        CREATE_BENCHMARK(256),
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

#define CREATE_BENCHMARKS(T) add_benchmarks_type<Benchmark, T>(name, benchmarks, stream, size, #T)

template<class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t                                   stream,
                    size_t                                        size)
{
    using custom_float2  = benchmark_utils::custom_type<float, float>;
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    CREATE_BENCHMARKS(int);
    CREATE_BENCHMARKS(float);
    CREATE_BENCHMARKS(double);
    CREATE_BENCHMARKS(int8_t);
    CREATE_BENCHMARKS(long long);
    CREATE_BENCHMARKS(custom_float2);
    CREATE_BENCHMARKS(custom_double2);
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

    std::cout << "benchmark_block_shuffle" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;

    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<offset>("offset", benchmarks, stream, size);
    add_benchmarks<rotate>("rotate", benchmarks, stream, size);
    add_benchmarks<up>("up", benchmarks, stream, size);
    add_benchmarks<down>("down", benchmarks, stream, size);

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
