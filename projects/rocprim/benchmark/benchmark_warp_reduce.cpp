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

#include "../common/utils_device_ptr.hpp"

#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_reduce.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

template<bool AllReduce, typename T, unsigned int VirtualWaveSize, unsigned int Trials>
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void warp_reduce_kernel(const T* d_input, T* d_output)
{
    if constexpr(VirtualWaveSize <= rocprim::arch::wavefront::max_size())
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        auto value = d_input[i];

        using wreduce_t = rocprim::warp_reduce<T, VirtualWaveSize, AllReduce>;
        __shared__ typename wreduce_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wreduce_t().reduce(value, value, storage);
        }

        d_output[i] = value;
    }
}

template<typename T, typename Flag, unsigned int VirtualWaveSize, unsigned int Trials>
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void segmented_warp_reduce_kernel(const T* d_input, Flag* d_flags, T* d_output)
{
    if constexpr(VirtualWaveSize <= rocprim::arch::wavefront::max_size())
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        auto value = d_input[i];
        auto flag  = d_flags[i];

        using wreduce_t = rocprim::warp_reduce<T, VirtualWaveSize>;
        __shared__ typename wreduce_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wreduce_t().head_segmented_reduce(value, value, flag, storage);
        }

        d_output[i] = value;
    }
}

template<bool         AllReduce,
         bool         Segmented,
         unsigned int VirtualWaveSize,
         unsigned int BlockSize,
         unsigned int Trials,
         typename T,
         typename Flag>
inline auto execute_warp_reduce_kernel(
    T* input, T* output, Flag* /* flags */, size_t size, hipStream_t stream) ->
    typename std::enable_if<!Segmented>::type
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(warp_reduce_kernel<AllReduce, T, VirtualWaveSize, Trials>),
                       dim3(size / BlockSize),
                       dim3(BlockSize),
                       0,
                       stream,
                       input,
                       output);
    HIP_CHECK(hipGetLastError());
}

template<bool         AllReduce,
         bool         Segmented,
         unsigned int VirtualWaveSize,
         unsigned int BlockSize,
         unsigned int Trials,
         typename T,
         typename Flag>
inline auto
    execute_warp_reduce_kernel(T* input, T* output, Flag* flags, size_t size, hipStream_t stream) ->
    typename std::enable_if<Segmented>::type
{
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(segmented_warp_reduce_kernel<T, Flag, VirtualWaveSize, Trials>),
        dim3(size / BlockSize),
        dim3(BlockSize),
        0,
        stream,
        input,
        flags,
        output);
    HIP_CHECK(hipGetLastError());
}

template<bool AllReduce,
         bool Segmented,
         typename T,
         unsigned int VirtualWaveSize,
         unsigned int BlockSize,
         unsigned int Trials = 100>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;

    using flag_type = unsigned char;

    // Calculate the number of elements
    size_t N = bytes / sizeof(T);

    const auto size = BlockSize * ((N + BlockSize - 1) / BlockSize);

    const auto     random_range = limit_random_range<T>(0, 10);
    std::vector<T> input
        = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());
    std::vector<flag_type>        flags = get_random_data<flag_type>(size, 0, 1, seed.get_1());
    common::device_ptr<T>         d_input(input);
    common::device_ptr<flag_type> d_flags(flags);
    common::device_ptr<T>         d_output(size);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            execute_warp_reduce_kernel<AllReduce, Segmented, VirtualWaveSize, BlockSize, Trials>(
                d_input.get(),
                d_output.get(),
                d_flags.get(),
                size,
                stream);
        });

    state.set_throughput(Trials * size, sizeof(T));
}

#define CREATE_BENCHMARK(T, WS, BS)                                                           \
    executor.queue_fn(                                                                        \
        bench_naming::format_name("{lvl:warp,algo:reduce,key_type:" #T ",broadcast_result:"   \
                                  + std::string(AllReduce ? "true" : "false")                 \
                                  + ",segmented:" + std::string(Segmented ? "true" : "false") \
                                  + ",ws:" #WS ",cfg:{bs:" #BS "}}")                          \
            .c_str(),                                                                         \
        run_benchmark<AllReduce, Segmented, T, WS, BS>);

// clang-format off
#define BENCHMARK_TYPE(type)       \
    CREATE_BENCHMARK(type, 32, 64) \
    CREATE_BENCHMARK(type, 37, 64) \
    CREATE_BENCHMARK(type, 61, 64) \
    CREATE_BENCHMARK(type, 64, 64)
// clang-format on

template<bool AllReduce, bool Segmented>
void add_benchmarks(benchmark_utils::executor& executor)
{
    BENCHMARK_TYPE(int)
    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(rocprim::uint128_t)
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0);

    add_benchmarks<false, false>(executor);
    add_benchmarks<true, false>(executor);
    add_benchmarks<false, true>(executor);

    executor.run();
}
