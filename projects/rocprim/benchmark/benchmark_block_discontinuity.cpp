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

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/block/block_discontinuity.hpp>
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

template<typename Runner,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         WithTile,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* d_input, T* d_output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, WithTile, Trials>(d_input, d_output);
}

struct flag_heads
{
    template<typename T,
             unsigned int BlockSize,
             unsigned int ItemsPerThread,
             bool         WithTile,
             unsigned int Trials>
    __device__
    static void run(const T* d_input, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_discontinuity<T, BlockSize> bdiscontinuity;
            bool                                       head_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.flag_heads(head_flags, T(123), input, rocprim::equal_to<T>());
            }
            else
            {
                bdiscontinuity.flag_heads(head_flags, input, rocprim::equal_to<T>());
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += head_flags[i];
            }
            rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct flag_tails
{
    template<typename T,
             unsigned int BlockSize,
             unsigned int ItemsPerThread,
             bool         WithTile,
             unsigned int Trials>
    __device__
    static void run(const T* d_input, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_discontinuity<T, BlockSize> bdiscontinuity;
            bool                                       tail_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.flag_tails(tail_flags, T(123), input, rocprim::equal_to<T>());
            }
            else
            {
                bdiscontinuity.flag_tails(tail_flags, input, rocprim::equal_to<T>());
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += tail_flags[i];
            }
            rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct flag_heads_and_tails
{
    template<typename T,
             unsigned int BlockSize,
             unsigned int ItemsPerThread,
             bool         WithTile,
             unsigned int Trials>
    __device__
    static void run(const T* d_input, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_discontinuity<T, BlockSize> bdiscontinuity;
            bool                                       head_flags[ItemsPerThread];
            bool                                       tail_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.flag_heads_and_tails(head_flags,
                                                    T(123),
                                                    tail_flags,
                                                    T(234),
                                                    input,
                                                    rocprim::equal_to<T>());
            }
            else
            {
                bdiscontinuity.flag_heads_and_tails(head_flags,
                                                    tail_flags,
                                                    input,
                                                    rocprim::equal_to<T>());
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += head_flags[i];
                input[i] += tail_flags[i];
            }
            rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template<typename Benchmark,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         WithTile,
         unsigned int Trials = 100>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;
    const auto& stream = state.stream;

    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);

    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    const auto     random_range = limit_random_range<T>(0, 10);
    std::vector<T> input
        = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());
    T* d_input;
    T* d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            kernel<Benchmark, T, BlockSize, ItemsPerThread, WithTile, Trials>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input, d_output);
            HIP_CHECK(hipGetLastError());
        });

    state.set_throughput(size * Trials, sizeof(T));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE)                                                 \
    executor.queue_fn(bench_naming::format_name("{lvl:block,algo:discontinuity,subalgo:" + name \
                                                + ",key_type:" #T ",cfg:{bs:" #BS ",ipt:" #IPT  \
                                                  ",with_tile:" #WITH_TILE "}}")                \
                          .c_str(),                                                             \
                      run_benchmark<Benchmark, T, BS, IPT, WITH_TILE>);

#define BENCHMARK_TYPE(type, block, bool)  \
    CREATE_BENCHMARK(type, block, 1, bool) \
    CREATE_BENCHMARK(type, block, 2, bool) \
    CREATE_BENCHMARK(type, block, 3, bool) \
    CREATE_BENCHMARK(type, block, 4, bool) \
    CREATE_BENCHMARK(type, block, 8, bool)

template<typename Benchmark>
void add_benchmarks(const std::string& name, benchmark_utils::executor& executor)
{
    BENCHMARK_TYPE(int, 256, false)
    BENCHMARK_TYPE(int, 256, true)
    BENCHMARK_TYPE(int8_t, 256, false)
    BENCHMARK_TYPE(int8_t, 256, true)
    BENCHMARK_TYPE(uint8_t, 256, false)
    BENCHMARK_TYPE(uint8_t, 256, true)
    BENCHMARK_TYPE(rocprim::half, 256, false)
    BENCHMARK_TYPE(rocprim::half, 256, true)
    BENCHMARK_TYPE(long long, 256, false)
    BENCHMARK_TYPE(long long, 256, true)
    BENCHMARK_TYPE(rocprim::int128_t, 256, false)
    BENCHMARK_TYPE(rocprim::int128_t, 256, true)
    BENCHMARK_TYPE(rocprim::uint128_t, 256, false)
    BENCHMARK_TYPE(rocprim::uint128_t, 256, true)
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 512 * benchmark_utils::MiB, 1, 0);

    add_benchmarks<flag_heads>("flag_heads", executor);
    add_benchmarks<flag_tails>("flag_tails", executor);
    add_benchmarks<flag_heads_and_tails>("flag_heads_and_tails", executor);

    executor.run();
}
