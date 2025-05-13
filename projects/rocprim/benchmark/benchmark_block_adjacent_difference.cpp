// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/block/block_adjacent_difference.hpp>
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

template<typename Benchmark,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         WithTile,
         typename... Args>
__global__ __launch_bounds__(BlockSize)
void kernel(Args... args)
{
    Benchmark::template run<BlockSize, ItemsPerThread, WithTile>(args...);
}

struct subtract_left
{
    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; ++trial)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_diff_t().subtract_left(input, output, rocprim::minus<>{}, T(123), storage);
            }
            else
            {
                adjacent_diff_t().subtract_left(input, output, rocprim::minus<>{}, storage);
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }
            rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_left_partial
{
    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void
        run(const T* d_input, const unsigned int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        unsigned int tile_size = tile_sizes[blockIdx.x];

        // Try to evenly distribute the length of tile_sizes between all the trials
        const auto tile_size_diff = (BlockSize * ItemsPerThread) / trials + 1;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; ++trial)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_diff_t().subtract_left_partial(input,
                                                        output,
                                                        rocprim::minus<>{},
                                                        T(123),
                                                        tile_size,
                                                        storage);
            }
            else
            {
                adjacent_diff_t().subtract_left_partial(input,
                                                        output,
                                                        rocprim::minus<>{},
                                                        tile_size,
                                                        storage);
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }

            // Change the tile_size to even out the distribution
            tile_size = (tile_size + tile_size_diff) % (BlockSize * ItemsPerThread);
            rocprim::syncthreads();
        }
        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_right
{
    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; ++trial)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_diff_t().subtract_right(input,
                                                 output,
                                                 rocprim::minus<>{},
                                                 T(123),
                                                 storage);
            }
            else
            {
                adjacent_diff_t().subtract_right(input, output, rocprim::minus<>{}, storage);
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }
            rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_right_partial
{
    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void
        run(const T* d_input, const unsigned int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize>;
        __shared__ typename adjacent_diff_t::storage_type storage;

        unsigned int tile_size = tile_sizes[blockIdx.x];
        // Try to evenly distribute the length of tile_sizes between all the trials
        const auto tile_size_diff = (BlockSize * ItemsPerThread) / trials + 1;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < trials; ++trial)
        {
            T output[ItemsPerThread];
            adjacent_diff_t().subtract_right_partial(input,
                                                     output,
                                                     rocprim::minus<>{},
                                                     tile_size,
                                                     storage);

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }
            // Change the tile_size to even out the distribution
            tile_size = (tile_size + tile_size_diff) % (BlockSize * ItemsPerThread);
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
auto run_benchmark(benchmark_utils::state&& state)
    -> std::enable_if_t<!std::is_same<Benchmark, subtract_left_partial>::value
                        && !std::is_same<Benchmark, subtract_right_partial>::value>
{
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;
    const auto& stream = state.stream;

    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);

    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     num_blocks      = (N + items_per_block - 1) / items_per_block;
    // Round up size to the next multiple of items_per_block
    const auto size = num_blocks * items_per_block;

    const auto           random_range = limit_random_range<T>(0, 10);
    const std::vector<T> input
        = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(input.size());

    state.run(
        [&]
        {
            kernel<Benchmark, BlockSize, ItemsPerThread, WithTile>
                <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                   d_output.get(),
                                                                   Trials);
            HIP_CHECK(hipGetLastError());
        });

    state.set_throughput(size * Trials, sizeof(T));
}

template<typename Benchmark,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         WithTile,
         unsigned int Trials = 100>
auto run_benchmark(benchmark_utils::state&& state)
    -> std::enable_if_t<std::is_same<Benchmark, subtract_left_partial>::value
                        || std::is_same<Benchmark, subtract_right_partial>::value>
{
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;
    const auto& stream = state.stream;

    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);

    static constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto            num_blocks      = (N + items_per_block - 1) / items_per_block;
    // Round up size to the next multiple of items_per_block
    const auto size = num_blocks * items_per_block;

    const auto           random_range_input      = limit_random_range<T>(0, 10);
    const auto           random_range_tile_sizes = limit_random_range<T>(0, items_per_block);
    const std::vector<T> input                   = get_random_data<T>(size,
                                                    random_range_input.first,
                                                    random_range_input.second,
                                                    seed.get_0());
    const std::vector<unsigned int> tile_sizes
        = get_random_data<unsigned int>(num_blocks,
                                        random_range_tile_sizes.first,
                                        random_range_tile_sizes.second,
                                        seed.get_1());

    common::device_ptr<T>            d_input(input);
    common::device_ptr<unsigned int> d_tile_sizes(tile_sizes);
    common::device_ptr<T>            d_output(input.size());

    state.run(
        [&]
        {
            kernel<Benchmark, BlockSize, ItemsPerThread, WithTile>
                <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                   d_tile_sizes.get(),
                                                                   d_output.get(),
                                                                   Trials);
            HIP_CHECK(hipGetLastError());
        });

    state.set_throughput(size * Trials, sizeof(T));
}

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE)                                         \
    executor.queue_fn(                                                                  \
        bench_naming::format_name("{lvl:block,algo:adjacent_difference,subalgo:" + name \
                                  + ",key_type:" #T ",cfg:{bs:" #BS ",ipt:" #IPT        \
                                    ",with_tile:" #WITH_TILE "}}")                      \
            .c_str(),                                                                   \
        run_benchmark<Benchmark, T, BS, IPT, WITH_TILE>);

#define BENCHMARK_TYPE(type, block, with_tile)   \
    CREATE_BENCHMARK(type, block, 1, with_tile)  \
    CREATE_BENCHMARK(type, block, 3, with_tile)  \
    CREATE_BENCHMARK(type, block, 4, with_tile)  \
    CREATE_BENCHMARK(type, block, 8, with_tile)  \
    CREATE_BENCHMARK(type, block, 16, with_tile) \
    CREATE_BENCHMARK(type, block, 32, with_tile)

template<typename Benchmark>
void add_benchmarks(const std::string& name, benchmark_utils::executor& executor)
{
    BENCHMARK_TYPE(int, 256, false)
    BENCHMARK_TYPE(float, 256, false)
    BENCHMARK_TYPE(int8_t, 256, false)
    BENCHMARK_TYPE(rocprim::half, 256, false)
    BENCHMARK_TYPE(long long, 256, false)
    BENCHMARK_TYPE(double, 256, false)
    BENCHMARK_TYPE(rocprim::int128_t, 256, false)
    BENCHMARK_TYPE(rocprim::uint128_t, 256, false)

    if(!std::is_same<Benchmark, subtract_right_partial>::value)
    {
        BENCHMARK_TYPE(int, 256, true)
        BENCHMARK_TYPE(float, 256, true)
        BENCHMARK_TYPE(int8_t, 256, true)
        BENCHMARK_TYPE(rocprim::half, 256, true)
        BENCHMARK_TYPE(long long, 256, true)
        BENCHMARK_TYPE(double, 256, true)
        BENCHMARK_TYPE(rocprim::int128_t, 256, true)
        BENCHMARK_TYPE(rocprim::uint128_t, 256, true)
    }
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 512 * benchmark_utils::MiB, 1, 0);

    add_benchmarks<subtract_left>("subtract_left", executor);
    add_benchmarks<subtract_right>("subtract_right", executor);
    add_benchmarks<subtract_left_partial>("subtract_left_partial", executor);
    add_benchmarks<subtract_right_partial>("subtract_right_partial", executor);

    executor.run();
}
