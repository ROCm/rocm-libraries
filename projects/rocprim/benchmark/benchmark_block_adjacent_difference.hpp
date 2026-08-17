// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

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

template<typename Subalgo,
         typename T,
         unsigned int                                 BlockSize,
         unsigned int                                 ItemsPerThread,
         bool                                         WithTile,
         rocprim::block_adjacent_difference_algorithm Algorithm,
         typename... Args>
__global__ __launch_bounds__(BlockSize)
void kernel(Args... args)
{
    Subalgo::template run<T, BlockSize, ItemsPerThread, WithTile, Algorithm>(args...);
}

struct subtract_left
{
    template<typename T,
             unsigned int                                 BlockSize,
             unsigned int                                 ItemsPerThread,
             bool                                         WithTile,
             rocprim::block_adjacent_difference_algorithm Algorithm>
    __device__
    static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize, 1, 1, Algorithm>;
        __shared__
        typename adjacent_diff_t::storage_type storage;

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
    template<typename T,
             unsigned int                                 BlockSize,
             unsigned int                                 ItemsPerThread,
             bool                                         WithTile,
             rocprim::block_adjacent_difference_algorithm Algorithm>
    __device__
    static void
        run(const T* d_input, const unsigned int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize, 1, 1, Algorithm>;
        __shared__
        typename adjacent_diff_t::storage_type storage;

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
    template<typename T,
             unsigned int                                 BlockSize,
             unsigned int                                 ItemsPerThread,
             bool                                         WithTile,
             rocprim::block_adjacent_difference_algorithm Algorithm>
    __device__
    static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize, 1, 1, Algorithm>;
        __shared__
        typename adjacent_diff_t::storage_type storage;

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
    template<typename T,
             unsigned int                                 BlockSize,
             unsigned int                                 ItemsPerThread,
             bool                                         WithTile,
             rocprim::block_adjacent_difference_algorithm Algorithm>
    __device__
    static void
        run(const T* d_input, const unsigned int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        using adjacent_diff_t = rocprim::block_adjacent_difference<T, BlockSize, 1, 1, Algorithm>;
        __shared__
        typename adjacent_diff_t::storage_type storage;

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

template<rocprim::block_adjacent_difference_algorithm Algorithm>
std::string get_algorithm_name()
{
    switch(Algorithm)
    {
        case rocprim::block_adjacent_difference_algorithm::adjacent_difference_crosslane:
            return "crosslane";
        case rocprim::block_adjacent_difference_algorithm::adjacent_difference_shared_mem:
            return "shared_mem";
            // Not using `default: ...` because it kills effectiveness of -Wswitch
    }
    return "unknown_algorithm";
}

template<typename Subalgo>
std::string get_subalgo_name()
{
    if constexpr(std::is_same_v<Subalgo, subtract_left>)
        return "subtract_left";
    else if constexpr(std::is_same_v<Subalgo, subtract_right>)
        return "subtract_right";
    else if constexpr(std::is_same_v<Subalgo, subtract_left_partial>)
        return "subtract_left_partial";
    else if constexpr(std::is_same_v<Subalgo, subtract_right_partial>)
        return "subtract_right_partial";
    else
        static_assert(sizeof(Subalgo) == 0, "Unknown subalgo");
}

template<typename Subalgo,
         typename T,
         unsigned int                                 BlockSize,
         unsigned int                                 ItemsPerThread,
         bool                                         WithTile,
         rocprim::block_adjacent_difference_algorithm Algorithm,
         unsigned int                                 Trials = 100,
         typename Config                                     = rocprim::default_config>
struct block_adjacent_difference_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_adjacent_difference")
            .add("subalgo", get_subalgo_name<Subalgo>())
            .add("key_type", primbench::name<T>())
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("ipt", ItemsPerThread)
                     .add("with_tile", WithTile)
                     .add("method", get_algorithm_name<Algorithm>()));
    }

    void run(primbench::state& state) override
    {
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;
        const auto& stream = state.stream;

        auto seeds = primbench::seeds<2>(seed);

        size_t         N               = bytes / sizeof(T);
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     num_blocks      = (N + items_per_block - 1) / items_per_block;
        const auto     items           = num_blocks * items_per_block;

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        if constexpr(std::is_same_v<Subalgo, subtract_left_partial>
                     || std::is_same_v<Subalgo, subtract_right_partial>)
        {
            // Partial variant (with tile_sizes)
            const auto random_range_input         = limit_random_range<T>(0, 10);
            const auto random_range_tile_sizes    = limit_random_range<T>(0, items_per_block);
            const std::vector<T>            input = get_random_data<T>(items,
                                                            random_range_input.first,
                                                            random_range_input.second,
                                                            seeds[0]);
            const std::vector<unsigned int> tile_sizes
                = get_random_data<unsigned int>(num_blocks,
                                                random_range_tile_sizes.first,
                                                random_range_tile_sizes.second,
                                                seeds[1]);

            common::device_ptr<T>            d_input(input);
            common::device_ptr<unsigned int> d_tile_sizes(tile_sizes);
            common::device_ptr<T>            d_output(input.size());

            state.run(
                [&]
                {
                    kernel<Subalgo, T, BlockSize, ItemsPerThread, WithTile, Algorithm>
                        <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                           d_tile_sizes.get(),
                                                                           d_output.get(),
                                                                           Trials);
                });
        }
        else
        {
            // Full variant (without tile_sizes)
            const auto           random_range = limit_random_range<T>(0, 10);
            const std::vector<T> input
                = get_random_data<T>(items, random_range.first, random_range.second, seeds[0]);

            common::device_ptr<T> d_input(input);
            common::device_ptr<T> d_output(input.size());

            state.run(
                [&]
                {
                    kernel<Subalgo, T, BlockSize, ItemsPerThread, WithTile, Algorithm>
                        <<<dim3(num_blocks), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                           d_output.get(),
                                                                           Trials);
                });
        }
    }
};
