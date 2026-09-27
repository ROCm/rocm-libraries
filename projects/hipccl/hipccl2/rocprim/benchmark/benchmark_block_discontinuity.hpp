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

#include <hip/hip_runtime.h>

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

template<typename Subalgo,
         typename T,
         unsigned int                                 BlockSize,
         unsigned int                                 ItemsPerThread,
         bool                                         WithTile,
         rocprim::block_adjacent_difference_algorithm Algorithm,
         unsigned int                                 Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* d_input, T* d_output)
{
    Subalgo::template run<T, BlockSize, ItemsPerThread, WithTile, Algorithm, Trials>(d_input,
                                                                                     d_output);
}

struct flag_heads
{
    template<typename T,
             unsigned int                                 BlockSize,
             unsigned int                                 ItemsPerThread,
             bool                                         WithTile,
             rocprim::block_adjacent_difference_algorithm Algorithm,
             unsigned int                                 Trials>
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
            rocprim::block_discontinuity<T, BlockSize, 1, 1, Algorithm> bdiscontinuity;
            bool                                                        head_flags[ItemsPerThread];
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
             unsigned int                                 BlockSize,
             unsigned int                                 ItemsPerThread,
             bool                                         WithTile,
             rocprim::block_adjacent_difference_algorithm Algorithm,
             unsigned int                                 Trials>
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
            rocprim::block_discontinuity<T, BlockSize, 1, 1, Algorithm> bdiscontinuity;
            bool                                                        tail_flags[ItemsPerThread];
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
             unsigned int                                 BlockSize,
             unsigned int                                 ItemsPerThread,
             bool                                         WithTile,
             rocprim::block_adjacent_difference_algorithm Algorithm,
             unsigned int                                 Trials>
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
            rocprim::block_discontinuity<T, BlockSize, 1, 1, Algorithm> bdiscontinuity;
            bool                                                        head_flags[ItemsPerThread];
            bool                                                        tail_flags[ItemsPerThread];
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
    if constexpr(std::is_same_v<Subalgo, flag_heads>)
        return "flag_heads";
    else if constexpr(std::is_same_v<Subalgo, flag_tails>)
        return "flag_tails";
    else if constexpr(std::is_same_v<Subalgo, flag_heads_and_tails>)
        return "flag_heads_and_tails";
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
struct block_discontinuity_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_discontinuity")
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

        size_t         N               = bytes / sizeof(T);
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items = items_per_block * ((N + items_per_block - 1) / items_per_block);

        const auto     random_range = limit_random_range<T>(0, 10);
        std::vector<T> input
            = get_random_data<T>(items, random_range.first, random_range.second, seed);
        T* d_input;
        T* d_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), items * sizeof(T)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                kernel<Subalgo, T, BlockSize, ItemsPerThread, WithTile, Algorithm, Trials>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input,
                                                                                    d_output);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};
