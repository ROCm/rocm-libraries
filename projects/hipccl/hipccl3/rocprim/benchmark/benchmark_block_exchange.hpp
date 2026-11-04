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

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/block/block_exchange.hpp>
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdint.h>
#include <string>
#include <vector>

template<typename Subalgo,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* d_input, const unsigned int* d_ranks, T* d_output)
{
    Subalgo::template run<T, BlockSize, ItemsPerThread, Trials>(d_input, d_ranks, d_output);
}

struct blocked_to_striped
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_striped(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct striped_to_blocked
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.striped_to_blocked(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct blocked_to_warp_striped
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_warp_striped(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct warp_striped_to_blocked
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.warp_striped_to_blocked(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct scatter_to_blocked
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int* d_ranks, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T            input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);
        rocprim::block_load_direct_striped<BlockSize>(lid, d_ranks + block_offset, ranks);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_blocked(input, input, ranks);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct scatter_to_striped
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int* d_ranks, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T            input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);
        rocprim::block_load_direct_striped<BlockSize>(lid, d_ranks + block_offset, ranks);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_striped(input, input, ranks);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template<typename Subalgo>
std::string get_subalgo_name()
{
    if constexpr(std::is_same_v<Subalgo, blocked_to_striped>)
        return "blocked_to_striped";
    else if constexpr(std::is_same_v<Subalgo, striped_to_blocked>)
        return "striped_to_blocked";
    else if constexpr(std::is_same_v<Subalgo, blocked_to_warp_striped>)
        return "blocked_to_warp_striped";
    else if constexpr(std::is_same_v<Subalgo, warp_striped_to_blocked>)
        return "warp_striped_to_blocked";
    else if constexpr(std::is_same_v<Subalgo, scatter_to_blocked>)
        return "scatter_to_blocked";
    else if constexpr(std::is_same_v<Subalgo, scatter_to_striped>)
        return "scatter_to_striped";
    else
        static_assert(sizeof(Subalgo) == 0, "Unknown subalgo");
}

template<typename Subalgo,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials = 100,
         typename Config     = rocprim::default_config>
struct block_exchange_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_exchange")
            .add("subalgo", get_subalgo_name<Subalgo>())
            .add("key_type", primbench::name<T>())
            .add("cfg", primbench::json{}.add("bs", BlockSize).add("ipt", ItemsPerThread));
    }

    void run(primbench::state& state) override
    {
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;
        const auto& stream = state.stream;

        size_t         N               = bytes / sizeof(T);
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items = items_per_block * ((N + items_per_block - 1) / items_per_block);

        std::vector<T> input(items);

        // Fill input
        for(size_t i = 0; i < items; ++i)
        {
            input[i] = T(i);
        }
        std::vector<unsigned int> ranks(items);

        // Fill ranks (for scatter operations)
        engine_type gen(seed);
        for(size_t bi = 0; bi < items / items_per_block; ++bi)
        {
            auto block_ranks = ranks.begin() + bi * items_per_block;
            std::iota(block_ranks, block_ranks + items_per_block, 0);
            std::shuffle(block_ranks, block_ranks + items_per_block, gen);
        }
        common::device_ptr<T>            d_input(input);
        common::device_ptr<unsigned int> d_ranks(ranks);
        common::device_ptr<T>            d_output(items);

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                kernel<Subalgo, T, BlockSize, ItemsPerThread, Trials>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                    d_ranks.get(),
                                                                                    d_output.get());
            });
    }
};
