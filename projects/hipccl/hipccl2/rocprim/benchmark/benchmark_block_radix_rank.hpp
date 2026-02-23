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

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_radix_rank.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

#include <chrono>
#include <stdint.h>
#include <string>
#include <vector>

template<typename T,
         unsigned int                        BlockSize,
         unsigned int                        ItemsPerThread,
         unsigned int                        RadixBits,
         bool                                Descending,
         rocprim::block_radix_rank_algorithm Algorithm,
         unsigned int                        Trials>
__global__ __launch_bounds__(BlockSize)
void rank_kernel(const T* keys_input, unsigned int* ranks_output)
{
    using rank_type = rocprim::block_radix_rank<BlockSize, RadixBits, Algorithm>;

    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, keys_input + block_offset, keys);

    unsigned int ranks[ItemsPerThread];

    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; ++trial)
    {
        ROCPRIM_SHARED_MEMORY
        typename rank_type::storage_type storage;
        unsigned int                     begin_bit = 0;
        const unsigned int               end_bit   = sizeof(T) * 8;

        while(begin_bit < end_bit)
        {
            const unsigned pass_bits = min(RadixBits, end_bit - begin_bit);
            if constexpr(Descending)
            {
                rank_type().rank_keys_desc(keys, ranks, storage, begin_bit, pass_bits);
            }
            else
            {
                rank_type().rank_keys(keys, ranks, storage, begin_bit, pass_bits);
            }
            begin_bit += RadixBits;
        }
    }

    rocprim::block_store_direct_striped<BlockSize>(lid, ranks_output + block_offset, ranks);
}

template<rocprim::block_radix_rank_algorithm Algorithm>
std::string get_algorithm_name()
{
    switch(Algorithm)
    {
        case rocprim::block_radix_rank_algorithm::basic: return "basic";
        case rocprim::block_radix_rank_algorithm::basic_memoize: return "basic_memoize";
        case rocprim::block_radix_rank_algorithm::match:
            return "match";
            // Not using `default: ...` because it kills effectiveness of -Wswitch
    }
    return "unknown_algorithm";
}

template<typename T,
         size_t                              BlockSize,
         size_t                              ItemsPerThread,
         rocprim::block_radix_rank_algorithm Algorithm,
         size_t                              RadixBits  = 4,
         bool                                Descending = false,
         size_t                              Trials     = 10,
         typename Config                                = rocprim::default_config>
struct block_radix_rank_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_radix_rank")
            .add("key_type", primbench::name<T>())
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("ipt", ItemsPerThread)
                     .add("method", get_algorithm_name<Algorithm>()));
    }

    void run(primbench::state& state) override
    {
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;
        const auto& stream = state.stream;

        size_t           N               = bytes / sizeof(T);
        constexpr size_t items_per_block = BlockSize * ItemsPerThread;
        const size_t     grid_size       = ((N + items_per_block - 1) / items_per_block);
        const size_t     items           = items_per_block * grid_size;

        std::vector<T> input = get_random_data<T>(items,
                                                  common::generate_limits<T>::min(),
                                                  common::generate_limits<T>::max(),
                                                  seed);

        common::device_ptr<T>            d_input(input);
        common::device_ptr<unsigned int> d_output(items);

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                rank_kernel<T, BlockSize, ItemsPerThread, RadixBits, Descending, Algorithm, Trials>
                    <<<dim3(grid_size), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                      d_output.get());
            });
    }
};
