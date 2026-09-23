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

#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_run_length_decode.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <chrono>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

template<typename ItemT,
         typename OffsetT,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread,
         unsigned Trials>
__global__ __launch_bounds__(BlockSize)
void block_run_length_decode_kernel(const ItemT*   d_run_items,
                                    const OffsetT* d_run_offsets,
                                    ItemT*         d_decoded_items,
                                    bool           enable_store = false)
{
    using BlockRunLengthDecodeT
        = rocprim::block_run_length_decode<ItemT, BlockSize, RunsPerThread, DecodedItemsPerThread>;

    ItemT   run_items[RunsPerThread];
    OffsetT run_offsets[RunsPerThread];

    const unsigned global_thread_idx = BlockSize * hipBlockIdx_x + hipThreadIdx_x;
    rocprim::block_load_direct_blocked(global_thread_idx, d_run_items, run_items);
    rocprim::block_load_direct_blocked(global_thread_idx, d_run_offsets, run_offsets);

    BlockRunLengthDecodeT block_run_length_decode(run_items, run_offsets);

    const OffsetT total_decoded_size
        = d_run_offsets[(hipBlockIdx_x + 1) * BlockSize * RunsPerThread]
          - d_run_offsets[hipBlockIdx_x * BlockSize * RunsPerThread];

#pragma nounroll
    for(unsigned i = 0; i < Trials; ++i)
    {
        OffsetT decoded_window_offset = 0;
        while(decoded_window_offset < total_decoded_size)
        {
            ItemT decoded_items[DecodedItemsPerThread];
            block_run_length_decode.run_length_decode(decoded_items, decoded_window_offset);

            if(enable_store)
            {
                rocprim::block_store_direct_blocked(global_thread_idx,
                                                    d_decoded_items + decoded_window_offset,
                                                    decoded_items);
            }

            decoded_window_offset += BlockSize * DecodedItemsPerThread;
        }
    }
}

template<typename ItemT,
         typename OffsetT,
         unsigned MinRunLength,
         unsigned MaxRunLength,
         unsigned BlockSize,
         unsigned RunsPerThread,
         unsigned DecodedItemsPerThread,
         unsigned Trials = 100,
         typename Config = rocprim::default_config>
struct block_run_length_decode_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_run_length_decode")
            .add("item_type", primbench::name<ItemT>())
            .add("offset_type", primbench::name<OffsetT>())
            .add("min_run_length", MinRunLength)
            .add("max_run_length", MaxRunLength)
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("run_per_thread", RunsPerThread)
                     .add("decoded_items_per_thread", DecodedItemsPerThread));
    }

    void run(primbench::state& state) override
    {
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;
        const auto& stream = state.stream;

        size_t         N               = bytes / sizeof(ItemT);
        constexpr auto runs_per_block  = BlockSize * RunsPerThread;
        const auto     target_num_runs = 2 * N / (MinRunLength + MaxRunLength);
        const auto     num_runs
            = runs_per_block * ((target_num_runs + runs_per_block - 1) / runs_per_block);

        std::vector<ItemT>   run_items(num_runs);
        std::vector<OffsetT> run_offsets(num_runs + 1);

        engine_type prng(seed);
        using ItemDistribution = std::conditional_t<rocprim::is_integral<ItemT>::value,
                                                    common::uniform_int_distribution<ItemT>,
                                                    std::uniform_real_distribution<ItemT>>;
        ItemDistribution                          run_item_dist(0, 100);
        common::uniform_int_distribution<OffsetT> run_length_dist(MinRunLength, MaxRunLength);

        for(size_t i = 0; i < num_runs; ++i)
        {
            run_items[i] = run_item_dist(prng);
        }
        for(size_t i = 1; i < num_runs + 1; ++i)
        {
            const OffsetT next_run_length = run_length_dist(prng);
            run_offsets[i]                = run_offsets[i - 1] + next_run_length;
        }
        const OffsetT output_items = run_offsets.back();

        common::device_ptr<ItemT> d_run_items(run_items);

        common::device_ptr<OffsetT> d_run_offsets(run_offsets);

        common::device_ptr<ItemT> d_output(output_items);

        state.set_items(output_items * Trials);
        state.add_writes<ItemT>(output_items * Trials);

        state.run(
            [&]
            {
                block_run_length_decode_kernel<ItemT,
                                               OffsetT,
                                               BlockSize,
                                               RunsPerThread,
                                               DecodedItemsPerThread,
                                               Trials>
                    <<<dim3(num_runs / runs_per_block), dim3(BlockSize), 0, stream>>>(
                        d_run_items.get(),
                        d_run_offsets.get(),
                        d_output.get());
            });
    }
};
