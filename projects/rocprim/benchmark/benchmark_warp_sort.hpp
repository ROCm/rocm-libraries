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

#include <hip/hip_runtime.h>

#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_sort.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

template<typename K, unsigned int BlockSize, unsigned int WarpSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void warp_sort_kernel(K* input_keys, K* output_keys)
{
    const unsigned int flat_tid        = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset    = blockIdx.x * items_per_block;

    K keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(flat_tid, input_keys + block_offset, keys);

    rocprim::warp_sort<K, WarpSize> wsort;
    wsort.sort(keys);

    rocprim::block_store_direct_blocked(flat_tid, output_keys + block_offset, keys);
}

template<typename K,
         typename V,
         unsigned int BlockSize,
         unsigned int WarpSize,
         unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void warp_sort_by_key_kernel(K* input_keys, V* input_values, K* output_keys, V* output_values)
{
    const unsigned int flat_tid        = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset    = blockIdx.x * items_per_block;

    K keys[ItemsPerThread];
    V values[ItemsPerThread];

    rocprim::block_load_direct_striped<BlockSize>(flat_tid, input_keys + block_offset, keys);
    rocprim::block_load_direct_striped<BlockSize>(flat_tid, input_values + block_offset, values);

    rocprim::warp_sort<K, WarpSize, V> wsort;
    wsort.sort(keys, values);

    rocprim::block_store_direct_blocked(flat_tid, output_keys + block_offset, keys);
    rocprim::block_store_direct_blocked(flat_tid, output_values + block_offset, values);
}

template<typename Key,
         unsigned int BlockSize,
         unsigned int WarpSize,
         unsigned int ItemsPerThread = 1,
         typename Value              = rocprim::empty_type,
         unsigned int Trials         = 100,
         typename Config             = rocprim::default_config>
struct warp_sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        auto j = primbench::json{}
                     .add("lvl", "warp")
                     .add("algo", "warp_sort")
                     .add("key_type", primbench::name<Key>())
                     .add("ws", WarpSize)
                     .add("cfg", primbench::json{}.add("bs", BlockSize).add("ipt", ItemsPerThread));

        if constexpr(SortByKey)
        {
            j.add("value_type", primbench::name<Value>());
        }

        return j;
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        auto seeds = primbench::seeds<2>(seed);

        size_t items = bytes / sizeof(Key);

        constexpr auto items_per_block = BlockSize * ItemsPerThread;

        items = BlockSize * ((items + items_per_block - 1) / items_per_block);

        // Generate input data
        const auto       key_range = limit_random_range<Key>(0, 10'000);
        std::vector<Key> input_key
            = get_random_data<Key>(items, key_range.first, key_range.second, seeds[0]);

        // Device memory
        Key*   d_input_key    = nullptr;
        Key*   d_output_key   = nullptr;
        Value* d_input_value  = nullptr;
        Value* d_output_value = nullptr;

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_key), items * sizeof(Key)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_key), items * sizeof(Key)));
        HIP_CHECK(
            hipMemcpy(d_input_key, input_key.data(), items * sizeof(Key), hipMemcpyHostToDevice));

        state.set_items(items);
        state.add_reads<Key>(items * Trials);
        if constexpr(SortByKey)
        {
            state.add_reads<Value>(items * Trials);
        }

        if constexpr(SortByKey)
        {
            run_sort_by_key(state,
                            stream,
                            items,
                            seeds,
                            d_input_key,
                            d_output_key,
                            d_input_value,
                            d_output_value);
        }
        else
        {
            run_sort_no_value(state, stream, items, d_input_key, d_output_key);
        }

        HIP_CHECK(hipFree(d_input_key));
        HIP_CHECK(hipFree(d_output_key));
        HIP_CHECK(hipFree(d_input_value));
        HIP_CHECK(hipFree(d_output_value));
    }

private:
    static constexpr bool SortByKey = !std::is_same_v<Value, rocprim::empty_type>;

    static void run_sort_by_key(primbench::state&          state,
                                hipStream_t                stream,
                                size_t                     items,
                                const primbench::seeds<2>& seeds,
                                Key*                       d_input_key,
                                Key*                       d_output_key,
                                Value*&                    d_input_value,
                                Value*&                    d_output_value)
    {
        constexpr auto items_per_block = BlockSize * ItemsPerThread;

        // Generate and copy values
        const auto         val_range = limit_random_range<Value>(0, 10'000);
        std::vector<Value> input_value
            = get_random_data<Value>(items, val_range.first, val_range.second, seeds[1]);

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_value), items * sizeof(Value)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_value), items * sizeof(Value)));
        HIP_CHECK(hipMemcpy(d_input_value,
                            input_value.data(),
                            items * sizeof(Value),
                            hipMemcpyHostToDevice));

        state.run(
            [&]
            {
                ROCPRIM_NO_UNROLL
                for(unsigned int trial = 0; trial < Trials; ++trial)
                {
                    warp_sort_by_key_kernel<Key, Value, BlockSize, WarpSize, ItemsPerThread>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(
                            d_input_key,
                            d_input_value,
                            d_output_key,
                            d_output_value);
                }
            });
    }

    static void run_sort_no_value(primbench::state& state,
                                  hipStream_t       stream,
                                  size_t            items,
                                  Key*              d_input_key,
                                  Key*              d_output_key)
    {
        constexpr auto items_per_block = BlockSize * ItemsPerThread;

        state.run(
            [&]
            {
                ROCPRIM_NO_UNROLL
                for(unsigned int trial = 0; trial < Trials; ++trial)
                {
                    warp_sort_kernel<Key, BlockSize, WarpSize, ItemsPerThread>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(
                            d_input_key,
                            d_output_key);
                }
            });
    }
};
