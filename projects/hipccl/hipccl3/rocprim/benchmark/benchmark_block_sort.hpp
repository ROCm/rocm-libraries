// MIT License
//
// Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprim/block/block_sort.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>
#include <rocprim/types/tuple.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

template<typename KeyType,
         typename ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm,
         std::enable_if_t<std::is_same<ValueType, rocprim::empty_type>::value, bool> = true>
__global__ __launch_bounds__(BlockSize)
void sort_kernel(const KeyType* input, KeyType* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    KeyType keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    rocprim::block_sort<KeyType, BlockSize, ItemsPerThread, ValueType, block_sort_algorithm> bsort;
    bsort.sort(keys);

    rocprim::block_store_direct_blocked(lid, output + block_offset, keys);
}

template<typename KeyType,
         typename ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm,
         std::enable_if_t<!std::is_same<ValueType, rocprim::empty_type>::value, bool> = true>
__global__ __launch_bounds__(BlockSize)
void sort_kernel(const KeyType* input, KeyType* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    KeyType   keys[ItemsPerThread];
    ValueType values[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        values[item] = block_offset + lid * ItemsPerThread + item;
    }

    rocprim::block_sort<KeyType, BlockSize, ItemsPerThread, ValueType, block_sort_algorithm> bsort;
    bsort.sort(keys, values);

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        keys[item] = keys[item] + static_cast<KeyType>(values[item]);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, keys);
}

/// Turns a type T into a stable sortable type, given a certain order.
template<class T>
struct stablify
{
    using type = ::rocprim::tuple<T, int>;

    __host__ __device__
    static inline type make(T data, int ordering)
    {
        return ::rocprim::make_tuple(data, ordering);
    }

    __host__ __device__
    static inline T unmake(type a)
    {
        using rocprim::get;
        return get<0>(a);
    }

    __host__ __device__
    static inline bool compare(type a, type b)
    {
        using rocprim::get;
        return (get<0>(a) < get<0>(b)) || ((get<0>(a) == get<0>(b)) && (get<1>(a) < get<1>(b)));
    }
};

template<typename KeyType,
         typename ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm>
__global__ __launch_bounds__(BlockSize)
void stable_sort_kernel(const KeyType* input, KeyType* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    KeyType keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    using stable_key      = stablify<KeyType>;
    using stable_key_type = typename stable_key::type;
    stable_key_type stable_keys[ItemsPerThread];

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        stable_keys[item] = stable_key::make(keys[item], (ItemsPerThread * lid) + item);
    }

    rocprim::block_sort<stable_key_type,
                        BlockSize,
                        ItemsPerThread,
                        rocprim::empty_type,
                        block_sort_algorithm>
        bsort;
    bsort.sort(stable_keys, stable_key::compare);

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        keys[item] = stable_key::unmake(stable_keys[item]);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, keys);
}

template<typename KeyType,
         typename ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm,
         const bool                    stable = false>
struct block_sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_sort")
            .add("key_type", primbench::name<KeyType>())
            .add("value_type", primbench::name<ValueType>())
            .add("stable", stable)
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("ipt", ItemsPerThread)
                     .add("method", get_block_sort_method_name(block_sort_algorithm)));
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        size_t     N     = bytes / sizeof(KeyType);
        const auto items = items_per_block * ((N + items_per_block - 1) / items_per_block);

        std::vector<KeyType> input
            = get_random_data<KeyType>(items,
                                       common::generate_limits<KeyType>::min(),
                                       common::generate_limits<KeyType>::max(),
                                       seed);

        common::device_ptr<KeyType> d_input(input);
        common::device_ptr<KeyType> d_output(items);

        static constexpr auto stable_tag = rocprim::detail::bool_constant<stable>{};

        state.set_items(items);
        state.add_reads<KeyType>(items);

        state.run(
            [&] { dispatch_block_sort(stable_tag, items, stream, d_input.get(), d_output.get()); });

        // state.gbench_state.counters["sorted_size"]
        //     = benchmark::Counter(BlockSize * ItemsPerThread,
        //                          benchmark::Counter::kDefaults,
        //                          benchmark::Counter::OneK::kIs1024);
    }

private:
    static constexpr bool with_values = !std::is_same<ValueType, rocprim::empty_type>::value;
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    static std::string get_block_sort_method_name(rocprim::block_sort_algorithm alg)
    {
        switch(alg)
        {
            case rocprim::block_sort_algorithm::merge_sort: return "merge_sort";
            case rocprim::block_sort_algorithm::stable_merge_sort: return "stable_merge_sort";
            case rocprim::block_sort_algorithm::bitonic_sort:
                return "bitonic_sort";
                // Not using `default: ...` because it kills effectiveness of -Wswitch
        }
        return "unknown_algorithm";
    }

    static auto dispatch_block_sort(std::false_type /*stable_sort*/,
                                    size_t            items,
                                    const hipStream_t stream,
                                    KeyType*          d_input,
                                    KeyType*          d_output)
    {
        sort_kernel<KeyType, ValueType, BlockSize, ItemsPerThread, block_sort_algorithm>
            <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input, d_output);
    }

    static auto dispatch_block_sort(std::true_type /*stable_sort*/,
                                    size_t            items,
                                    const hipStream_t stream,
                                    KeyType*          d_input,
                                    KeyType*          d_output)
    {
        stable_sort_kernel<KeyType, ValueType, BlockSize, ItemsPerThread, block_sort_algorithm>
            <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input, d_output);
    }
};
