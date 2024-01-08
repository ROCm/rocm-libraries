// MIT License
//
// Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_BLOCK_SORT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_BLOCK_SORT_PARALLEL_HPP_

#include <cstddef>
#include <string>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/detail/various.hpp>

#include "benchmark_utils.hpp"

template<class KeyType,
         class ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm,
         std::enable_if_t<std::is_same<ValueType, rocprim::empty_type>::value, bool> = true>
__global__ __launch_bounds__(BlockSize) void sort_kernel(const KeyType* input, KeyType* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    KeyType keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    rocprim::block_sort<KeyType, BlockSize, ItemsPerThread, ValueType, block_sort_algorithm> bsort;
    bsort.sort(keys);

    rocprim::block_store_direct_blocked(lid, output + block_offset, keys);
}

template<class KeyType,
         class ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm,
         std::enable_if_t<!std::is_same<ValueType, rocprim::empty_type>::value, bool> = true>
__global__ __launch_bounds__(BlockSize) void sort_kernel(const KeyType* input, KeyType* output)
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

template<class KeyType,
         class ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm>
__global__ __launch_bounds__(BlockSize) void stable_sort_kernel(const KeyType* input,
                                                                KeyType*       output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    KeyType keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    using stable_key_type = rocprim::tuple<KeyType, unsigned int>;
    stable_key_type stable_keys[ItemsPerThread];

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        stable_keys[item] = rocprim::make_tuple(keys[item], ItemsPerThread * lid + item);
    }

    // Special comparison that preserves relative order of equal keys
    auto stable_compare_function
        = [](const stable_key_type& a, const stable_key_type& b) mutable -> bool
    {
        const bool ab = rocprim::less<KeyType>{}(rocprim::get<0>(a), rocprim::get<0>(b));
        return ab
               || (!rocprim::less<KeyType>{}(rocprim::get<0>(b), rocprim::get<0>(a))
                   && (rocprim::get<1>(a) < rocprim::get<1>(b)));
    };

    rocprim::block_sort<stable_key_type,
                        BlockSize,
                        ItemsPerThread,
                        rocprim::empty_type,
                        block_sort_algorithm>
        bsort;
    bsort.sort(stable_keys, stable_compare_function);

    ROCPRIM_UNROLL
    for(unsigned int item = 0; item < ItemsPerThread; ++item)
    {
        keys[item] = rocprim::get<0>(stable_keys[item]);
    }

    rocprim::block_store_direct_blocked(lid, output + block_offset, keys);
}

template<class KeyType,
         class ValueType,
         unsigned int                  BlockSize,
         unsigned int                  ItemsPerThread,
         rocprim::block_sort_algorithm block_sort_algorithm,
         const bool                    stable = false>
struct block_sort_benchmark : public config_autotune_interface
{
private:
    static constexpr bool with_values = !std::is_same<ValueType, rocprim::empty_type>::value;
    static constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    static const char* get_block_sort_method_name(rocprim::block_sort_algorithm alg)
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

public:
    std::string sort_key() const override
    {
        using namespace std::string_literals;
        return std::string((with_values ? "_pairs"s : "_keys"s) + (stable ? "_stable"s : ""s)
                           + pad_string(std::to_string(items_per_block), 5) + ", " + name());
    }

    std::string name() const override
    {
        return bench_naming::format_name(
            "{lvl:block,algo:sort,key_type:" + std::string(Traits<KeyType>::name()) + ",value_type:"
            + std::string(Traits<ValueType>::name()) + ",stable:" + (stable ? "true" : "false")
            + ",cfg:{bs:" + std::to_string(BlockSize) + ",ipt:" + std::to_string(ItemsPerThread)
            + ",method:" + std::string(get_block_sort_method_name(block_sort_algorithm)) + "}}");
    }

    static constexpr unsigned int batch_size        = 10;
    static constexpr unsigned int warmup_size       = 5;
    static constexpr bool         debug_synchronous = false;

    static auto dispatch_block_sort(std::false_type /*stable_sort*/,
                             size_t            size,
                             const hipStream_t stream,
                             KeyType*          d_input,
                             KeyType*          d_output)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                sort_kernel<KeyType, ValueType, BlockSize, ItemsPerThread, block_sort_algorithm>),
            dim3(size / items_per_block),
            dim3(BlockSize),
            0,
            stream,
            d_input,
            d_output);
    }

    static auto dispatch_block_sort(std::true_type /*stable_sort*/,
                             size_t            size,
                             const hipStream_t stream,
                             KeyType*          d_input,
                             KeyType*          d_output)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(stable_sort_kernel<KeyType,
                                                              ValueType,
                                                              BlockSize,
                                                              ItemsPerThread,
                                                              block_sort_algorithm>),
                           dim3(size / items_per_block),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_input,
                           d_output);
    }

    void run(benchmark::State& state, const std::size_t N, const hipStream_t stream) const override
    {
        const auto size = items_per_block * ((N + items_per_block - 1) / items_per_block);

        std::vector<KeyType> input;
        if(std::is_floating_point<KeyType>::value)
        {
            input = get_random_data<KeyType>(size, (KeyType)-1000, (KeyType) + 1000);
        }
        else
        {
            input = get_random_data<KeyType>(size,
                                             std::numeric_limits<KeyType>::min(),
                                             std::numeric_limits<KeyType>::max());
        }
        KeyType* d_input;
        KeyType* d_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(KeyType)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(KeyType)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(KeyType), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        static constexpr auto stable_tag = rocprim::detail::bool_constant<stable>{};

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Run
        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                dispatch_block_sort(stable_tag, size, stream, d_input, d_output);
            }
            HIP_CHECK(hipGetLastError());

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(KeyType));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        state.counters["sorted_size"] = benchmark::Counter(BlockSize * ItemsPerThread,
                                                           benchmark::Counter::kDefaults,
                                                           benchmark::Counter::OneK::kIs1024);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

#endif // ROCPRIM_BENCHMARK_BLOCK_SORT_PARALLEL_HPP_
