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

#include <rocprim/block/block_histogram.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

template<typename Method,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int BinSize,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Method::template run<T, BlockSize, ItemsPerThread, BinSize, Trials>(input, output);
}

template<rocprim::block_histogram_algorithm algorithm>
struct histogram
{
    static constexpr auto algorithm_type = algorithm;
    template<typename T,
             unsigned int BlockSize,
             unsigned int ItemsPerThread,
             unsigned int BinSize,
             unsigned int Trials>
    __device__
    static void run(const T* input, T* output)
    {
        // TODO: Move global_offset into final loop
        const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
        unsigned int       global_offset = blockIdx.x * BinSize;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; ++k)
        {
            values[k] = input[index + k];
        }

        using bhistogram_t
            = rocprim::block_histogram<T, BlockSize, ItemsPerThread, BinSize, algorithm>;
        __shared__
        T                                   histogram[BinSize];
        __shared__
        typename bhistogram_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            bhistogram_t().histogram(values, histogram, storage);
            for(unsigned int k = 0; k < ItemsPerThread; ++k)
            {
                values[k] = BinSize - 1 - values[k];
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int offset = 0; offset < BinSize; offset += BlockSize)
        {
            if(offset + threadIdx.x < BinSize)
            {
                output[global_offset + threadIdx.x] = histogram[offset + threadIdx.x];
                global_offset += BlockSize;
            }
        }
    }
};

template<typename Method>
std::string get_method_name()
{
    using histogram_atomic_t = histogram<rocprim::block_histogram_algorithm::using_atomic>;
    using histogram_sort_t   = histogram<rocprim::block_histogram_algorithm::using_sort>;

    if constexpr(std::is_same_v<Method, histogram_atomic_t>)
        return "using_atomic";
    else if constexpr(std::is_same_v<Method, histogram_sort_t>)
        return "using_sort";
    else
        static_assert(sizeof(Method) == 0, "Unknown method");
}

template<typename Method,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int BinSize = BlockSize,
         unsigned int Trials  = 100,
         typename Config      = rocprim::default_config>
struct block_histogram_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_histogram")
            .add("key_type", primbench::name<T>())
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("ipt", ItemsPerThread)
                     .add("method", get_method_name<Method>()));
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;

        // Ensure items is a multiple of BlockSize
        size_t         N               = bytes / sizeof(T);
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items    = items_per_block * ((N + items_per_block - 1) / items_per_block);
        const auto     bin_size = BinSize * ((N + items_per_block - 1) / items_per_block);

        // Allocate and fill memory
        std::vector<T>        input(items, 0.0f);
        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(bin_size);

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                kernel<Method, T, BlockSize, ItemsPerThread, BinSize, Trials>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                    d_output.get());
            });
    }
};
