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

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/block/block_reduce.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

template<typename Method,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Method::template run<T, BlockSize, ItemsPerThread, Trials>(input, output);
}

template<rocprim::block_reduce_algorithm algorithm>
struct reduce
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        T values[ItemsPerThread];
        T reduced_value;
        for(unsigned int k = 0; k < ItemsPerThread; ++k)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using breduce_t = rocprim::block_reduce<T, BlockSize, algorithm>;
        __shared__
        typename breduce_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            breduce_t().reduce(values, reduced_value, storage);
            values[0] = reduced_value;
        }

        if(threadIdx.x == 0)
        {
            output[blockIdx.x] = reduced_value;
        }
    }
};

template<typename Method>
std::string get_method_name()
{
    using reduce_uwr_t  = reduce<rocprim::block_reduce_algorithm::using_warp_reduce>;
    using reduce_rr_t   = reduce<rocprim::block_reduce_algorithm::raking_reduce>;
    using reduce_rrco_t = reduce<rocprim::block_reduce_algorithm::raking_reduce_commutative_only>;

    if constexpr(std::is_same_v<Method, reduce_uwr_t>)
        return "using_warp_reduce";
    else if constexpr(std::is_same_v<Method, reduce_rr_t>)
        return "raking_reduce";
    else if constexpr(std::is_same_v<Method, reduce_rrco_t>)
        return "raking_reduce_commutative_only";
    else
        static_assert(sizeof(Method) == 0, "Unknown method");
}

template<typename Method,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials = 100,
         typename Config     = rocprim::default_config>
struct block_reduce_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_reduce")
            .add("key_type", primbench::name<T>())
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("ipt", ItemsPerThread)
                     .add("method", get_method_name<Method>()));
    }

    void run(primbench::state& state) override
    {
        const auto& bytes  = state.size;
        const auto& stream = state.stream;

        // Ensure items is a multiple of BlockSize
        size_t         N               = bytes / sizeof(T);
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items = items_per_block * ((N + items_per_block - 1) / items_per_block);

        // Allocate and fill memory
        std::vector<T>        input(items, T(1));
        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(items);

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                kernel<Method, T, BlockSize, ItemsPerThread, Trials>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                    d_output.get());
            });
    }
};
