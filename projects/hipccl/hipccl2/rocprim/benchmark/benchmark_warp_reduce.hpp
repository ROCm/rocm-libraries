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

#include <rocprim/config.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_reduce.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

template<bool AllReduce, typename T, unsigned int VirtualWaveSize, unsigned int Trials>
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void warp_reduce_kernel(const T* d_input, T* d_output)
{
    if constexpr(VirtualWaveSize <= rocprim::arch::wavefront::max_size())
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        auto value = d_input[i];

        using wreduce_t = rocprim::warp_reduce<T, VirtualWaveSize, AllReduce>;
        __shared__
        typename wreduce_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wreduce_t().reduce(value, value, storage);
        }

        d_output[i] = value;
    }
}

template<typename T, typename Flag, unsigned int VirtualWaveSize, unsigned int Trials>
__global__ __launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void segmented_warp_reduce_kernel(const T* d_input, Flag* d_flags, T* d_output)
{
    if constexpr(VirtualWaveSize <= rocprim::arch::wavefront::max_size())
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        auto value = d_input[i];
        auto flag  = d_flags[i];

        using wreduce_t = rocprim::warp_reduce<T, VirtualWaveSize>;
        __shared__
        typename wreduce_t::storage_type storage;
        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            wreduce_t().head_segmented_reduce(value, value, flag, storage);
        }

        d_output[i] = value;
    }
}

template<bool AllReduce,
         bool Segmented,
         typename T,
         unsigned int VirtualWaveSize,
         unsigned int BlockSize,
         unsigned int Trials = 100,
         typename Config     = rocprim::default_config>
struct warp_reduce_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "warp")
            .add("algo", "warp_reduce")
            .add("key_type", primbench::name<T>())
            .add("broadcast_result", AllReduce)
            .add("segmented", Segmented)
            .add("ws", VirtualWaveSize)
            .add("cfg", primbench::json{}.add("bs", BlockSize));
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        auto seeds = primbench::seeds<2>(seed);

        using flag_type = unsigned char;

        size_t     N     = bytes / sizeof(T);
        const auto items = BlockSize * ((N + BlockSize - 1) / BlockSize);

        const auto     random_range = limit_random_range<T>(0, 10);
        std::vector<T> input
            = get_random_data<T>(items, random_range.first, random_range.second, seeds[0]);
        std::vector<flag_type>        flags = get_random_data<flag_type>(items, 0, 1, seeds[1]);
        common::device_ptr<T>         d_input(input);
        common::device_ptr<flag_type> d_flags(flags);
        common::device_ptr<T>         d_output(items);

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                if constexpr(Segmented)
                {

                    segmented_warp_reduce_kernel<T, flag_type, VirtualWaveSize, Trials>
                        <<<dim3(items / BlockSize), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                  d_flags.get(),
                                                                                  d_output.get());
                }
                else
                {

                    warp_reduce_kernel<AllReduce, T, VirtualWaveSize, Trials>
                        <<<dim3(items / BlockSize), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                  d_output.get());
                }
            });
    }
};
