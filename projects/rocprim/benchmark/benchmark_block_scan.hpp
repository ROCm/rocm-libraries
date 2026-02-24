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

#include <rocprim/block/block_scan.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

template<typename Runner,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(input, output);
}

template<rocprim::block_scan_algorithm algorithm>
struct inclusive_scan
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; ++k)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using bscan_t = rocprim::block_scan<T, BlockSize, algorithm>;
        __shared__
        typename bscan_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            bscan_t().inclusive_scan(values, values, storage);
        }

        for(unsigned int k = 0; k < ItemsPerThread; ++k)
        {
            output[i * ItemsPerThread + k] = values[k];
        }
    }
};

template<rocprim::block_scan_algorithm algorithm>
struct exclusive_scan
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        using U              = typename std::remove_reference<T>::type;

        T values[ItemsPerThread];
        U init = U(100);

        for(unsigned int k = 0; k < ItemsPerThread; ++k)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using bscan_t = rocprim::block_scan<T, BlockSize, algorithm>;
        __shared__
        typename bscan_t::storage_type storage;

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            bscan_t().exclusive_scan(values, values, init, storage);
        }

        for(unsigned int k = 0; k < ItemsPerThread; ++k)
        {
            output[i * ItemsPerThread + k] = values[k];
        }
    }
};

using inclusive_scan_uws_t = inclusive_scan<rocprim::block_scan_algorithm::using_warp_scan>;
using exclusive_scan_uws_t = exclusive_scan<rocprim::block_scan_algorithm::using_warp_scan>;
using inclusive_scan_rts_t = inclusive_scan<rocprim::block_scan_algorithm::reduce_then_scan>;
using exclusive_scan_rts_t = exclusive_scan<rocprim::block_scan_algorithm::reduce_then_scan>;

template<typename Benchmark>
std::string get_subalgo_name()
{
    if constexpr(std::is_same_v<Benchmark, inclusive_scan_uws_t>)
        return "using_warp_scan";
    else if constexpr(std::is_same_v<Benchmark, exclusive_scan_uws_t>)
        return "using_warp_scan";
    else if constexpr(std::is_same_v<Benchmark, inclusive_scan_rts_t>)
        return "reduce_then_scan";
    else if constexpr(std::is_same_v<Benchmark, exclusive_scan_rts_t>)
        return "reduce_then_scan";
    else
        static_assert(sizeof(Benchmark) == 0, "Unknown subalgo");
}

template<typename Benchmark>
std::string get_method_name()
{
    if constexpr(std::is_same_v<Benchmark, inclusive_scan_uws_t>)
        return "inclusive_scan";
    else if constexpr(std::is_same_v<Benchmark, inclusive_scan_rts_t>)
        return "inclusive_scan";
    else if constexpr(std::is_same_v<Benchmark, exclusive_scan_uws_t>)
        return "exclusive_scan";
    else if constexpr(std::is_same_v<Benchmark, exclusive_scan_rts_t>)
        return "exclusive_scan";
    else
        static_assert(sizeof(Benchmark) == 0, "Unknown method");
}

template<typename Benchmark,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials = 100>
struct block_scan_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_scan")
            .add("subalgo", get_subalgo_name<Benchmark>())
            .add("key_type", primbench::name<T>())
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("ipt", ItemsPerThread)
                     .add("method", get_method_name<Benchmark>()));
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
                kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                    d_output.get());
            });
    }
};
