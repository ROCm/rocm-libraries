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
#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_radix_sort.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

template<typename T>
using select_decomposer_t = std::conditional_t<common::is_custom_type<T>::value,
                                               custom_type_decomposer<T>,
                                               rocprim::identity_decomposer>;

template<typename T,
         unsigned int BlockSize,
         unsigned int RadixBitsPerPass,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void sort_keys_kernel(const T* input, T* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; ++trial)
    {
        rocprim::block_radix_sort<T,
                                  BlockSize,
                                  ItemsPerThread,
                                  rocprim::empty_type,
                                  1,
                                  1,
                                  RadixBitsPerPass>
            sort;
        sort.sort(keys, 0, sizeof(T) * 8, select_decomposer_t<T>{});
    }

    rocprim::block_store_direct_striped<BlockSize>(lid, output + block_offset, keys);
}

template<typename T,
         unsigned int BlockSize,
         unsigned int RadixBitsPerPass,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void sort_pairs_kernel(const T* input, T* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    T values[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        values[i] = keys[i] + T(1);
    }

    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; ++trial)
    {
        rocprim::block_radix_sort<T, BlockSize, ItemsPerThread, T, 1, 1, RadixBitsPerPass> sort;
        sort.sort(keys, values, 0, sizeof(T) * 8, select_decomposer_t<T>{});
    }

    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        keys[i] += values[i];
    }
    rocprim::block_store_direct_striped<BlockSize>(lid, output + block_offset, keys);
}

template<benchmark_kinds BenchmarkKind>
std::string get_benchmark_kind_name()
{
    switch(BenchmarkKind)
    {
        case benchmark_kinds::sort_keys: return "keys";
        case benchmark_kinds::sort_pairs:
            return "pairs";
            // Not using `default: ...` because it kills effectiveness of -Wswitch
    }
    return "unknown_benchmark_kind";
}

template<typename T,
         benchmark_kinds BenchmarkKind,
         unsigned int    BlockSize,
         unsigned int    RadixBitsPerPass,
         unsigned int    ItemsPerThread,
         unsigned int    Trials = 10,
         typename Config        = rocprim::default_config>
struct block_radix_sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "block")
            .add("algo", "block_radix_sort")
            .add("key_type", primbench::name<T>())
            .add("subalgo", get_benchmark_kind_name<BenchmarkKind>())
            .add("cfg",
                 primbench::json{}
                     .add("bs", BlockSize)
                     .add("rb", RadixBitsPerPass)
                     .add("ipt", ItemsPerThread));
    }

    void run(primbench::state& state) override
    {
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;
        const auto& stream = state.stream;

        size_t         N               = bytes / sizeof(T);
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items = items_per_block * ((N + items_per_block - 1) / items_per_block);

        std::vector<T> input = get_random_data<T>(items,
                                                  common::generate_limits<T>::min(),
                                                  common::generate_limits<T>::max(),
                                                  seed);

        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(items);

        state.set_items(items * Trials);
        state.add_reads<T>(items * Trials);

        state.run(
            [&]
            {
                if constexpr(BenchmarkKind == benchmark_kinds::sort_keys)
                {
                    sort_keys_kernel<T, BlockSize, RadixBitsPerPass, ItemsPerThread, Trials>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(
                            d_input.get(),
                            d_output.get());
                }
                else if constexpr(BenchmarkKind == benchmark_kinds::sort_pairs)
                {
                    sort_pairs_kernel<T, BlockSize, RadixBitsPerPass, ItemsPerThread, Trials>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(
                            d_input.get(),
                            d_output.get());
                }
            });
    }
};
