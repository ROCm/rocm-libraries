// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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

template<typename T,
         benchmark_kinds BenchmarkKind,
         unsigned int    BlockSize,
         unsigned int    RadixBitsPerPass,
         unsigned int    ItemsPerThread,
         unsigned int    Trials = 10>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;
    const auto& stream = state.stream;

    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);

    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    std::vector<T> input = get_random_data<T>(size,
                                              common::generate_limits<T>::min(),
                                              common::generate_limits<T>::max(),
                                              seed.get_0());

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(size);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            if constexpr(BenchmarkKind == benchmark_kinds::sort_keys)
            {
                sort_keys_kernel<T, BlockSize, RadixBitsPerPass, ItemsPerThread, Trials>
                    <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                   d_output.get());
            }
            else if constexpr(BenchmarkKind == benchmark_kinds::sort_pairs)
            {
                sort_pairs_kernel<T, BlockSize, RadixBitsPerPass, ItemsPerThread, Trials>
                    <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                                   d_output.get());
            }
            HIP_CHECK(hipGetLastError());
        });

    state.set_throughput(size * Trials, sizeof(T));
}

#define CREATE_BENCHMARK(T, BS, RB, IPT)                                                       \
    executor.queue_fn(                                                                         \
        bench_naming::format_name("{lvl:block,algo:radix_sort,key_type:" #T ",subalgo:" + name \
                                  + ",cfg:{bs:" #BS ",rb:" #RB ",ipt:" #IPT "}}")              \
            .c_str(),                                                                          \
        run_benchmark<T, BenchmarkKind, BS, RB, IPT>);

#define BENCHMARK_TYPE(type, block, radix_bits)  \
    CREATE_BENCHMARK(type, block, radix_bits, 1) \
    CREATE_BENCHMARK(type, block, radix_bits, 2) \
    CREATE_BENCHMARK(type, block, radix_bits, 3) \
    CREATE_BENCHMARK(type, block, radix_bits, 4) \
    CREATE_BENCHMARK(type, block, radix_bits, 8)

template<benchmark_kinds BenchmarkKind>
void add_benchmarks(const std::string& name, benchmark_utils::executor& executor)
{
    using custom_int_type = common::custom_type<int, int>;

    BENCHMARK_TYPE(int, 64, 3)
    BENCHMARK_TYPE(int, 512, 3)

    BENCHMARK_TYPE(int, 64, 4)
    BENCHMARK_TYPE(int, 128, 4)
    BENCHMARK_TYPE(int, 192, 4)
    BENCHMARK_TYPE(int, 256, 4)
    BENCHMARK_TYPE(int, 320, 4)
    BENCHMARK_TYPE(int, 512, 4)

    BENCHMARK_TYPE(int8_t, 64, 3)
    BENCHMARK_TYPE(int8_t, 512, 3)

    BENCHMARK_TYPE(int8_t, 64, 4)
    BENCHMARK_TYPE(int8_t, 128, 4)
    BENCHMARK_TYPE(int8_t, 192, 4)
    BENCHMARK_TYPE(int8_t, 256, 4)
    BENCHMARK_TYPE(int8_t, 320, 4)
    BENCHMARK_TYPE(int8_t, 512, 4)

    BENCHMARK_TYPE(uint8_t, 64, 3)
    BENCHMARK_TYPE(uint8_t, 512, 3)

    BENCHMARK_TYPE(uint8_t, 64, 4)
    BENCHMARK_TYPE(uint8_t, 128, 4)
    BENCHMARK_TYPE(uint8_t, 192, 4)
    BENCHMARK_TYPE(uint8_t, 256, 4)
    BENCHMARK_TYPE(uint8_t, 320, 4)
    BENCHMARK_TYPE(uint8_t, 512, 4)

    BENCHMARK_TYPE(rocprim::half, 64, 3)
    BENCHMARK_TYPE(rocprim::half, 512, 3)

    BENCHMARK_TYPE(rocprim::half, 64, 4)
    BENCHMARK_TYPE(rocprim::half, 128, 4)
    BENCHMARK_TYPE(rocprim::half, 192, 4)
    BENCHMARK_TYPE(rocprim::half, 256, 4)
    BENCHMARK_TYPE(rocprim::half, 320, 4)
    BENCHMARK_TYPE(rocprim::half, 512, 4)

    BENCHMARK_TYPE(long long, 64, 3)
    BENCHMARK_TYPE(long long, 512, 3)

    BENCHMARK_TYPE(long long, 64, 4)
    BENCHMARK_TYPE(long long, 128, 4)
    BENCHMARK_TYPE(long long, 192, 4)
    BENCHMARK_TYPE(long long, 256, 4)
    BENCHMARK_TYPE(long long, 320, 4)
    BENCHMARK_TYPE(long long, 512, 4)

    BENCHMARK_TYPE(custom_int_type, 64, 3)
    BENCHMARK_TYPE(custom_int_type, 512, 3)

    BENCHMARK_TYPE(custom_int_type, 64, 4)
    BENCHMARK_TYPE(custom_int_type, 128, 4)
    BENCHMARK_TYPE(custom_int_type, 192, 4)
    BENCHMARK_TYPE(custom_int_type, 256, 4)
    BENCHMARK_TYPE(custom_int_type, 320, 4)
    BENCHMARK_TYPE(custom_int_type, 512, 4)

    BENCHMARK_TYPE(rocprim::int128_t, 64, 3)
    BENCHMARK_TYPE(rocprim::int128_t, 512, 3)

    BENCHMARK_TYPE(rocprim::int128_t, 64, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 128, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 192, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 256, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 320, 4)
    BENCHMARK_TYPE(rocprim::int128_t, 512, 4)

    BENCHMARK_TYPE(rocprim::uint128_t, 64, 3)
    BENCHMARK_TYPE(rocprim::uint128_t, 512, 3)

    BENCHMARK_TYPE(rocprim::uint128_t, 64, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 128, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 192, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 256, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 320, 4)
    BENCHMARK_TYPE(rocprim::uint128_t, 512, 4)
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 512 * benchmark_utils::MiB, 1, 0);

    add_benchmarks<benchmark_kinds::sort_keys>("keys", executor);
    add_benchmarks<benchmark_kinds::sort_pairs>("pairs", executor);

    executor.run();
}
