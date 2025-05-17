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

#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>
// rocPRIM
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
         typename Value              = Key,
         bool         SortByKey      = false,
         unsigned int Trials         = 100>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;

    // Calculate the number of elements
    size_t size = bytes / sizeof(Key);

    // Make sure size is a multiple of items_per_block
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    size                           = BlockSize * ((size + items_per_block - 1) / items_per_block);
    // Allocate and fill memory
    const auto       random_range = limit_random_range<Key>(0, 10'000);
    std::vector<Key> input_key
        = get_random_data<Key>(size, random_range.first, random_range.second, seed.get_0());
    std::vector<Value> input_value(size_t(1));
    if(SortByKey)
    {
        const auto random_range = limit_random_range<Value>(0, 10'000);
        input_value
            = get_random_data<Value>(size, random_range.first, random_range.second, seed.get_1());
    }
    Key*   d_input_key    = nullptr;
    Key*   d_output_key   = nullptr;
    Value* d_input_value  = nullptr;
    Value* d_output_value = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_key), size * sizeof(Key)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_key), size * sizeof(Key)));
    if(SortByKey)
    {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input_value), size * sizeof(Value)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output_value), size * sizeof(Value)));
    }
    HIP_CHECK(hipMemcpy(d_input_key, input_key.data(), size * sizeof(Key), hipMemcpyHostToDevice));
    if(SortByKey)
        HIP_CHECK(hipMemcpy(d_input_value,
                            input_value.data(),
                            size * sizeof(Value),
                            hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            if(SortByKey)
            {
                ROCPRIM_NO_UNROLL
                for(unsigned int trial = 0; trial < Trials; ++trial)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(warp_sort_by_key_kernel<Key,
                                                                               Value,
                                                                               BlockSize,
                                                                               WarpSize,
                                                                               ItemsPerThread>),
                                       dim3(size / items_per_block),
                                       dim3(BlockSize),
                                       0,
                                       stream,
                                       d_input_key,
                                       d_input_value,
                                       d_output_key,
                                       d_output_value);
                }
            }
            else
            {
                ROCPRIM_NO_UNROLL
                for(unsigned int trial = 0; trial < Trials; ++trial)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(warp_sort_kernel<Key, BlockSize, WarpSize, ItemsPerThread>),
                        dim3(size / items_per_block),
                        dim3(BlockSize),
                        0,
                        stream,
                        d_input_key,
                        d_output_key);
                }
            }
        });

    auto type_size = SortByKey ? sizeof(Key) + sizeof(Value) : sizeof(Key);

    state.set_throughput(size * Trials, type_size);

    HIP_CHECK(hipFree(d_input_key));
    HIP_CHECK(hipFree(d_output_key));
    HIP_CHECK(hipFree(d_input_value));
    HIP_CHECK(hipFree(d_output_value));
}

#define CREATE_SORT_BENCHMARK(K, BS, WS, IPT)                                                      \
    executor.queue_fn(bench_naming::format_name("{lvl:warp,algo:sort,key_type:" #K ",value_type:"  \
                                                + std::string(Traits<rocprim::empty_type>::name()) \
                                                + ",ws:" #WS ",cfg:{bs:" #BS ",ipt:" #IPT "}}")    \
                          .c_str(),                                                                \
                      run_benchmark<K, BS, WS, IPT>);

#define CREATE_SORTBYKEY_BENCHMARK(K, V, BS, WS, IPT)                                        \
    executor.queue_fn(bench_naming::format_name("{lvl:warp,algo:sort,key_type:" #K           \
                                                ",value_type:" #V ",ws:" #WS ",cfg:{bs:" #BS \
                                                ",ipt:" #IPT "}}")                           \
                          .c_str(),                                                          \
                      run_benchmark<K, BS, WS, IPT, V, true>);

// clang-format off
#define BENCHMARK_TYPE(type)                \
    CREATE_SORT_BENCHMARK(type, 64, 64, 1)  \
    CREATE_SORT_BENCHMARK(type, 64, 64, 2)  \
    CREATE_SORT_BENCHMARK(type, 64, 64, 4)  \
    CREATE_SORT_BENCHMARK(type, 128, 64, 1) \
    CREATE_SORT_BENCHMARK(type, 128, 64, 2) \
    CREATE_SORT_BENCHMARK(type, 128, 64, 4) \
    CREATE_SORT_BENCHMARK(type, 256, 64, 1) \
    CREATE_SORT_BENCHMARK(type, 256, 64, 2) \
    CREATE_SORT_BENCHMARK(type, 256, 64, 4) \
    CREATE_SORT_BENCHMARK(type, 64, 32, 1)  \
    CREATE_SORT_BENCHMARK(type, 64, 32, 2)  \
    CREATE_SORT_BENCHMARK(type, 64, 16, 1)  \
    CREATE_SORT_BENCHMARK(type, 64, 16, 2)  \
    CREATE_SORT_BENCHMARK(type, 64, 16, 4)
// clang-format on

// clang-format off
#define BENCHMARK_KEY_TYPE(type, value)                 \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64, 1)  \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64, 2)  \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 64, 64, 4)  \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 1) \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 2) \
    CREATE_SORTBYKEY_BENCHMARK(type, value, 256, 64, 4)
// clang-format on

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0);

    using custom_double2    = common::custom_type<double, double>;
    using custom_int_double = common::custom_type<int, double>;

    using custom_int2            = common::custom_type<int, int>;
    using custom_char_double     = common::custom_type<char, double>;
    using custom_longlong_double = common::custom_type<long long, double>;

    BENCHMARK_TYPE(int)
    BENCHMARK_TYPE(float)
    BENCHMARK_TYPE(double)
    BENCHMARK_TYPE(int8_t)
    BENCHMARK_TYPE(uint8_t)
    BENCHMARK_TYPE(rocprim::half)
    BENCHMARK_TYPE(rocprim::int128_t)
    BENCHMARK_TYPE(rocprim::uint128_t)

    BENCHMARK_KEY_TYPE(float, float)
    BENCHMARK_KEY_TYPE(unsigned int, int)
    BENCHMARK_KEY_TYPE(int, custom_double2)
    BENCHMARK_KEY_TYPE(int, custom_int_double)
    BENCHMARK_KEY_TYPE(custom_int2, custom_double2)
    BENCHMARK_KEY_TYPE(custom_int2, custom_char_double)
    BENCHMARK_KEY_TYPE(custom_int2, custom_longlong_double)
    BENCHMARK_KEY_TYPE(int8_t, int8_t)
    BENCHMARK_KEY_TYPE(uint8_t, uint8_t)
    BENCHMARK_KEY_TYPE(rocprim::half, rocprim::half)
    BENCHMARK_KEY_TYPE(rocprim::int128_t, rocprim::int128_t)
    BENCHMARK_KEY_TYPE(rocprim::uint128_t, rocprim::uint128_t)

    executor.run();
}
