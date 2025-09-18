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
#include "../common/utils_device_ptr.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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
        __shared__ typename bscan_t::storage_type storage;

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
        __shared__ typename bscan_t::storage_type storage;

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

template<typename Benchmark,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials = 100>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& bytes  = state.bytes;
    const auto& stream = state.stream;

    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);
    // Make sure size is a multiple of BlockSize
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size = items_per_block * ((N + items_per_block - 1) / items_per_block);
    // Allocate and fill memory
    std::vector<T> input(size, T(1));
    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(size);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                               d_output.get());
            HIP_CHECK(hipGetLastError());
        });

    state.set_throughput(size * Trials, sizeof(T));
}

#define CREATE_BENCHMARK(T, BS, IPT)                                                             \
    executor.queue_fn(bench_naming::format_name("{lvl:block,algo:scan,subalgo:" + algorithm_name \
                                                + ",key_type:" #T ",cfg:{bs:" #BS ",ipt:" #IPT   \
                                                  ",method:"                                     \
                                                + method_name + "}}")                            \
                          .c_str(),                                                              \
                      run_benchmark<Benchmark, T, BS, IPT>);

#define BENCHMARK_TYPE(type, block)   \
    CREATE_BENCHMARK(type, block, 1)  \
    CREATE_BENCHMARK(type, block, 2)  \
    CREATE_BENCHMARK(type, block, 3)  \
    CREATE_BENCHMARK(type, block, 4)  \
    CREATE_BENCHMARK(type, block, 8)  \
    CREATE_BENCHMARK(type, block, 11) \
    CREATE_BENCHMARK(type, block, 16)

template<typename Benchmark>
void add_benchmarks(const std::string&         method_name,
                    const std::string&         algorithm_name,
                    benchmark_utils::executor& executor)
{
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    // When block size is less than or equal to warp size
    BENCHMARK_TYPE(int, 64)
    BENCHMARK_TYPE(float, 64)
    BENCHMARK_TYPE(double, 64)
    BENCHMARK_TYPE(int8_t, 64)
    BENCHMARK_TYPE(uint8_t, 64)
    BENCHMARK_TYPE(rocprim::half, 64)

    BENCHMARK_TYPE(int, 256)
    BENCHMARK_TYPE(float, 256)
    BENCHMARK_TYPE(double, 256)
    BENCHMARK_TYPE(int8_t, 256)
    BENCHMARK_TYPE(uint8_t, 256)
    BENCHMARK_TYPE(rocprim::half, 256)

    CREATE_BENCHMARK(custom_float2, 256, 1)
    CREATE_BENCHMARK(custom_float2, 256, 4)
    CREATE_BENCHMARK(custom_float2, 256, 8)

    CREATE_BENCHMARK(float2, 256, 1)
    CREATE_BENCHMARK(float2, 256, 4)
    CREATE_BENCHMARK(float2, 256, 8)

    CREATE_BENCHMARK(custom_double2, 256, 1)
    CREATE_BENCHMARK(custom_double2, 256, 4)
    CREATE_BENCHMARK(custom_double2, 256, 8)

    CREATE_BENCHMARK(double2, 256, 1)
    CREATE_BENCHMARK(double2, 256, 4)
    CREATE_BENCHMARK(double2, 256, 8)

    CREATE_BENCHMARK(float4, 256, 1)
    CREATE_BENCHMARK(float4, 256, 4)
    CREATE_BENCHMARK(float4, 256, 8)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 8)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 8)
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 512 * benchmark_utils::MiB, 1, 0);

    using inclusive_scan_uws_t = inclusive_scan<rocprim::block_scan_algorithm::using_warp_scan>;
    add_benchmarks<inclusive_scan_uws_t>("inclusive_scan", "using_warp_scan", executor);

    using exclusive_scan_uws_t = exclusive_scan<rocprim::block_scan_algorithm::using_warp_scan>;
    add_benchmarks<exclusive_scan_uws_t>("exclusive_scan", "using_warp_scan", executor);

    using inclusive_scan_rts_t = inclusive_scan<rocprim::block_scan_algorithm::reduce_then_scan>;
    add_benchmarks<inclusive_scan_rts_t>("inclusive_scan", "reduce_then_scan", executor);

    using exclusive_scan_rts_t = exclusive_scan<rocprim::block_scan_algorithm::reduce_then_scan>;
    add_benchmarks<exclusive_scan_rts_t>("exclusive_scan", "reduce_then_scan", executor);

    executor.run();
}
