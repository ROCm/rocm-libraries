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
#include <rocprim/block/block_reduce.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
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
        __shared__ typename breduce_t::storage_type storage;

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

#define CREATE_BENCHMARK(T, BS, IPT)                                                    \
    executor.queue_fn(bench_naming::format_name("{lvl:block,algo:reduce,key_type:" #T   \
                                                ",cfg:{bs:" #BS ",ipt:" #IPT ",method:" \
                                                + name + "}}")                          \
                          .c_str(),                                                     \
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
void add_benchmarks(const std::string& name, benchmark_utils::executor& executor)
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
    BENCHMARK_TYPE(rocprim::int128_t, 64)
    BENCHMARK_TYPE(rocprim::uint128_t, 64)

    BENCHMARK_TYPE(int, 256)
    BENCHMARK_TYPE(float, 256)
    BENCHMARK_TYPE(double, 256)
    BENCHMARK_TYPE(int8_t, 256)
    BENCHMARK_TYPE(uint8_t, 256)
    BENCHMARK_TYPE(rocprim::half, 256)
    BENCHMARK_TYPE(rocprim::int128_t, 256)
    BENCHMARK_TYPE(rocprim::uint128_t, 256)

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
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0);

    using reduce_uwr_t = reduce<rocprim::block_reduce_algorithm::using_warp_reduce>;
    add_benchmarks<reduce_uwr_t>("using_warp_reduce", executor);

    using reduce_rr_t = reduce<rocprim::block_reduce_algorithm::raking_reduce>;
    add_benchmarks<reduce_rr_t>("raking_reduce", executor);

    using reduce_rrco_t = reduce<rocprim::block_reduce_algorithm::raking_reduce_commutative_only>;
    add_benchmarks<reduce_rrco_t>("raking_reduce_commutative_only", executor);

    executor.run();
}
