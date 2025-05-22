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

#include "../common/utils_device_ptr.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/block/block_histogram.hpp>
#include <rocprim/config.hpp>
#include <rocprim/types.hpp>

template<typename Runner,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int BinSize,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, BinSize, Trials>(input, output);
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
        __shared__ T histogram[BinSize];
        __shared__ typename bhistogram_t::storage_type storage;

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

template<typename Benchmark,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int BinSize = BlockSize,
         unsigned int Trials  = 100>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;

    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);
    // Make sure size is a multiple of BlockSize
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size     = items_per_block * ((N + items_per_block - 1) / items_per_block);
    const auto     bin_size = BinSize * ((N + items_per_block - 1) / items_per_block);
    // Allocate and fill memory
    std::vector<T> input(size, 0.0f);
    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(bin_size);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            kernel<Benchmark, T, BlockSize, ItemsPerThread, BinSize, Trials>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                               d_output.get());
            HIP_CHECK(hipGetLastError());
        });

    state.set_throughput(size * Trials, sizeof(T));
}

#define CREATE_BENCHMARK(Benchmark, method, T, BS, IPT)                                  \
    executor.queue_fn(bench_naming::format_name("{lvl:block,algo:histogram,key_type:" #T \
                                                ",cfg:{bs:" #BS ",ipt:" #IPT ",method:"  \
                                                + std::string(method) + "}}")            \
                          .c_str(),                                                      \
                      run_benchmark<Benchmark, T, BS, IPT>);

#define BENCHMARK_TYPE(Benchmark, method, T, BS)  \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 1) \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 2) \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 3) \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 4) \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 8) \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 16)

#define BENCHMARK_TYPE_128(Benchmark, method, T, BS) \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 1)    \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 2)    \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 3)    \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 4)    \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 8)    \
    CREATE_BENCHMARK(Benchmark, method, T, BS, 12)

#define BENCHMARK_ATOMIC()                                                      \
    BENCHMARK_TYPE(histogram_atomic_t, "using_atomic", int, 256)                \
    BENCHMARK_TYPE(histogram_atomic_t, "using_atomic", int, 320)                \
    BENCHMARK_TYPE(histogram_atomic_t, "using_atomic", int, 512)                \
                                                                                \
    BENCHMARK_TYPE(histogram_atomic_t, "using_atomic", unsigned long long, 256) \
    BENCHMARK_TYPE(histogram_atomic_t, "using_atomic", unsigned long long, 320)

#define BENCHMARK_SORT()                                                       \
    BENCHMARK_TYPE(histogram_sort_t, "using_sort", int, 256)                   \
    BENCHMARK_TYPE(histogram_sort_t, "using_sort", int, 320)                   \
    BENCHMARK_TYPE(histogram_sort_t, "using_sort", int, 512)                   \
                                                                               \
    BENCHMARK_TYPE(histogram_sort_t, "using_sort", unsigned long long, 256)    \
    BENCHMARK_TYPE(histogram_sort_t, "using_sort", unsigned long long, 320)    \
                                                                               \
    BENCHMARK_TYPE_128(histogram_sort_t, "using_sort", rocprim::int128_t, 256) \
    BENCHMARK_TYPE_128(histogram_sort_t, "using_sort", rocprim::uint128_t, 256)

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 512 * benchmark_utils::MiB, 1, 0);

#ifndef BENCHMARK_CONFIG_TUNING
    using histogram_atomic_t = histogram<rocprim::block_histogram_algorithm::using_atomic>;
    using histogram_sort_t   = histogram<rocprim::block_histogram_algorithm::using_sort>;

    BENCHMARK_ATOMIC()
    BENCHMARK_SORT()
#endif

    executor.run();
}
