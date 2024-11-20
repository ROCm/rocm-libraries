// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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
// CmdParser
#include "cmdparser.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_radix_sort.hpp>
#include <rocprim/block/block_store_func.hpp>

#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <cstdio>
#include <cstdlib>

#ifndef DEFAULT_N
const size_t DEFAULT_BYTES = 1024 * 1024 * 128 * 4;
#endif

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

namespace rp = rocprim;

template<class T>
using select_decomposer_t = std::
    conditional_t<is_custom_type<T>::value, custom_type_decomposer<T>, rp::identity_decomposer>;

template<class T,
         unsigned int BlockSize,
         unsigned int RadixBitsPerPass,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(const T* input, T* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);

    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        rp::block_radix_sort<T,
                             BlockSize,
                             ItemsPerThread,
                             rocprim::empty_type,
                             1,
                             1,
                             RadixBitsPerPass>
            sort;
        sort.sort(keys, 0, sizeof(T) * 8, select_decomposer_t<T>{});
    }

    rp::block_store_direct_striped<BlockSize>(lid, output + block_offset, keys);
}

template<class T,
         unsigned int BlockSize,
         unsigned int RadixBitsPerPass,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(const T* input, T* output)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    T values[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, input + block_offset, keys);
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        values[i] = keys[i] + T(1);
    }

    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        rp::block_radix_sort<T, BlockSize, ItemsPerThread, T, 1, 1, RadixBitsPerPass> sort;
        sort.sort(keys, values, 0, sizeof(T) * 8, select_decomposer_t<T>{});
    }

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        keys[i] += values[i];
    }
    rp::block_store_direct_striped<BlockSize>(lid, output + block_offset, keys);
}

template<class T,
         unsigned int BlockSize,
         unsigned int RadixBitsPerPass,
         unsigned int ItemsPerThread,
         unsigned int Trials = 10>
void run_benchmark(benchmark::State&   state,
                   benchmark_kinds     benchmark_kind,
                   size_t              bytes,
                   const managed_seed& seed,
                   hipStream_t         stream)
{
    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);
    
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

    std::vector<T> input = get_random_data<T>(size,
                                              generate_limits<T>::min(),
                                              generate_limits<T>::max(),
                                              seed.get_0());

    T*  d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(T)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        if(benchmark_kind == benchmark_kinds::sort_keys)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    sort_keys_kernel<T, BlockSize, RadixBitsPerPass, ItemsPerThread, Trials>),
                dim3(size / items_per_block),
                dim3(BlockSize),
                0,
                stream,
                d_input,
                d_output);
        }
        else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    sort_pairs_kernel<T, BlockSize, RadixBitsPerPass, ItemsPerThread, Trials>),
                dim3(size / items_per_block),
                dim3(BlockSize),
                0,
                stream,
                d_input,
                d_output);
        }
        HIP_CHECK(hipGetLastError());

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, RB, IPT)                                                       \
    benchmark::RegisterBenchmark(                                                              \
        bench_naming::format_name("{lvl:block,algo:radix_sort,key_type:" #T ",subalgo:" + name \
                                  + ",cfg:{bs:" #BS ",rb:" #RB ",ipt:" #IPT "}}")              \
            .c_str(),                                                                          \
        run_benchmark<T, BS, RB, IPT>,                                                         \
        benchmark_kind,                                                                        \
        bytes,                                                                                  \
        seed,                                                                                  \
        stream)

#define BENCHMARK_TYPE(type, block, radix_bits)                                                 \
    CREATE_BENCHMARK(type, block, radix_bits, 1), CREATE_BENCHMARK(type, block, radix_bits, 2), \
        CREATE_BENCHMARK(type, block, radix_bits, 3),                                           \
        CREATE_BENCHMARK(type, block, radix_bits, 4), CREATE_BENCHMARK(type, block, radix_bits, 8)

void add_benchmarks(benchmark_kinds                               benchmark_kind,
                    const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    size_t                                        bytes,
                    const managed_seed&                           seed,
                    hipStream_t                                   stream)
{
    using custom_int_type = custom_type<int, int>;

    std::vector<benchmark::internal::Benchmark*> bs = {
        BENCHMARK_TYPE(int, 64, 3),
        BENCHMARK_TYPE(int, 512, 3),

        BENCHMARK_TYPE(int, 64, 4),
        BENCHMARK_TYPE(int, 128, 4),
        BENCHMARK_TYPE(int, 192, 4),
        BENCHMARK_TYPE(int, 256, 4),
        BENCHMARK_TYPE(int, 320, 4),
        BENCHMARK_TYPE(int, 512, 4),

        BENCHMARK_TYPE(int8_t, 64, 3),
        BENCHMARK_TYPE(int8_t, 512, 3),

        BENCHMARK_TYPE(int8_t, 64, 4),
        BENCHMARK_TYPE(int8_t, 128, 4),
        BENCHMARK_TYPE(int8_t, 192, 4),
        BENCHMARK_TYPE(int8_t, 256, 4),
        BENCHMARK_TYPE(int8_t, 320, 4),
        BENCHMARK_TYPE(int8_t, 512, 4),

        BENCHMARK_TYPE(uint8_t, 64, 3),
        BENCHMARK_TYPE(uint8_t, 512, 3),

        BENCHMARK_TYPE(uint8_t, 64, 4),
        BENCHMARK_TYPE(uint8_t, 128, 4),
        BENCHMARK_TYPE(uint8_t, 192, 4),
        BENCHMARK_TYPE(uint8_t, 256, 4),
        BENCHMARK_TYPE(uint8_t, 320, 4),
        BENCHMARK_TYPE(uint8_t, 512, 4),

        BENCHMARK_TYPE(rocprim::half, 64, 3),
        BENCHMARK_TYPE(rocprim::half, 512, 3),

        BENCHMARK_TYPE(rocprim::half, 64, 4),
        BENCHMARK_TYPE(rocprim::half, 128, 4),
        BENCHMARK_TYPE(rocprim::half, 192, 4),
        BENCHMARK_TYPE(rocprim::half, 256, 4),
        BENCHMARK_TYPE(rocprim::half, 320, 4),
        BENCHMARK_TYPE(rocprim::half, 512, 4),

        BENCHMARK_TYPE(long long, 64, 3),
        BENCHMARK_TYPE(long long, 512, 3),

        BENCHMARK_TYPE(long long, 64, 4),
        BENCHMARK_TYPE(long long, 128, 4),
        BENCHMARK_TYPE(long long, 192, 4),
        BENCHMARK_TYPE(long long, 256, 4),
        BENCHMARK_TYPE(long long, 320, 4),
        BENCHMARK_TYPE(long long, 512, 4),

        BENCHMARK_TYPE(custom_int_type, 64, 3),
        BENCHMARK_TYPE(custom_int_type, 512, 3),

        BENCHMARK_TYPE(custom_int_type, 64, 4),
        BENCHMARK_TYPE(custom_int_type, 128, 4),
        BENCHMARK_TYPE(custom_int_type, 192, 4),
        BENCHMARK_TYPE(custom_int_type, 256, 4),
        BENCHMARK_TYPE(custom_int_type, 320, 4),
        BENCHMARK_TYPE(custom_int_type, 512, 4),
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t bytes = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("bytes", std::to_string(bytes));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks(benchmark_kinds::sort_keys, "keys", benchmarks, bytes, seed, stream);
    add_benchmarks(benchmark_kinds::sort_pairs, "pairs", benchmarks, bytes, seed, stream);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
