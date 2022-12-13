// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <limits>
#include <string>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

namespace rp = rocprim;

template<typename T,
         unsigned int                   BlockSize,
         unsigned int                   ItemsPerThread,
         unsigned int                   RadixBits,
         bool                           Descending,
         rp::block_radix_rank_algorithm Algorithm,
         unsigned int                   Trials>
__global__ __launch_bounds__(BlockSize) void rank_kernel(const T*      keys_input,
                                                         unsigned int* ranks_output)
{
    using rank_type = rp::block_radix_rank<BlockSize, RadixBits, Algorithm>;

    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    rp::block_load_direct_striped<BlockSize>(lid, keys_input + block_offset, keys);

    unsigned int ranks[ItemsPerThread];

    ROCPRIM_NO_UNROLL
    for(unsigned int trial = 0; trial < Trials; ++trial)
    {
        ROCPRIM_SHARED_MEMORY typename rank_type::storage_type storage;
        unsigned int                                           begin_bit = 0;
        const unsigned int                                     end_bit   = sizeof(T) * 8;

        while(begin_bit < end_bit)
        {
            const unsigned pass_bits = min(RadixBits, end_bit - begin_bit);
            if ROCPRIM_IF_CONSTEXPR(Descending)
            {
                rank_type().rank_keys_desc(keys, ranks, storage, begin_bit, pass_bits);
            }
            else
            {
                rank_type().rank_keys(keys, ranks, storage, begin_bit, pass_bits);
            }
            begin_bit += RadixBits;
        }
    }

    rp::block_store_direct_striped<BlockSize>(lid, ranks_output + block_offset, ranks);
}

template<typename T,
         unsigned int                   BlockSize,
         unsigned int                   ItemsPerThread,
         rp::block_radix_rank_algorithm Algorithm,
         unsigned int                   RadixBits  = 4,
         bool                           Descending = false,
         unsigned int                   Trials     = 10>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     grid_size       = ((N + items_per_block - 1) / items_per_block);
    const unsigned int     size            = items_per_block * grid_size;

    std::vector<T> input;
    if ROCPRIM_IF_CONSTEXPR(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, static_cast<T>(-1000), static_cast<T>(1000));
    }
    else
    {
        input = get_random_data<T>(size,
                                   std::numeric_limits<T>::min(),
                                   std::numeric_limits<T>::max());
    }

    T*            d_input;
    unsigned int* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(HIP_KERNEL_NAME(rank_kernel<T,
                                                       BlockSize,
                                                       ItemsPerThread,
                                                       RadixBits,
                                                       Descending,
                                                       Algorithm,
                                                       Trials>),
                           dim3(grid_size),
                           dim3(BlockSize),
                           0,
                           stream,
                           d_input,
                           d_output);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IPT, KIND)                                                  \
    benchmark::RegisterBenchmark(                                                           \
        bench_naming::format_name("{lvl:block,algo:radix_rank,key_type:" #T ",cfg:{bs:" #BS \
                                  ",ipt:" #IPT ",method:" #KIND "}}")                       \
            .c_str(),                                                                       \
        run_benchmark<T, BS, IPT, KIND>,                                                    \
        stream,                                                                             \
        size)

// clang-format off
#define CREATE_BENCHMARK_KINDS(type, block, ipt)                                       \
    CREATE_BENCHMARK(type, block, ipt, rp::block_radix_rank_algorithm::basic),         \
    CREATE_BENCHMARK(type, block, ipt, rp::block_radix_rank_algorithm::basic_memoize), \
    CREATE_BENCHMARK(type, block, ipt, rp::block_radix_rank_algorithm::match)

#define BENCHMARK_TYPE(type, block)          \
    CREATE_BENCHMARK_KINDS(type, block, 1),  \
    CREATE_BENCHMARK_KINDS(type, block, 4),  \
    CREATE_BENCHMARK_KINDS(type, block, 8),  \
    CREATE_BENCHMARK_KINDS(type, block, 12), \
    CREATE_BENCHMARK_KINDS(type, block, 16), \
    CREATE_BENCHMARK_KINDS(type, block, 20)
// clang-format on

void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t                                   stream,
                    size_t                                        size)
{
    std::vector<benchmark::internal::Benchmark*> bs = {
        BENCHMARK_TYPE(int, 128),
        BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(int, 512),

        BENCHMARK_TYPE(uint8_t, 128),
        BENCHMARK_TYPE(uint8_t, 256),
        BENCHMARK_TYPE(uint8_t, 512),

        BENCHMARK_TYPE(long long, 128),
        BENCHMARK_TYPE(long long, 256),
        BENCHMARK_TYPE(long long, 512),
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks(benchmarks, stream, size);

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
