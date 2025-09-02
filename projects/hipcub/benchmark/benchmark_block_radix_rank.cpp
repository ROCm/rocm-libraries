// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_benchmark_header.hpp"

// HIP API
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_radix_rank.hpp>
#include <hipcub/block/block_store.hpp>

#include <hipcub/block/radix_rank_sort_operations.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

enum class RadixRankAlgorithm
{
    RADIX_RANK_BASIC,
    RADIX_RANK_MEMOIZE,
    RADIX_RANK_MATCH,
};

template<class T,
         unsigned int       RadixBits,
         bool               Descending,
         RadixRankAlgorithm BenchmarkKind,
         unsigned int       BlockSize,
         unsigned int       ItemsPerThread,
         unsigned int       Trials>
__global__ __launch_bounds__(BlockSize) void rank_kernel(const T* keys_input, int* ranks_output)
{
    const unsigned int lid          = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    hipcub::LoadDirectBlocked(lid, keys_input + block_offset, keys);

    using KeyTraits      = hipcub::Traits<T>;
    using UnsignedBits   = typename KeyTraits::UnsignedBits;
    using DigitExtractor = hipcub::BFEDigitExtractor<T>;

    UnsignedBits(&unsigned_keys)[ItemsPerThread]
        = reinterpret_cast<UnsignedBits(&)[ItemsPerThread]>(keys);

    using RankType = std::conditional_t<
        BenchmarkKind == RadixRankAlgorithm::RADIX_RANK_MATCH,
        hipcub::BlockRadixRankMatch<BlockSize, RadixBits, Descending>,
        hipcub::BlockRadixRank<BlockSize,
                               RadixBits,
                               Descending,
                               BenchmarkKind == RadixRankAlgorithm::RADIX_RANK_MEMOIZE>>;

#pragma unroll
    for(unsigned int key = 0; key < ItemsPerThread; key++)
    {
        unsigned_keys[key] = KeyTraits::TwiddleIn(unsigned_keys[key]);
    }

    int ranks[ItemsPerThread];

#pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        __shared__ typename RankType::TempStorage storage;
        RankType                                  rank(storage);
        unsigned                                  begin_bit = 0;
        const unsigned                            end_bit   = sizeof(T) * 8;

        while(begin_bit < end_bit)
        {
            const unsigned pass_bits = min(RadixBits, end_bit - begin_bit);
            DigitExtractor digit_extractor(begin_bit, pass_bits);

            rank.RankKeys(unsigned_keys, ranks, digit_extractor);
            begin_bit += RadixBits;
        }
    }

    hipcub::StoreDirectBlocked(lid, ranks_output + block_offset, ranks);
}

template<class T,
         RadixRankAlgorithm BenchmarkKind,
         unsigned int       BlockSize,
         unsigned int       ItemsPerThread,
         unsigned int       Trials = 10>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    std::vector<T> input
        = benchmark_utils::get_random_data<T>(size,
                                              benchmark_utils::generate_limits<T>::min(),
                                              benchmark_utils::generate_limits<T>::max());

    T*   d_input;
    int* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rank_kernel<T, 4, false, BenchmarkKind, BlockSize, ItemsPerThread, Trials>),
            dim3(size / items_per_block),
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

#define CREATE_BENCHMARK(T, KIND, BS, IPT)                                                     \
    benchmark::RegisterBenchmark(std::string("block_radix_rank<data_type:" #T ",kind:" #KIND   \
                                             ",block_size:" #BS ",items_per_thread:" #IPT ">." \
                                             + name)                                           \
                                     .c_str(),                                                 \
                                 &run_benchmark<T, KIND, BS, IPT>,                             \
                                 stream,                                                       \
                                 size)

// clang-format off
#define CREATE_BENCHMARK_KINDS(type, block, ipt)                                \
    CREATE_BENCHMARK(type, RadixRankAlgorithm::RADIX_RANK_BASIC, block, ipt),   \
    CREATE_BENCHMARK(type, RadixRankAlgorithm::RADIX_RANK_MEMOIZE, block, ipt), \
    CREATE_BENCHMARK(type, RadixRankAlgorithm::RADIX_RANK_MATCH, block, ipt)

#define BENCHMARK_TYPE(type, block)          \
    CREATE_BENCHMARK_KINDS(type, block, 1),  \
    CREATE_BENCHMARK_KINDS(type, block, 4),  \
    CREATE_BENCHMARK_KINDS(type, block, 8),  \
    CREATE_BENCHMARK_KINDS(type, block, 16), \
    CREATE_BENCHMARK_KINDS(type, block, 32)
// clang-format on

void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
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
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    std::cout << "benchmark_block_radix_rank" << std::endl;
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks("rank", benchmarks, stream, size);

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
