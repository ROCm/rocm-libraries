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

#include "benchmark_utils.hpp"

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_radix_rank.hpp>
#include <hipcub/block/block_store.hpp>

#include <hipcub/block/radix_rank_sort_operations.hpp>

constexpr unsigned int Trials = 10;

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
         unsigned int       ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void rank_kernel(const T* keys_input, int* ranks_output)
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

inline const char* get_algorithm_name(RadixRankAlgorithm algorithm)
{
    switch(algorithm)
    {
        case RadixRankAlgorithm::RADIX_RANK_BASIC: return "basic";
        case RadixRankAlgorithm::RADIX_RANK_MATCH: return "match";
        case RadixRankAlgorithm::RADIX_RANK_MEMOIZE: return "memoize";
    }

    return "unknown algorithm";
}

template<class T,
         RadixRankAlgorithm BenchmarkKind,
         unsigned int       BlockSize,
         unsigned int       ItemsPerThread>
class block_radix_rank_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_radix_rank")
            .add("subalgo", get_algorithm_name(BenchmarkKind))
            .add("lvl", "block")
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
        const size_t           items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        std::vector<T> input
            = benchmark_utils::get_random_data<T>(items,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        T*   d_input;
        int* d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(Trials * items);
        state.add_writes<T>(Trials * items);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(
                        rank_kernel<T, 4, false, BenchmarkKind, BlockSize, ItemsPerThread>),
                    dim3(items / items_per_block),
                    dim3(BlockSize),
                    0,
                    stream,
                    d_input,
                    d_output);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

#define CREATE_BENCHMARK(T, KIND, BS, IPT) \
    executor.queue<block_radix_rank_benchmark<T, KIND, BS, IPT>>()

#define CREATE_BENCHMARK_KINDS(type, block, ipt)                                \
    CREATE_BENCHMARK(type, RadixRankAlgorithm::RADIX_RANK_BASIC, block, ipt),   \
    CREATE_BENCHMARK(type, RadixRankAlgorithm::RADIX_RANK_MEMOIZE, block, ipt), \
    CREATE_BENCHMARK(type, RadixRankAlgorithm::RADIX_RANK_MATCH, block, ipt)

#define BENCHMARK_TYPE(type, block)                                                      \
    CREATE_BENCHMARK_KINDS(type, block, 1), CREATE_BENCHMARK_KINDS(type, block, 4),      \
        CREATE_BENCHMARK_KINDS(type, block, 8), CREATE_BENCHMARK_KINDS(type, block, 16), \
        CREATE_BENCHMARK_KINDS(type, block, 32)

void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int, 128);
    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(int, 512);

    BENCHMARK_TYPE(uint8_t, 128);
    BENCHMARK_TYPE(uint8_t, 256);
    BENCHMARK_TYPE(uint8_t, 512);

    BENCHMARK_TYPE(int64_t, 128);
    BENCHMARK_TYPE(int64_t, 256);
    BENCHMARK_TYPE(int64_t, 512);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks(executor);

    executor.run();
}
