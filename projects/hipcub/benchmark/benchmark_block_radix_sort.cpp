// MIT License
//
// Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hipcub/block/block_radix_sort.hpp>
#include <hipcub/block/block_store.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

struct helper_blocked_blocked
{
    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread, typename InputIteratorT>
    HIPCUB_DEVICE static void
        load(int linear_id, InputIteratorT block_iter, T (&items)[ItemsPerThread])
    {
        hipcub::LoadDirectStriped<BlockSize>(linear_id, block_iter, items);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(T (&keys)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread> sort;
        sort.Sort(keys);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(T (&keys)[ItemsPerThread], T (&values)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread, T> sort;
        sort.Sort(keys, values);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread> sort;
        sort.Sort(keys, benchmark_utils::custom_type_decomposer<custom_t>{});
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread],
                                   benchmark_utils::custom_type<T> (&values)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread, custom_t> sort;
        sort.Sort(keys, values, benchmark_utils::custom_type_decomposer<custom_t>{});
    }
};

struct helper_blocked_striped
{
    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread, typename InputIteratorT>
    HIPCUB_DEVICE static void
        load(int linear_id, InputIteratorT block_iter, T (&items)[ItemsPerThread])
    {
        hipcub::LoadDirectBlocked(linear_id, block_iter, items);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(T (&keys)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread> sort;
        sort.SortBlockedToStriped(keys);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(T (&keys)[ItemsPerThread], T (&values)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread, T> sort;
        sort.SortBlockedToStriped(keys, values);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread> sort;
        sort.SortBlockedToStriped(keys, benchmark_utils::custom_type_decomposer<custom_t>{});
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread],
                                   benchmark_utils::custom_type<T> (&values)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread, custom_t> sort;
        sort.SortBlockedToStriped(keys,
                                  values,
                                  benchmark_utils::custom_type_decomposer<custom_t>{});
    }
};

template<class Helper,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void sort_keys_kernel(const T* input, T* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    Helper::template load<BlockSize>(lid, input + block_offset, keys);

#pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        Helper::template sort<BlockSize>(keys);
    }

    hipcub::StoreDirectStriped<BlockSize>(lid, output + block_offset, keys);
}

template<class Helper,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize) void sort_pairs_kernel(const T* input, T* output)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    T values[ItemsPerThread];
    Helper::template load<BlockSize>(lid, input + block_offset, keys);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        values[i] = keys[i] + T(1);
    }

#pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        Helper::template sort<BlockSize>(keys, values);
    }

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        keys[i] += values[i];
    }

    hipcub::StoreDirectStriped<BlockSize>(lid, output + block_offset, keys);
}

template<class Helper,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials = 10>
void run_benchmark(benchmark::State& state,
                   benchmark_kinds   benchmark_kind,
                   hipStream_t       stream,
                   size_t            N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    std::vector<T> input
        = benchmark_utils::get_random_data<T>(size,
                                              benchmark_utils::generate_limits<T>::min(),
                                              benchmark_utils::generate_limits<T>::max());

    T* d_input;
    T* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if(benchmark_kind == benchmark_kinds::sort_keys)
        {
            sort_keys_kernel<Helper, T, BlockSize, ItemsPerThread, Trials>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input, d_output);
        } else if(benchmark_kind == benchmark_kinds::sort_pairs)
        {
            sort_pairs_kernel<Helper, T, BlockSize, ItemsPerThread, Trials>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input, d_output);
        }
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

#define CREATE_BENCHMARK(T, BS, IPT)                                                             \
    benchmark::RegisterBenchmark(std::string("block_radix_sort<data_type:" #T ",block_size:" #BS \
                                             ",items_per_thread:" #IPT ">.sub_algorithm_name:"   \
                                             + name)                                             \
                                     .c_str(),                                                   \
                                 &run_benchmark<Helper, T, BS, IPT>,                             \
                                 benchmark_kind,                                                 \
                                 stream,                                                         \
                                 size)

// clang-format off
#define BENCHMARK_TYPE(type, block)   \
    CREATE_BENCHMARK(type, block, 1), \
    CREATE_BENCHMARK(type, block, 3), \
    CREATE_BENCHMARK(type, block, 4), \
    CREATE_BENCHMARK(type, block, 8)
// clang-format on

template<typename Helper>
void add_benchmarks(benchmark_kinds                               benchmark_kind,
                    const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t                                   stream,
                    size_t                                        size)
{
    using custom_int_t = benchmark_utils::custom_type<int>;

    std::vector<benchmark::internal::Benchmark*> bs = {
        BENCHMARK_TYPE(int, 64),           BENCHMARK_TYPE(int, 128),
        BENCHMARK_TYPE(int, 192),          BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(int, 320),          BENCHMARK_TYPE(int, 512),

        BENCHMARK_TYPE(int8_t, 64),        BENCHMARK_TYPE(int8_t, 128),
        BENCHMARK_TYPE(int8_t, 192),       BENCHMARK_TYPE(int8_t, 256),
        BENCHMARK_TYPE(int8_t, 320),       BENCHMARK_TYPE(int8_t, 512),

        BENCHMARK_TYPE(long long, 64),     BENCHMARK_TYPE(long long, 128),
        BENCHMARK_TYPE(long long, 192),    BENCHMARK_TYPE(long long, 256),
        BENCHMARK_TYPE(long long, 320),    BENCHMARK_TYPE(long long, 512),

        BENCHMARK_TYPE(custom_int_t, 64),  BENCHMARK_TYPE(custom_int_t, 128),
        BENCHMARK_TYPE(custom_int_t, 192), BENCHMARK_TYPE(custom_int_t, 256),
        BENCHMARK_TYPE(custom_int_t, 320), BENCHMARK_TYPE(custom_int_t, 512),
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

    std::cout << "benchmark_block_radix_sort" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    // clang-format off
    add_benchmarks<helper_blocked_blocked>(
        benchmark_kinds::sort_keys, "sort(keys)", benchmarks, stream, size);
    add_benchmarks<helper_blocked_blocked>(
        benchmark_kinds::sort_pairs, "sort(keys, values)", benchmarks, stream, size);
    add_benchmarks<helper_blocked_striped>(
        benchmark_kinds::sort_keys, "sort_to_striped(keys)", benchmarks, stream, size);
    add_benchmarks<helper_blocked_striped>(
        benchmark_kinds::sort_pairs, "sort_to_striped(keys, values)", benchmarks, stream, size);
    // clang-format on

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
