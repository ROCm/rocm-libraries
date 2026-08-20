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

#include "benchmark_utils.hpp"

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_radix_sort.hpp>
#include <hipcub/block/block_store.hpp>

constexpr unsigned int Trials = 10;

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

struct helper_blocked_blocked
{
    static const char* get_algorithm_name(benchmark_kinds algorithm)
    {
        switch(algorithm)
        {
            case benchmark_kinds::sort_keys: return "sort(keys)";
            case benchmark_kinds::sort_pairs: return "sort(keys, values)";
        }

        return "unknown algorithm";
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread, typename InputIteratorT>
    HIPCUB_DEVICE
    static void load(int linear_id, InputIteratorT block_iter, T (&items)[ItemsPerThread])
    {
        hipcub::LoadDirectStriped<BlockSize>(linear_id, block_iter, items);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(T (&keys)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread> sort;
        sort.Sort(keys);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(T (&keys)[ItemsPerThread], T (&values)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread, T> sort;
        sort.Sort(keys, values);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread> sort;
        sort.Sort(keys, benchmark_utils::custom_type_decomposer<custom_t>{});
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread],
                     benchmark_utils::custom_type<T> (&values)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread, custom_t> sort;
        sort.Sort(keys, values, benchmark_utils::custom_type_decomposer<custom_t>{});
    }
};

struct helper_blocked_striped
{
    static const char* get_algorithm_name(benchmark_kinds algorithm)
    {
        switch(algorithm)
        {
            case benchmark_kinds::sort_keys: return "sort_to_striped(keys)";
            case benchmark_kinds::sort_pairs: return "sort_to_striped(keys, values)";
        }

        return "unknown algorithm";
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread, typename InputIteratorT>
    HIPCUB_DEVICE
    static void load(int linear_id, InputIteratorT block_iter, T (&items)[ItemsPerThread])
    {
        hipcub::LoadDirectBlocked(linear_id, block_iter, items);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(T (&keys)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread> sort;
        sort.SortBlockedToStriped(keys);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(T (&keys)[ItemsPerThread], T (&values)[ItemsPerThread])
    {
        hipcub::BlockRadixSort<T, BlockSize, ItemsPerThread, T> sort;
        sort.SortBlockedToStriped(keys, values);
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread> sort;
        sort.SortBlockedToStriped(keys, benchmark_utils::custom_type_decomposer<custom_t>{});
    }

    template<unsigned int BlockSize, class T, unsigned int ItemsPerThread>
    HIPCUB_DEVICE
    static void sort(benchmark_utils::custom_type<T> (&keys)[ItemsPerThread],
                     benchmark_utils::custom_type<T> (&values)[ItemsPerThread])
    {
        using custom_t = benchmark_utils::custom_type<T>;
        hipcub::BlockRadixSort<custom_t, BlockSize, ItemsPerThread, custom_t> sort;
        sort.SortBlockedToStriped(keys,
                                  values,
                                  benchmark_utils::custom_type_decomposer<custom_t>{});
    }
};

template<class Helper, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void sort_keys_kernel(const T* input, T* output)
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

template<class Helper, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void sort_pairs_kernel(const T* input, T* output)
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
         unsigned int    BlockSize,
         unsigned int    ItemsPerThread,
         benchmark_kinds BenchmarkKind>
class block_radix_sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_radix_sort")
            .add("subalgo", Helper::get_algorithm_name(BenchmarkKind))
            .add("lvl", "block")
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const size_t   items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        std::vector<T> input
            = benchmark_utils::get_random_data<T>(items,
                                                  benchmark_utils::generate_limits<T>::min(),
                                                  benchmark_utils::generate_limits<T>::max());

        T* d_input;
        T* d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(Trials * items);
        state.add_writes<T>(Trials * items);

        state.run(
            [&]
            {
                if constexpr(BenchmarkKind == benchmark_kinds::sort_keys)
                {
                    sort_keys_kernel<Helper, T, BlockSize, ItemsPerThread>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input,
                                                                                        d_output);
                }
                else if(BenchmarkKind == benchmark_kinds::sort_pairs)
                {
                    sort_pairs_kernel<Helper, T, BlockSize, ItemsPerThread>
                        <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_input,
                                                                                        d_output);
                }
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

#define CREATE_BENCHMARK(T, BS, IPT) \
    executor.queue<block_radix_sort_benchmark<Helper, T, BS, IPT, BenchmarkKind>>()

#define BENCHMARK_TYPE(type, block)                                     \
    CREATE_BENCHMARK(type, block, 1), CREATE_BENCHMARK(type, block, 3), \
        CREATE_BENCHMARK(type, block, 4), CREATE_BENCHMARK(type, block, 8)

template<typename Helper, benchmark_kinds BenchmarkKind>
void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int, 64);
    BENCHMARK_TYPE(int, 128);
    BENCHMARK_TYPE(int, 192);
    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(int, 320);
    BENCHMARK_TYPE(int, 512);

    BENCHMARK_TYPE(int8_t, 64);
    BENCHMARK_TYPE(int8_t, 128);
    BENCHMARK_TYPE(int8_t, 192);
    BENCHMARK_TYPE(int8_t, 256);
    BENCHMARK_TYPE(int8_t, 320);
    BENCHMARK_TYPE(int8_t, 512);

    BENCHMARK_TYPE(int64_t, 64);
    BENCHMARK_TYPE(int64_t, 128);
    BENCHMARK_TYPE(int64_t, 192);
    BENCHMARK_TYPE(int64_t, 256);
    BENCHMARK_TYPE(int64_t, 320);
    BENCHMARK_TYPE(int64_t, 512);

    BENCHMARK_TYPE(custom_int_t, 64);
    BENCHMARK_TYPE(custom_int_t, 128);
    BENCHMARK_TYPE(custom_int_t, 192);
    BENCHMARK_TYPE(custom_int_t, 256);
    BENCHMARK_TYPE(custom_int_t, 320);
    BENCHMARK_TYPE(custom_int_t, 512);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<helper_blocked_blocked, benchmark_kinds::sort_keys>(executor);
    add_benchmarks<helper_blocked_blocked, benchmark_kinds::sort_pairs>(executor);
    add_benchmarks<helper_blocked_striped, benchmark_kinds::sort_keys>(executor);
    add_benchmarks<helper_blocked_striped, benchmark_kinds::sort_pairs>(executor);

    executor.run();
}
