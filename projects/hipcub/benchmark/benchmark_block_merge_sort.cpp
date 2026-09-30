// MIT License
//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../test/hipcub/test_utils_sort_comparator.hpp"

#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_merge_sort.hpp>
#include <hipcub/block/block_store.hpp>

constexpr unsigned int Trials = 10;

enum class benchmark_kinds
{
    sort_keys,
    sort_pairs
};

template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, class CompareOp>
__global__ __launch_bounds__(BlockSize)
void sort_keys_kernel(const T* input, T* output, CompareOp compare_op)
{
    const unsigned int lid          = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    hipcub::LoadDirectStriped<BlockSize>(lid, input + block_offset, keys);

#pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        hipcub::BlockMergeSort<T, BlockSize, ItemsPerThread> sort;
        sort.Sort(keys, compare_op);
    }

    hipcub::StoreDirectStriped<BlockSize>(lid, output + block_offset, keys);
}

template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, class CompareOp>
__global__ __launch_bounds__(BlockSize)
void sort_pairs_kernel(const T* input, T* output, CompareOp compare_op)
{
    const unsigned int lid          = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

    T keys[ItemsPerThread];
    T values[ItemsPerThread];
    hipcub::LoadDirectStriped<BlockSize>(lid, input + block_offset, keys);

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        values[i] = keys[i] + T(1);
    }

#pragma nounroll
    for(unsigned int trial = 0; trial < Trials; trial++)
    {
        hipcub::BlockMergeSort<T, BlockSize, ItemsPerThread, T> sort;
        sort.Sort(keys, values, compare_op);
    }

    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        keys[i] += values[i];
    }
    hipcub::StoreDirectStriped<BlockSize>(lid, output + block_offset, keys);
}

inline const char* get_algorithm_name(benchmark_kinds benchmark)
{
    switch(benchmark)
    {
        case benchmark_kinds::sort_keys: return "sort(keys)";
        case benchmark_kinds::sort_pairs: return "sort(keys, values)";
    }

    return "unknown benchmark kind";
}

template<benchmark_kinds BenchmarkKind,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class CompareOp = test_utils::less>
class merge_sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_merge_sort")
            .add("subalgo", get_algorithm_name(BenchmarkKind))
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items
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
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(sort_keys_kernel<T, BlockSize, ItemsPerThread, CompareOp>),
                        dim3(items / items_per_block),
                        dim3(BlockSize),
                        0,
                        stream,
                        d_input,
                        d_output,
                        CompareOp());
                }
                else if constexpr(BenchmarkKind == benchmark_kinds::sort_pairs)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(sort_pairs_kernel<T, BlockSize, ItemsPerThread, CompareOp>),
                        dim3(items / items_per_block),
                        dim3(BlockSize),
                        0,
                        stream,
                        d_input,
                        d_output,
                        CompareOp());
                }
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

#define CREATE_BENCHMARK(T, BS, IPT) \
    executor.queue<merge_sort_benchmark<BenchmarkKind, T, BS, IPT>>()

#define BENCHMARK_TYPE(type, block)                                         \
    CREATE_BENCHMARK(type, block, 1), CREATE_BENCHMARK(type, block, 2),     \
        CREATE_BENCHMARK(type, block, 3), CREATE_BENCHMARK(type, block, 4), \
        CREATE_BENCHMARK(type, block, 8)

template<benchmark_kinds BenchmarkKind>
void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int, 64);
    BENCHMARK_TYPE(int, 128);
    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(int, 512);

    BENCHMARK_TYPE(int8_t, 64);
    BENCHMARK_TYPE(int8_t, 128);
    BENCHMARK_TYPE(int8_t, 256);
    BENCHMARK_TYPE(int8_t, 512);

    BENCHMARK_TYPE(uint8_t, 64);
    BENCHMARK_TYPE(uint8_t, 128);
    BENCHMARK_TYPE(uint8_t, 256);
    BENCHMARK_TYPE(uint8_t, 512);

    BENCHMARK_TYPE(int64_t, 64);
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

    add_benchmarks<benchmark_kinds::sort_keys>(executor);
    add_benchmarks<benchmark_kinds::sort_pairs>(executor);

    executor.run();
}
