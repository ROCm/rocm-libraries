// MIT License
//
// Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/block/block_histogram.hpp>

constexpr unsigned int Trials = 100;

template<class Runner,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int BinSize>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, BinSize>(input, output);
}

template<hipcub::BlockHistogramAlgorithm algorithm>
struct histogram
{
    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int BinSize>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int index = ((hipBlockIdx_x * BlockSize) + hipThreadIdx_x) * ItemsPerThread;
        unsigned int       global_offset = hipBlockIdx_x * BinSize;

        T values[ItemsPerThread];
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[index + k];
        }

        using bhistogram_t
            = hipcub::BlockHistogram<T, BlockSize, ItemsPerThread, BinSize, algorithm>;
        __shared__
        T                                  histogram[BinSize];
        __shared__
        typename bhistogram_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bhistogram_t(storage).Histogram(values, histogram);
        }

#pragma unroll
        for(unsigned int offset = 0; offset < BinSize; offset += BlockSize)
        {
            if(offset + hipThreadIdx_x < BinSize)
            {
                output[global_offset + hipThreadIdx_x] = histogram[offset + hipThreadIdx_x];
                global_offset += BlockSize;
            }
        }
    }
};

using histogram_a_t = histogram<hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_ATOMIC>;
using histogram_s_t = histogram<hipcub::BlockHistogramAlgorithm::BLOCK_HISTO_SORT>;

template<class T>
struct histogram_algorithm_name;

template<>
struct histogram_algorithm_name<histogram_a_t>
{
    static constexpr const char* value = "using_atomic";
};

template<>
struct histogram_algorithm_name<histogram_s_t>
{
    static constexpr const char* value = "using_sort";
};

template<class Benchmark,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int BinSize = BlockSize>
class block_histogram_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_histogram")
            .add("subalgo", histogram_algorithm_name<Benchmark>::value)
            .add("lvl", "block")
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread);

        //  BinSize is always equal to BlockSize
        // .add("bin_size", BinSize);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        // Make sure size is a multiple of BlockSize
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);
        const auto bin_size = BinSize * ((input_items + items_per_block - 1) / items_per_block);

        // Allocate and fill memory
        std::vector<T> input(items, 0.0f);
        T*             d_input;
        T*             d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, bin_size * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(Trials * items);
        state.add_writes<T>(Trials * items);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, BinSize>),
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

#define CREATE_BENCHMARK(T, BS, IPT) \
    executor.queue<block_histogram_benchmark<Benchmark, T, BS, IPT>>()

#define BENCHMARK_TYPE(type, block)                                         \
    CREATE_BENCHMARK(type, block, 1), CREATE_BENCHMARK(type, block, 2),     \
        CREATE_BENCHMARK(type, block, 3), CREATE_BENCHMARK(type, block, 4), \
        CREATE_BENCHMARK(type, block, 8), CREATE_BENCHMARK(type, block, 16)

template<class Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(int, 320);
    BENCHMARK_TYPE(int, 512);

    BENCHMARK_TYPE(unsigned long long, 256);
    BENCHMARK_TYPE(unsigned long long, 320);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<histogram_a_t>(executor);
    add_benchmarks<histogram_s_t>(executor);

    executor.run();
}
