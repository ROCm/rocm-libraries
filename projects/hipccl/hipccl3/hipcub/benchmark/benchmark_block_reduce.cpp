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

#include <hipcub/block/block_reduce.hpp>
#include <hipcub/thread/thread_operators.hpp>

constexpr unsigned int Trials = 100;

template<class Runner, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread>(input, output);
}

template<hipcub::BlockReduceAlgorithm algorithm>
struct reduce
{
    static const char* get_algorithm_name()
    {
        switch(algorithm)
        {
            case hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING: return "block_reduce_raking";
            case hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY:
                return "block_reduce_raking_commutative_only";
            case hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS:
                return "block_reduce_warp_reductions";
        }

        return "unknown algorithm";
    }

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        T values[ItemsPerThread];
        T reduced_value;
        for(unsigned int k = 0; k < ItemsPerThread; k++)
        {
            values[k] = input[i * ItemsPerThread + k];
        }

        using breduce_t = hipcub::BlockReduce<T, BlockSize, algorithm>;
        __shared__ typename breduce_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            reduced_value = breduce_t(storage).Reduce(values, hipcub::Sum());
            values[0]     = reduced_value;
        }

        if(hipThreadIdx_x == 0)
        {
            output[hipBlockIdx_x] = reduced_value;
        }
    }
};

template<class Benchmark, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
class block_reduce_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_reduce")
            .add("subalgo", Benchmark::get_algorithm_name())
            .add("lvl", "block")
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        // Make sure size is a multiple of BlockSize
        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        // Allocate and fill memory
        std::vector<T> input(items, T(1));
        T*             d_input;
        T*             d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        state.set_items(Trials * items);
        state.add_writes<T>(Trials * items);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread>),
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

#define CREATE_BENCHMARK(T, BS, IPT) executor.queue<block_reduce_benchmark<Benchmark, T, BS, IPT>>()

#define BENCHMARK_TYPE(type, block)                                          \
    CREATE_BENCHMARK(type, block, 1), CREATE_BENCHMARK(type, block, 2),      \
        CREATE_BENCHMARK(type, block, 3), CREATE_BENCHMARK(type, block, 4),  \
        CREATE_BENCHMARK(type, block, 8), CREATE_BENCHMARK(type, block, 11), \
        CREATE_BENCHMARK(type, block, 16)

template<class Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int, 64);
    BENCHMARK_TYPE(float, 64);
    BENCHMARK_TYPE(double, 64);
    BENCHMARK_TYPE(int8_t, 64);
    BENCHMARK_TYPE(uint8_t, 64);

    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(float, 256);
    BENCHMARK_TYPE(double, 256);
    BENCHMARK_TYPE(int8_t, 256);
    BENCHMARK_TYPE(uint8_t, 256);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                    = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch    = 100;
    settings.noise_tolerance_percent = 2;

    primbench::executor executor(argc, argv, settings);

    // using_warp_scan
    using reduce_uwr_t = reduce<hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;
    add_benchmarks<reduce_uwr_t>(executor);

    // raking reduce
    using reduce_rr_t = reduce<hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>;
    add_benchmarks<reduce_rr_t>(executor);

    // raking reduce commutative only
    using reduce_rrco_t
        = reduce<hipcub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    add_benchmarks<reduce_rrco_t>(executor);

    executor.run();
}
