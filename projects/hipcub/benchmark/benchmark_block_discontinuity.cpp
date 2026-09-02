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

#include <hipcub/block/block_discontinuity.hpp>
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>
#include <hipcub/thread/thread_operators.hpp> //to use hipcub::Equality

constexpr unsigned int Trials = 100;

template<class T>
struct custom_flag_op1
{
    HIPCUB_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        return (a == b);
    }
};

template<class Runner, class T, unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* d_input, T* d_output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, WithTile>(d_input, d_output);
}

struct flag_heads
{
    static constexpr const char* name = "flag_heads";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile>
    __device__
    static void run(const T* d_input, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockDiscontinuity<T, BlockSize> bdiscontinuity;
            bool                                     head_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.FlagHeads(head_flags, input, hipcub::Equality(), T(123));
            }
            else
            {
                bdiscontinuity.FlagHeads(head_flags, input, hipcub::Equality());
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] += head_flags[i];
            }
            __syncthreads();
        }
        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct flag_tails
{
    static constexpr const char* name = "flag_tails";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile>
    __device__
    static void run(const T* d_input, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockDiscontinuity<T, BlockSize> bdiscontinuity;
            bool                                     tail_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.FlagTails(tail_flags, input, hipcub::Equality(), T(123));
            }
            else
            {
                bdiscontinuity.FlagTails(tail_flags, input, hipcub::Equality());
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] += tail_flags[i];
            }
            __syncthreads();
        }
        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct flag_heads_and_tails
{
    static constexpr const char* name = "flag_heads_and_tails";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile>
    __device__
    static void run(const T* d_input, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockDiscontinuity<T, BlockSize> bdiscontinuity;
            bool                                     head_flags[ItemsPerThread];
            bool                                     tail_flags[ItemsPerThread];
            if(WithTile)
            {
                bdiscontinuity.FlagHeadsAndTails(head_flags,
                                                 T(123),
                                                 tail_flags,
                                                 T(234),
                                                 input,
                                                 hipcub::Equality());
            } else
            {
                bdiscontinuity.FlagHeadsAndTails(head_flags, tail_flags, input, hipcub::Equality());
            }

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] += head_flags[i];
                input[i] += tail_flags[i];
            }
            __syncthreads();
        }
        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template<class Benchmark,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         WithTile>
class block_discontinuity_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_discontinuity")
            .add("subalgo", Benchmark::name)
            .add("lvl", "block")
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread)
            .add("with_tile", WithTile);
    }

    void run(primbench::state& state) override
    {
        const size_t input_items = state.size;
        const auto&  stream      = state.stream;

        constexpr auto items_per_block = BlockSize * ItemsPerThread;
        const auto     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        std::vector<T> input = benchmark_utils::get_random_data<T>(items, T(0), T(10));
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
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, WithTile>),
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

#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE) \
    executor.queue<block_discontinuity_benchmark<Benchmark, T, BS, IPT, WITH_TILE>>()

#define BENCHMARK_TYPE(type, block, bool)                                               \
    CREATE_BENCHMARK(type, block, 1, bool), CREATE_BENCHMARK(type, block, 2, bool),     \
        CREATE_BENCHMARK(type, block, 3, bool), CREATE_BENCHMARK(type, block, 4, bool), \
        CREATE_BENCHMARK(type, block, 8, bool)

template<class Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    BENCHMARK_TYPE(int, 256, false);
    BENCHMARK_TYPE(int, 256, true);
    BENCHMARK_TYPE(int8_t, 256, false);
    BENCHMARK_TYPE(int8_t, 256, true);
    BENCHMARK_TYPE(uint8_t, 256, false);
    BENCHMARK_TYPE(uint8_t, 256, true);
    BENCHMARK_TYPE(int64_t, 256, false);
    BENCHMARK_TYPE(int64_t, 256, true);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 1000;
    settings.batch_window_size    = 3;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<flag_heads>(executor);
    add_benchmarks<flag_tails>(executor);
    add_benchmarks<flag_heads_and_tails>(executor);

    executor.run();
}
