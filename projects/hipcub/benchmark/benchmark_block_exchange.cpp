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

#include <hipcub/block/block_exchange.hpp>
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>

constexpr unsigned int Trials = 100;

template<class Runner, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* d_input, const unsigned int* d_ranks, T* d_output)
{
    Runner::template run<T, BlockSize, ItemsPerThread>(d_input, d_ranks, d_output);
}

struct blocked_to_striped
{
    static constexpr const char* name = "blocked_to_striped";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectBlocked(lid, d_input + block_offset, input);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.BlockedToStriped(input, input);
            __syncthreads(); // extra sync needed because of loop. In normal usage
                // sync with be cared for by the load and store functions
                // (outside the loop).
        }
        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct striped_to_blocked
{
    static constexpr const char* name = "striped_to_blocked";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.StripedToBlocked(input, input);
            __syncthreads(); // extra sync needed because of loop. In normal usage
                // sync with be cared for by the load and store functions
                // (outside the loop).
        }
        hipcub::StoreDirectBlocked(lid, d_output + block_offset, input);
    }
};

struct blocked_to_warp_striped
{
    static constexpr const char* name = "blocked_to_warp_striped";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectBlocked(lid, d_input + block_offset, input);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.BlockedToWarpStriped(input, input);
            __syncthreads(); // extra sync needed because of loop. In normal usage
                // sync with be cared for by the load and store functions
                // (outside the loop).
        }
        hipcub::StoreDirectWarpStriped(lid, d_output + block_offset, input);
    }
};

struct warp_striped_to_blocked
{
    static constexpr const char* name = "warp_striped_to_blocked";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectWarpStriped(lid, d_input + block_offset, input);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.WarpStripedToBlocked(input, input);
            __syncthreads(); // extra sync needed because of loop. In normal usage
                // sync with be cared for by the load and store functions
                // (outside the loop).
        }
        hipcub::StoreDirectBlocked(lid, d_output + block_offset, input);
    }
};

struct scatter_to_blocked
{
    static constexpr const char* name = "scatter_to_blocked";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* d_input, const unsigned int* d_ranks, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T            input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);
        hipcub::LoadDirectStriped<BlockSize>(lid, d_ranks + block_offset, ranks);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.ScatterToBlocked(input, input, ranks);
            __syncthreads(); // extra sync needed because of loop. In normal usage
                // sync with be cared for by the load and store functions
                // (outside the loop).
        }
        hipcub::StoreDirectBlocked(lid, d_output + block_offset, input);
    }
};

struct scatter_to_striped
{
    static constexpr const char* name = "scatter_to_striped";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* d_input, const unsigned int* d_ranks, T* d_output)
    {
        const unsigned int lid          = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T            input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);
        hipcub::LoadDirectStriped<BlockSize>(lid, d_ranks + block_offset, ranks);

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.ScatterToStriped(input, input, ranks);
            __syncthreads(); // extra sync needed because of loop. In normal usage
                // sync with be cared for by the load and store functions
                // (outside the loop).
        }
        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template<class Benchmark, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
class block_exchange_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_exchange")
            .add("subalgo", Benchmark::name)
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
        const auto     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        std::vector<T> input(items);

        // Fill input
        for(size_t i = 0; i < items; i++)
        {
            input[i] = T(i);
        }

        // Fill ranks (for scatter operations)
        std::vector<unsigned int> ranks(items);
        std::mt19937              gen;
        for(size_t bi = 0; bi < items / items_per_block; bi++)
        {
            auto block_ranks = ranks.begin() + bi * items_per_block;
            std::iota(block_ranks, block_ranks + items_per_block, 0);
            std::shuffle(block_ranks, block_ranks + items_per_block, gen);
        }
        T*            d_input;
        unsigned int* d_ranks;
        T*            d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_ranks, items * sizeof(unsigned int)));
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), items * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(d_ranks, ranks.data(), items * sizeof(unsigned int), hipMemcpyHostToDevice));
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
                                   d_ranks,
                                   d_output);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_ranks));
        HIP_CHECK(hipFree(d_output));
    }
};

#define CREATE_BENCHMARK(T, BS, IPT) \
    executor.queue<block_exchange_benchmark<Benchmark, T, BS, IPT>>()

#define BENCHMARK_TYPE(type, block)                                         \
    CREATE_BENCHMARK(type, block, 1), CREATE_BENCHMARK(type, block, 2),     \
        CREATE_BENCHMARK(type, block, 3), CREATE_BENCHMARK(type, block, 4), \
        CREATE_BENCHMARK(type, block, 7), CREATE_BENCHMARK(type, block, 8)

template<class Benchmark>
void add_benchmarks(primbench::executor& executor)
{

    BENCHMARK_TYPE(int, 256);
    BENCHMARK_TYPE(int8_t, 256);
    BENCHMARK_TYPE(int64_t, 256);
    BENCHMARK_TYPE(custom_float2, 256);
    BENCHMARK_TYPE(custom_double2, 256);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<blocked_to_striped>(executor);
    add_benchmarks<striped_to_blocked>(executor);
    add_benchmarks<blocked_to_warp_striped>(executor);
    add_benchmarks<warp_striped_to_blocked>(executor);
    add_benchmarks<scatter_to_blocked>(executor);
    add_benchmarks<scatter_to_striped>(executor);

    executor.run();
}
