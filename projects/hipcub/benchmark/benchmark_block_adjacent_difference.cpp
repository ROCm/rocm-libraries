// MIT License
//
// Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/block/block_adjacent_difference.hpp>
#include <hipcub/block/block_load.hpp>
#include <hipcub/block/block_store.hpp>

constexpr unsigned int Trials = 100;

template<class Benchmark,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         WithTile,
         typename... Args>
__global__ __launch_bounds__(BlockSize)
void kernel(Args... args)
{
    Benchmark::template run<BlockSize, ItemsPerThread, WithTile>(args...);
}

template<class T>
struct minus
{
    HIPCUB_HOST_DEVICE
    inline constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

struct subtract_left
{
    static constexpr const char* name = "subtract_left";

    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

        hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

#pragma nounroll
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_difference.SubtractLeft(input, output, minus<T>{}, T(123));
            }
            else
            {
                adjacent_difference.SubtractLeft(input, output, minus<T>{});
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }

            __syncthreads();
        }

        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_left_partial_tile
{
    static constexpr const char* name = "subtract_left_partial_tile";

    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void run(const T* d_input, const int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

        hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

        int tile_size = tile_sizes[blockIdx.x];

        // Try to evenly distribute the length of tile_sizes between all the trials
        const auto tile_size_diff = (BlockSize * ItemsPerThread) / trials + 1;

#pragma nounroll
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];

            if(WithTile)
            {
                adjacent_difference.SubtractLeftPartialTile(input,
                                                            output,
                                                            minus<T>{},
                                                            tile_size,
                                                            T(123));
            }
            else
            {
                adjacent_difference.SubtractLeftPartialTile(input, output, minus<T>{}, tile_size);
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }

            // Change the tile_size to even out the distribution
            tile_size = (tile_size + tile_size_diff) % (BlockSize * ItemsPerThread);
            __syncthreads();
        }

        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_right
{
    static constexpr const char* name = "subtract_right";

    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void run(const T* d_input, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

        hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

#pragma nounroll
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];
            if(WithTile)
            {
                adjacent_difference.SubtractRight(input, output, minus<T>{}, T(123));
            }
            else
            {
                adjacent_difference.SubtractRight(input, output, minus<T>{});
            }

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }

            __syncthreads();
        }

        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct subtract_right_partial_tile
{
    static constexpr const char* name = "subtract_right_partial_tile";

    template<unsigned int BlockSize, unsigned int ItemsPerThread, bool WithTile, typename T>
    __device__
    static void run(const T* d_input, const int* tile_sizes, T* d_output, unsigned int trials)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

        hipcub::BlockAdjacentDifference<T, BlockSize> adjacent_difference;

        int tile_size = tile_sizes[blockIdx.x];

        // Try to evenly distribute the length of tile_sizes between all the trials
        const auto tile_size_diff = (BlockSize * ItemsPerThread) / trials + 1;

#pragma nounroll
        for(unsigned int trial = 0; trial < trials; trial++)
        {
            T output[ItemsPerThread];

            adjacent_difference.SubtractRightPartialTile(input, output, minus<T>{}, tile_size);

            for(unsigned int i = 0; i < ItemsPerThread; ++i)
            {
                input[i] += output[i];
            }

            // Change the tile_size to even out the distribution
            tile_size = (tile_size + tile_size_diff) % (BlockSize * ItemsPerThread);
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
class block_adjacent_difference_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_adjacent_difference")
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
        const auto     num_blocks      = (input_items + items_per_block - 1) / items_per_block;

        // Round up items to the next multiple of items_per_block
        const auto items = num_blocks * items_per_block;

        const std::vector<T> input = benchmark_utils::get_random_data<T>(items, T(0), T(10));
        T*                   d_input;
        T*                   d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(input[0]),
                            hipMemcpyHostToDevice));

        state.set_items(Trials * items);
        state.add_writes<T>(Trials * items);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(kernel<Benchmark, BlockSize, ItemsPerThread, WithTile>),
                    dim3(num_blocks),
                    dim3(BlockSize),
                    0,
                    stream,
                    d_input,
                    d_output,
                    Trials);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

template<class Benchmark,
         class T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         bool         WithTile>
class block_adjacent_difference_partial_tile_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_adjacent_difference")
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
        const auto     num_blocks      = (input_items + items_per_block - 1) / items_per_block;

        // Round up items to the next multiple of items_per_block
        const auto items = num_blocks * items_per_block;

        const std::vector<T>   input = benchmark_utils::get_random_data<T>(items, T(0), T(10));
        const std::vector<int> tile_sizes
            = benchmark_utils::get_random_data<int>(num_blocks, 0, items_per_block);

        T*   d_input;
        int* d_tile_sizes;
        T*   d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_tile_sizes, tile_sizes.size() * sizeof(tile_sizes[0])));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(input[0]),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_tile_sizes,
                            tile_sizes.data(),
                            tile_sizes.size() * sizeof(tile_sizes[0]),
                            hipMemcpyHostToDevice));

        state.set_items(Trials * items);
        state.add_writes<T>(Trials * items);

        state.run(
            [&]
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(kernel<Benchmark, BlockSize, ItemsPerThread, WithTile>),
                    dim3(num_blocks),
                    dim3(BlockSize),
                    0,
                    stream,
                    d_input,
                    d_tile_sizes,
                    d_output,
                    Trials);
            });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_tile_sizes));
        HIP_CHECK(hipFree(d_output));
    }
};

// or use block_adjacent_difference_partial_tile_benchmark
#define CREATE_BENCHMARK(T, BS, IPT, WITH_TILE)                                             \
    executor.queue<std::conditional_t<                                                      \
        is_partial,                                                                         \
        block_adjacent_difference_partial_tile_benchmark<Benchmark, T, BS, IPT, WITH_TILE>, \
        block_adjacent_difference_benchmark<Benchmark, T, BS, IPT, WITH_TILE>>>()

#define BENCHMARK_TYPE(type, block, with_tile)                                                    \
    CREATE_BENCHMARK(type, block, 1, with_tile), CREATE_BENCHMARK(type, block, 3, with_tile),     \
        CREATE_BENCHMARK(type, block, 4, with_tile), CREATE_BENCHMARK(type, block, 8, with_tile), \
        CREATE_BENCHMARK(type, block, 16, with_tile), CREATE_BENCHMARK(type, block, 32, with_tile)

template<class Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    constexpr bool is_partial = std::is_same_v<Benchmark, subtract_left_partial_tile>
                                || std::is_same_v<Benchmark, subtract_right_partial_tile>;

    BENCHMARK_TYPE(int, 256, false);
    BENCHMARK_TYPE(float, 256, false);
    BENCHMARK_TYPE(int8_t, 256, false);
    BENCHMARK_TYPE(int64_t, 256, false);
    BENCHMARK_TYPE(double, 256, false);

    if(!std::is_same<Benchmark, subtract_right_partial_tile>::value)
    {
        BENCHMARK_TYPE(int, 256, true);
        BENCHMARK_TYPE(float, 256, true);
        BENCHMARK_TYPE(int8_t, 256, true);
        BENCHMARK_TYPE(int64_t, 256, true);
        BENCHMARK_TYPE(double, 256, true);
    }
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<subtract_left>(executor);
    add_benchmarks<subtract_right>(executor);
    add_benchmarks<subtract_left_partial_tile>(executor);
    add_benchmarks<subtract_right_partial_tile>(executor);

    executor.run();
}
