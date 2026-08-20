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

#include <hipcub/block/block_shuffle.hpp>

constexpr unsigned int Trials = 100;

template<class Runner, class T, unsigned int BlockSize, unsigned int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* input, T* output)
{
    Runner::template run<T, BlockSize, ItemsPerThread>(input, output);
}

struct offset
{
    static constexpr const char* name = "offset";

    template<class T, unsigned int BlockSize, unsigned int /* ItemsPerThread */>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T value = input[tid];

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Offset(value, value, 1);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        output[tid] = value;
    }

    static constexpr bool uses_ipt = false;
};

struct rotate
{
    static constexpr const char* name = "rotate";

    template<class T, unsigned int BlockSize, unsigned int /* ItemsPerThread */>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T value = input[tid];

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Rotate(value, value, 1);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        output[tid] = value;
    }

    static constexpr bool uses_ipt = false;
};

struct up
{
    static constexpr const char* name = "up";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values[i] = input[ItemsPerThread * tid + i];
        }

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Up(values, values);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[ItemsPerThread * tid + i] = values[i];
        }
    }

    static constexpr bool uses_ipt = true;
};

struct down
{
    static constexpr const char* name = "down";

    template<class T, unsigned int BlockSize, unsigned int ItemsPerThread>
    __device__
    static void run(const T* input, T* output)
    {
        const unsigned int tid = hipBlockIdx_x * BlockSize + hipThreadIdx_x;

        T values[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values[i] = input[ItemsPerThread * tid + i];
        }

        using bshuffle_t = hipcub::BlockShuffle<T, BlockSize>;
        __shared__ typename bshuffle_t::TempStorage storage;

#pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            bshuffle_t(storage).Down(values, values);

            // sync is required because of loop since
            // temporary storage is accessed next iteration
            __syncthreads();
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[ItemsPerThread * tid + i] = values[i];
        }
    }

    static constexpr bool uses_ipt = true;
};

template<class Benchmark, class T, unsigned int BlockSize, unsigned int ItemsPerThread = 1>
class block_shuffle_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "block_shuffle")
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

#define CREATE_BENCHMARK_IPT(BS, IPT) \
    executor.queue<block_shuffle_benchmark<Benchmark, T, BS, IPT>>()

#define CREATE_BENCHMARK(BS) executor.queue<block_shuffle_benchmark<Benchmark, T, BS>>()

template<class Benchmark, class T, std::enable_if_t<Benchmark::uses_ipt, bool> = true>
void add_benchmarks_type(primbench::executor& executor)
{
    CREATE_BENCHMARK_IPT(256, 1);
    CREATE_BENCHMARK_IPT(256, 3);
    CREATE_BENCHMARK_IPT(256, 4);
    CREATE_BENCHMARK_IPT(256, 8);
    CREATE_BENCHMARK_IPT(256, 16);
    CREATE_BENCHMARK_IPT(256, 32);
}

template<class Benchmark, class T, std::enable_if_t<!Benchmark::uses_ipt, bool> = true>
void add_benchmarks_type(primbench::executor& executor)
{
    CREATE_BENCHMARK(256);
}

#define CREATE_BENCHMARKS(T) add_benchmarks_type<Benchmark, T>(executor)

template<class Benchmark>
void add_benchmarks(primbench::executor& executor)
{
    CREATE_BENCHMARKS(int);
    CREATE_BENCHMARKS(float);
    CREATE_BENCHMARKS(double);
    CREATE_BENCHMARKS(int8_t);
    CREATE_BENCHMARKS(int64_t);
    CREATE_BENCHMARKS(custom_float2);
    CREATE_BENCHMARKS(custom_double2);
}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    add_benchmarks<offset>(executor);
    add_benchmarks<rotate>(executor);
    add_benchmarks<up>(executor);
    add_benchmarks<down>(executor);

    executor.run();
}
