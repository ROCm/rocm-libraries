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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
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
#include "hipcub/block/block_exchange.hpp"
#include "hipcub/block/block_load.hpp"
#include "hipcub/block/block_store.hpp"


#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<
    class Runner,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials
>
__global__
__launch_bounds__(BlockSize, HIPCUB_DEFAULT_MIN_WARPS_PER_EU)
void kernel(const T * d_input, const unsigned int * d_ranks, T * d_output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(d_input, d_ranks, d_output);
}

struct blocked_to_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectBlocked(lid, d_input + block_offset, input);


        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.BlockedToStriped(input, input);
            __syncthreads(); // extra sync needed because of loop. In normal usage sync with be cared for by the load and store functions (outside the loop).
        }
        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct striped_to_blocked
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.StripedToBlocked(input, input);
            __syncthreads();// extra sync needed because of loop. In normal usage sync with be cared for by the load and store functions (outside the loop).
        }
        hipcub::StoreDirectBlocked(lid, d_output + block_offset, input);
    }
};

struct blocked_to_warp_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectBlocked(lid, d_input + block_offset, input);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.BlockedToWarpStriped(input, input);
            __syncthreads();// extra sync needed because of loop. In normal usage sync with be cared for by the load and store functions (outside the loop).
        }
        hipcub::StoreDirectWarpStriped(lid, d_output + block_offset, input);
    }
};

struct warp_striped_to_blocked
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int *, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        hipcub::LoadDirectWarpStriped(lid, d_input + block_offset, input);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.WarpStripedToBlocked(input, input);
            __syncthreads(); // extra sync needed because of loop. In normal usage sync with be cared for by the load and store functions (outside the loop).
        }
        hipcub::StoreDirectBlocked(lid, d_output + block_offset, input);
    }
};

struct scatter_to_blocked
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int * d_ranks, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);
        hipcub::LoadDirectStriped<BlockSize>(lid, d_ranks + block_offset, ranks);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.ScatterToBlocked(input, input, ranks);
            __syncthreads();// extra sync needed because of loop. In normal usage sync with be cared for by the load and store functions (outside the loop).
        }
        hipcub::StoreDirectBlocked(lid, d_output + block_offset, input);
    }
};

struct scatter_to_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __device__
    static void run(const T * d_input, const unsigned int * d_ranks, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        hipcub::LoadDirectStriped<BlockSize>(lid, d_input + block_offset, input);
        hipcub::LoadDirectStriped<BlockSize>(lid, d_ranks + block_offset, ranks);

        #pragma nounroll
        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            hipcub::BlockExchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.ScatterToStriped(input, input, ranks);
            __syncthreads(); // extra sync needed because of loop. In normal usage sync with be cared for by the load and store functions (outside the loop).
        }
        hipcub::StoreDirectStriped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template<
    class Benchmark,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

    std::vector<T> input(size);
    // Fill input
    for(size_t i = 0; i < size; i++)
    {
        input[i] = T(i);
    }
    std::vector<unsigned int> ranks(size);
    // Fill ranks (for scatter operations)
    std::mt19937 gen;
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        auto block_ranks = ranks.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, gen);
    }
    T * d_input;
    unsigned int * d_ranks;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_ranks, size * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            d_ranks, ranks.data(),
            size * sizeof(unsigned int),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>),
            dim3(size/items_per_block), dim3(BlockSize), 0, stream,
            d_input, d_ranks, d_output
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * Trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * Trials * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_ranks));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IPT) \
benchmark::RegisterBenchmark( \
    (std::string("block_exchange<" #T ", " #BS ", " #IPT ">.") + name).c_str(), \
    run_benchmark<Benchmark, T, BS, IPT>, \
    stream, size \
)

#define BENCHMARK_TYPE(type, block) \
    CREATE_BENCHMARK(type, block, 1), \
    CREATE_BENCHMARK(type, block, 2), \
    CREATE_BENCHMARK(type, block, 3), \
    CREATE_BENCHMARK(type, block, 4), \
    CREATE_BENCHMARK(type, block, 7), \
    CREATE_BENCHMARK(type, block, 8)

template<class Benchmark>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    using custom_float2 = benchmark_utils::custom_type<float, float>;
    using custom_double2 = benchmark_utils::custom_type<double, double>;

    std::vector<benchmark::internal::Benchmark*> bs =
    {
        BENCHMARK_TYPE(int, 256),
        BENCHMARK_TYPE(int8_t, 256),
        BENCHMARK_TYPE(long long, 256),
        BENCHMARK_TYPE(custom_float2, 256),
        BENCHMARK_TYPE(custom_double2, 256),
    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<blocked_to_striped>("blocked_to_striped", benchmarks, stream, size);
    add_benchmarks<striped_to_blocked>("striped_to_blocked", benchmarks, stream, size);
    add_benchmarks<blocked_to_warp_striped>("blocked_to_warp_striped", benchmarks, stream, size);
    add_benchmarks<warp_striped_to_blocked>("warp_striped_to_blocked", benchmarks, stream, size);
    add_benchmarks<scatter_to_blocked>("scatter_to_blocked", benchmarks, stream, size);
    add_benchmarks<scatter_to_striped>("scatter_to_striped", benchmarks, stream, size);

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
