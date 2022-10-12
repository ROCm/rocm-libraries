// MIT License
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include "hipcub/warp/warp_exchange.hpp"


#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<
    class T,
    unsigned BlockSize,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize,
    template<class, unsigned, unsigned> class Op
>
__global__
__launch_bounds__(BlockSize)
void warp_exchange_kernel(T* d_output)
{
    T thread_data[ItemsPerThread];
    #pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = static_cast<T>(i);
    }

    using WarpExchangeT = ::hipcub::WarpExchange<
        T,
        ItemsPerThread,
        ::benchmark_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
    >;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];
    const unsigned warp_id = hipThreadIdx_x / LogicalWarpSize;

    WarpExchangeT warp_exchange(temp_storage[warp_id]);
    Op<
        T,
        ItemsPerThread,
        ::benchmark_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
    >{}(warp_exchange, thread_data);

    #pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned global_idx =
            (BlockSize * hipBlockIdx_x + hipThreadIdx_x) * ItemsPerThread + i;
        d_output[global_idx] = thread_data[i];
    }
}

template<
    class T,
    class OffsetT,
    unsigned BlockSize,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize
>
__global__
__launch_bounds__(BlockSize)
void warp_exchange_scatter_to_striped_kernel(T* d_output)
{
    const unsigned warp_id = hipThreadIdx_x / LogicalWarpSize;
    T thread_data[ItemsPerThread];
    OffsetT thread_ranks[ItemsPerThread];
    #pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = static_cast<T>(i);
        thread_ranks[i] = static_cast<OffsetT>(LogicalWarpSize - warp_id * ItemsPerThread - i - 1);
    }

    using WarpExchangeT = ::hipcub::WarpExchange<
        T,
        ItemsPerThread,
        ::benchmark_utils::DeviceSelectWarpSize<LogicalWarpSize>::value
    >;
    constexpr unsigned warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];

    WarpExchangeT(temp_storage[warp_id]).ScatterToStriped(thread_data, thread_ranks);

    #pragma unroll
    for (unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned striped_global_idx =
            BlockSize * ItemsPerThread * hipBlockIdx_x + BlockSize * i + hipThreadIdx_x;
        d_output[striped_global_idx] = thread_data[i];
    }
}

template<
    class T,
    unsigned BlockSize,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize,
    template<class, unsigned, unsigned> class Op
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned trials = 100;
    constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
    const unsigned size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    T * d_output;
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < trials; ++i)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(warp_exchange_kernel<
                    T,
                    BlockSize,
                    ItemsPerThread,
                    LogicalWarpSize,
                    Op
                    >
                ),
                dim3(size / items_per_block), dim3(BlockSize), 0, stream, d_output
            );
        }

        HIP_CHECK(hipPeekAtLastError())
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * trials * size);

    HIP_CHECK(hipFree(d_output));
}

template<
    class T,
    class OffsetT,
    unsigned BlockSize,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize
>
void run_benchmark_scatter_to_striped(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned trials = 100;
    constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
    const unsigned size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    T * d_output;
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < trials; ++i)
        {
            hipLaunchKernelGGL(
              HIP_KERNEL_NAME(warp_exchange_scatter_to_striped_kernel<
                    T,
                    OffsetT,
                    BlockSize,
                    ItemsPerThread,
                    LogicalWarpSize
                    >
                ),
                dim3(size / items_per_block), dim3(BlockSize), 0, stream, d_output
            );
        }

        HIP_CHECK(hipPeekAtLastError())
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * trials * size);

    HIP_CHECK(hipFree(d_output));
}

template<
    class T,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize
>
struct StripedToBlockedOp
{
    __device__ void operator()(
        ::hipcub::WarpExchange<T, ItemsPerThread, LogicalWarpSize> &warp_exchange,
        T (&thread_data)[ItemsPerThread]
    ) const
    {
        warp_exchange.StripedToBlocked(thread_data, thread_data);
    }
};

template<
    class T,
    unsigned ItemsPerThread,
    unsigned LogicalWarpSize
>
struct BlockedToStripedOp
{
    __device__ void operator()(
        ::hipcub::WarpExchange<T, ItemsPerThread, LogicalWarpSize> &warp_exchange,
        T (&thread_data)[ItemsPerThread]
    ) const
    {
        warp_exchange.BlockedToStriped(thread_data, thread_data);
    }
};

#define CREATE_BENCHMARK_STRIPED_TO_BLOCKED(T, BS, IT, WS) \
benchmark::RegisterBenchmark( \
    "warp_exchange_striped_to_blocked<Datatype:" #T ",Block Size:" #BS ",Items Per Thread:" #IT ",Warp Size:" #WS ">.", \
    &run_benchmark<T, BS, IT, WS, StripedToBlockedOp>, \
    stream, size \
)

#define CREATE_BENCHMARK_BLOCKED_TO_STRIPED(T, BS, IT, WS) \
benchmark::RegisterBenchmark( \
    "warp_exchange_blocked_to_striped<Datatype:" #T ",Block Size:" #BS ",Items Per Thread:" #IT ",Warp Size:" #WS ">.", \
    &run_benchmark<T, BS, IT, WS, BlockedToStripedOp>, \
    stream, size \
)


#define CREATE_BENCHMARK_SCATTER_TO_STRIPED(T, OFFSET_T, BS, IT, WS) \
benchmark::RegisterBenchmark( \
    "warp_exchange_scatter_to_striped<Datatype:" #T ",Offset Type:" #OFFSET_T ",Block Size:" #BS ",Items Per Thread:" #IT ",Warp Size:" #WS ">.", \
    &run_benchmark_scatter_to_striped<T, OFFSET_T, BS, IT, WS>, \
    stream, size \
)

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

    std::cout << "benchmark_warp_exchange" << std::endl;

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 16),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 16),
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 16),

        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 32),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 32),
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 32),

        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 32),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 32),
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 256, 4, 32),
    };

    if (::benchmark_utils::is_warp_size_supported(64))
    {
        std::vector<benchmark::internal::Benchmark*> additional_benchmarks{
            CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 64),
            CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 64),
            CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 64),

            CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 64),
            CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 64),
            CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 256, 4, 64)
        };
        benchmarks.insert(
            benchmarks.end(),
            additional_benchmarks.begin(),
            additional_benchmarks.end()
        );
    }

    // Use manual timing
    for (auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if (trials > 0)
    {
        for (auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
