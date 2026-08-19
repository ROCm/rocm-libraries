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

#include "common_benchmark_header.hpp"

// HIP API
#include <hipcub/warp/warp_exchange.hpp>

#include <type_traits>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__device__ auto warp_exchange_benchmark(T* d_output)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    T thread_data[ItemsPerThread];
#pragma unroll
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i] = static_cast<T>(i);
    }

    using WarpExchangeT                                           = ::hipcub::WarpExchange<T,
                                                 ItemsPerThread,
                                                 LogicalWarpSize,
                                                 1, // ARCH
                                                 Algorithm>;
    constexpr unsigned                             warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];
    const unsigned                                 warp_id = threadIdx.x / LogicalWarpSize;

    WarpExchangeT warp_exchange(temp_storage[warp_id]);
    Op{}(warp_exchange, thread_data);

#pragma unroll
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned global_idx = (BlockSize * blockIdx.x + threadIdx.x) * ItemsPerThread + i;
        d_output[global_idx]      = thread_data[i];
    }
}

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__device__ auto warp_exchange_benchmark(T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_exchange_kernel(T* d_output)
{
    warp_exchange_benchmark<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm, Op>(d_output);
}

template<class OffsetT,
         unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T>
__device__ auto warp_exchange_scatter_to_striped_benchmark(T* d_output)
    -> std::enable_if_t<benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{
    const unsigned warp_id = threadIdx.x / LogicalWarpSize;
    T              thread_data[ItemsPerThread];
    OffsetT        thread_ranks[ItemsPerThread];
#pragma unroll
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        thread_data[i]  = static_cast<T>(i);
        thread_ranks[i] = static_cast<OffsetT>(LogicalWarpSize - warp_id * ItemsPerThread - i - 1);
    }

    using WarpExchangeT = ::hipcub::WarpExchange<T, ItemsPerThread, LogicalWarpSize>;
    constexpr unsigned                             warps_in_block = BlockSize / LogicalWarpSize;
    __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_in_block];

    WarpExchangeT(temp_storage[warp_id]).ScatterToStriped(thread_data, thread_ranks);

#pragma unroll
    for(unsigned i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned striped_global_idx
            = BlockSize * ItemsPerThread * blockIdx.x + BlockSize * i + threadIdx.x;
        d_output[striped_global_idx] = thread_data[i];
    }
}

template<class OffsetT,
         unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T>
__device__ auto warp_exchange_scatter_to_striped_benchmark(T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<class OffsetT,
         unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T>
__global__ __launch_bounds__(BlockSize) void warp_exchange_scatter_to_striped_kernel(T* d_output)
{
    warp_exchange_scatter_to_striped_benchmark<OffsetT, BlockSize, ItemsPerThread, LogicalWarpSize>(
        d_output);
}

template<class T,
         unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned trials          = 100;
    constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
    const unsigned     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    T* d_output;
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < trials; ++i)
        {
            warp_exchange_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm, Op>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_output);
        }

        HIP_CHECK(hipPeekAtLastError())
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * trials * size);

    HIP_CHECK(hipFree(d_output));
}

template<class T,
         class OffsetT,
         unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize>
void run_benchmark_scatter_to_striped(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned trials          = 100;
    constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
    const unsigned     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    T* d_output;
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < trials; ++i)
        {
            warp_exchange_scatter_to_striped_kernel<OffsetT,
                                                    BlockSize,
                                                    ItemsPerThread,
                                                    LogicalWarpSize>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_output);
        }

        HIP_CHECK(hipPeekAtLastError())
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds
            = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * trials * size);

    HIP_CHECK(hipFree(d_output));
}

struct StripedToBlockedOp
{
    template<class WarpExchangeT, class T, unsigned ItemsPerThread>
    __device__ void operator()(WarpExchangeT& warp_exchange, T (&thread_data)[ItemsPerThread]) const
    {
        warp_exchange.StripedToBlocked(thread_data, thread_data);
    }
};

struct BlockedToStripedOp
{
    template<class WarpExchangeT, class T, unsigned ItemsPerThread>
    __device__ void operator()(WarpExchangeT& warp_exchange, T (&thread_data)[ItemsPerThread]) const
    {
        warp_exchange.BlockedToStriped(thread_data, thread_data);
    }
};

#define CREATE_BENCHMARK_STRIPED_TO_BLOCKED(T, BS, IT, WS, ALG)                                  \
    benchmark::RegisterBenchmark(std::string("warp_exchange_striped_to_blocked<data_type:" #T    \
                                             ",block_size:" #BS ",items_per_thread:" #IT         \
                                             ",warp_size:" #WS ",sub_algorithm_name:" #ALG ">.") \
                                     .c_str(),                                                   \
                                 &run_benchmark<T, BS, IT, WS, ALG, StripedToBlockedOp>,         \
                                 stream,                                                         \
                                 size)

#define CREATE_BENCHMARK_BLOCKED_TO_STRIPED(T, BS, IT, WS, ALG)                                  \
    benchmark::RegisterBenchmark(std::string("warp_exchange_blocked_to_striped<data_type:" #T    \
                                             ",block_size:" #BS ",items_per_thread:" #IT         \
                                             ",warp_size:" #WS ",sub_algorithm_name:" #ALG ">.") \
                                     .c_str(),                                                   \
                                 &run_benchmark<T, BS, IT, WS, ALG, BlockedToStripedOp>,         \
                                 stream,                                                         \
                                 size)

#define CREATE_BENCHMARK_SCATTER_TO_STRIPED(T, OFFSET_T, BS, IT, WS)                          \
    benchmark::RegisterBenchmark(std::string("warp_exchange_scatter_to_striped<data_type:" #T \
                                             ",offset_type:" #OFFSET_T ",block_size:" #BS     \
                                             ",items_per_thread:" #IT ",warp_size:" #WS ">.") \
                                     .c_str(),                                                \
                                 &run_benchmark_scatter_to_striped<T, OFFSET_T, BS, IT, WS>,  \
                                 stream,                                                      \
                                 size)

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");

    std::cout << "benchmark_warp_exchange" << std::endl;

    // HIP
    hipStream_t     stream = 0; // default
    hipDeviceProp_t devProp;
    int             device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM),
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 16),
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 32),
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 256, 4, 32),

        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE),

// CUB requires WS == IPT for WARP_EXCHANGE_SHUFFLE
#ifdef HIPCUB_ROCPRIM_API
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE),
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE),
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE),
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE),
#endif
    };

#ifdef HIPCUB_ROCPRIM_API
    if(::benchmark_utils::is_warp_size_supported(64))
    {
        std::vector<benchmark::internal::Benchmark*> additional_benchmarks{
            CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM),
            CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE),
            CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM),
            CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE),
            CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 64),

            CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM),
            CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE),
            CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM),
            CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE),
            CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 256, 4, 64)};
        benchmarks.insert(benchmarks.end(),
                          additional_benchmarks.begin(),
                          additional_benchmarks.end());
    }
#endif

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
