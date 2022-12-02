// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>

#include <vector>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"

// HIP API
#include <hip/hip_runtime.h>
#include <rocprim/warp/warp_exchange.hpp>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

struct BlockedToStripedOp
{
    template<
        class warp_exchange_type,
        class T,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&items)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage)
    {
        warp_exchange.blocked_to_striped(items, items, storage);
    }
};

struct StripedToBlockedOp
{
    template<
        class warp_exchange_type,
        class T,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&items)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage)
    {
        warp_exchange.striped_to_blocked(items, items, storage);
    }
};

struct BlockedToStripedShuffleOp
{
    template<
        class warp_exchange_type,
        class T,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&items)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/)
    {
        warp_exchange.blocked_to_striped_shuffle(items, items);
    }
};

struct StripedToBlockedShuffleOp
{
    template<
        class warp_exchange_type,
        class T,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&items)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& /*storage*/)
    {
        warp_exchange.striped_to_blocked_shuffle(items, items);
    }
};

struct ScatterToStripedOp
{
    template<
        class T,
        class OffsetT,
        class warp_exchange_type,
        unsigned int ItemsPerThread
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    const OffsetT (&ranks)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage)
    {
        warp_exchange.scatter_to_striped(thread_data, thread_data, ranks, storage);
    }
};

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int LogicalWarpSize,
    class Op
>
__global__
__launch_bounds__(BlockSize)
auto warp_exchange_kernel(T* d_output, unsigned int trials) -> typename std::enable_if<!std::is_same<Op, ScatterToStripedOp>::value, void>::type
{
    T thread_data[ItemsPerThread];

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        // generate unique value each data-element
        thread_data[i] = static_cast<T>(hipThreadIdx_x*ItemsPerThread+i);
    }

    using warp_exchange_type = ::rocprim::warp_exchange<
        T,
        ItemsPerThread,
        DeviceSelectWarpSize<LogicalWarpSize>::value
    >;
    constexpr unsigned int warps_in_block = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = hipThreadIdx_x / LogicalWarpSize;
    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage[warps_in_block];

    ROCPRIM_NO_UNROLL
    for(unsigned int i = 0; i < trials; i++)
    {
        Op{}(warp_exchange_type(), thread_data, storage[warp_id]);
        ::rocprim::wave_barrier();
    }

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const unsigned int global_idx =
            (BlockSize * hipBlockIdx_x + hipThreadIdx_x) * ItemsPerThread + i;
        d_output[global_idx] = thread_data[i];
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int LogicalWarpSize,
    class Op
    >
__global__
__launch_bounds__(BlockSize)
    auto warp_exchange_kernel(T* d_output, unsigned int trials) -> typename std::enable_if<std::is_same<Op, ScatterToStripedOp>::value, void>::type
{
    T thread_data[ItemsPerThread];
    unsigned int thread_ranks[ItemsPerThread];
    constexpr unsigned int warps_in_block = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = hipThreadIdx_x / LogicalWarpSize;
    const unsigned int lane_id = hipThreadIdx_x % LogicalWarpSize;

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        // generate unique value each data-element
        thread_data[i] = static_cast<T>(hipThreadIdx_x*ItemsPerThread+i);
        // generate unique destination location for each data-element
        const unsigned int s_lane_id = i % 2 == 0 ? LogicalWarpSize - 1 - lane_id : lane_id;
        thread_ranks[i] = s_lane_id*ItemsPerThread+i; // scatter values in warp across whole storage
    }

    using warp_exchange_type = ::rocprim::warp_exchange<
        T,
        ItemsPerThread,
        DeviceSelectWarpSize<LogicalWarpSize>::value
        >;
    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage[warps_in_block];

    ROCPRIM_NO_UNROLL
    for(unsigned int i = 0; i < trials; i++)
    {
        Op{}(warp_exchange_type(), thread_data, thread_ranks, storage[warp_id]);
        ::rocprim::wave_barrier();
    }

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        const unsigned int global_idx =
            (BlockSize * hipBlockIdx_x + hipThreadIdx_x) * ItemsPerThread + i;
        d_output[global_idx] = thread_data[i];
    }
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int LogicalWarpSize,
    class Op
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr unsigned int trials = 200;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    T * d_output;
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_exchange_kernel<
                    T,
                    BlockSize,
                    ItemsPerThread,
                    LogicalWarpSize,
                    Op
                >
            ),
            dim3(size / items_per_block), dim3(BlockSize), 0, stream,
            d_output, trials
        );

        HIP_CHECK(hipPeekAtLastError());

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * trials * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * trials * size);

    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(T, BS, IT, WS, OP)                                                       \
    benchmark::RegisterBenchmark(bench_naming::format_name("{lvl:warp,algo:exchange,key_type:" #T \
                                                           ",operation:" #OP ",ws:" #WS           \
                                                           ",cfg:{bs:" #BS ",ipt:" #IT "}}")      \
                                     .c_str(),                                                    \
                                 &run_benchmark<T, BS, IT, WS, OP>,                               \
                                 stream,                                                          \
                                 size)

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{
        CREATE_BENCHMARK(int, 256,  1, 16, BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256,  1, 32, BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256,  4, 16, BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256,  4, 32, BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 16, BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 32, BlockedToStripedOp),

        CREATE_BENCHMARK(int, 256,  1, 16, StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256,  1, 32, StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256,  4, 16, StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256,  4, 32, StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 16, 16, StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 16, 32, StripedToBlockedOp),

        CREATE_BENCHMARK(int, 256,  1, 16, BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256,  1, 32, BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256,  4, 16, BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256,  4, 32, BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 16, BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 32, BlockedToStripedShuffleOp),

        CREATE_BENCHMARK(int, 256,  1, 16, StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256,  1, 32, StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256,  4, 16, StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256,  4, 32, StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 16, StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 32, StripedToBlockedShuffleOp),

        CREATE_BENCHMARK(int, 256,  1, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256,  1, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256,  4, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256,  4, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 32, ScatterToStripedOp)
    };

    if(is_warp_size_supported(64))
    {
        std::vector<benchmark::internal::Benchmark*> additional_benchmarks{
            CREATE_BENCHMARK(int, 256,  1, 64, BlockedToStripedOp),
            CREATE_BENCHMARK(int, 256,  4, 64, BlockedToStripedOp),
            CREATE_BENCHMARK(int, 256, 16, 64, BlockedToStripedOp),

            CREATE_BENCHMARK(int, 256,  1, 64, StripedToBlockedOp),
            CREATE_BENCHMARK(int, 256,  4, 64, StripedToBlockedOp),
            CREATE_BENCHMARK(int, 256, 16, 64, StripedToBlockedOp),

            CREATE_BENCHMARK(int, 256,  1, 64, BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(int, 256,  4, 64, BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(int, 256, 16, 64, BlockedToStripedShuffleOp),

            CREATE_BENCHMARK(int, 256,  1, 64, StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(int, 256,  4, 64, StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(int, 256, 16, 64, StripedToBlockedShuffleOp),

            CREATE_BENCHMARK(int, 256,  1, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(int, 256,  4, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(int, 256, 16, 64, ScatterToStripedOp)
        };
        benchmarks.insert(
            benchmarks.end(),
            additional_benchmarks.begin(),
            additional_benchmarks.end()
        );
    }

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
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
