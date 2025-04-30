// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"
// CmdParser
#include "cmdparser.hpp"

#include "../common/utils.hpp"
#include "../common/warp_exchange.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>
#include <rocprim/warp/warp_exchange.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#ifndef DEFAULT_BYTES
const size_t DEFAULT_BYTES = 1024 * 1024 * 32 * 4;
#endif

struct ScatterToStripedOp
{
    template<typename T, typename OffsetT, typename warp_exchange_type, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void operator()(warp_exchange_type warp_exchange,
                    T (&thread_data)[ItemsPerThread],
                    const OffsetT (&ranks)[ItemsPerThread],
                    typename warp_exchange_type::storage_type& storage) const
    {
        warp_exchange.scatter_to_striped(thread_data, thread_data, ranks, storage);
    }
};

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int LogicalWarpSize,
         typename Op,
         typename T>
__device__
auto warp_exchange_benchmark(T* d_output, unsigned int trials)
    -> std::enable_if_t<common::device_test_enabled_for_warp_size_v<LogicalWarpSize>
                        && !std::is_same<Op, ScatterToStripedOp>::value>
{
    T thread_data[ItemsPerThread];

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        // generate unique value each data-element
        thread_data[i] = static_cast<T>(threadIdx.x * ItemsPerThread + i);
    }

    using warp_exchange_type = ::rocprim::warp_exchange<T, ItemsPerThread, LogicalWarpSize>;
    constexpr unsigned int warps_in_block = BlockSize / LogicalWarpSize;
    const unsigned int warp_id = threadIdx.x / LogicalWarpSize;
    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage[warps_in_block];

    ROCPRIM_NO_UNROLL
    for(unsigned int i = 0; i < trials; ++i)
    {
        Op{}(warp_exchange_type(), thread_data, storage[warp_id]);
        ::rocprim::wave_barrier();
    }

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned int global_idx = (BlockSize * blockIdx.x + threadIdx.x) * ItemsPerThread + i;
        d_output[global_idx]          = thread_data[i];
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int LogicalWarpSize,
         typename Op,
         typename T>
__device__
auto warp_exchange_benchmark(T* d_output, unsigned int trials)
    -> std::enable_if_t<common::device_test_enabled_for_warp_size_v<LogicalWarpSize>
                        && std::is_same<Op, ScatterToStripedOp>::value>
{
    T                      thread_data[ItemsPerThread];
    unsigned int           thread_ranks[ItemsPerThread];
    constexpr unsigned int warps_in_block = BlockSize / LogicalWarpSize;
    const unsigned int     warp_id        = threadIdx.x / LogicalWarpSize;
    const unsigned int     lane_id        = threadIdx.x % LogicalWarpSize;

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        // generate unique value each data-element
        thread_data[i] = static_cast<T>(threadIdx.x * ItemsPerThread + i);
        // generate unique destination location for each data-element
        const unsigned int s_lane_id = i % 2 == 0 ? LogicalWarpSize - 1 - lane_id : lane_id;
        thread_ranks[i]
            = s_lane_id * ItemsPerThread + i; // scatter values in warp across whole storage
    }

    using warp_exchange_type = ::rocprim::warp_exchange<T, ItemsPerThread, LogicalWarpSize>;
    ROCPRIM_SHARED_MEMORY typename warp_exchange_type::storage_type storage[warps_in_block];

    ROCPRIM_NO_UNROLL
    for(unsigned int i = 0; i < trials; ++i)
    {
        Op{}(warp_exchange_type(), thread_data, thread_ranks, storage[warp_id]);
        ::rocprim::wave_barrier();
    }

    ROCPRIM_UNROLL
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
    {
        const unsigned int global_idx = (BlockSize * blockIdx.x + threadIdx.x) * ItemsPerThread + i;
        d_output[global_idx]          = thread_data[i];
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int LogicalWarpSize,
         typename Op,
         typename T>
__device__
auto warp_exchange_benchmark(T* /*d_output*/, unsigned int /*trials*/)
    -> std::enable_if_t<!common::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int LogicalWarpSize,
         typename Op,
         typename T>
__global__ __launch_bounds__(BlockSize)
void warp_exchange_kernel(T* d_output, unsigned int trials)
{
    warp_exchange_benchmark<BlockSize, ItemsPerThread, LogicalWarpSize, Op>(d_output, trials);
}

template<typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int LogicalWarpSize,
         typename Op>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t bytes)
{
    // Calculate the number of elements
    size_t N = bytes / sizeof(T);

    constexpr unsigned int trials          = 200;
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    T* d_output;
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        warp_exchange_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Op>
            <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_output, trials);

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
                                 bytes)

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t bytes  = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("bytes", std::to_string(bytes));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{
        CREATE_BENCHMARK(int, 256, 1, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 1, 32, common::BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 4, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 4, 32, common::BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 32, common::BlockedToStripedOp),
        CREATE_BENCHMARK(int, 256, 32, 32, common::BlockedToStripedOp),

        CREATE_BENCHMARK(int, 256, 1, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 1, 32, common::StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 4, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 4, 32, common::StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 16, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 16, 32, common::StripedToBlockedOp),
        CREATE_BENCHMARK(int, 256, 32, 32, common::StripedToBlockedOp),

        CREATE_BENCHMARK(int, 256, 1, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 1, 32, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 4, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 4, 32, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 32, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(int, 256, 32, 32, common::BlockedToStripedShuffleOp),

        CREATE_BENCHMARK(int, 256, 1, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 1, 32, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 4, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 4, 32, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 16, 32, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(int, 256, 32, 32, common::StripedToBlockedShuffleOp),

        CREATE_BENCHMARK(int, 256, 1, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256, 1, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256, 4, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256, 4, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(int, 256, 16, 32, ScatterToStripedOp),

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::BlockedToStripedOp),

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::StripedToBlockedOp),

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::BlockedToStripedShuffleOp),

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::StripedToBlockedShuffleOp),

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, ScatterToStripedOp),

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::BlockedToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::BlockedToStripedOp),

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::StripedToBlockedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::StripedToBlockedOp),

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::BlockedToStripedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::BlockedToStripedShuffleOp),

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::StripedToBlockedShuffleOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::StripedToBlockedShuffleOp),

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, ScatterToStripedOp),
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, ScatterToStripedOp)};

    int hip_device = 0;
    HIP_CHECK(::rocprim::detail::get_device_from_stream(stream, hip_device));
    if(is_warp_size_supported(64, hip_device))
    {
        std::vector<benchmark::internal::Benchmark*> additional_benchmarks{
            CREATE_BENCHMARK(int, 256, 1, 64, common::BlockedToStripedOp),
            CREATE_BENCHMARK(int, 256, 4, 64, common::BlockedToStripedOp),
            CREATE_BENCHMARK(int, 256, 16, 64, common::BlockedToStripedOp),
            CREATE_BENCHMARK(int, 256, 64, 64, common::BlockedToStripedOp),

            CREATE_BENCHMARK(int, 256, 1, 64, common::StripedToBlockedOp),
            CREATE_BENCHMARK(int, 256, 4, 64, common::StripedToBlockedOp),
            CREATE_BENCHMARK(int, 256, 16, 64, common::StripedToBlockedOp),
            CREATE_BENCHMARK(int, 256, 64, 64, common::StripedToBlockedOp),

            CREATE_BENCHMARK(int, 256, 1, 64, common::BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(int, 256, 4, 64, common::BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(int, 256, 16, 64, common::BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(int, 256, 64, 64, common::BlockedToStripedShuffleOp),

            CREATE_BENCHMARK(int, 256, 1, 64, common::StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(int, 256, 4, 64, common::StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(int, 256, 16, 64, common::StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(int, 256, 64, 64, common::StripedToBlockedShuffleOp),

            CREATE_BENCHMARK(int, 256, 1, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(int, 256, 4, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(int, 256, 16, 64, ScatterToStripedOp),

            CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::BlockedToStripedOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::BlockedToStripedOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::BlockedToStripedOp),

            CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::StripedToBlockedOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::StripedToBlockedOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::StripedToBlockedOp),

            CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::BlockedToStripedShuffleOp),

            CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::StripedToBlockedShuffleOp),

            CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, ScatterToStripedOp),

            CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::BlockedToStripedOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::BlockedToStripedOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::BlockedToStripedOp),

            CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::StripedToBlockedOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::StripedToBlockedOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::StripedToBlockedOp),

            CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::BlockedToStripedShuffleOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::BlockedToStripedShuffleOp),

            CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::StripedToBlockedShuffleOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::StripedToBlockedShuffleOp),

            CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, ScatterToStripedOp),
            CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, ScatterToStripedOp)};
        benchmarks.insert(benchmarks.end(),
                          additional_benchmarks.begin(),
                          additional_benchmarks.end());
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
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
