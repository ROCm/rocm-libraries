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

#include "../common/utils.hpp"
#include "../common/utils_device_ptr.hpp"
#include "../common/warp_exchange.hpp"

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
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;

    // Calculate the number of elements
    size_t N = bytes / sizeof(T);

    constexpr uint64_t trials          = 200;
    constexpr uint64_t items_per_block = BlockSize * ItemsPerThread;
    const uint64_t     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    common::device_ptr<T> d_output(size);

    state.run(
        [&]
        {
            warp_exchange_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Op>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_output.get(),
                                                                               trials);
        });

    state.set_throughput(trials * size, sizeof(T));
}

#define CREATE_BENCHMARK(T, BS, IT, WS, OP)                                                  \
    executor.queue_fn(bench_naming::format_name("{lvl:warp,algo:exchange,key_type:" #T       \
                                                ",operation:" #OP ",ws:" #WS ",cfg:{bs:" #BS \
                                                ",ipt:" #IT "}}")                            \
                          .c_str(),                                                          \
                      run_benchmark<T, BS, IT, WS, OP>);

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0);

    CREATE_BENCHMARK(int, 256, 1, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int, 256, 1, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int, 256, 4, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int, 256, 4, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int, 256, 16, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int, 256, 16, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(int, 256, 32, 32, common::BlockedToStripedOp)

    CREATE_BENCHMARK(int, 256, 1, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int, 256, 1, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int, 256, 4, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int, 256, 4, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int, 256, 16, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int, 256, 16, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(int, 256, 32, 32, common::StripedToBlockedOp)

    CREATE_BENCHMARK(int, 256, 1, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int, 256, 1, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int, 256, 4, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int, 256, 4, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int, 256, 16, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int, 256, 16, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(int, 256, 32, 32, common::BlockedToStripedShuffleOp)

    CREATE_BENCHMARK(int, 256, 1, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int, 256, 1, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int, 256, 4, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int, 256, 4, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int, 256, 16, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int, 256, 16, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(int, 256, 32, 32, common::StripedToBlockedShuffleOp)

    CREATE_BENCHMARK(int, 256, 1, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(int, 256, 1, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(int, 256, 4, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(int, 256, 4, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(int, 256, 16, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(int, 256, 16, 32, ScatterToStripedOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::BlockedToStripedOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::StripedToBlockedOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::BlockedToStripedShuffleOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, common::StripedToBlockedShuffleOp)

    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 32, ScatterToStripedOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::BlockedToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::BlockedToStripedOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::StripedToBlockedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::StripedToBlockedOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::BlockedToStripedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::BlockedToStripedShuffleOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, common::StripedToBlockedShuffleOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, common::StripedToBlockedShuffleOp)

    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 32, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 16, ScatterToStripedOp)
    CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 32, ScatterToStripedOp)

    int hip_device = 0;
    HIP_CHECK(::rocprim::detail::get_device_from_stream(hipStreamDefault, hip_device));
    if(is_warp_size_supported(64, hip_device))
    {
        CREATE_BENCHMARK(int, 256, 1, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(int, 256, 4, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(int, 256, 16, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(int, 256, 64, 64, common::BlockedToStripedOp)

        CREATE_BENCHMARK(int, 256, 1, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(int, 256, 4, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(int, 256, 16, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(int, 256, 64, 64, common::StripedToBlockedOp)

        CREATE_BENCHMARK(int, 256, 1, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(int, 256, 4, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(int, 256, 16, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(int, 256, 64, 64, common::BlockedToStripedShuffleOp)

        CREATE_BENCHMARK(int, 256, 1, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(int, 256, 4, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(int, 256, 16, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(int, 256, 64, 64, common::StripedToBlockedShuffleOp)

        CREATE_BENCHMARK(int, 256, 1, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(int, 256, 4, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(int, 256, 16, 64, ScatterToStripedOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::BlockedToStripedOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::StripedToBlockedOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::BlockedToStripedShuffleOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, common::StripedToBlockedShuffleOp)

        CREATE_BENCHMARK(rocprim::int128_t, 256, 1, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 4, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::int128_t, 256, 16, 64, ScatterToStripedOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::BlockedToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::BlockedToStripedOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::StripedToBlockedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::StripedToBlockedOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::BlockedToStripedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::BlockedToStripedShuffleOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, common::StripedToBlockedShuffleOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, common::StripedToBlockedShuffleOp)

        CREATE_BENCHMARK(rocprim::uint128_t, 256, 1, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 4, 64, ScatterToStripedOp)
        CREATE_BENCHMARK(rocprim::uint128_t, 256, 16, 64, ScatterToStripedOp)
    }

    executor.run();
}
