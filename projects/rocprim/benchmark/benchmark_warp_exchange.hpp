// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils.hpp"
#include "../common/utils_device_ptr.hpp"
#include "../common/warp_exchange.hpp"

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
auto warp_exchange_device_fn(T* d_output, unsigned int trials)
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
    constexpr unsigned int                    warps_in_block = BlockSize / LogicalWarpSize;
    const unsigned int                        warp_id        = threadIdx.x / LogicalWarpSize;
    ROCPRIM_SHARED_MEMORY
    typename warp_exchange_type::storage_type storage[warps_in_block];

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
auto warp_exchange_device_fn(T* d_output, unsigned int trials)
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
    ROCPRIM_SHARED_MEMORY
    typename warp_exchange_type::storage_type storage[warps_in_block];

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
auto warp_exchange_device_fn(T* /*d_output*/, unsigned int /*trials*/)
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
    warp_exchange_device_fn<BlockSize, ItemsPerThread, LogicalWarpSize, Op>(d_output, trials);
}

template<typename Op>
std::string get_operation_name()
{
    if constexpr(std::is_same_v<Op, common::BlockedToStripedOp>)
        return "common::BlockedToStripedOp";
    else if constexpr(std::is_same_v<Op, common::StripedToBlockedOp>)
        return "common::StripedToBlockedOp";
    else if constexpr(std::is_same_v<Op, common::BlockedToStripedShuffleOp>)
        return "common::BlockedToStripedShuffleOp";
    else if constexpr(std::is_same_v<Op, common::StripedToBlockedShuffleOp>)
        return "common::StripedToBlockedShuffleOp";
    else if constexpr(std::is_same_v<Op, ScatterToStripedOp>)
        return "ScatterToStripedOp";
    else
        static_assert(sizeof(Op) == 0, "Unknown operation");
}

template<typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int LogicalWarpSize,
         typename Op,
         typename Config = rocprim::default_config>
struct warp_exchange_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return meta_impl();
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;

        size_t             N               = bytes / sizeof(T);
        constexpr uint64_t items_per_block = BlockSize * ItemsPerThread;
        const uint64_t     items = items_per_block * ((N + items_per_block - 1) / items_per_block);

        constexpr uint64_t trials = 200;

        common::device_ptr<T> d_output(items);

        state.set_items(items);
        state.add_reads<T>(trials * items);

        state.run(
            [&]
            {
                warp_exchange_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Op>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_output.get(),
                                                                                    trials);
            });
    }

private:
    template<typename op = Op>
    primbench::json meta_impl() const
    {
        return primbench::json{}
            .add("lvl", "warp")
            .add("algo", "warp_exchange")
            .add("key_type", primbench::name<T>())
            .add("operation", get_operation_name<op>())
            .add("ws", LogicalWarpSize)
            .add("cfg", primbench::json{}.add("bs", BlockSize).add("ipt", ItemsPerThread));
    }
};
