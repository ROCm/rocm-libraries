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

#include "benchmark_utils.hpp"

#include <hipcub/warp/warp_exchange.hpp>

#include <type_traits>

constexpr const char* get_algorithm_name(hipcub::WarpExchangeAlgorithm algorithm)
{
    switch(algorithm)
    {
        case hipcub::WarpExchangeAlgorithm::WARP_EXCHANGE_SMEM: return "warp_exchange_smem";
        case hipcub::WarpExchangeAlgorithm::WARP_EXCHANGE_SHUFFLE: return "warp_exchange_shuffle";
    }
    return "unknown_algorithm";
}

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__device__
auto warp_exchange_benchmark(T* d_output)
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
    __shared__
    typename WarpExchangeT::TempStorage            temp_storage[warps_in_block];
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
__device__
auto warp_exchange_benchmark(T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<unsigned                        BlockSize,
         unsigned                        ItemsPerThread,
         unsigned                        LogicalWarpSize,
         ::hipcub::WarpExchangeAlgorithm Algorithm,
         class Op,
         class T>
__global__ __launch_bounds__(BlockSize)
void warp_exchange_kernel(T* d_output)
{
    warp_exchange_benchmark<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm, Op>(d_output);
}

template<class OffsetT,
         unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T>
__device__
auto warp_exchange_scatter_to_striped_benchmark(T* d_output)
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
    constexpr unsigned                  warps_in_block = BlockSize / LogicalWarpSize;
    __shared__
    typename WarpExchangeT::TempStorage temp_storage[warps_in_block];

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
__device__
auto warp_exchange_scatter_to_striped_benchmark(T* /*d_output*/)
    -> std::enable_if_t<!benchmark_utils::device_test_enabled_for_warp_size_v<LogicalWarpSize>>
{}

template<class OffsetT,
         unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize,
         class T>
__global__ __launch_bounds__(BlockSize)
void warp_exchange_scatter_to_striped_kernel(T* d_output)
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
class exchange_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "warp_exchange")
            .add("subalgo", get_algorithm_name(Algorithm))
            .add("data_type", primbench::name<T>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread)
            .add("warp_size", LogicalWarpSize)
            .add("op", Op::name)
            .add("lvl", "warp");
    }

    void run(primbench::state& state) override
    {
        const size_t& input_items = state.size;
        const auto&   stream      = state.stream;

        constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
        const unsigned     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        T* d_output;
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(
            [&]
            {
                warp_exchange_kernel<BlockSize, ItemsPerThread, LogicalWarpSize, Algorithm, Op>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_output);
            });

        HIP_CHECK(hipFree(d_output));
    }
};

template<class T,
         class OffsetT,
         unsigned BlockSize,
         unsigned ItemsPerThread,
         unsigned LogicalWarpSize>
class scatter_to_striped_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "warp_exchange")
            .add("subalgo", "scatter_to_striped")
            .add("data_type", primbench::name<T>())
            .add("offset_type", primbench::name<OffsetT>())
            .add("block_size", BlockSize)
            .add("items_per_thread", ItemsPerThread)
            .add("warp_size", LogicalWarpSize)
            .add("lvl", "warp");
    }

    void run(primbench::state& state) override
    {
        const size_t& input_items = state.size;
        const auto&   stream      = state.stream;

        constexpr unsigned items_per_block = BlockSize * ItemsPerThread;
        const unsigned     items
            = items_per_block * ((input_items + items_per_block - 1) / items_per_block);

        T* d_output;
        HIP_CHECK(hipMalloc(&d_output, items * sizeof(T)));

        state.set_items(items);
        state.add_writes<T>(items);

        state.run(
            [&]
            {
                warp_exchange_scatter_to_striped_kernel<OffsetT,
                                                        BlockSize,
                                                        ItemsPerThread,
                                                        LogicalWarpSize>
                    <<<dim3(items / items_per_block), dim3(BlockSize), 0, stream>>>(d_output);
            });

        HIP_CHECK(hipFree(d_output));
    }
};

struct StripedToBlockedOp
{
    static constexpr const char* name = "striped_to_blocked_op";

    template<class WarpExchangeT, class T, unsigned ItemsPerThread>
    __device__
    void operator()(WarpExchangeT& warp_exchange, T (&thread_data)[ItemsPerThread]) const
    {
        warp_exchange.StripedToBlocked(thread_data, thread_data);
    }
};

struct BlockedToStripedOp
{
    static constexpr const char* name = "blocked_to_striped_op";

    template<class WarpExchangeT, class T, unsigned ItemsPerThread>
    __device__
    void operator()(WarpExchangeT& warp_exchange, T (&thread_data)[ItemsPerThread]) const
    {
        warp_exchange.BlockedToStriped(thread_data, thread_data);
    }
};

#define CREATE_BENCHMARK_STRIPED_TO_BLOCKED(T, BS, IT, WS, ALG) \
    executor.queue<exchange_benchmark<T, BS, IT, WS, ALG, StripedToBlockedOp>>();

#define CREATE_BENCHMARK_BLOCKED_TO_STRIPED(T, BS, IT, WS, ALG) \
    executor.queue<exchange_benchmark<T, BS, IT, WS, ALG, BlockedToStripedOp>>();

#define CREATE_BENCHMARK_SCATTER_TO_STRIPED(T, OFFSET_T, BS, IT, WS) \
    executor.queue<scatter_to_striped_benchmark<T, OFFSET_T, BS, IT, WS>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 32 * primbench::MiB; // In items
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    // Add benchmarks
    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SMEM);
    CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 16);
    CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 32);
    CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 256, 4, 32);

    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 16, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE);

// CUB requires WS == IPT for WARP_EXCHANGE_SHUFFLE
#ifdef HIPCUB_ROCPRIM_API
    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 16, ::hipcub::WARP_EXCHANGE_SHUFFLE);
    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE);
    CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE);
    CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 32, ::hipcub::WARP_EXCHANGE_SHUFFLE);
#endif

#ifdef HIPCUB_ROCPRIM_API
    if(::benchmark_utils::is_warp_size_supported(64))
    {
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM);
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE);
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM);
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 128, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE);
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 128, 4, 64);

        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM);
        CREATE_BENCHMARK_STRIPED_TO_BLOCKED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE);
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SMEM);
        CREATE_BENCHMARK_BLOCKED_TO_STRIPED(int, 256, 4, 64, ::hipcub::WARP_EXCHANGE_SHUFFLE);
        CREATE_BENCHMARK_SCATTER_TO_STRIPED(int, int, 256, 4, 64);
    }
#endif

    executor.run();
}
