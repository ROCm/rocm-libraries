// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/block/block_exchange.hpp>
#include <rocprim/block/block_load_func.hpp>
#include <rocprim/block/block_store_func.hpp>
#include <rocprim/config.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdint.h>
#include <string>
#include <vector>

template<typename Runner,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials>
__global__ __launch_bounds__(BlockSize)
void kernel(const T* d_input, const unsigned int* d_ranks, T* d_output)
{
    Runner::template run<T, BlockSize, ItemsPerThread, Trials>(d_input, d_ranks, d_output);
}

struct blocked_to_striped
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_striped(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct striped_to_blocked
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.striped_to_blocked(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct blocked_to_warp_striped
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_warp_striped(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct warp_striped_to_blocked
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int*, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.warp_striped_to_blocked(input, input);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct scatter_to_blocked
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int* d_ranks, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T            input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);
        rocprim::block_load_direct_striped<BlockSize>(lid, d_ranks + block_offset, ranks);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_blocked(input, input, ranks);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

struct scatter_to_striped
{
    template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread, unsigned int Trials>
    __device__
    static void run(const T* d_input, const unsigned int* d_ranks, T* d_output)
    {
        const unsigned int lid          = threadIdx.x;
        const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;

        T            input[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rocprim::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);
        rocprim::block_load_direct_striped<BlockSize>(lid, d_ranks + block_offset, ranks);

        ROCPRIM_NO_UNROLL
        for(unsigned int trial = 0; trial < Trials; ++trial)
        {
            rocprim::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_striped(input, input, ranks);
            ::rocprim::syncthreads();
        }

        rocprim::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, input);
    }
};

template<typename Benchmark,
         typename T,
         unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int Trials = 100>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;
    const auto& stream = state.stream;

    // Calculate the number of elements N
    size_t N = bytes / sizeof(T);

    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto     size = items_per_block * ((N + items_per_block - 1) / items_per_block);

    std::vector<T> input(size);
    // Fill input
    for(size_t i = 0; i < size; ++i)
    {
        input[i] = T(i);
    }
    std::vector<unsigned int> ranks(size);
    // Fill ranks (for scatter operations)
    engine_type gen(seed.get_0());
    for(size_t bi = 0; bi < size / items_per_block; ++bi)
    {
        auto block_ranks = ranks.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, gen);
    }
    common::device_ptr<T>            d_input(input);
    common::device_ptr<unsigned int> d_ranks(ranks);
    common::device_ptr<T>            d_output(size);
    HIP_CHECK(hipDeviceSynchronize());

    state.run(
        [&]
        {
            kernel<Benchmark, T, BlockSize, ItemsPerThread, Trials>
                <<<dim3(size / items_per_block), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                               d_ranks.get(),
                                                                               d_output.get());
            HIP_CHECK(hipGetLastError());
        });

    state.set_throughput(size * Trials, sizeof(T));
}

#define CREATE_BENCHMARK(T, BS, IPT)                                                           \
    executor.queue_fn(bench_naming::format_name("{lvl:block,algo:exchange,subalgo:" + name     \
                                                + ",key_type:" #T ",cfg:{bs:" #BS ",ipt:" #IPT \
                                                  "}}")                                        \
                          .c_str(),                                                            \
                      run_benchmark<Benchmark, T, BS, IPT>);

#define BENCHMARK_TYPE(type, block)  \
    CREATE_BENCHMARK(type, block, 1) \
    CREATE_BENCHMARK(type, block, 2) \
    CREATE_BENCHMARK(type, block, 3) \
    CREATE_BENCHMARK(type, block, 4) \
    CREATE_BENCHMARK(type, block, 7) \
    CREATE_BENCHMARK(type, block, 8)

template<typename Benchmark>
void add_benchmarks(const std::string& name, benchmark_utils::executor& executor)
{
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    BENCHMARK_TYPE(int, 256)
    BENCHMARK_TYPE(int8_t, 256)
    BENCHMARK_TYPE(rocprim::half, 256)
    BENCHMARK_TYPE(long long, 256)
    BENCHMARK_TYPE(custom_float2, 256)
    BENCHMARK_TYPE(float2, 256)
    BENCHMARK_TYPE(custom_double2, 256)
    BENCHMARK_TYPE(double2, 256)
    BENCHMARK_TYPE(float4, 256)
    BENCHMARK_TYPE(rocprim::int128_t, 256)
    BENCHMARK_TYPE(rocprim::uint128_t, 256)
}

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0);

    add_benchmarks<blocked_to_striped>("blocked_to_striped", executor);
    add_benchmarks<striped_to_blocked>("striped_to_blocked", executor);
    add_benchmarks<blocked_to_warp_striped>("blocked_to_warp_striped", executor);
    add_benchmarks<warp_striped_to_blocked>("warp_striped_to_blocked", executor);
    add_benchmarks<scatter_to_blocked>("scatter_to_blocked", executor);
    add_benchmarks<scatter_to_striped>("scatter_to_striped", executor);

    executor.run();
}
