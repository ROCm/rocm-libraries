// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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
#include <chrono>
#include <vector>
#include <locale>
#include <codecvt>
#include <string>

// Google Benchmark
#include "benchmark/benchmark.h"

// CmdParser
#include "cmdparser.hpp"

#include "benchmark_utils.hpp"

// HC API
#include <hcc/hc.hpp>
// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

// rocPRIM
#include <block/block_exchange.hpp>
#include <block/block_load.hpp>
#include <block/block_store.hpp>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

namespace rp = rocprim;

struct blocked_to_striped
{
    template<
        class T,
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        unsigned int Trials
    >
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        T output[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_striped(input, output);

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] = output[i] + ((i + trial) % 2 == 0 ? +trial : -trial);
            }
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, output);
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
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        T output[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.striped_to_blocked(input, output);

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] = output[i] + ((i + trial) % 2 == 0 ? +trial : -trial);
            }
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, output);
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
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        T output[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.blocked_to_warp_striped(input, output);

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] = output[i] + ((i + trial) % 2 == 0 ? +trial : -trial);
            }
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, output);
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
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        T output[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.warp_striped_to_blocked(input, output);

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] = output[i] + ((i + trial) % 2 == 0 ? +trial : -trial);
            }
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, output);
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
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        T output[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                ranks[i] = trial % 2 == 0
                    ? (ItemsPerThread - 1 - i) * BlockSize + lid
                    : (BlockSize - 1 - lid) * ItemsPerThread + i;
                ranks[i] = ItemsPerThread * BlockSize - 1 - ranks[i];
            }

            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_blocked(input, output, ranks);

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] = output[i] + ((i + trial) % 2 == 0 ? +trial : -trial);
            }
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, output);
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
    __global__
    static void kernel(const T * d_input, T * d_output)
    {
        const unsigned int lid = hipThreadIdx_x;
        const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;

        T input[ItemsPerThread];
        T output[ItemsPerThread];
        unsigned int ranks[ItemsPerThread];
        rp::block_load_direct_striped<BlockSize>(lid, d_input + block_offset, input);

        for(unsigned int trial = 0; trial < Trials; trial++)
        {
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                ranks[i] = trial % 2 == 0
                    ? (ItemsPerThread - 1 - i) * BlockSize + lid
                    : (BlockSize - 1 - lid) * ItemsPerThread + i;
                ranks[i] = ItemsPerThread * BlockSize - 1 - ranks[i];
            }

            rp::block_exchange<T, BlockSize, ItemsPerThread> exchange;
            exchange.scatter_to_striped(input, output, ranks);

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                input[i] = output[i] + ((i + trial) % 2 == 0 ? +trial : -trial);
            }
        }

        rp::block_store_direct_striped<BlockSize>(lid, d_output + block_offset, output);
    }
};

template<
    class K,
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Trials = 100
>
void run_benchmark(benchmark::State& state, hipStream_t stream, size_t N)
{
    constexpr auto items_per_block = BlockSize * ItemsPerThread;
    const auto size = items_per_block * ((N + items_per_block - 1)/items_per_block);

    std::vector<T> input = get_random_data<T>(size, (T)0, (T)1000);
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(K::template kernel<T, BlockSize, ItemsPerThread, Trials>),
            dim3(size/items_per_block), dim3(BlockSize), 0, stream,
            d_input, d_output
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
    HIP_CHECK(hipFree(d_output));
}

template<class K>
void add_benchmarks(const std::string& name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    hipStream_t stream,
                    size_t size)
{
    auto n = [=](const std::string& x) { return std::string("block_exchange") + x + "." + name; };
    std::vector<benchmark::internal::Benchmark*> bs =
    {
        benchmark::RegisterBenchmark(
            n("<int, 256, 2>").c_str(),
            run_benchmark<K, int, 256, 2>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<int, 256, 3>").c_str(),
            run_benchmark<K, int, 256, 3>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<int, 256, 4>").c_str(),
            run_benchmark<K, int, 256, 4>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<int, 256, 7>").c_str(),
            run_benchmark<K, int, 256, 7>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<int, 256, 8>").c_str(),
            run_benchmark<K, int, 256, 8>,
            stream, size
        ),

        benchmark::RegisterBenchmark(
            n("<long long, 256, 2>").c_str(),
            run_benchmark<K, long long, 256, 2>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<long long, 256, 3>").c_str(),
            run_benchmark<K, long long, 256, 3>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<long long, 256, 4>").c_str(),
            run_benchmark<K, long long, 256, 4>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<long long, 256, 7>").c_str(),
            run_benchmark<K, long long, 256, 7>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            n("<long long, 256, 8>").c_str(),
            run_benchmark<K, long long, 256, 8>,
            stream, size
        ),
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

    // HC
    hc::accelerator_view* acc_view;
    HIP_CHECK(hipHccGetAcceleratorView(stream, &acc_view));
    auto acc = acc_view->get_accelerator();
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::cout << "[HC] Device name: " << conv.to_bytes(acc.get_description()) << std::endl;

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
