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

// HC API
#include <hcc/hc.hpp>
// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

// rocPRIM
#include <warp/warp_sort.hpp>

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

template<unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hc_warp_sort(benchmark::State& state, hc::accelerator_view acc_view, size_t N)
{
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    for (auto _ : state)
    {
        std::vector<float> x(size, 1.0f);
        hc::array_view<float, 1> av_x(size, x.data());
        av_x.synchronize_to(acc_view);
        acc_view.wait();

        auto start = std::chrono::high_resolution_clock::now();
        auto event = hc::parallel_for_each(
            acc_view,
            hc::extent<1>(size).tile(BlockSize),
            [=](hc::tiled_index<1> i) [[hc]]
            {
                float value = av_x[i];
                rp::warp_sort<float, WarpSize> wscan;
                wscan.sort(value);
                av_x[i] = value;
            }
        );
        event.wait();
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

template<unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hc_warp_sort_by_key(benchmark::State& state, hc::accelerator_view acc_view, size_t N)
{
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    for (auto _ : state)
    {
        std::vector<float> x(size, 1.0f);
        hc::array_view<float, 1> av_x(size, x.data());
        av_x.synchronize_to(acc_view);
        acc_view.wait();

        auto start = std::chrono::high_resolution_clock::now();
        auto event = hc::parallel_for_each(
            acc_view,
            hc::extent<1>(size).tile(BlockSize),
            [=](hc::tiled_index<1> i) [[hc]]
            {
                float value1 = av_x[i];
                rp::warp_sort<float, WarpSize, float> wscan;
                wscan.sort(value1, value1);
                av_x[i] = value1;
            }
        );
        event.wait();
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

template<class T, unsigned int WarpSize>
__global__
void warp_sort_kernel(T * input)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = input[i];
    rp::warp_sort<T, WarpSize> wscan;
    wscan.sort(value);
    input[i] = value;
}

template<unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hip_warp_sort(benchmark::State& state, hipStream_t stream, size_t N)
{
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    for (auto _ : state)
    {
        std::vector<float> x(size, 1.0f);
        float * d_x;
        HIP_CHECK(hipMalloc(&d_x, size * sizeof(float)));
        HIP_CHECK(
            hipMemcpy(
                d_x, x.data(),
                size * sizeof(float),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_sort_kernel<float, WarpSize>),
            dim3(size/BlockSize), dim3(BlockSize), 0, stream,
            d_x
        );
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_x));
    }
}

template<class T, unsigned int WarpSize>
__global__
void warp_sort_by_key_kernel(T * input)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = input[i];
    rp::warp_sort<T, WarpSize, T> wscan;
    wscan.sort(value, value);
    input[i] = value;
}

template<unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hip_warp_sort_by_key(benchmark::State& state, hipStream_t stream, size_t N)
{
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    for (auto _ : state)
    {
        std::vector<float> x(size, 1.0f);
        float * d_x;
        HIP_CHECK(hipMalloc(&d_x, size * sizeof(float)));
        HIP_CHECK(
            hipMemcpy(
                d_x, x.data(),
                size * sizeof(float),
                hipMemcpyHostToDevice
            )
        );
        HIP_CHECK(hipDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(warp_sort_by_key_kernel<float, WarpSize>),
            dim3(size/BlockSize), dim3(BlockSize), 0, stream,
            d_x
        );
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_x));
    }
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
    std::cout << "[HC]  Device name: " << conv.to_bytes(acc.get_description()) << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        benchmark::RegisterBenchmark(
            "warp_sort_hc", // name
            benchmark_hc_warp_sort<64, 256>, // func
            *acc_view, size // arguments for func
        ),
        benchmark::RegisterBenchmark(
            "warp_sort_by_key_hc", // name
            benchmark_hc_warp_sort_by_key<64, 256>, // func
            *acc_view, size // arguments for func
        ),
        benchmark::RegisterBenchmark(
            "warp_sort_hip",
            benchmark_hip_warp_sort<64, 256>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            "warp_sort_by_key_hip",
            benchmark_hip_warp_sort_by_key<64, 256>,
            stream, size
        )
    };

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
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
