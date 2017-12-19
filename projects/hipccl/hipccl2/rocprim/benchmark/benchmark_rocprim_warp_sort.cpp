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

template<class T, unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hc_warp_sort(benchmark::State& state, hc::accelerator_view acc_view, size_t N)
{
    // Make sure size is a multiple of BlockSize
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    // Allocate and fill memory
    std::vector<T> input(size, 1.0f);
    std::vector<T> output(size, -1.0f);
    hc::array_view<T, 1> av_input(size, input.data());
    hc::array_view<T, 1> av_output(size, output.data());
    av_input.synchronize_to(acc_view);
    av_output.synchronize_to(acc_view);
    acc_view.wait();

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto event = hc::parallel_for_each(
            acc_view,
            hc::extent<1>(size).tile(BlockSize),
            [=](hc::tiled_index<1> i) [[hc]]
            {
                T value = av_input[i];
                rp::warp_sort<T, WarpSize> wsort;
                wsort.sort(value);
                av_output[i] = value;
            }
        );
        event.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * size);
}

template<class T, unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hc_warp_sort_by_key(benchmark::State& state, hc::accelerator_view acc_view, size_t N)
{
    // Make sure size is a multiple of BlockSize
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    // Allocate and fill memory
    std::vector<T> input(size, 1.0f);
    std::vector<T> output(size, -1.0f);
    hc::array_view<T, 1> av_input(size, input.data());
    hc::array_view<T, 1> av_output(size, output.data());
    av_input.synchronize_to(acc_view);
    av_output.synchronize_to(acc_view);
    acc_view.wait();

    for (auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto event = hc::parallel_for_each(
            acc_view,
            hc::extent<1>(size).tile(BlockSize),
            [=](hc::tiled_index<1> i) [[hc]]
            {
                T value = av_input[i];
                rp::warp_sort<T, WarpSize, T> wsort;
                wsort.sort(value, value);
                av_output[i] = value;
            }
        );
        event.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * size);
}

template<class T, unsigned int WarpSize>
__global__
void warp_sort_kernel(const T* input, T* output)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = input[i];
    rp::warp_sort<T, WarpSize> wsort;
    wsort.sort(value);
    output[i] = value;
}

template<class T, unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hip_warp_sort(benchmark::State& state, hipStream_t stream, size_t N)
{
    // Make sure size is a multiple of BlockSize
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    // Allocate and fill memory
    std::vector<T> input;
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }
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
            HIP_KERNEL_NAME(warp_sort_kernel<T, WarpSize>),
            dim3(size/BlockSize), dim3(BlockSize), 0, stream,
            d_input, d_output
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

template<class T, unsigned int WarpSize>
__global__
void warp_sort_by_key_kernel(const T* input, T* output)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    auto value = input[i];
    rp::warp_sort<T, WarpSize, T> wsort;
    wsort.sort(value, value);
    output[i] = value;
}

template<class T, unsigned int WarpSize, unsigned int BlockSize>
void benchmark_hip_warp_sort_by_key(benchmark::State& state, hipStream_t stream, size_t N)
{
    // Make sure size is a multiple of BlockSize
    const auto size = BlockSize * ((N + BlockSize - 1)/BlockSize);
    // Allocate and fill memory
    std::vector<T> input;
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }
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
            HIP_KERNEL_NAME(warp_sort_by_key_kernel<T, WarpSize>),
            dim3(size/BlockSize), dim3(BlockSize), 0, stream,
            d_input, d_output
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
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
            benchmark_hc_warp_sort<float, 64, 256>, // func
            *acc_view, size // arguments for func
        ),
        benchmark::RegisterBenchmark(
            "warp_sort_by_key_hc", // name
            benchmark_hc_warp_sort_by_key<float, 64, 256>, // func
            *acc_view, size // arguments for func
        ),
        benchmark::RegisterBenchmark(
            "warp_sort_hip",
            benchmark_hip_warp_sort<float, 64, 256>,
            stream, size
        ),
        benchmark::RegisterBenchmark(
            "warp_sort_by_key_hip",
            benchmark_hip_warp_sort_by_key<float, 64, 256>,
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
