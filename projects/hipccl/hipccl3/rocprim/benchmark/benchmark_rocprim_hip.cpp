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

// Google Benchmark
#include "benchmark/benchmark.h"

// CmdParser
#include "cmdparser.hpp"

// HIP API
#include <hip/hip_runtime.h>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 100;
#endif

template <class T>
__global__
void saxpy_kernel(const T * x, T * y, const T a, const size_t size)
{
    const unsigned int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < size)
    {
        y[i] += a * x[i];
    }
}

void saxpy_function(benchmark::State& state, size_t N) {
    for (auto _ : state) {
        const float a = 100.0f;
        std::vector<float> x(N, 2.0f);
        std::vector<float> y(N, 1.0f);

        float * d_x;
        float * d_y;
        hipMalloc(&d_x, N * sizeof(float));
        hipMalloc(&d_y, N * sizeof(float));
        hipMemcpy(
            d_x, x.data(),
            N * sizeof(float),
            hipMemcpyHostToDevice
        );

        hipMemcpy(
            d_y, y.data(),
            N * sizeof(float),
            hipMemcpyHostToDevice
        );
        hipDeviceSynchronize();

        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(saxpy_kernel<float>),
            dim3((N + 255)/256), dim3(256), 0, 0,
            d_x, d_y, a, N
        );
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        hipPeekAtLastError();

        hipMemcpy(
            y.data(), d_y,
            N * sizeof(float),
            hipMemcpyDeviceToHost
        );

        hipDeviceSynchronize();
        hipFree(d_x);
        hipFree(d_y);
    }
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.run_and_exit_if_error();

    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    std::cout << "Device name: " << devProp.name << std::endl;

    benchmark::RegisterBenchmark("Saxpy_HC", saxpy_function, size)->UseManualTime();
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
