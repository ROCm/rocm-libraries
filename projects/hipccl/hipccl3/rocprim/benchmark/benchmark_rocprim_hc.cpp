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

#ifndef DEFAULT_N
const size_t DEFAULT_N = 100;
#endif

void saxpy_function(benchmark::State& state, hc::accelerator_view acc_view, size_t N) {
    for (auto _ : state) {
        const float a = 100.0f;
        std::vector<float> x(N, 2.0f);
        std::vector<float> y(N, 1.0f);

        hc::array_view<float, 1> av_x(N, x.data());
        hc::array_view<float, 1> av_y(N, y.data());

        av_x.synchronize_to(acc_view);
        av_y.synchronize_to(acc_view);

        auto start = std::chrono::high_resolution_clock::now();
        auto event = hc::parallel_for_each(
            acc_view,
            hc::extent<1>(N),
            [=](hc::index<1> i) [[hc]]
            {
                av_y[i] = a * av_x[i] + av_y[i];
            }
        );
        event.wait();
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
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

    hc::accelerator acc;
    auto acc_view = acc.get_default_view();
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::cout << "Device name: " << conv.to_bytes(acc.get_description()) << std::endl;

    benchmark::RegisterBenchmark("Saxpy_HC", saxpy_function, acc_view, size)->UseManualTime();
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
