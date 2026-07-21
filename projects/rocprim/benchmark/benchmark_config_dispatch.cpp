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

#include "benchmark_config_dispatch.hpp"

#include "primbench.hpp"

int main(int argc, char* argv[])
{
    // Set the number of bytes to 1, because we want throughput stats,
    // even though config_dispatch doesn't create an input array.
    primbench::settings settings;
    settings.size = 1;
    primbench::executor executor(argc, argv, settings);

    // Queue stream variation benchmarks.
    executor.queue<config_dispatch_benchmark>("default_stream");
    executor.queue<config_dispatch_benchmark>("per_thread_stream");
    executor.queue<config_dispatch_benchmark>("explicit_stream");
    executor.queue<config_dispatch_benchmark>("async_stream");

    // Queue empty kernel baseline, using the same benchmark struct.
    executor.queue<config_dispatch_benchmark>("empty_kernel");

    executor.run();
}
