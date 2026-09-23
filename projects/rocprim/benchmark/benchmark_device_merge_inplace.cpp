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

#include "benchmark_device_merge_inplace.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(Value) \
    executor.queue<device_merge_inplace_benchmark<Value, inplace_runner<Value>>>();

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 128 * primbench::MiB;
    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(int16_t)
    CREATE_BENCHMARK(int32_t)
    CREATE_BENCHMARK(int64_t)
    CREATE_BENCHMARK(rocprim::int128_t)

    executor.run();
}
