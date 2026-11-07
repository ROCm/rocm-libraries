// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_device_partial_sort.hpp"
#include "benchmark_utils.hpp"
#include "primbench.hpp"

#include "../common/utils_custom_type.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

#define CREATE_BENCHMARK_PARTIAL_SORT(TYPE, SMALL_N) \
    executor.queue<device_partial_sort_benchmark<TYPE>>(SMALL_N);

#define CREATE_BENCHMARK(TYPE) \
    {CREATE_BENCHMARK_PARTIAL_SORT(TYPE, true) CREATE_BENCHMARK_PARTIAL_SORT(TYPE, false)}

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = 128 * primbench::MiB;
    settings.min_gpu_ms_per_batch = 1000;
    settings.batch_window_size    = 3;
    primbench::executor executor(argc, argv, settings, primbench::flags::sync);

    benchmark_types::queue_type<(benchmark_types::Type_Category::device_sort)>(
        executor,
        [&](auto type_tag) { CREATE_BENCHMARK(typename decltype(type_tag)::type) });

    executor.run();
}
