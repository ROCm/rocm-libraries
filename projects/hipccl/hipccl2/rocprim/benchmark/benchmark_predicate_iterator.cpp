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

#include "benchmark_predicate_iterator.hpp"

#include "primbench.hpp"

#define CREATE_BENCHMARK(B, T, C) \
    executor                      \
        .queue<predicate_iterator_benchmark<B<T, less_than<T, C>, common::increment_by<5>>, C>>();

// clang-format off
#define CREATE_TYPED_BENCHMARK(T)                \
    CREATE_BENCHMARK(transform_it, T, 0)         \
    CREATE_BENCHMARK(read_predicate_it, T, 0)    \
    CREATE_BENCHMARK(write_predicate_it, T, 0)   \
    CREATE_BENCHMARK(transform_it, T, 25)        \
    CREATE_BENCHMARK(read_predicate_it, T, 25)   \
    CREATE_BENCHMARK(write_predicate_it, T, 25)  \
    CREATE_BENCHMARK(transform_it, T, 50)        \
    CREATE_BENCHMARK(read_predicate_it, T, 50)   \
    CREATE_BENCHMARK(write_predicate_it, T, 50)  \
    CREATE_BENCHMARK(transform_it, T, 75)        \
    CREATE_BENCHMARK(read_predicate_it, T, 75)   \
    CREATE_BENCHMARK(write_predicate_it, T, 75)  \
    CREATE_BENCHMARK(transform_it, T, 100)       \
    CREATE_BENCHMARK(read_predicate_it, T, 100)  \
    CREATE_BENCHMARK(write_predicate_it, T, 100)
// clang-format on

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size = 512 * primbench::MiB;
    primbench::executor executor(argc, argv, settings);

    CREATE_TYPED_BENCHMARK(int8_t)
    CREATE_TYPED_BENCHMARK(int16_t)
    CREATE_TYPED_BENCHMARK(int32_t)
    CREATE_TYPED_BENCHMARK(int64_t)
    CREATE_TYPED_BENCHMARK(custom_i64_i64)
    CREATE_TYPED_BENCHMARK(rocprim::int128_t)
    CREATE_TYPED_BENCHMARK(rocprim::uint128_t)

    executor.run();
}
