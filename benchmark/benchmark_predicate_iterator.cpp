// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/predicate_iterator.hpp"
#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_transform.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/iterator/predicate_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <vector>

template<typename T, int C>
struct less_than
{
    __device__
    bool operator()(T value) const
    {
        return value < T{C};
    }
};

template<typename T, typename Predicate, typename Transform>
struct transform_op
{
    __device__
    auto operator()(T v) const
    {
        return Predicate{}(v) ? Transform{}(v) : v;
    }
};

template<typename T, typename Predicate, typename Transform>
struct transform_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t size, const hipStream_t stream)
    {
        auto t_it
            = rocprim::make_transform_iterator(d_input, transform_op<T, Predicate, Transform>{});
        HIP_CHECK(rocprim::transform(t_it, d_output, size, rocprim::identity<T>{}, stream));
    }
};

template<typename T, typename Predicate, typename Transform>
struct read_predicate_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t size, const hipStream_t stream)
    {
        auto t_it = rocprim::make_transform_iterator(d_input, Transform{});
        auto r_it = rocprim::make_predicate_iterator(t_it, d_input, Predicate{});
        HIP_CHECK(rocprim::transform(r_it, d_output, size, rocprim::identity<T>{}, stream));
    }
};

template<typename T, typename Predicate, typename Transform>
struct write_predicate_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t size, const hipStream_t stream)
    {
        auto t_it = rocprim::make_transform_iterator(d_input, Transform{});
        auto w_it = rocprim::make_predicate_iterator(d_output, d_input, Predicate{});
        HIP_CHECK(rocprim::transform(t_it, w_it, size, rocprim::identity<T>{}, stream));
    }
};

template<typename IteratorBenchmark>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;

    using T = typename IteratorBenchmark::value_type;

    // Calculate the number of elements
    size_t size = bytes / sizeof(T);

    const auto     random_range = limit_random_range<T>(0, 99);
    std::vector<T> input
        = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());
    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(size);
    HIP_CHECK(hipDeviceSynchronize());

    state.run([&] { IteratorBenchmark{}(d_input.get(), d_output.get(), size, stream); });

    state.set_throughput(size, sizeof(T));
}

#define CREATE_BENCHMARK(B, T, C)                                                                \
    executor.queue_fn(bench_naming::format_name("{lvl:device,algo:" #B ",p:p" #C ",key_type:" #T \
                                                ",cfg:default_config}")                          \
                          .c_str(),                                                              \
                      run_benchmark<B<T, less_than<T, C>, common::increment_by<5>>>);

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
    benchmark_utils::executor executor(argc, argv, 512 * benchmark_utils::MiB, 10, 5);

    using custom_128 = common::custom_type<int64_t, int64_t>;

    CREATE_TYPED_BENCHMARK(int8_t)
    CREATE_TYPED_BENCHMARK(int16_t)
    CREATE_TYPED_BENCHMARK(int32_t)
    CREATE_TYPED_BENCHMARK(int64_t)
    CREATE_TYPED_BENCHMARK(custom_128)
    CREATE_TYPED_BENCHMARK(rocprim::int128_t)
    CREATE_TYPED_BENCHMARK(rocprim::uint128_t)

    executor.run();
}
