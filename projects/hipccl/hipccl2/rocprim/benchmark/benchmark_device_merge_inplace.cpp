// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <rocprim/device/device_merge_inplace.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <random>
#include <sstream>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

template<typename T, T increment>
struct random_monotonic_iterator
{
    const unsigned int seed;
    random_monotonic_iterator(unsigned int seed) : seed(seed) {}

    using difference_type = std::ptrdiff_t;
    using value_type      = T;

    // not all integral types are valid for int distribution
    using dist_value_type
        = std::conditional_t<rocprim::is_integral<T>::value
                                 && !common::is_valid_for_int_distribution<T>::value,
                             int,
                             T>;

    using dist_type = std::conditional_t<rocprim::is_integral<T>::value,
                                         common::uniform_int_distribution<dist_value_type>,
                                         std::uniform_real_distribution<T>>;

    std::mt19937 engine{seed};
    dist_type    dist{dist_value_type{0}, dist_value_type{increment}};

    dist_value_type value = dist_value_type{0};

    int operator*() const
    {
        return limit_cast<value_type>(value);
    }

    random_monotonic_iterator& operator++()
    {
        // prefix
        value += dist(engine);
        return *this;
    }

    random_monotonic_iterator operator++(int)
    {
        // postfix
        random_monotonic_iterator retval{*this};
        value += dist(engine);
        return retval;
    }
};

template<class ValueT>
struct inplace_runner
{
    using value_type = ValueT;
    using compare_op_type =
        typename std::conditional<std::is_same<value_type, rocprim::half>::value,
                                  half_less,
                                  rocprim::less<value_type>>::type;

    value_type* d_data;
    size_t      left_size;
    size_t      right_size;
    hipStream_t stream;

    common::device_ptr<void> d_temporary_storage;
    size_t temporary_storage_bytes = 0;

    compare_op_type compare_op{};

    inplace_runner(value_type* data, size_t left_size, size_t right_size, hipStream_t stream)
        : d_data(data), left_size(left_size), right_size(right_size), stream(stream)
    {}

    void prepare()
    {
        HIP_CHECK(rocprim::merge_inplace(d_temporary_storage.get(),
                                         temporary_storage_bytes,
                                         d_data,
                                         left_size,
                                         right_size,
                                         compare_op,
                                         stream));
        d_temporary_storage.resize(temporary_storage_bytes);
    }

    void run()
    {
        HIP_CHECK(rocprim::merge_inplace(d_temporary_storage.get(),
                                         temporary_storage_bytes,
                                         d_data,
                                         left_size,
                                         right_size,
                                         compare_op,
                                         stream));
    }
};

template<class ValueT, class RunnerT>
void run_merge_inplace_benchmarks(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& size_a = state.bytes;
    const auto& size_b = state.bytes;
    const auto& seed   = state.seed;

    using value_type  = ValueT;
    using runner_type = RunnerT;

    size_t                  total_size = size_a + size_b;
    std::vector<value_type> h_data(total_size);

    auto gen_a_it = random_monotonic_iterator<value_type, 4>{seed.get_0()};
    auto gen_b_it = random_monotonic_iterator<value_type, 4>{seed.get_1()};

    // generate left array
    for(size_t i = 0; i < size_a; ++i)
    {
        h_data[i] = static_cast<value_type>(*(gen_a_it++));
    }

    // generate right array
    for(size_t i = 0; i < size_b; ++i)
    {
        h_data[size_a + i] = static_cast<value_type>(*(gen_b_it++));
    }

    common::device_ptr<value_type> d_data(total_size);

    runner_type runner{d_data.get(), size_a, size_b, stream};

    runner.prepare();

    state.run_before_every_iteration([&] { d_data.store(h_data); });

    state.run([&] { runner.run(); });

    state.set_throughput(total_size, sizeof(value_type));
}

#define CREATE_BENCHMARK(Value)                                                       \
    executor.queue_fn(                                                                \
        bench_naming::format_name("{lvl:device,algo:merge_inplace,value_type:" #Value \
                                  ",cfg:default_config}")                             \
            .c_str(),                                                                 \
        [=](benchmark_utils::state&& state)                                           \
        {                                                                             \
            run_merge_inplace_benchmarks<Value, inplace_runner<Value>>(               \
                std::forward<benchmark_utils::state>(state));                         \
        });

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 1, 0);

    CREATE_BENCHMARK(int8_t)
    CREATE_BENCHMARK(int16_t)
    CREATE_BENCHMARK(int32_t)
    CREATE_BENCHMARK(int64_t)
    CREATE_BENCHMARK(rocprim::int128_t)

    executor.run();
}
