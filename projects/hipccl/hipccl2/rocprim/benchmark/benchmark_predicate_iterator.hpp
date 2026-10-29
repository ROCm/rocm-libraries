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

#pragma once

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/predicate_iterator.hpp"
#include "../common/utils_custom_type.hpp"
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

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

    void operator()(T* d_input, T* d_output, const size_t items, const hipStream_t stream)
    {
        auto t_it
            = rocprim::make_transform_iterator(d_input, transform_op<T, Predicate, Transform>{});
        HIP_CHECK(rocprim::transform(t_it, d_output, items, rocprim::identity<T>{}, stream));
    }
};

template<typename T, typename Predicate, typename Transform>
struct read_predicate_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t items, const hipStream_t stream)
    {
        auto t_it = rocprim::make_transform_iterator(d_input, Transform{});
        auto r_it = rocprim::make_predicate_iterator(t_it, d_input, Predicate{});
        HIP_CHECK(rocprim::transform(r_it, d_output, items, rocprim::identity<T>{}, stream));
    }
};

template<typename T, typename Predicate, typename Transform>
struct write_predicate_it
{
    using value_type = T;

    void operator()(T* d_input, T* d_output, const size_t items, const hipStream_t stream)
    {
        auto t_it = rocprim::make_transform_iterator(d_input, Transform{});
        auto w_it = rocprim::make_predicate_iterator(d_output, d_input, Predicate{});
        HIP_CHECK(rocprim::transform(t_it, w_it, items, rocprim::identity<T>{}, stream));
    }
};

template<typename T>
struct subalgo_name;

template<typename T, typename Predicate, typename Transform>
struct subalgo_name<transform_it<T, Predicate, Transform>>
{
    static std::string value()
    {
        return "transform";
    }
};

template<typename T, typename Predicate, typename Transform>
struct subalgo_name<read_predicate_it<T, Predicate, Transform>>
{
    static std::string value()
    {
        return "read_predicate";
    }
};

template<typename T, typename Predicate, typename Transform>
struct subalgo_name<write_predicate_it<T, Predicate, Transform>>
{
    static std::string value()
    {
        return "write_predicate";
    }
};

template<typename Subalgo>
std::string get_subalgo_name()
{
    return subalgo_name<Subalgo>::value();
}

template<typename IteratorBenchmark, int C, typename Config = rocprim::default_config>
struct predicate_iterator_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "predicate_iterator")
            .add("subalgo", get_subalgo_name<IteratorBenchmark>())
            .add("percent", C)
            .add("key_type", primbench::name<T>())
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        size_t items = bytes / sizeof(T);

        const auto     random_range = limit_random_range<T>(0, 99);
        std::vector<T> input
            = get_random_data<T>(items, random_range.first, random_range.second, seed);
        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(items);

        state.set_items(items);
        state.add_reads<T>(items);

        state.run([&] { IteratorBenchmark{}(d_input.get(), d_output.get(), items, stream); });
    }

private:
    using T = typename IteratorBenchmark::value_type;
};
