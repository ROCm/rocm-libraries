// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/utils_data_generation.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_find_if.hpp>
#include <rocprim/functional.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

template<typename Config>
constexpr auto config_name()
{
    if constexpr(std::is_same_v<Config, rocprim::default_config>)
    {
        return std::string("default");
    }
    else
    {
        auto config = Config();
        return primbench::json{}
            .add("bs", config.kernel_config.block_size)
            .add("ipt", config.kernel_config.items_per_thread);
    }
}

template<typename T>
struct value_predicate_t
{
    T                    value_;
    rocprim::equal_to<T> equal_op{};

    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
    bool                 operator()(const T x) const
    {
        return equal_op(x, value_);
    }
};

template<typename T, typename Config = rocprim::default_config>
struct device_find_if_benchmark : public primbench::benchmark_interface
{
    device_find_if_benchmark(double first_occurrence)
    {
        m_first_occurrences.push_back(first_occurrence);
    }

    device_find_if_benchmark(const std::vector<double>& first_occurrences)
        : m_first_occurrences(first_occurrences)
    {}

    primbench::json meta() const override
    {
        auto j = primbench::json{}
                     .add("lvl", "device")
                     .add("algo", "device_find_if")
                     .add("value_type", primbench::name<T>())
                     .add("cfg", config_name<Config>());

        if(m_first_occurrences.size() == 1)
        {
            j.add("first_occurrence", m_first_occurrences[0]);
        }

        return j;
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using type        = T;
        using output_type = size_t;
        using predicate_t = value_predicate_t<T>;

        const size_t items = bytes / sizeof(type);

        const type find_value{42};
        auto       predicate = predicate_t{find_value};

        primbench::log("Generating inputs");
        std::vector<std::vector<type>> inputs(m_first_occurrences.size());
        for(size_t fi = 0; fi < m_first_occurrences.size(); ++fi)
        {
            std::vector<type> input(items);
            const auto        starting_point = m_first_occurrences[fi];
            const size_t      items1
                = starting_point >= 1.0 ? items : static_cast<size_t>(items * starting_point);
            const size_t items2 = items - items1;
            if(items1 > 0)
            {
                auto input1 = get_random_data<type>(items1, find_value + 1, 101, seed);
                std::copy(input1.begin(), input1.end(), input.begin());
            }
            if(items2 > 0)
            {
                auto input2 = get_random_data<type>(items2, 0, find_value, seed + 1);
                std::copy(input2.begin(), input2.end(), input.begin() + items1);
            }
        }

        primbench::log("Creating d_inputs");
        std::vector<type*> d_inputs(m_first_occurrences.size());
        primbench::log("Filling d_inputs");
        for(size_t fi = 0; fi < m_first_occurrences.size(); ++fi)
        {
            type* d_input;
            HIP_CHECK(hipMalloc(&d_input, items * sizeof(*d_input)));
            HIP_CHECK(hipMemcpyAsync(d_input,
                                     inputs[fi].data(),
                                     inputs[fi].size() * sizeof(*d_input),
                                     hipMemcpyHostToDevice,
                                     stream));
            d_inputs[fi] = d_input;
        }

        output_type* d_output;
        primbench::log("Allocating d_output");
        HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        const auto launch = [&](const type* d_input)
        {
            HIP_CHECK(rocprim::find_if<Config>(d_temporary_storage,
                                               temporary_storage_bytes,
                                               d_input,
                                               d_output,
                                               items,
                                               predicate,
                                               stream));
        };

        size_t max_temporary_storage_bytes = 0;
        primbench::log("Calculating max_temporary_storage_bytes");
        launch(d_inputs[0]);
        max_temporary_storage_bytes
            = std::max(max_temporary_storage_bytes, temporary_storage_bytes);
        temporary_storage_bytes = max_temporary_storage_bytes;
        primbench::log("Allocating d_temporary_storage");
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        // Only a part of data (before the first occurrence) must be actually processed. In ideal
        // cases when no thread blocks do unneeded work (i.e. exit early once the match is found),
        // performance for different values of first_occurrence must be similar.
        size_t sum_effective_size = 0;
        for(double first_occurrence : m_first_occurrences)
        {
            sum_effective_size += static_cast<size_t>(items * first_occurrence);
        }

        state.set_items(sum_effective_size);
        state.add_reads<type>(sum_effective_size);

        state.run(
            [&]
            {
                for(size_t fi = 0; fi < m_first_occurrences.size(); ++fi)
                {
                    launch(d_inputs[fi]);
                }
            });

        for(size_t fi = 0; fi < m_first_occurrences.size(); ++fi)
        {
            HIP_CHECK(hipFree(d_inputs[fi]));
        }
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temporary_storage));
    }

private:
    std::vector<double> m_first_occurrences;
};

template<typename T, unsigned int BlockSize>
struct device_find_if_benchmark_generator
{

    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        using generated_config = rocprim::find_if_config<BlockSize, ItemsPerThread>;

        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            std::vector<double> first_occurrences{0.1, 0.5, 1.0};
            storage.emplace_back(
                std::make_unique<device_find_if_benchmark<T, generated_config>>(first_occurrences));
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        static constexpr unsigned int min_items_per_thread = 1;
        static constexpr unsigned int max_items_per_thread = 16;
        static_for_each<make_index_range<unsigned int, min_items_per_thread, max_items_per_thread>,
                        create_ipt>(storage);
    }
};
