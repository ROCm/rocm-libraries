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

#pragma once

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_find_first_of.hpp>
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

template<typename T, typename Config = rocprim::default_config>
struct device_find_first_of_benchmark : public primbench::benchmark_interface
{
    device_find_first_of_benchmark(size_t keys_size, double first_occurrence)
    {
        m_keys_sizes.push_back(keys_size);
        m_first_occurrences.push_back(first_occurrence);
    }

    device_find_first_of_benchmark(const std::vector<size_t>& keys_sizes,
                                   const std::vector<double>& first_occurrences)
        : m_keys_sizes(keys_sizes), m_first_occurrences(first_occurrences)
    {}

    primbench::json meta() const override
    {
        auto j = primbench::json{}
                     .add("lvl", "device")
                     .add("algo", "device_find_first_of")
                     .add("value_type", primbench::name<T>())
                     .add("cfg", config_name<Config>());

        if(m_keys_sizes.size() == 1)
        {
            j.add("keys_size", m_keys_sizes[0]);
        }
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
        using key_type    = T;
        using output_type = size_t;

        const size_t items = bytes / sizeof(type);

        primbench::log("Calculating max_keys_size");
        const size_t max_keys_size = *std::max_element(m_keys_sizes.begin(), m_keys_sizes.end());

        primbench::log("Generating key_input");
        std::vector<key_type> key_input = get_random_data<key_type>(max_keys_size, 0, 100, seed);
        primbench::log("Generating input");
        std::vector<type> input
            = get_random_data<type>(items, 101, common::generate_limits<type>::max(), seed);

        primbench::log("Creating d_inputs");
        std::vector<type*> d_inputs(m_first_occurrences.size());
        primbench::log("Filling d_inputs");
        for(size_t fi = 0; fi < m_first_occurrences.size(); ++fi)
        {
            type* d_input;
            HIP_CHECK(hipMalloc(&d_input, items * sizeof(*d_input)));
            HIP_CHECK(hipMemcpyAsync(d_input,
                                     input.data(),
                                     input.size() * sizeof(*d_input),
                                     hipMemcpyHostToDevice,
                                     stream));
            // Set the first occurrence of keys in input
            const size_t p = static_cast<size_t>(items * m_first_occurrences[fi]);
            if(p < items)
            {
                const type key = key_input[0];
                HIP_CHECK(hipMemcpyAsync(d_input + p,
                                         &key,
                                         sizeof(*d_input),
                                         hipMemcpyHostToDevice,
                                         stream));
            }
            d_inputs[fi] = d_input;
        }

        key_type*    d_key_input;
        output_type* d_output;
        primbench::log("Allocating d_key_input and d_output");
        HIP_CHECK(hipMalloc(&d_key_input, max_keys_size * sizeof(*d_key_input)));
        HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));

        primbench::log("Copying key_input to d_key_input");
        HIP_CHECK(hipMemcpy(d_key_input,
                            key_input.data(),
                            key_input.size() * sizeof(*d_key_input),
                            hipMemcpyHostToDevice));

        ::rocprim::equal_to<type> compare_op;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        const auto launch = [&](size_t key_size, const type* d_input)
        {
            HIP_CHECK(rocprim::find_first_of<Config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_input,
                                                     d_key_input,
                                                     d_output,
                                                     input.size(),
                                                     key_size,
                                                     compare_op,
                                                     stream));
        };

        size_t max_temporary_storage_bytes = 0;
        primbench::log("Calculating max_temporary_storage_bytes");
        for(size_t keys_size : m_keys_sizes)
        {
            launch(keys_size, d_inputs[0]);
            max_temporary_storage_bytes
                = std::max(max_temporary_storage_bytes, temporary_storage_bytes);
        }
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

        // TODO: Remove?
        // Each input is read once but all keys are read by all threads so performance is likely
        // compute-bound or bound by cache bandwidth for reading keys rather than reading inputs.
        // Let's additionally report the rate of comparisons to see if it reaches a plateau with
        // increasing keys_size.
        // size_t sum_keys_size = 0;
        // for(size_t keys_size : m_keys_sizes)
        // {
        //     sum_keys_size += keys_size;
        // }
        // state.gbench_state.counters["comparisons_per_second"] = benchmark::Counter(
        //     static_cast<double>(state.gbench_state.iterations() * state.batch_iterations
        //                         * sum_effective_size * sum_keys_size),
        //     benchmark::Counter::kIsRate);

        state.set_items(sum_effective_size);
        state.add_reads<type>(sum_effective_size);

        state.run(
            [&]
            {
                for(size_t fi = 0; fi < m_first_occurrences.size(); ++fi)
                {
                    for(size_t keys_size : m_keys_sizes)
                    {
                        launch(keys_size, d_inputs[fi]);
                    }
                }
            });

        for(size_t fi = 0; fi < m_first_occurrences.size(); ++fi)
        {
            HIP_CHECK(hipFree(d_inputs[fi]));
        }
        HIP_CHECK(hipFree(d_key_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temporary_storage));
    }

private:
    std::vector<size_t> m_keys_sizes;
    std::vector<double> m_first_occurrences;
};

template<typename T, unsigned int BlockSize>
struct device_find_first_of_benchmark_generator
{

    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        using generated_config = rocprim::find_first_of_config<BlockSize, ItemsPerThread>;

        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            std::vector<size_t> keys_sizes{1, 10, 100, 1000};
            std::vector<double> first_occurrences{0.1, 0.5, 1.0};
            storage.emplace_back(
                std::make_unique<device_find_first_of_benchmark<T, generated_config>>(
                    keys_sizes,
                    first_occurrences));
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
