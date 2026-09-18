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
#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_find_end.hpp>
#include <rocprim/functional.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

template<typename Key = int, typename Config = rocprim::default_config>
struct device_find_end_benchmark : public primbench::benchmark_interface
{
    device_find_end_benchmark(size_t key_size, bool repeating)
        : m_key_size(key_size), m_repeating(repeating)
    {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_find_end")
            .add("repeating", m_repeating)
            .add("key_size", m_key_size)
            .add("value_type", primbench::name<Key>())
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type    = Key;
        using output_type = size_t;

        size_t items    = bytes / sizeof(key_type);
        size_t key_size = std::min(items, m_key_size);

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(key_size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed);

        std::vector<key_type> input(items);
        if(m_repeating)
        {
            // Repeating similar pattern without early exits.
            keys_input[0] = 0;
            for(size_t i = 0; i < items; ++i)
            {
                input[i] = keys_input[i % key_size];
            }
            keys_input[0] = 1;
        }
        else
        {
            input = get_random_data<key_type>(items,
                                              common::generate_limits<key_type>::min(),
                                              common::generate_limits<key_type>::max(),
                                              seed + 1);
        }

        common::device_ptr<key_type>    d_keys_input(keys_input);
        common::device_ptr<key_type>    d_input(input);
        common::device_ptr<output_type> d_output(1);

        rocprim::equal_to<key_type> compare_op;

        size_t temporary_storage_bytes = 0;

        HIP_CHECK(rocprim::find_end(nullptr,
                                    temporary_storage_bytes,
                                    d_input.get(),
                                    d_keys_input.get(),
                                    d_output.get(),
                                    items,
                                    m_key_size,
                                    compare_op,
                                    stream,
                                    false));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<key_type>(items);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::find_end(d_temporary_storage.get(),
                                            temporary_storage_bytes,
                                            d_input.get(),
                                            d_keys_input.get(),
                                            d_output.get(),
                                            items,
                                            m_key_size,
                                            compare_op,
                                            stream,
                                            false));
            });
    }

private:
    size_t m_key_size  = 10;
    bool   m_repeating = false;
};
