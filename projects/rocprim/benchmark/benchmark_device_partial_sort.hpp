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
#include <rocprim/device/device_partial_sort.hpp>
#include <rocprim/functional.hpp>

#include <cstddef>
#include <string>
#include <vector>

template<typename Key = int, typename Config = rocprim::default_config>
struct device_partial_sort_benchmark : public primbench::benchmark_interface
{
    device_partial_sort_benchmark(bool small_n) : m_small_n(small_n) {}

    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_partial_sort")
            .add("small_n", m_small_n)
            .add("key_type", primbench::name<Key>())
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {

        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type = Key;

        size_t items = bytes / sizeof(key_type);

        size_t middle = 10;

        if(!m_small_n)
        {
            middle = items / 2;
        }

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(items,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed);

        key_type* d_keys_input;
        key_type* d_keys_new_data;
        HIP_CHECK(hipMalloc(&d_keys_input, items * sizeof(*d_keys_input)));
        HIP_CHECK(hipMalloc(&d_keys_new_data, items * sizeof(*d_keys_new_data)));

        HIP_CHECK(hipMemcpy(d_keys_new_data,
                            keys_input.data(),
                            items * sizeof(*d_keys_input),
                            hipMemcpyHostToDevice));

        rocprim::less<key_type> lesser_op;
        void*                   d_temporary_storage     = nullptr;
        size_t                  temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::partial_sort(d_temporary_storage,
                                        temporary_storage_bytes,
                                        d_keys_input,
                                        middle,
                                        items,
                                        lesser_op,
                                        stream,
                                        false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        state.run_before_every_iteration(
            [&]
            {
                HIP_CHECK(hipMemcpy(d_keys_input,
                                    d_keys_new_data,
                                    items * sizeof(*d_keys_input),
                                    hipMemcpyDeviceToDevice));
            });

        state.set_items(items);
        state.add_reads<key_type>(items);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::partial_sort(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input,
                                                middle,
                                                items,
                                                lesser_op,
                                                stream,
                                                false));
            });

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_new_data));
    }

private:
    bool m_small_n = false;
};
