// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

template<typename Key    = int,
         typename Value  = rocprim::empty_type,
         typename Config = rocprim::default_config>
struct device_radix_sort_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_radix_sort")
            .add("key_type", primbench::name<Key>())
            .add("value_type", primbench::name<Value>())
            .add("cfg", "default");
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using key_type   = Key;
        using value_type = Value;

        size_t items = bytes / sizeof(key_type);

        // Keys
        std::vector<key_type> keys_input
            = get_random_data<key_type>(items,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed);

        // TODO: Replace all hipMallocs and hipFrees in this file with common::device_ptr

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), items * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), items * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            items * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        if constexpr(std::is_same_v<value_type, rocprim::empty_type>)
        {
            // Keys-only
            HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                        temporary_storage_bytes,
                                        d_keys_input,
                                        d_keys_output,
                                        static_cast<Value*>(nullptr),
                                        static_cast<Value*>(nullptr),
                                        items,
                                        stream));

            HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

            state.set_items(items);
            state.add_reads<key_type>(items);

            state.run(
                [&]
                {
                    HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input,
                                                d_keys_output,
                                                static_cast<Value*>(nullptr),
                                                static_cast<Value*>(nullptr),
                                                items,
                                                stream));
                });
        }
        else
        {
            // Key-value pairs
            std::vector<value_type> values_input(items);
            for(size_t i = 0; i < items; ++i)
                values_input[i] = value_type(i);

            value_type* d_values_input;
            value_type* d_values_output;
            HIP_CHECK(
                hipMalloc(reinterpret_cast<void**>(&d_values_input), items * sizeof(value_type)));
            HIP_CHECK(
                hipMalloc(reinterpret_cast<void**>(&d_values_output), items * sizeof(value_type)));
            HIP_CHECK(hipMemcpy(d_values_input,
                                values_input.data(),
                                items * sizeof(value_type),
                                hipMemcpyHostToDevice));

            HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                        temporary_storage_bytes,
                                        d_keys_input,
                                        d_keys_output,
                                        d_values_input,
                                        d_values_output,
                                        items,
                                        stream));

            HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

            state.set_items(items);
            state.add_reads<key_type>(items);
            state.add_reads<value_type>(items);

            state.run(
                [&]
                {
                    HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                                temporary_storage_bytes,
                                                d_keys_input,
                                                d_keys_output,
                                                d_values_input,
                                                d_values_output,
                                                items,
                                                stream));
                });

            HIP_CHECK(hipFree(d_values_input));
            HIP_CHECK(hipFree(d_values_output));
        }

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }

private:
    template<typename K = Key, typename V = Value>
    static hipError_t invoke_radix_sort(void*       d_temporary_storage,
                                        size_t&     temp_storage_bytes,
                                        K*          keys_input,
                                        K*          keys_output,
                                        V*          values_input,
                                        V*          values_output,
                                        size_t      items,
                                        hipStream_t stream)
    {
        if constexpr(std::is_same_v<V, rocprim::empty_type>)
        {
            if constexpr(common::is_custom_type<K>::value)
            {
                return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                        temp_storage_bytes,
                                                        keys_input,
                                                        keys_output,
                                                        items,
                                                        custom_type_decomposer<K>{},
                                                        stream);
            }
            else
            {
                return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                        temp_storage_bytes,
                                                        keys_input,
                                                        keys_output,
                                                        items,
                                                        0,
                                                        sizeof(K) * 8,
                                                        stream);
            }
        }
        else
        {
            if constexpr(common::is_custom_type<K>::value)
            {
                return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                         temp_storage_bytes,
                                                         keys_input,
                                                         keys_output,
                                                         values_input,
                                                         values_output,
                                                         items,
                                                         custom_type_decomposer<K>{},
                                                         stream);
            }
            else
            {
                return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                         temp_storage_bytes,
                                                         keys_input,
                                                         keys_output,
                                                         values_input,
                                                         values_output,
                                                         items,
                                                         0,
                                                         sizeof(K) * 8,
                                                         stream);
            }
        }
    }
};
