// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
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
struct device_radix_sort_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name(
            "{lvl:device,algo:radix_sort,key_type:" + std::string(Traits<Key>::name())
            + ",value_type:" + std::string(Traits<Value>::name()) + ",cfg: default_config}");
    }

    // keys benchmark
    template<typename val = Value>
    auto do_run(benchmark_utils::state&& state) const
        -> std::enable_if_t<std::is_same<val, ::rocprim::empty_type>::value, void>
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using key_type = Key;

        // Calculate the number of elements
        size_t size = bytes / sizeof(key_type);

        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed.get_0());

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));

        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                    temporary_storage_bytes,
                                    d_keys_input,
                                    d_keys_output,
                                    static_cast<Value*>(nullptr),
                                    static_cast<Value*>(nullptr),
                                    size,
                                    stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                            temporary_storage_bytes,
                                            d_keys_input,
                                            d_keys_output,
                                            static_cast<Value*>(nullptr),
                                            static_cast<Value*>(nullptr),
                                            size,
                                            stream));
            });

        state.set_throughput(size, sizeof(key_type));

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }

    // pairs benchmark
    template<typename val = Value>
    auto do_run(benchmark_utils::state&& state) const
        -> std::enable_if_t<!std::is_same<val, ::rocprim::empty_type>::value, void>
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using key_type   = Key;
        using value_type = Value;

        // Calculate the number of elements
        size_t size = bytes / sizeof(key_type);

        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed.get_0());

        std::vector<value_type> values_input(size);
        for(size_t i = 0; i < size; ++i)
        {
            values_input[i] = value_type(i);
        }

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        value_type* d_values_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_input), size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_output), size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                    temporary_storage_bytes,
                                    d_keys_input,
                                    d_keys_output,
                                    d_values_input,
                                    d_values_output,
                                    size,
                                    stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK(invoke_radix_sort(d_temporary_storage,
                                            temporary_storage_bytes,
                                            d_keys_input,
                                            d_keys_output,
                                            d_values_input,
                                            d_values_output,
                                            size,
                                            stream));
            });

        state.set_throughput(size, sizeof(key_type) + sizeof(value_type));

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }

    void run(benchmark_utils::state&& state) override
    {
        do_run(std::forward<benchmark_utils::state>(state));
    }

private:
    template<typename K = Key, typename V = Value>
    static auto invoke_radix_sort(void*       d_temporary_storage,
                                  size_t&     temp_storage_bytes,
                                  K*          keys_input,
                                  K*          keys_output,
                                  V*          values_input,
                                  V*          values_output,
                                  size_t      size,
                                  hipStream_t stream)
        -> std::enable_if_t<!common::is_custom_type<K>::value
                                && std::is_same<V, rocprim::empty_type>::value,
                            hipError_t>
    {
        (void)values_input;
        (void)values_output;
        return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                temp_storage_bytes,
                                                keys_input,
                                                keys_output,
                                                size,
                                                0,
                                                sizeof(K) * 8,
                                                stream);
    }

    template<typename K = Key, typename V = Value>
    static auto invoke_radix_sort(void*       d_temporary_storage,
                                  size_t&     temp_storage_bytes,
                                  K*          keys_input,
                                  K*          keys_output,
                                  V*          values_input,
                                  V*          values_output,
                                  size_t      size,
                                  hipStream_t stream)
        -> std::enable_if_t<common::is_custom_type<K>::value
                                && std::is_same<V, rocprim::empty_type>::value,
                            hipError_t>
    {
        (void)values_input;
        (void)values_output;
        return rocprim::radix_sort_keys<Config>(d_temporary_storage,
                                                temp_storage_bytes,
                                                keys_input,
                                                keys_output,
                                                size,
                                                custom_type_decomposer<K>{},
                                                stream);
    }

    template<typename K = Key, typename V = Value>
    static auto invoke_radix_sort(void*       d_temporary_storage,
                                  size_t&     temp_storage_bytes,
                                  K*          keys_input,
                                  K*          keys_output,
                                  V*          values_input,
                                  V*          values_output,
                                  size_t      size,
                                  hipStream_t stream)
        -> std::enable_if_t<!common::is_custom_type<K>::value
                                && !std::is_same<V, rocprim::empty_type>::value,
                            hipError_t>
    {
        return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                 temp_storage_bytes,
                                                 keys_input,
                                                 keys_output,
                                                 values_input,
                                                 values_output,
                                                 size,
                                                 0,
                                                 sizeof(K) * 8,
                                                 stream);
    }

    template<typename K = Key, typename V = Value>
    static auto invoke_radix_sort(void*       d_temporary_storage,
                                  size_t&     temp_storage_bytes,
                                  K*          keys_input,
                                  K*          keys_output,
                                  V*          values_input,
                                  V*          values_output,
                                  size_t      size,
                                  hipStream_t stream)
        -> std::enable_if_t<common::is_custom_type<K>::value
                                && !std::is_same<V, rocprim::empty_type>::value,
                            hipError_t>
    {
        return rocprim::radix_sort_pairs<Config>(d_temporary_storage,
                                                 temp_storage_bytes,
                                                 keys_input,
                                                 keys_output,
                                                 values_input,
                                                 values_output,
                                                 size,
                                                 custom_type_decomposer<K>{},
                                                 stream);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_PARALLEL_HPP_
