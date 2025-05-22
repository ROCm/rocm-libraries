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

#ifndef ROCPRIM_BENCHMARK_DEVICE_PARTIAL_SORT_COPY_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_PARTIAL_SORT_COPY_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/device_partial_sort.hpp>
#include <rocprim/functional.hpp>

#include <cstddef>
#include <string>
#include <vector>

template<typename Key = int, typename Config = rocprim::default_config>
struct device_partial_sort_copy_benchmark : public benchmark_utils::autotune_interface
{
    bool small_n = false;

    device_partial_sort_copy_benchmark(bool SmallN)
    {
        small_n = SmallN;
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:partial_sort_copy,nth:" + (small_n ? "small"s : "half"s)
            + ",key_type:" + std::string(Traits<Key>::name()) + ",cfg:default_config}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using key_type = Key;

        // Calculate the number of elements
        size_t size   = bytes / sizeof(key_type);
        size_t middle = 10;

        if(!small_n)
        {
            middle = size / 2;
        }

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        common::generate_limits<key_type>::min(),
                                        common::generate_limits<key_type>::max(),
                                        seed.get_0());

        common::device_ptr<key_type> d_keys_input(keys_input);
        common::device_ptr<key_type> d_keys_output(size);

        rocprim::less<key_type> lesser_op;
        common::device_ptr<void> d_temporary_storage;
        size_t                  temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::partial_sort_copy(d_temporary_storage.get(),
                                             temporary_storage_bytes,
                                             d_keys_input.get(),
                                             d_keys_output.get(),
                                             middle,
                                             size,
                                             lesser_op,
                                             stream,
                                             false));

        d_temporary_storage.resize(temporary_storage_bytes);

        state.run(
            [&]
            {
                HIP_CHECK(rocprim::partial_sort_copy(d_temporary_storage.get(),
                                                     temporary_storage_bytes,
                                                     d_keys_input.get(),
                                                     d_keys_output.get(),
                                                     middle,
                                                     size,
                                                     lesser_op,
                                                     stream,
                                                     false));
            });

        state.set_throughput(size, sizeof(key_type));
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_PARTIAL_SORT_COPY_PARALLEL_HPP_
