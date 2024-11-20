// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SEARCH_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SEARCH_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_search.hpp>

#include <string>
#include <vector>

#include <cstddef>

template<typename Key = int, typename Config = rocprim::default_config>
struct device_search_benchmark : public config_autotune_interface
{
    size_t key_size_  = 10;
    bool   repeating_ = false;

    device_search_benchmark(size_t KeySize, bool repeating)
    {
        key_size_  = KeySize;
        repeating_ = repeating;
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:search,value_pattern:" + (repeating_ ? "repeating"s : "random"s)
            + ",key_size:" + std::to_string(key_size_)
            + ",value_type:" + std::string(Traits<Key>::name()) + ",cfg:default_config}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using key_type    = Key;
        using output_type = size_t;

        // Calculate the number of elements
        size_t size     = bytes / sizeof(key_type);
        size_t key_size = std::min(size, key_size_);

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(key_size,
                                        generate_limits<key_type>::min(),
                                        generate_limits<key_type>::max(),
                                        seed.get_0());

        std::vector<key_type> input(size);
        if(repeating_)
        {
            // Repeating similar pattern without early exits.
            keys_input[key_size - 1] = 0;
            for(size_t i = 0; i < size; i++)
            {
                input[i] = keys_input[i % key_size];
            }
            keys_input[key_size - 1] = 1;
        }
        else
        {
            input = get_random_data<key_type>(size,
                                              generate_limits<key_type>::min(),
                                              generate_limits<key_type>::max(),
                                              seed.get_0() + 1);
        }

        key_type*    d_keys_input;
        key_type*    d_input;
        output_type* d_output;
        HIP_CHECK(hipMalloc(&d_keys_input, key_size * sizeof(*d_keys_input)));
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
        HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));

        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(*d_input), hipMemcpyHostToDevice));

        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            key_size * sizeof(*d_keys_input),
                            hipMemcpyHostToDevice));

        rocprim::equal_to<key_type> compare_op;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        HIP_CHECK(rocprim::search(d_temporary_storage,
                                  temporary_storage_bytes,
                                  d_input,
                                  d_keys_input,
                                  d_output,
                                  size,
                                  key_size,
                                  compare_op,
                                  stream,
                                  false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rocprim::search(d_temporary_storage,
                                      temporary_storage_bytes,
                                      d_input,
                                      d_keys_input,
                                      d_output,
                                      size,
                                      key_size,
                                      compare_op,
                                      stream,
                                      false));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(rocprim::search(d_temporary_storage,
                                          temporary_storage_bytes,
                                          d_input,
                                          d_keys_input,
                                          d_output,
                                          size,
                                          key_size,
                                          compare_op,
                                          stream,
                                          false));
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(*d_input));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_SEARCH_PARALLEL_HPP_
