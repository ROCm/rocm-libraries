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

#ifndef ROCPRIM_BENCHMARK_DEVICE_NTH_ELEMENT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_NTH_ELEMENT_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_nth_element.hpp>

#include <string>
#include <vector>

#include <cstddef>

template<typename Key = int, typename Config = rocprim::default_config>
struct device_nth_element_benchmark : public config_autotune_interface
{
    bool small_n = false;

    device_nth_element_benchmark(bool SmallN)
    {
        small_n = SmallN;
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:nth_element,nth:" + (small_n ? "small"s : "large"s)
            + ",key_type:" + std::string(Traits<Key>::name()) + ",cfg:default_config}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using key_type = Key;

        // Calculate the number of elements 
        size_t size = bytes / sizeof(key_type);
        size_t nth = 10;

        if(!small_n)
        {
            nth = size / 2;
        }

        // Generate data
        std::vector<key_type> keys_input
            = get_random_data<key_type>(size,
                                        generate_limits<key_type>::min(),
                                        generate_limits<key_type>::max(),
                                        seed.get_0());

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(*d_keys_input)));
        HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(*d_keys_output)));

        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(*d_keys_input),
                            hipMemcpyHostToDevice));

        ::rocprim::less<key_type> lesser_op;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rocprim::nth_element(d_temporary_storage,
                                       temporary_storage_bytes,
                                       d_keys_input,
                                       d_keys_output,
                                       nth,
                                       size,
                                       lesser_op,
                                       stream,
                                       false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rocprim::nth_element(d_temporary_storage,
                                           temporary_storage_bytes,
                                           d_keys_input,
                                           d_keys_output,
                                           nth,
                                           size,
                                           lesser_op,
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
                HIP_CHECK(rocprim::nth_element(d_temporary_storage,
                                               temporary_storage_bytes,
                                               d_keys_input,
                                               d_keys_output,
                                               nth,
                                               size,
                                               lesser_op,
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

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(*d_keys_input));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_NTH_ELEMENT_PARALLEL_HPP_
