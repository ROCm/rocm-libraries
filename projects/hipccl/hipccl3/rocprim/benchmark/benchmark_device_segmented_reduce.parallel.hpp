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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_REDUCE_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SEGMENTED_REDUCE_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_segmented_reduce.hpp>

#include <iostream>
#include <limits>
#include <locale>
#include <string>
#include <vector>

namespace rp = rocprim;

constexpr const char* get_reduce_method_name(rocprim::block_reduce_algorithm alg)
{
    switch(alg)
    {
        case rocprim::block_reduce_algorithm::raking_reduce: return "raking_reduce";
        case rocprim::block_reduce_algorithm::raking_reduce_commutative_only:
            return "raking_reduce_commutative_only";
        case rocprim::block_reduce_algorithm::using_warp_reduce:
            return "using_warp_reduce";
            // Not using `default: ...` because it kills effectiveness of -Wswitch
    }
    return "unknown_algorithm";
}

template<typename Config>
std::string config_name()
{
    const rocprim::detail::reduce_config_params config = Config();
    return "{bs:" + std::to_string(config.reduce_config.block_size)
           + ",ipt:" + std::to_string(config.reduce_config.items_per_thread)
           + ",method:" + std::string(get_reduce_method_name(config.block_reduce_method)) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T              = int,
         typename BinaryFunction = rocprim::plus<T>,
         typename Config         = rocprim::default_config>
struct device_segmented_reduce_benchmark : public config_autotune_interface
{

    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:segmented_reduce,key_type:"
                                         + std::string(Traits<T>::name())
                                         + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    void run_benchmark(benchmark::State&   state,
                       size_t              desired_segment,
                       size_t              bytes,
                       const managed_seed& seed,
                       hipStream_t         stream) const
    {
        using offset_type = int;
        using value_type  = T;

        // Calculate the number of elements
        size_t size = bytes / sizeof(T);

        // Generate data
        engine_type gen(seed.get_0());

        const double avg_segment_length = static_cast<double>(size) / desired_segment;
        std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

        std::vector<offset_type> offsets;
        unsigned int             segments_count = 0;
        size_t                   offset         = 0;
        while(offset < size)
        {
            const size_t segment_length = std::round(segment_length_dis(gen));
            offsets.push_back(offset);
            segments_count++;
            offset += segment_length;
        }
        offsets.push_back(size);

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

        offset_type* d_offsets;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_offsets),
                            (segments_count + 1) * sizeof(offset_type)));
        HIP_CHECK(hipMemcpy(d_offsets,
                            offsets.data(),
                            (segments_count + 1) * sizeof(offset_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_input), size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        value_type* d_aggregates_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_aggregates_output),
                            segments_count * sizeof(value_type)));

        rocprim::plus<value_type> reduce_op;
        value_type                init(0);

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        HIP_CHECK(rp::segmented_reduce<Config>(d_temporary_storage,
                                               temporary_storage_bytes,
                                               d_values_input,
                                               d_aggregates_output,
                                               segments_count,
                                               d_offsets,
                                               d_offsets + 1,
                                               reduce_op,
                                               init,
                                               stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rp::segmented_reduce<Config>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_values_input,
                                                   d_aggregates_output,
                                                   segments_count,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   reduce_op,
                                                   init,
                                                   stream));
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
                HIP_CHECK(rp::segmented_reduce<Config>(d_temporary_storage,
                                                       temporary_storage_bytes,
                                                       d_values_input,
                                                       d_aggregates_output,
                                                       segments_count,
                                                       d_offsets,
                                                       d_offsets + 1,
                                                       reduce_op,
                                                       init,
                                                       stream));
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

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(value_type));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_offsets));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_aggregates_output));
    }

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        constexpr std::array<size_t, 5> desired_segments{1, 10, 100, 1000, 10000};

        for(const auto desired_segment : desired_segments)
        {
            run_benchmark(state, desired_segment, bytes, seed, stream);
        }
    }
};

#endif