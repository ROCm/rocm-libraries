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

#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_segmented_reduce.hpp>

#include <iostream>
#include <limits>
#include <locale>
#include <numeric>
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
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread)
           + ",method:" + std::string(get_reduce_method_name(config.block_reduce_method)) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T,
         typename BinaryFunction = rocprim::plus<T>,
         typename Config         = rocprim::default_config>
struct device_segmented_reduce_benchmark : public benchmark_utils::autotune_interface
{
private:
    std::vector<size_t> desired_segments;
    size_t              total_size;

public:
    device_segmented_reduce_benchmark()
    {
        this->desired_segments = std::vector<size_t>{1, 10, 100, 1000, 10000};
    }

    device_segmented_reduce_benchmark(size_t desired_segment)
    {
        desired_segments.push_back(desired_segment);
    }

    std::string name() const override
    {
        return bench_naming::format_name(
            "{lvl:device,algo:segmented_reduce,key_type:" + std::string(Traits<T>::name())
            + (desired_segments.size() == 1
                   ? ",segment_count:" + std::to_string(desired_segments[0])
                   : "")
            + ",cfg:" + config_name<Config>() + "}");
    }

    void run_benchmark(benchmark_utils::state&& state, size_t desired_segment)
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

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

        common::device_ptr<offset_type> d_offsets(offsets);

        common::device_ptr<value_type> d_values_input(values_input);

        common::device_ptr<value_type> d_aggregates_output(segments_count);

        rocprim::plus<value_type> reduce_op;
        value_type                init(0);

        size_t temporary_storage_bytes = 0;

        HIP_CHECK(rp::segmented_reduce<Config>(nullptr,
                                               temporary_storage_bytes,
                                               d_values_input.get(),
                                               d_aggregates_output.get(),
                                               segments_count,
                                               d_offsets.get(),
                                               d_offsets.get() + 1,
                                               reduce_op,
                                               init,
                                               stream));

        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK(rp::segmented_reduce<Config>(d_temporary_storage.get(),
                                                       temporary_storage_bytes,
                                                       d_values_input.get(),
                                                       d_aggregates_output.get(),
                                                       segments_count,
                                                       d_offsets.get(),
                                                       d_offsets.get() + 1,
                                                       reduce_op,
                                                       init,
                                                       stream));
            });

        total_size += size;
    }

    void run(benchmark_utils::state&& state) override
    {
        total_size = 0;

        for(const auto desired_segment : desired_segments)
        {
            run_benchmark(std::forward<benchmark_utils::state>(state), desired_segment);
        }

        state.set_throughput(total_size, sizeof(T));
    }
};

#endif