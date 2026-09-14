// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common/utils_device_ptr.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/device_segmented_scan.hpp>

#include <numeric>
#include <vector>

constexpr inline const char* get_block_scan_algorithm_name(rocprim::block_scan_algorithm alg)
{
    switch(alg)
    {
        case rocprim::block_scan_algorithm::using_warp_scan:
            return "block_scan_algorithm::using_warp_scan";
        case rocprim::block_scan_algorithm::reduce_then_scan:
            return "block_scan_algorithm::reduce_then_scan";
            // Not using `default: ...` because it kills effectiveness of -Wswitch
    }
    return "default_algorithm";
}

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
            .add("ipt", config.kernel_config.items_per_thread)
            .add("method", get_block_scan_algorithm_name(config.block_scan_method));
    }
}

template<bool Exclusive,
         typename T,
         typename BinaryFunction = rocprim::plus<T>,
         typename Config         = rocprim::default_config>
struct device_segmented_scan_benchmark : public primbench::benchmark_interface
{
    device_segmented_scan_benchmark()
        : m_desired_segments(std::vector<size_t>{1, 10, 100, 1000, 10000})
    {}

    device_segmented_scan_benchmark(size_t desired_segment)
    {
        m_desired_segments.push_back(desired_segment);
    }

    primbench::json meta() const override
    {
        auto j = primbench::json{}
                     .add("lvl", "device")
                     .add("algo", "device_segmented_scan")
                     .add("key_type", primbench::name<T>())
                     .add("exclusive", Exclusive)
                     .add("cfg", config_name<Config>());

        if(m_desired_segments.size() == 1)
        {
            j.add("segment_count", m_desired_segments[0]);
        }

        return j;
    }

    template<class OffsetIterator>
    inline hipError_t run_device_segmented_scan(void*             temporary_storage,
                                                size_t&           storage_size,
                                                T*                input,
                                                T*                output,
                                                const size_t      segments,
                                                OffsetIterator    begin_offsets,
                                                OffsetIterator    end_offsets,
                                                const T           initial_value,
                                                const hipStream_t stream,
                                                const bool        debug_synchronous = false)
    {
        if constexpr(Exclusive)
        {
            return rocprim::segmented_exclusive_scan<Config>(temporary_storage,
                                                             storage_size,
                                                             input,
                                                             output,
                                                             segments,
                                                             begin_offsets,
                                                             end_offsets,
                                                             initial_value,
                                                             BinaryFunction{},
                                                             stream,
                                                             debug_synchronous);
        }
        else
        {
            return rocprim::segmented_inclusive_scan<Config>(temporary_storage,
                                                             storage_size,
                                                             input,
                                                             output,
                                                             segments,
                                                             begin_offsets,
                                                             end_offsets,
                                                             BinaryFunction{},
                                                             stream,
                                                             debug_synchronous);
        }
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using offset_type = int;
        using value_type  = T;

        const size_t items            = bytes / sizeof(T);
        const int    num_input_arrays = m_desired_segments.size();

        // Shared input values, generated once
        engine_type gen(seed);

        std::vector<value_type> values_input(items);
        std::iota(values_input.begin(), values_input.end(), 0);
        common::device_ptr<value_type> d_values_input(values_input);

        // Build offsets per desired segment count
        std::vector<std::vector<offset_type>> offsets_arrays(num_input_arrays);
        std::vector<unsigned int>             segments_counts(num_input_arrays);
        unsigned int                          max_segments_count = 0;

        for(int i = 0; i < num_input_arrays; ++i)
        {
            const size_t desired_segment = m_desired_segments[i];

            // The minimal average length should at least be 1 to prevent infinite loop.
            const double avg_segment_length
                = std::max(1.0, static_cast<double>(items) / desired_segment);
            std::uniform_real_distribution<double> segment_length_dis(0, avg_segment_length * 2);

            std::vector<offset_type>& offsets        = offsets_arrays[i];
            unsigned int              segments_count = 0;
            size_t                    offset         = 0;
            while(offset < items)
            {
                const size_t segment_length = std::round(segment_length_dis(gen));
                offsets.push_back(offset);
                segments_count++;
                offset += segment_length;
            }
            offsets.push_back(items);

            segments_counts[i] = segments_count;
            max_segments_count = std::max(max_segments_count, segments_count);
        }

        std::vector<common::device_ptr<offset_type>> d_offsets_arrays(num_input_arrays);
        for(int i = 0; i < num_input_arrays; ++i)
        {
            d_offsets_arrays[i].store(offsets_arrays[i]);
        }

        // Shared, reused across all runs (sized to the largest)
        common::device_ptr<value_type> d_values_output(items);

        value_type init(5);

        const auto dispatch_input = [&](void*        d_temp_storage,
                                        size_t&      temp_storage_size_bytes,
                                        offset_type* d_offsets,
                                        unsigned int segments_count)
        {
            HIP_CHECK(run_device_segmented_scan(d_temp_storage,
                                                temp_storage_size_bytes,
                                                d_values_input.get(),
                                                d_values_output.get(),
                                                segments_count,
                                                d_offsets,
                                                d_offsets + 1,
                                                init,
                                                stream));
        };

        // Query temp storage size per run, track the max
        std::vector<size_t> temp_storage_bytes_per_run(num_input_arrays);
        size_t              max_temporary_storage_bytes = 0;

        for(int i = 0; i < num_input_arrays; ++i)
        {
            size_t temp_storage_size_bytes = 0;
            dispatch_input(nullptr,
                           temp_storage_size_bytes,
                           d_offsets_arrays[i].get(),
                           segments_counts[i]);

            temp_storage_bytes_per_run[i] = temp_storage_size_bytes;
            max_temporary_storage_bytes
                = std::max(max_temporary_storage_bytes, temp_storage_size_bytes);
        }

        common::device_ptr<void> d_temp_storage(max_temporary_storage_bytes);

        state.set_items(items);
        state.add_reads<T>(items * num_input_arrays);

        state.run(
            [&]
            {
                for(int i = 0; i < num_input_arrays; ++i)
                {
                    dispatch_input(d_temp_storage.get(),
                                   temp_storage_bytes_per_run[i],
                                   d_offsets_arrays[i].get(),
                                   segments_counts[i]);
                }
            });
    }

private:
    std::vector<size_t> m_desired_segments;
};
