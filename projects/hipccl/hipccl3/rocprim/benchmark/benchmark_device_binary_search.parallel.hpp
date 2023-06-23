// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_BINARY_SEARCH_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_BINARY_SEARCH_PARALLEL_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "benchmark_utils.hpp"
#include <benchmark/benchmark.h>
#include <hip/hip_runtime_api.h>
#include <rocprim/device/device_binary_search.hpp>

struct binary_search_subalgorithm
{
    std::string name() const
    {
        return "binary_search";
    }
};

struct lower_bound_subalgorithm
{
    std::string name() const
    {
        return "lower_bound";
    }
};

struct upper_bound_subalgorithm
{
    std::string name() const
    {
        return "upper_bound";
    }
};

template<class Config = rocprim::default_config, class... Args>
hipError_t dispatch_binary_search(binary_search_subalgorithm, Args&&... args)
{
    return rocprim::binary_search<Config>(std::forward<Args>(args)...);
}

template<class Config = rocprim::default_config, class... Args>
hipError_t dispatch_binary_search(upper_bound_subalgorithm, Args&&... args)
{
    return rocprim::upper_bound<Config>(std::forward<Args>(args)...);
}

template<class Config = rocprim::default_config, class... Args>
hipError_t dispatch_binary_search(lower_bound_subalgorithm, Args&&... args)
{
    return rocprim::lower_bound<Config>(std::forward<Args>(args)...);
}

template<class SubAlgorithm, class T, class OutputType, class Config>
struct device_binary_search_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:" + SubAlgorithm{}.name()
                                         + ",value_type:" + std::string(Traits<T>::name())
                                         + ",output_type:" + std::string(Traits<OutputType>::name())
                                         + ",cfg:{bs:" + std::to_string(Config::block_size)
                                         + ",ipt:" + std::to_string(Config::items_per_thread)
                                         + "}}");
    }

    void run(benchmark::State& state,
             const std::size_t haystack_size,
             const hipStream_t stream) const override
    {
        using compare_op_t        = rocprim::less<T>;
        const auto   needles_size = haystack_size / 10;
        compare_op_t compare_op;

        std::vector<T> haystack(haystack_size);
        std::iota(haystack.begin(), haystack.end(), 0);

        std::vector<T> needles = get_random_data<T>(needles_size, T(0), T(haystack_size));
        T*             d_haystack;
        T*             d_needles;
        OutputType*    d_output;
        HIP_CHECK(hipMalloc(&d_haystack, haystack_size * sizeof(*d_haystack)));
        HIP_CHECK(hipMalloc(&d_needles, needles_size * sizeof(*d_needles)));
        HIP_CHECK(hipMalloc(&d_output, needles_size * sizeof(*d_output)));
        HIP_CHECK(hipMemcpy(d_haystack,
                            haystack.data(),
                            haystack_size * sizeof(*d_haystack),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_needles,
                            needles.data(),
                            needles_size * sizeof(*d_needles),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage = nullptr;
        size_t temporary_storage_bytes;
        HIP_CHECK(dispatch_binary_search<Config>(SubAlgorithm{},
                                                 d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_haystack,
                                                 d_needles,
                                                 d_output,
                                                 haystack_size,
                                                 needles_size,
                                                 compare_op,
                                                 stream));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        // Warm-up
        HIP_CHECK(dispatch_binary_search<Config>(SubAlgorithm{},
                                                 d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_haystack,
                                                 d_needles,
                                                 d_output,
                                                 haystack_size,
                                                 needles_size,
                                                 compare_op,
                                                 stream));
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            HIP_CHECK(dispatch_binary_search<Config>(SubAlgorithm{},
                                                     d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_haystack,
                                                     d_needles,
                                                     d_output,
                                                     haystack_size,
                                                     needles_size,
                                                     compare_op,
                                                     stream));

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

        state.SetBytesProcessed(state.iterations() * needles_size * sizeof(T));
        state.SetItemsProcessed(state.iterations() * needles_size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_haystack));
        HIP_CHECK(hipFree(d_needles));
        HIP_CHECK(hipFree(d_output));
    }
};

#endif // ROCPRIM_BENCHMARK_BINARY_SEARCH_PARALLEL_HPP_
