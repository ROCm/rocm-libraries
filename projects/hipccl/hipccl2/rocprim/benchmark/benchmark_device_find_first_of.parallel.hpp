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

#ifndef ROCPRIM_BENCHMARK_DEVICE_FIND_FIRST_OF_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_FIND_FIRST_OF_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/device_find_first_of.hpp>

#include <cstddef>
#include <string>
#include <vector>

template<typename Config>
std::string config_name()
{
    const rocprim::detail::find_first_of_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T, typename Config = rocprim::default_config>
struct device_find_first_of_benchmark : public config_autotune_interface
{
    std::vector<size_t> keys_sizes;
    std::vector<double> first_occurrences;

    device_find_first_of_benchmark(size_t keys_size, double first_occurrence)
    {
        keys_sizes.push_back(keys_size);
        first_occurrences.push_back(first_occurrence);
    }

    device_find_first_of_benchmark(const std::vector<size_t>& keys_sizes,
                                   const std::vector<double>& first_occurrences)
    {
        this->keys_sizes        = keys_sizes;
        this->first_occurrences = first_occurrences;
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:find_first_of,"s
            + (keys_sizes.size() == 1 ? "keys_size:"s + std::to_string(keys_sizes[0]) : ""s)
            + (first_occurrences.size() == 1
                   ? ",first_occurrence:"s + std::to_string(first_occurrences[0])
                   : ""s)
            + ",value_type:"s + std::string(Traits<T>::name()) + ",cfg:" + config_name<Config>()
            + "}");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 2;

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using type        = T;
        using key_type    = T;
        using output_type = size_t;

        const size_t size = bytes / sizeof(type);

        const size_t max_keys_size = *std::max_element(keys_sizes.begin(), keys_sizes.end());

        // Generate data
        std::vector<key_type> key_input
            = get_random_data<key_type>(max_keys_size, 0, 100, seed.get_0());
        std::vector<type> input
            = get_random_data<type>(size, 101, generate_limits<type>::max(), seed.get_0());

        std::vector<type*> d_inputs(first_occurrences.size());
        for(size_t fi = 0; fi < first_occurrences.size(); ++fi)
        {
            type* d_input;
            HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
            HIP_CHECK(hipMemcpyAsync(d_input,
                                     input.data(),
                                     input.size() * sizeof(*d_input),
                                     hipMemcpyHostToDevice,
                                     stream));
            // Set the first occurrence of keys in input
            const size_t p = static_cast<size_t>(size * first_occurrences[fi]);
            if(p < size)
            {
                const type key = key_input[0];
                HIP_CHECK(hipMemcpyAsync(d_input + p,
                                         &key,
                                         sizeof(*d_input),
                                         hipMemcpyHostToDevice,
                                         stream));
            }
            d_inputs[fi] = d_input;
        }

        key_type*    d_key_input;
        output_type* d_output;
        HIP_CHECK(hipMalloc(&d_key_input, max_keys_size * sizeof(*d_key_input)));
        HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));

        HIP_CHECK(hipMemcpy(d_key_input,
                            key_input.data(),
                            key_input.size() * sizeof(*d_key_input),
                            hipMemcpyHostToDevice));

        ::rocprim::equal_to<type> compare_op;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        auto run = [&](size_t key_size, const type* d_input)
        {
            HIP_CHECK(rocprim::find_first_of<Config>(d_temporary_storage,
                                                     temporary_storage_bytes,
                                                     d_input,
                                                     d_key_input,
                                                     d_output,
                                                     input.size(),
                                                     key_size,
                                                     compare_op,
                                                     stream));
        };

        size_t max_temporary_storage_bytes = 0;
        for(size_t keys_size : keys_sizes)
        {
            run(keys_size, d_inputs[0]);
            max_temporary_storage_bytes
                = std::max(max_temporary_storage_bytes, temporary_storage_bytes);
        }
        temporary_storage_bytes = max_temporary_storage_bytes;
        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            for(size_t fi = 0; fi < first_occurrences.size(); ++fi)
            {
                for(size_t keys_size : keys_sizes)
                {
                    run(keys_size, d_inputs[fi]);
                }
            }
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
                for(size_t fi = 0; fi < first_occurrences.size(); ++fi)
                {
                    for(size_t keys_size : keys_sizes)
                    {
                        run(keys_size, d_inputs[fi]);
                    }
                }
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

        // Only a part of data (before the first occurrence) must be actually processed. In ideal
        // cases when no thread blocks do unneeded work (i.e. exit early once the match is found),
        // performance for different values of first_occurrence must be similar.
        size_t sum_effective_size = 0;
        for(double first_occurrence : first_occurrences)
        {
            sum_effective_size += static_cast<size_t>(size * first_occurrence);
        }
        size_t sum_keys_size = 0;
        for(size_t keys_size : keys_sizes)
        {
            sum_keys_size += keys_size;
        }
        state.SetBytesProcessed(state.iterations() * batch_size * sum_effective_size
                                * sizeof(*d_inputs[0]));
        state.SetItemsProcessed(state.iterations() * batch_size * sum_effective_size);
        // Each input is read once but all keys are read by all threads so performance is likely
        // compute-bound or bound by cache bandwidth for reading keys rather than reading inputs.
        // Let's additionally report the rate of comparisons to see if it reaches a plateau with
        // increasing keys_size.
        state.counters["comparisons_per_second"]
            = benchmark::Counter(static_cast<double>(state.iterations() * batch_size
                                                     * sum_effective_size * sum_keys_size),
                                 benchmark::Counter::kIsRate);

        for(size_t fi = 0; fi < first_occurrences.size(); ++fi)
        {
            HIP_CHECK(hipFree(d_inputs[fi]));
        }
        HIP_CHECK(hipFree(d_key_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temporary_storage));
    }
};

template<typename T, unsigned int BlockSize>
struct device_find_first_of_benchmark_generator
{

    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        using generated_config = rocprim::find_first_of_config<BlockSize, ItemsPerThread>;

        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            std::vector<size_t> keys_sizes{1, 10, 100, 1000};
            std::vector<double> first_occurrences{0.1, 0.5, 1.0};
            storage.emplace_back(
                std::make_unique<device_find_first_of_benchmark<T, generated_config>>(
                    keys_sizes,
                    first_occurrences));
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static constexpr unsigned int min_items_per_thread = 1;
        static constexpr unsigned int max_items_per_thread = 16;
        static_for_each<make_index_range<unsigned int, min_items_per_thread, max_items_per_thread>,
                        create_ipt>(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_FIND_FIRST_OF_PARALLEL_HPP_
