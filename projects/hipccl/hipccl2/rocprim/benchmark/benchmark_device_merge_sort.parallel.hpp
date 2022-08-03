// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_MERGE_SORT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_MERGE_SORT_PARALLEL_HPP_

#include <cstddef>
#include <string>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

namespace rp = rocprim;

template<typename Key   = int,
         typename Value = rocprim::empty_type,
         typename Config
         = rocprim::detail::default_merge_sort_config<ROCPRIM_TARGET_ARCH, Key, Value>>
struct device_merge_sort_benchmark : public config_autotune_interface
{
    static std::string get_name_pattern()
    {
        return R"---((?P<algo>\S*)\<)---"
               R"---((?P<key_type>\S*),(?:\s*(?P<value_type>\S*),)?\s*merge_sort_config\<\s*)---"
               R"---((?P<sort_block_size>[0-9]+),\s*(?P<sort_items_per_thread>[0-9]+),\s*(?P<merge_block_size>[0-9]+)\>\>)---";
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return std::string(
            "device_merge_sort_"
            + (std::is_same<Value, rocprim::empty_type>::value ? "keys"s : "pairs"s) + "<"
            + std::string(Traits<Key>::name()) + ", "
            + (std::is_same<Value, rocprim::empty_type>::value
                   ? ""s
                   : std::string(Traits<Value>::name()) + ", ")
            + "merge_sort_config<" + pad_string(std::to_string(Config::sort_config::block_size), 4)
            + ", " + pad_string(std::to_string(Config::sort_config::items_per_thread), 2) + ", "
            + pad_string(std::to_string(Config::merge_impl1_config::block_size), 4) + ">>");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    // keys benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        using key_type = Key;

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = get_random_data<key_type>(size, (key_type)-1000, (key_type) + 1000);
        }
        else
        {
            keys_input = get_random_data<key_type>(size,
                                                   std::numeric_limits<key_type>::min(),
                                                   std::numeric_limits<key_type>::max());
        }

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        ::rocprim::less<key_type> lesser_op;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::merge_sort<Config>(d_temporary_storage,
                                         temporary_storage_bytes,
                                         d_keys_input,
                                         d_keys_output,
                                         size,
                                         lesser_op,
                                         stream,
                                         false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rp::merge_sort<Config>(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             size,
                                             lesser_op,
                                             stream,
                                             false));
        }
        HIP_CHECK(hipDeviceSynchronize());

        for(auto _ : state)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(rp::merge_sort<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input,
                                                 d_keys_output,
                                                 size,
                                                 lesser_op,
                                                 stream,
                                                 false));
            }
            HIP_CHECK(hipDeviceSynchronize());

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds
                = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            state.SetIterationTime(elapsed_seconds.count());
        }
        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }

    // pairs benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<!std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        using key_type   = Key;
        using value_type = Value;

        // Generate data
        std::vector<key_type> keys_input;
        if(std::is_floating_point<key_type>::value)
        {
            keys_input = get_random_data<key_type>(size, (key_type)-1000, (key_type) + 1000);
        }
        else
        {
            keys_input = get_random_data<key_type>(size,
                                                   std::numeric_limits<key_type>::min(),
                                                   std::numeric_limits<key_type>::max());
        }

        std::vector<value_type> values_input(size);
        std::iota(values_input.begin(), values_input.end(), 0);

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

        ::rocprim::less<key_type> lesser_op;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::merge_sort<Config>(d_temporary_storage,
                                         temporary_storage_bytes,
                                         d_keys_input,
                                         d_keys_output,
                                         d_values_input,
                                         d_values_output,
                                         size,
                                         lesser_op,
                                         stream,
                                         false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rp::merge_sort<Config>(d_temporary_storage,
                                             temporary_storage_bytes,
                                             d_keys_input,
                                             d_keys_output,
                                             d_values_input,
                                             d_values_output,
                                             size,
                                             lesser_op,
                                             stream,
                                             false));
        }
        HIP_CHECK(hipDeviceSynchronize());

        for(auto _ : state)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(rp::merge_sort<Config>(d_temporary_storage,
                                                 temporary_storage_bytes,
                                                 d_keys_input,
                                                 d_keys_output,
                                                 d_values_input,
                                                 d_values_output,
                                                 size,
                                                 lesser_op,
                                                 stream,
                                                 false));
            }
            HIP_CHECK(hipDeviceSynchronize());

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds
                = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            state.SetIterationTime(elapsed_seconds.count());
        }
        state.SetBytesProcessed(state.iterations() * batch_size * size
                                * (sizeof(key_type) + sizeof(value_type)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }

    void run(benchmark::State& state, size_t size, hipStream_t stream) const override
    {
        do_run(state, size, stream);
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<unsigned int MergeBlockSizeExponent,
         unsigned int SortBlockSizeExponent,
         typename Key,
         typename Value = rocprim::empty_type>
struct device_merge_sort_benchmark_generator
{
    template<unsigned int ItemsPerThreadExponent>
    struct create_ipt
    {
        static constexpr unsigned int merge_block_size = 1u << MergeBlockSizeExponent;
        static constexpr unsigned int sort_block_size  = 1u << SortBlockSizeExponent;
        static constexpr unsigned int items_per_thread = 1u << ItemsPerThreadExponent;

        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<device_merge_sort_benchmark<
                    Key,
                    Value,
                    rocprim::
                        merge_sort_config<merge_block_size, sort_block_size, items_per_thread>>>());
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        // Sort items per block must be divisible by merge_block_size, so make
        // the items per thread at least as large that the items_per_block
        // is equal to merge_block_size.
        static constexpr unsigned int min_items_per_thread_exponent
            = MergeBlockSizeExponent - std::min(SortBlockSizeExponent, MergeBlockSizeExponent);

        // Very large block sizes don't work with large items_per_blocks since
        // shared memory is limited
        static constexpr unsigned int max_items_per_thread_exponent
            = std::min(4u, 11u - SortBlockSizeExponent);

        static_for_each<make_index_range<unsigned int,
                                         min_items_per_thread_exponent,
                                         max_items_per_thread_exponent>,
                        create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_MERGE_SORT_PARALLEL_HPP_
