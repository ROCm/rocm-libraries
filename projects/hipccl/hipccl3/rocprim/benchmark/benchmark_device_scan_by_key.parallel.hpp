// MIT License
//
// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SCAN_BY_KEY_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SCAN_BY_KEY_PARALLEL_HPP_

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

template<typename Config>
std::string config_name()
{
    const rocprim::detail::scan_by_key_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread)
           + ",method:" + std::string(get_block_scan_method_name(config.block_scan_method)) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<bool Exclusive                = false,
         class Key                     = int,
         class Value                   = int,
         class ScanOp                  = rocprim::plus<Value>,
         class CompareOp               = rocprim::equal_to<Key>,
         unsigned int MaxSegmentLength = 1024,
         class Config                  = rocprim::default_config>
struct device_scan_by_key_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:scan_by_key,exclusive:" + (Exclusive ? "true"s : "false"s)
            + ",key_type:" + std::string(Traits<Key>::name())
            + ",value_type:" + std::string(Traits<Value>::name()) + ",max_segment_length:"
            + std::to_string(MaxSegmentLength) + ",cfg:" + config_name<Config>() + "}");
    }

    template<bool excl = Exclusive>
    auto run_device_scan_by_key(void*             temporary_storage,
                                size_t&           storage_size,
                                const Key*        keys,
                                const Value*      input,
                                Value*            output,
                                const Value       initial_value,
                                const size_t      input_size,
                                const ScanOp      scan_op,
                                const CompareOp   compare_op,
                                const hipStream_t stream,
                                const bool        debug = false) const ->
        typename std::enable_if<excl, hipError_t>::type
    {
        return rocprim::exclusive_scan_by_key<Config>(temporary_storage,
                                                      storage_size,
                                                      keys,
                                                      input,
                                                      output,
                                                      initial_value,
                                                      input_size,
                                                      scan_op,
                                                      compare_op,
                                                      stream,
                                                      debug);
    }

    template<bool excl = Exclusive>
    auto run_device_scan_by_key(void*        temporary_storage,
                                size_t&      storage_size,
                                const Key*   keys,
                                const Value* input,
                                Value*       output,
                                const Value /*initial_value*/,
                                const size_t      input_size,
                                const ScanOp      scan_op,
                                const CompareOp   compare_op,
                                const hipStream_t stream,
                                const bool        debug = false) const ->
        typename std::enable_if<!excl, hipError_t>::type
    {
        return rocprim::inclusive_scan_by_key<Config>(temporary_storage,
                                                      storage_size,
                                                      keys,
                                                      input,
                                                      output,
                                                      input_size,
                                                      scan_op,
                                                      compare_op,
                                                      stream,
                                                      debug);
    }

    void run(benchmark::State& state, size_t size, hipStream_t stream) const override
    {
        constexpr bool debug = false;

        const std::vector<Key> keys
            = get_random_segments<Key>(size, MaxSegmentLength, std::random_device{}());

        const std::vector<Value> input = get_random_data<Value>(size, Value(0), Value(1000));

        ScanOp    scan_op{};
        CompareOp compare_op{};

        Value  initial_value = Value(123);
        Value* d_input;
        Key*   d_keys;
        Value* d_output;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMalloc(&d_keys, keys.size() * sizeof(keys[0])));
        HIP_CHECK(hipMalloc(&d_output, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(input[0]),
                            hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(d_keys, keys.data(), keys.size() * sizeof(keys[0]), hipMemcpyHostToDevice));

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK((run_device_scan_by_key(d_temp_storage,
                                          temp_storage_size_bytes,
                                          d_keys,
                                          d_input,
                                          d_output,
                                          initial_value,
                                          size,
                                          scan_op,
                                          compare_op,
                                          stream,
                                          debug)));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        // Warm-up
        for(size_t i = 0; i < 5; i++)
        {
            HIP_CHECK((run_device_scan_by_key(d_temp_storage,
                                              temp_storage_size_bytes,
                                              d_keys,
                                              d_input,
                                              d_output,
                                              initial_value,
                                              size,
                                              scan_op,
                                              compare_op,
                                              stream,
                                              debug)));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        const unsigned int batch_size = 10;
        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK((run_device_scan_by_key(d_temp_storage,
                                                  temp_storage_size_bytes,
                                                  d_keys,
                                                  d_input,
                                                  d_output,
                                                  initial_value,
                                                  size,
                                                  scan_op,
                                                  compare_op,
                                                  stream,
                                                  debug)));
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

        state.SetBytesProcessed(state.iterations() * batch_size * size
                                * (sizeof(Key) + sizeof(Value)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_keys));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename KeyType, typename ValueType, rocprim::block_scan_algorithm BlockScanAlgorithm>
struct device_scan_by_key_benchmark_generator
{
    template<typename index_range>
    struct create_block_scan_algorithm
    {
        template<unsigned int BlockSizeExponent>
        struct create_block_size
        {
            template<unsigned int ItemsPerThread>
            struct create_ipt
            {
                void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
                {
                    storage.emplace_back(std::make_unique<device_scan_by_key_benchmark<
                                             false,
                                             KeyType,
                                             ValueType,
                                             rocprim::plus<ValueType>,
                                             rocprim::equal_to<KeyType>,
                                             1024,
                                             rocprim::scan_by_key_config_v2<
                                                 block_size,
                                                 ItemsPerThread,
                                                 rocprim::block_load_method::block_load_transpose,
                                                 rocprim::block_store_method::block_store_transpose,
                                                 BlockScanAlgorithm>>>());
                }
            };

            void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
            {
                // Limit items per thread to not over-use shared memory
                static constexpr unsigned int max_items_per_thread = ::rocprim::min<size_t>(
                    65536 / (block_size * (sizeof(KeyType) + sizeof(ValueType))),
                    24);
                static_for_each<make_index_range<unsigned int, 1, max_items_per_thread>,
                                create_ipt>(storage);
            }

            static constexpr unsigned int block_size = 1u << BlockSizeExponent;
        };

        static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            static_for_each<index_range, create_block_size>(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        // Block sizes 64, 128, 256
        create_block_scan_algorithm<make_index_range<unsigned int, 6, 8>>::create(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SCAN_BY_KEY_PARALLEL_HPP_
