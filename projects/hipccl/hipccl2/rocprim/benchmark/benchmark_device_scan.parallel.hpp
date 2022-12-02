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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SCAN_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SCAN_PARALLEL_HPP_

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

template<bool ByKey                    = false,
         bool Exclusive                = false,
         class T                       = int,
         class BinaryFunction          = rocprim::plus<T>,
         unsigned int MaxSegmentLength = 1024,
         class Config = rocprim::detail::default_scan_config<ROCPRIM_TARGET_ARCH, T>>
struct device_scan_benchmark : public config_autotune_interface
{
    static const char* get_block_scan_method_name(rocprim::block_scan_algorithm alg)
    {
        switch(alg)
        {
            case rocprim::block_scan_algorithm::using_warp_scan:
                return "block_scan_algorithm::using_warp_scan";
            case rocprim::block_scan_algorithm::reduce_then_scan:
                return "block_scan_algorithm::reduce_then_scan";
                // Not using `default: ...` because it kills effectiveness of -Wswitch
        }
        return "unknown_algorithm";
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:scan" + (Exclusive ? "_exclusive"s : "_inclusive"s)
            + ",key_type:" + std::string(Traits<T>::name()) + ",value_type:"
            + std::string(ByKey ? Traits<T>::name() : Traits<rocprim::empty_type>::name())
            + ",max_segment_length:" + std::to_string(MaxSegmentLength)
            + ",cfg:{bs:" + std::to_string(Config::block_size)
            + ",ipt:" + std::to_string(Config::items_per_thread) + ",method:"
            + std::string(get_block_scan_method_name(Config::block_scan_method)) + "}}");
    }

    template<bool excl = Exclusive>
    auto run_device_scan(void*             temporary_storage,
                         size_t&           storage_size,
                         T*                input,
                         T*                output,
                         const T           initial_value,
                         const size_t      input_size,
                         BinaryFunction    scan_op,
                         const hipStream_t stream,
                         const bool        debug = false) const ->
        typename std::enable_if<excl, hipError_t>::type
    {
        return rocprim::exclusive_scan<Config>(temporary_storage,
                                               storage_size,
                                               input,
                                               output,
                                               initial_value,
                                               input_size,
                                               scan_op,
                                               stream,
                                               debug);
    }

    template<bool excl = Exclusive>
    auto run_device_scan(void*             temporary_storage,
                         size_t&           storage_size,
                         T*                input,
                         T*                output,
                         const T           initial_value,
                         const size_t      input_size,
                         BinaryFunction    scan_op,
                         const hipStream_t stream,
                         const bool        debug = false) const ->
        typename std::enable_if<!excl, hipError_t>::type
    {
        (void)initial_value;
        return rocprim::inclusive_scan<Config>(temporary_storage,
                                               storage_size,
                                               input,
                                               output,
                                               input_size,
                                               scan_op,
                                               stream,
                                               debug);
    }

    template<typename K, typename CompareFunction, bool excl = Exclusive>
    auto run_device_scan_by_key(void*                 temporary_storage,
                                size_t&               storage_size,
                                const K*              keys,
                                const T*              input,
                                T*                    output,
                                const T               initial_value,
                                const size_t          input_size,
                                const BinaryFunction  scan_op,
                                const CompareFunction compare,
                                const hipStream_t     stream,
                                const bool            debug = false) const ->
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
                                                      compare,
                                                      stream,
                                                      debug);
    }

    template<typename K, typename CompareFunction, bool excl = Exclusive>
    auto run_device_scan_by_key(void*    temporary_storage,
                                size_t&  storage_size,
                                const K* keys,
                                const T* input,
                                T*       output,
                                const T /*initial_value*/,
                                const size_t          input_size,
                                const BinaryFunction  scan_op,
                                const CompareFunction compare,
                                const hipStream_t     stream,
                                const bool            debug = false) const ->
        typename std::enable_if<!excl, hipError_t>::type
    {
        return rocprim::inclusive_scan_by_key<Config>(temporary_storage,
                                                      storage_size,
                                                      keys,
                                                      input,
                                                      output,
                                                      input_size,
                                                      scan_op,
                                                      compare,
                                                      stream,
                                                      debug);
    }

    void run_benchmark(benchmark::State& state,
                       size_t            size,
                       const hipStream_t stream,
                       BinaryFunction    scan_op) const
    {
        std::vector<T> input         = get_random_data<T>(size, T(0), T(1000));
        T              initial_value = T(123);
        T*             d_input;
        T*             d_output;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(T), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes;
        void*  d_temp_storage = nullptr;
        // Get size of d_temp_storage
        HIP_CHECK((run_device_scan(d_temp_storage,
                                   temp_storage_size_bytes,
                                   d_input,
                                   d_output,
                                   initial_value,
                                   size,
                                   scan_op,
                                   stream)));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < 5; i++)
        {
            HIP_CHECK((run_device_scan(d_temp_storage,
                                       temp_storage_size_bytes,
                                       d_input,
                                       d_output,
                                       initial_value,
                                       size,
                                       scan_op,
                                       stream)));
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
                HIP_CHECK((run_device_scan(d_temp_storage,
                                           temp_storage_size_bytes,
                                           d_input,
                                           d_output,
                                           initial_value,
                                           size,
                                           scan_op,
                                           stream)));
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

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    template<typename K, typename CompareFunction>
    void run_benchmark_by_key(benchmark::State&     state,
                              const size_t          size,
                              const hipStream_t     stream,
                              const BinaryFunction  scan_op,
                              const CompareFunction compare = CompareFunction()) const
    {
        constexpr bool       debug = false;
        const std::vector<T> input = get_random_data<T>(size, T(0), T(1000));

        const std::vector<K> keys
            = get_random_segments<K>(size, MaxSegmentLength, std::random_device{}());

        T  initial_value = T(123);
        T* d_input;
        K* d_keys;
        T* d_output;
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
        HIP_CHECK((run_device_scan_by_key<K, CompareFunction>(d_temp_storage,
                                                              temp_storage_size_bytes,
                                                              d_keys,
                                                              d_input,
                                                              d_output,
                                                              initial_value,
                                                              size,
                                                              scan_op,
                                                              compare,
                                                              stream,
                                                              debug)));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size_bytes));

        // Warm-up
        for(size_t i = 0; i < 5; i++)
        {
            HIP_CHECK((run_device_scan_by_key<K, CompareFunction>(d_temp_storage,
                                                                  temp_storage_size_bytes,
                                                                  d_keys,
                                                                  d_input,
                                                                  d_output,
                                                                  initial_value,
                                                                  size,
                                                                  scan_op,
                                                                  compare,
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
                HIP_CHECK((run_device_scan_by_key<K, CompareFunction>(d_temp_storage,
                                                                      temp_storage_size_bytes,
                                                                      d_keys,
                                                                      d_input,
                                                                      d_output,
                                                                      initial_value,
                                                                      size,
                                                                      scan_op,
                                                                      compare,
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

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_keys));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_temp_storage));
    }

    template<bool by_key = ByKey>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<!by_key, void>::type
    {
        run_benchmark(state, size, stream, BinaryFunction());
    }

    template<bool by_key = ByKey>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<by_key, void>::type
    {
        run_benchmark_by_key<int, rocprim::equal_to<int>>(state, size, stream, BinaryFunction());
    }

    void run(benchmark::State& state, size_t size, hipStream_t stream) const override
    {
        do_run(state, size, stream);
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

inline constexpr unsigned int get_max_items_per_thread(size_t bytes)
{
    if(bytes <= 2)
    {
        return 30;
    }
    else if(2 < bytes && bytes <= 6)
    {
        return 20;
    }
    else //(6 < bytes)
    {
        return 15;
    }
}

template<typename T, bool ByKey, bool Excl>
struct device_scan_benchmark_generator
{
    template<rocprim::block_scan_algorithm BlockScanAlgorithm, typename index_range>
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
                    static constexpr unsigned int block_size = 1u << BlockSizeExponent;
                    storage.emplace_back(
                        std::make_unique<device_scan_benchmark<
                            ByKey,
                            Excl,
                            T,
                            rocprim::plus<T>,
                            1024,
                            rocprim::scan_config<block_size,
                                                 ItemsPerThread,
                                                 true,
                                                 rocprim::block_load_method::block_load_transpose,
                                                 rocprim::block_store_method::block_store_transpose,
                                                 BlockScanAlgorithm>>>());
                }
            };

            void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
            {
                static constexpr unsigned int max_items_per_thread
                    = get_max_items_per_thread(sizeof(T));
                static_for_each<make_index_range<unsigned int, 1, max_items_per_thread>,
                                create_ipt>(storage);
            }
        };

        static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            static_for_each<index_range, create_block_size>(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static const rocprim::block_scan_algorithm using_warp_scan
            = rocprim::block_scan_algorithm::using_warp_scan;
        static const rocprim::block_scan_algorithm reduce_then_scan
            = rocprim::block_scan_algorithm::reduce_then_scan;

        // 64, 128, 256
        create_block_scan_algorithm<using_warp_scan, make_index_range<unsigned int, 6, 8>>::create(
            storage);
        // 256
        create_block_scan_algorithm<reduce_then_scan, make_index_range<unsigned int, 8, 8>>::create(
            storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SCAN_PARALLEL_HPP_
