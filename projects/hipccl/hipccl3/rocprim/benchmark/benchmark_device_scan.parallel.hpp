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

template<typename Config>
std::string config_name()
{
    const rocprim::detail::scan_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread)
           + ",method:" + std::string(get_block_scan_method_name(config.block_scan_method)) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<bool Exclusive = false,
         class T        = int,
         class ScanOp   = rocprim::plus<T>,
         class Config   = rocprim::default_config>
struct device_scan_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:scan,exclusive:" + (Exclusive ? "true"s : "false"s) + ",value_type:"
            + std::string(Traits<T>::name()) + ",cfg:" + config_name<Config>() + "}");
    }

    template<bool excl = Exclusive>
    auto run_device_scan(void*             temporary_storage,
                         size_t&           storage_size,
                         T*                input,
                         T*                output,
                         const T           initial_value,
                         const size_t      input_size,
                         ScanOp            scan_op,
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
                         ScanOp            scan_op,
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

    void run_benchmark(benchmark::State& state,
                       size_t            size,
                       const hipStream_t stream,
                       ScanOp            scan_op) const
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

    void run(benchmark::State& state, size_t size, hipStream_t stream) const override
    {
        run_benchmark(state, size, stream, ScanOp());
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename T, rocprim::block_scan_algorithm BlockScanAlgorithm>
struct device_scan_benchmark_generator
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
                    storage.emplace_back(std::make_unique<device_scan_benchmark<
                                             false,
                                             T,
                                             rocprim::plus<T>,
                                             rocprim::scan_config_v2<
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
                static constexpr unsigned int max_items_per_thread
                    = ::rocprim::min<size_t>(65536 / (block_size * sizeof(T)), 24);
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

#endif // ROCPRIM_BENCHMARK_DEVICE_SCAN_PARALLEL_HPP_
