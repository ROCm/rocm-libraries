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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SCAN_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SCAN_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"
#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/utils_custom_type.hpp"
#endif

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_scan.hpp>
#include <rocprim/functional.hpp>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <rocprim/block/block_load.hpp>
    #include <rocprim/block/block_scan.hpp>
    #include <rocprim/block/block_store.hpp>
#else
    #include <rocprim/functional.hpp>
    #include <rocprim/types.hpp>
#endif

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <memory>
#else
    #include <stdint.h>
#endif

template<typename Config>
std::string config_name()
{
    const rocprim::detail::scan_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread) + ",method:"
           + std::string(get_block_scan_algorithm_name(config.block_scan_method)) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<bool Exclusive,
         typename T,
         typename ScanOp,
         bool Deterministic,
         typename Config = rocprim::default_config>
struct device_scan_benchmark : public benchmark_utils::autotune_interface
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
        if constexpr(!Deterministic)
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
        else
        {
            return rocprim::deterministic_exclusive_scan<Config>(temporary_storage,
                                                                 storage_size,
                                                                 input,
                                                                 output,
                                                                 initial_value,
                                                                 input_size,
                                                                 scan_op,
                                                                 stream,
                                                                 debug);
        }
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
        if constexpr(!Deterministic)
        {
            return rocprim::inclusive_scan<Config>(temporary_storage,
                                                   storage_size,
                                                   input,
                                                   output,
                                                   input_size,
                                                   scan_op,
                                                   stream,
                                                   debug);
        }
        else
        {
            return rocprim::deterministic_inclusive_scan<Config>(temporary_storage,
                                                                 storage_size,
                                                                 input,
                                                                 output,
                                                                 input_size,
                                                                 scan_op,
                                                                 stream,
                                                                 debug);
        }
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        // Calculate the number of elements
        size_t size = bytes / sizeof(T);

        ScanOp         scan_op{};
        const auto     random_range = limit_random_range<T>(0, 1000);
        std::vector<T> input
            = get_random_data<T>(size, random_range.first, random_range.second, seed.get_0());
        T                     initial_value = T(123);
        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(size);
        HIP_CHECK(hipDeviceSynchronize());

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes;
        // Get size of d_temp_storage
        HIP_CHECK((run_device_scan(nullptr,
                                   temp_storage_size_bytes,
                                   d_input.get(),
                                   d_output.get(),
                                   initial_value,
                                   size,
                                   scan_op,
                                   stream)));
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        state.run(
            [&]
            {
                HIP_CHECK((run_device_scan(d_temp_storage.get(),
                                           temp_storage_size_bytes,
                                           d_input.get(),
                                           d_output.get(),
                                           initial_value,
                                           size,
                                           scan_op,
                                           stream)));
            });

        state.set_throughput(size, sizeof(T));
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
                void operator()(
                    std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
                {
                    storage.emplace_back(
                        std::make_unique<device_scan_benchmark<
                            false,
                            T,
                            rocprim::plus<T>,
                            false,
                            rocprim::scan_config<block_size,
                                                 ItemsPerThread,
                                                 rocprim::block_load_method::block_load_transpose,
                                                 rocprim::block_store_method::block_store_transpose,
                                                 BlockScanAlgorithm>>>());
                }
            };

            void operator()(
                std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
            {
                // Limit items per thread to not over-use shared memory
                static constexpr unsigned int max_items_per_thread
                    = ::rocprim::min<size_t>(65536 / (block_size * sizeof(T)) - 1, 24);
                static_for_each<make_index_range<unsigned int, 1, max_items_per_thread>,
                                create_ipt>(storage);
            }

            static constexpr unsigned int block_size = 1u << BlockSizeExponent;
        };

        static void
            create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        {
            static_for_each<index_range, create_block_size>(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        // Block sizes 64, 128, 256
        create_block_scan_algorithm<make_index_range<unsigned int, 6, 8>>::create(storage);
    }
};

#else

    #define CREATE_EXCL_INCL_BENCHMARK(EXCL, T, SCAN_OP) \
        executor.queue_instance(device_scan_benchmark<EXCL, T, SCAN_OP, Deterministic>());

    #define CREATE_BENCHMARK(T, SCAN_OP)              \
        CREATE_EXCL_INCL_BENCHMARK(false, T, SCAN_OP) \
        CREATE_EXCL_INCL_BENCHMARK(true, T, SCAN_OP)

template<bool Deterministic>
void add_benchmarks(benchmark_utils::executor& executor)
{
    using custom_float2  = common::custom_type<float, float>;
    using custom_double2 = common::custom_type<double, double>;

    CREATE_BENCHMARK(int, rocprim::plus<int>)
    CREATE_BENCHMARK(float, rocprim::plus<float>)
    CREATE_BENCHMARK(double, rocprim::plus<double>)
    CREATE_BENCHMARK(long long, rocprim::plus<long long>)
    CREATE_BENCHMARK(float2, rocprim::plus<float2>)
    CREATE_BENCHMARK(custom_float2, rocprim::plus<custom_float2>)
    CREATE_BENCHMARK(double2, rocprim::plus<double2>)
    CREATE_BENCHMARK(custom_double2, rocprim::plus<custom_double2>)
    CREATE_BENCHMARK(int8_t, rocprim::plus<int8_t>)
    CREATE_BENCHMARK(uint8_t, rocprim::plus<uint8_t>)
    CREATE_BENCHMARK(rocprim::half, rocprim::plus<rocprim::half>)
    CREATE_BENCHMARK(rocprim::int128_t, rocprim::plus<rocprim::int128_t>)
    CREATE_BENCHMARK(rocprim::uint128_t, rocprim::plus<rocprim::uint128_t>)
}

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_SCAN_PARALLEL_HPP_
