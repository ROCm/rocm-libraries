// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef BENCHMARK_CONFIG_TUNING
    #include "../common/utils_custom_type.hpp"
#endif

#include <hip/hip_runtime.h>

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
         typename ScanOp,
         bool Deterministic,
         typename Config = rocprim::default_config>
struct device_scan_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        auto algo = []
        {
            if constexpr(Deterministic)
            {
                return "device_scan_deterministic";
            }
            else
            {
                return "device_scan";
            }
        };

        return primbench::json{}
            .add("lvl", "device")
            .add("algo", algo())
            .add("exclusive", Exclusive)
            .add("value_type", primbench::name<T>())
            .add("cfg", config_name<Config>());
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
                         const bool        debug = false) const
    {
        if constexpr(excl)
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
        else
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
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        size_t items = bytes / sizeof(T);

        ScanOp         scan_op{};
        const auto     random_range = limit_random_range<T>(0, 1000);
        std::vector<T> input
            = get_random_data<T>(items, random_range.first, random_range.second, seed);
        T                     initial_value = T(123);
        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(items);

        // Allocate temporary storage memory
        size_t temp_storage_size_bytes;

        // Get size of d_temp_storage
        HIP_CHECK((run_device_scan(nullptr,
                                   temp_storage_size_bytes,
                                   d_input.get(),
                                   d_output.get(),
                                   initial_value,
                                   items,
                                   scan_op,
                                   stream)));
        common::device_ptr<void> d_temp_storage(temp_storage_size_bytes);

        state.set_items(items);
        state.add_reads<T>(items);

        state.run(
            [&]
            {
                HIP_CHECK((run_device_scan(d_temp_storage.get(),
                                           temp_storage_size_bytes,
                                           d_input.get(),
                                           d_output.get(),
                                           initial_value,
                                           items,
                                           scan_op,
                                           stream)));
            });
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
                    std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
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

            void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
            {
                // Limit items per thread to not over-use shared memory
                static constexpr unsigned int max_items_per_thread
                    = ::rocprim::min<size_t>(65536 / (block_size * sizeof(T)) - 1, 24);
                static_for_each<make_index_range<unsigned int, 1, max_items_per_thread>,
                                create_ipt>(storage);
            }

            static constexpr unsigned int block_size = 1u << BlockSizeExponent;
        };

        static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            static_for_each<index_range, create_block_size>(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        // Block sizes 64, 128, 256
        create_block_scan_algorithm<make_index_range<unsigned int, 6, 8>>::create(storage);
    }
};

#else

    #define CREATE_EXCL_INCL_BENCHMARK(EXCL, T) \
        executor.queue<device_scan_benchmark<EXCL, T, rocprim::plus<T>, Deterministic>>();

    #define CREATE_BENCHMARK(T)              \
        CREATE_EXCL_INCL_BENCHMARK(false, T) \
        CREATE_EXCL_INCL_BENCHMARK(true, T)

    #define BENCHMARK_TYPE_TUNING(T) \
        executor.queue<device_scan_benchmark<false, T, rocprim::plus<T>, false>>();

template<bool Deterministic>
void add_benchmarks(primbench::executor& executor)
{
    if constexpr(!Deterministic)
    {
        // Tuned types
        BENCHMARK_TYPE_TUNING(rocprim::int128_t)
        BENCHMARK_TYPE_TUNING(int64_t)
        BENCHMARK_TYPE_TUNING(int)
        BENCHMARK_TYPE_TUNING(short)
        BENCHMARK_TYPE_TUNING(int8_t)
        BENCHMARK_TYPE_TUNING(double)
        BENCHMARK_TYPE_TUNING(float)
        BENCHMARK_TYPE_TUNING(rocprim::half)
    }

    #ifndef BENCHMARK_AUTOTUNED_TYPES_ONLY
    // Not tuned types
    CREATE_EXCL_INCL_BENCHMARK(true, float)
    CREATE_EXCL_INCL_BENCHMARK(true, double)
    CREATE_BENCHMARK(float2)
    CREATE_BENCHMARK(double2)
    CREATE_BENCHMARK(uint8_t)
    CREATE_BENCHMARK(rocprim::uint128_t)

    // Not tuned custom types
    using custom_f32_f32 = common::custom_type<float, float>;
    using custom_f64_f64 = common::custom_type<double, double>;

    CREATE_BENCHMARK(custom_f32_f32)
    CREATE_BENCHMARK(custom_f64_f64)
    #endif
}

#endif // BENCHMARK_CONFIG_TUNING
