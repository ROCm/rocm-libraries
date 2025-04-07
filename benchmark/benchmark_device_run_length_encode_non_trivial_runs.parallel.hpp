// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_RUN_LENGTH_ENCODE_NON_TRIVIAL_RUNS_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_RUN_LENGTH_ENCODE_NON_TRIVIAL_RUNS_PARALLEL_HPP_

#include "benchmark_utils.hpp"

#include "../common/utils_device_ptr.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_run_length_encode.hpp>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <rocprim/block/block_load.hpp>
    #include <rocprim/block/block_scan.hpp>
    #include <rocprim/config.hpp>
    #include <rocprim/functional.hpp>
    #include <rocprim/types/tuple.hpp>
#endif

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <algorithm>
    #include <memory>
#endif

template<typename Config>
std::string non_trivial_runs_config_name()
{
    const rocprim::detail::non_trivial_runs_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread)
           + ",load_method:" + get_block_load_method_name(config.load_input_method) + "}";
}

template<>
inline std::string non_trivial_runs_config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T, size_t MaxLength, typename Config = rocprim::default_config>
struct device_non_trivial_runs_benchmark : public benchmark_utils::autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name(
            "{lvl:device,algo:run_length_encode,subalgo:non_trivial,key_type:"
            + std::string(Traits<T>::name()) + ",keys_max_length:" + std::to_string(MaxLength)
            + ",cfg:" + non_trivial_runs_config_name<Config>() + "}");
    }

    void run(benchmark_utils::state&& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.bytes;
        const auto& seed   = state.seed;

        using offset_type = unsigned int;
        using count_type  = unsigned int;

        constexpr std::array<int, 2> tuning_max_segment_lengths = {10, 1000};
        constexpr int num_input_arrays = is_tuning ? tuning_max_segment_lengths.size() : 1;

        constexpr size_t item_size = sizeof(T) + sizeof(offset_type) + sizeof(count_type);

        const size_t size = bytes / item_size;

        // Generate data
        std::vector<T> input[num_input_arrays];
        if(is_tuning)
        {
            for(size_t i = 0; i < tuning_max_segment_lengths.size(); ++i)
            {
                input[i] = get_random_segments_iota<T>(size,
                                                       tuning_max_segment_lengths[i],
                                                       seed.get_0());
            }
        }
        else
        {
            input[0] = get_random_segments_iota<T>(size, MaxLength, seed.get_0());
        }

        common::device_ptr<T> d_input[num_input_arrays];
        for(int i = 0; i < num_input_arrays; ++i)
        {
            d_input[i].store(input[i]);
        }

        common::device_ptr<offset_type> d_offsets_output(size);
        common::device_ptr<count_type>  d_counts_output(size);
        common::device_ptr<count_type>  d_runs_count_output(1);

        const auto dispatch = [&](void* d_temporary_storage, size_t& temporary_storage_bytes)
        {
            const auto dispatch_input = [&](T* d_input)
            {
                HIP_CHECK(
                    rocprim::run_length_encode_non_trivial_runs<Config>(d_temporary_storage,
                                                                        temporary_storage_bytes,
                                                                        d_input,
                                                                        size,
                                                                        d_offsets_output.get(),
                                                                        d_counts_output.get(),
                                                                        d_runs_count_output.get(),
                                                                        stream,
                                                                        false));
            };

            for(int i = 0; i < num_input_arrays; ++i)
            {
                dispatch_input(d_input[i].get());
            }
        };

        // Allocate temporary storage memory
        size_t temporary_storage_bytes = 0;
        dispatch(nullptr, temporary_storage_bytes);
        common::device_ptr<void> d_temporary_storage(temporary_storage_bytes);
        HIP_CHECK(hipDeviceSynchronize());

        state.run([&] { dispatch(d_temporary_storage.get(), temporary_storage_bytes); });

        state.set_throughput(size, sizeof(T) + sizeof(offset_type) + sizeof(count_type));
    }
    static constexpr bool is_tuning = !std::is_same<Config, rocprim::default_config>::value;
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename T, unsigned int BlockSize, ::rocprim::block_load_method BlockLoadMethod>
struct device_non_trivial_runs_benchmark_generator
{
    using OffsetCountPairT = ::rocprim::tuple<unsigned int, unsigned int>;

    static constexpr unsigned int max_shared_memory = TUNING_SHARED_MEMORY_MAX;
    static constexpr unsigned int max_size_per_element
        = std::max(sizeof(T), sizeof(OffsetCountPairT));
    static constexpr unsigned int max_items_per_thread
        = max_shared_memory / (BlockSize * max_size_per_element);
    static constexpr unsigned int min_items_per_thread_exponent = 3u;
    static constexpr unsigned int max_items_per_thread_exponent
        = std::max(static_cast<unsigned int>(rocprim::Log2<max_items_per_thread>::VALUE),
                   min_items_per_thread_exponent)
          - 1u;

    static constexpr bool is_load_warp_transpose
        = BlockLoadMethod == ::rocprim::block_load_method::block_load_warp_transpose;
    static constexpr bool is_warp_load_supp
        = is_load_warp_transpose && BlockSize == ROCPRIM_WARP_SIZE_64;

    template<int ItemsPerThreadExp>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
        {
            if(!is_load_warp_transpose || is_warp_load_supp)
            {
                using config = rocprim::non_trivial_runs_config<
                    BlockSize,
                    items_per_thread,
                    BlockLoadMethod,
                    rocprim::block_scan_algorithm::using_warp_scan>;
                storage.emplace_back(
                    std::make_unique<device_non_trivial_runs_benchmark<T, 0, config>>());
            }
        }

    private:
        static constexpr unsigned int items_per_thread = 1u << ItemsPerThreadExp;
    };

    static void create(std::vector<std::unique_ptr<benchmark_utils::autotune_interface>>& storage)
    {
        static_for_each<
            make_index_range<int, min_items_per_thread_exponent, max_items_per_thread_exponent>,
            create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_RUN_LENGTH_ENCODE_NON_TRIVIAL_RUNS_PARALLEL_HPP_
