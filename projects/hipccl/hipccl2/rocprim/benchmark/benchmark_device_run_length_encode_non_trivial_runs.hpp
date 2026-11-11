// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

inline const char* get_block_load_method_name(rocprim::block_load_method method)
{
    switch(method)
    {
        case rocprim::block_load_method::block_load_direct:
            return "block_load_method::block_load_direct";
        case rocprim::block_load_method::block_load_striped:
            return "block_load_method::block_load_striped";
        case rocprim::block_load_method::block_load_vectorize:
            return "block_load_method::block_load_vectorize";
        case rocprim::block_load_method::block_load_transpose:
            return "block_load_method::block_load_transpose";
        case rocprim::block_load_method::block_load_warp_transpose:
            return "block_load_method::block_load_warp_transpose";
    }
    return "default_method";
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
            .add("load_method", get_block_load_method_name(config.load_input_method));
    }
}

template<typename T, size_t MaxLength, typename Config = rocprim::default_config>
struct device_non_trivial_runs_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_run_length_encode_non_trivial_runs")
            .add("key_type", primbench::name<T>())
            .add("keys_max_length", MaxLength)
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using offset_type = unsigned int;
        using count_type  = unsigned int;

        constexpr std::array<int, 2> tuning_max_segment_lengths = {10, 1000};
        constexpr int num_input_arrays = is_tuning ? tuning_max_segment_lengths.size() : 1;

        constexpr size_t item_size = sizeof(T) + sizeof(offset_type) + sizeof(count_type);

        const size_t items = bytes / item_size;

        // Generate data
        std::vector<T> input[num_input_arrays];
        if(is_tuning)
        {
            for(size_t i = 0; i < tuning_max_segment_lengths.size(); ++i)
            {
                input[i] = get_random_segments_iota<T>(items, tuning_max_segment_lengths[i], seed);
            }
        }
        else
        {
            input[0] = get_random_segments_iota<T>(items, MaxLength, seed);
        }

        common::device_ptr<T> d_input[num_input_arrays];
        for(int i = 0; i < num_input_arrays; ++i)
        {
            d_input[i].store(input[i]);
        }

        common::device_ptr<offset_type> d_offsets_output(items);
        common::device_ptr<count_type>  d_counts_output(items);
        common::device_ptr<count_type>  d_runs_count_output(1);

        const auto dispatch = [&](void* d_temporary_storage, size_t& temporary_storage_bytes)
        {
            const auto dispatch_input = [&](T* d_input)
            {
                HIP_CHECK(
                    rocprim::run_length_encode_non_trivial_runs<Config>(d_temporary_storage,
                                                                        temporary_storage_bytes,
                                                                        d_input,
                                                                        items,
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

        state.set_items(items);
        state.add_reads<T>(items);
        state.add_reads<offset_type>(items);
        state.add_reads<count_type>(items);

        state.run([&] { dispatch(d_temporary_storage.get(), temporary_storage_bytes); });
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
        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
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

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        static_for_each<
            make_index_range<int, min_items_per_thread_exponent, max_items_per_thread_exponent>,
            create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING
