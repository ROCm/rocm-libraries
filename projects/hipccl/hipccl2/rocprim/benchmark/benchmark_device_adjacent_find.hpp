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

#include "../common/utils_data_generation.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_adjacent_find.hpp>
#include <rocprim/functional.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
            .add("ipt", config.kernel_config.items_per_thread);
    }
}

template<typename InputT,
         unsigned int FirstAdjPosDecimal,
         typename Config = rocprim::default_config>
struct device_adjacent_find_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_adjacent_find")
            .add("input_type", primbench::name<InputT>())
            .add("first_adj_pos", FirstAdjPosDecimal * 0.1f)
            .add("cfg", config_name<Config>());
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        using input_type  = InputT;
        using output_type = std::size_t;

        const size_t items = bytes / sizeof(input_type);

        if(items < 2)
        {
            std::cerr << "Must have at least two elements for adjacent_find to make sense\n";
            exit(EXIT_FAILURE);
        }

        // Get index of the first adjacent equal pair
        std::size_t first_adj_index = static_cast<std::size_t>(items * FirstAdjPosDecimal * 0.1f);
        if(first_adj_index >= items - 1)
        {
            first_adj_index = items - 2;
        }

        // Generate data ensuring there is no adjacent pair before first_adj_index
        std::vector<input_type> input(items);
        if(std::is_same<input_type, int8_t>::value)
        {
            // For int8_t that has a very limited range of values, iota initialization
            // seems to give a more reliable benchmark input
            std::iota(input.begin(), input.end(), 0);
        }
        else
        {
            input = get_random_data<input_type>(items,
                                                common::generate_limits<input_type>::min(),
                                                common::generate_limits<input_type>::max(),
                                                seed);
            std::vector<std::size_t> iota(items);
            std::iota(iota.begin(), iota.end(), 0);
            std::transform(iota.begin() + 1,
                           iota.begin() + first_adj_index + 1,
                           input.begin() + 1,
                           [&](std::size_t& idx)
                           {
                               while(input[idx] == input[idx - 1])
                               {
                                   input[idx] = get_random_value<input_type>(
                                       common::generate_limits<input_type>::min(),
                                       common::generate_limits<input_type>::max(),
                                       seed);
                               }
                               return input[idx];
                           });
        }

        // Insert first adjacent pair
        input[first_adj_index] = input[first_adj_index + 1];

        input_type*  d_input;
        output_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, items * sizeof(*d_input)));
        HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(*d_input),
                            hipMemcpyHostToDevice));

        std::size_t tmp_storage_size;
        void*       d_tmp_storage        = nullptr;
        auto        launch_adjacent_find = [&]()
        {
            HIP_CHECK(::rocprim::adjacent_find<Config>(d_tmp_storage,
                                                       tmp_storage_size,
                                                       d_input,
                                                       d_output,
                                                       items,
                                                       rocprim::equal_to<input_type>{},
                                                       stream,
                                                       false));
        };

        // Get size of temporary storage
        launch_adjacent_find();
        HIP_CHECK(hipMalloc(&d_tmp_storage, tmp_storage_size));

        state.set_items(first_adj_index);
        state.add_reads<input_type>(first_adj_index);

        state.run([&] { launch_adjacent_find(); });

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_tmp_storage));
    }
};

template<typename InputT, unsigned int BlockSize>
struct device_adjacent_find_benchmark_generator
{
    static constexpr unsigned int min_items_per_thread          = 1;
    static constexpr unsigned int max_items_per_thread_exponent = rocprim::Log2<32>::VALUE;

    template<unsigned int FirstAdjPosDecimal>
    struct create_pos
    {
        template<unsigned int ItemsPerThreadExp>
        struct create_ipt
        {
            static constexpr unsigned int items_per_thread = 1u << ItemsPerThreadExp;
            using generated_config = rocprim::adjacent_find_config<BlockSize, items_per_thread>;

            void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
            {
                storage.emplace_back(
                    std::make_unique<device_adjacent_find_benchmark<InputT,
                                                                    FirstAdjPosDecimal,
                                                                    generated_config>>());
            }
        };
        void operator()(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
        {
            static_for_each<
                make_index_range<unsigned int, min_items_per_thread, max_items_per_thread_exponent>,
                create_ipt>(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<primbench::benchmark_interface>>& storage)
    {
        static_for_each<std::integer_sequence<unsigned int, 1, 5, 9>, create_pos>(storage);
    }
};
