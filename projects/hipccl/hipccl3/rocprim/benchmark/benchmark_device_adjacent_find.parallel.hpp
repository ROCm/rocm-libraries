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

#ifndef ROCPRIM_BENCHMARK_DEVICE_ADJACENT_FIND_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_ADJACENT_FIND_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// gbench
#include <benchmark/benchmark.h>

// HIP
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/detail/various.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_adjacent_find.hpp>

// C++ Standard Library
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

template<typename Config>
std::string config_name()
{
    auto config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread) + "}";
}

template<>
inline std::string config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename InputT,
         unsigned int FirstAdjPosDecimal,
         typename Config = rocprim::default_config>
struct device_adjacent_find_benchmark : public config_autotune_interface
{

    std::string name() const override
    {

        using namespace std::string_literals;
        return bench_naming::format_name(
            "{lvl:device,algo:adjacent_find,input_type:" + std::string(Traits<InputT>::name())
            + ",first_adj_pos:" + std::to_string(FirstAdjPosDecimal * 0.1f)
            + ",cfg:" + config_name<Config>() + "}");
    }

    static constexpr size_t warmup_size = 5;
    static constexpr size_t batch_size  = 10;

    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using input_type  = InputT;
        using output_type = std::size_t;

        const size_t size = bytes / sizeof(input_type);

        // Get index of the first adjacent equal pair
        std::size_t first_adj_index = static_cast<std::size_t>(size * FirstAdjPosDecimal * 0.1f);
        if(first_adj_index >= size - 1)
        {
            first_adj_index = size - 2;
        }

        // Generate data ensuring there is no adjacent pair before first_adj_index
        std::vector<input_type> input(size);
        if(std::is_same<input_type, int8_t>::value)
        {
            // For int8_t that has a very limited range of values, iota initialization
            // seems to give a more reliable benchmark input
            std::iota(input.begin(), input.end(), 0);
        }
        else
        {
            input = get_random_data<input_type>(size,
                                                generate_limits<input_type>::min(),
                                                generate_limits<input_type>::max(),
                                                seed.get_0());
            std::vector<std::size_t> iota(size);
            std::iota(iota.begin(), iota.end(), 0);
            std::transform(iota.begin() + 1,
                           iota.begin() + first_adj_index + 1,
                           input.begin() + 1,
                           [&](std::size_t& idx)
                           {
                               while(input[idx] == input[idx - 1])
                               {
                                   input[idx] = get_random_value<input_type>(
                                       generate_limits<input_type>::min(),
                                       generate_limits<input_type>::max(),
                                       seed.get_0());
                               }
                               return input[idx];
                           });
        }

        // Insert first adjacent pair
        input[first_adj_index] = input[first_adj_index + 1];

        input_type*  d_input;
        output_type* d_output;
        HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
        HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(*d_input),
                            hipMemcpyHostToDevice));

        std::size_t tmp_storage_size;
        void*       d_tmp_storage        = nullptr;
        auto        launch_adjacent_find = [&]()
        {
            HIP_CHECK(::rocprim::adjacent_find(d_tmp_storage,
                                               tmp_storage_size,
                                               d_input,
                                               d_output,
                                               size,
                                               rocprim::equal_to<input_type>{},
                                               stream,
                                               false));
        };

        // Get size of tmporary storage
        launch_adjacent_find();
        HIP_CHECK(hipMalloc(&d_tmp_storage, tmp_storage_size));

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            launch_adjacent_find();
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Run
        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                launch_adjacent_find();
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * first_adj_index
                                * sizeof(*d_input));
        state.SetItemsProcessed(state.iterations() * batch_size * first_adj_index);

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_tmp_storage));
    }
};

template<typename InputT, unsigned int BlockSize>
struct device_adjacent_find_benchmark_generator
{
    static constexpr unsigned int min_items_per_thread = 1;
    static constexpr unsigned int max_items_per_thread_arg
        = TUNING_SHARED_MEMORY_MAX / (BlockSize * sizeof(InputT) * 2);

    template<unsigned int FirstAdjPosDecimal>
    struct create_pos
    {
        template<unsigned int ItemsPerThreadExp>
        struct create_ipt
        {
            static constexpr unsigned int items_per_thread = 1u << ItemsPerThreadExp;
            using generated_config = rocprim::adjacent_find_config<BlockSize, items_per_thread>;

            void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
            {
                storage.emplace_back(
                    std::make_unique<device_adjacent_find_benchmark<InputT,
                                                                    FirstAdjPosDecimal,
                                                                    generated_config>>());
            }
        };
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            static constexpr unsigned int max_items_per_thread_exponent
                = rocprim::Log2<max_items_per_thread_arg>::VALUE - 1;
            static_for_each<
                make_index_range<unsigned int, min_items_per_thread, max_items_per_thread_exponent>,
                create_ipt>(storage);
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static_for_each<std::integer_sequence<unsigned int, 1, 5, 9>, create_pos>(storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_ADJACENT_FIND_PARALLEL_HPP_
