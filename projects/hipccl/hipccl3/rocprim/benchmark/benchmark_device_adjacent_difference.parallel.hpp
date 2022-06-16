// MIT License
//
// Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_ADJACENT_DIFFERENCE_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_ADJACENT_DIFFERENCE_PARALLEL_HPP_

#include <cstddef>
#include <string>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime_api.h>

// rocPRIM
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_adjacent_difference.hpp>

#include "benchmark_utils.hpp"

template<typename T    = int,
         bool left     = false,
         bool in_place = false,
         typename Config
         = rocprim::detail::default_adjacent_difference_config<ROCPRIM_TARGET_ARCH, T>>
struct device_adjacent_difference_benchmark : public config_autotune_interface
{
    static std::string get_name_pattern()
    {
        return R"---((?P<algo>\S*)\<)---"
               R"---((?P<datatype>\S*),\s*adjacent_difference_config\<\s*)---"
               R"---((?P<block_size>[0-9]+),\s*(?P<items_per_thread>[0-9]+)\>\>)---";
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return std::string("device_adjacent_difference" + (left ? ""s : "_right"s)
                           + (in_place ? "_inplace"s : ""s) + "<" + std::string(Traits<T>::name())
                           + ", adjacent_difference_config<"
                           + pad_string(std::to_string(Config::block_size), 3) + ", "
                           + pad_string(std::to_string(Config::items_per_thread), 2) + ">>");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    template<typename InputIt, typename OutputIt, typename... Args>
    auto dispatch_adjacent_difference(std::true_type /*left*/,
                                      std::false_type /*in_place*/,
                                      void* const    temporary_storage,
                                      std::size_t&   storage_size,
                                      const InputIt  input,
                                      const OutputIt output,
                                      Args&&... args) const
    {
        return ::rocprim::adjacent_difference<Config>(temporary_storage,
                                                      storage_size,
                                                      input,
                                                      output,
                                                      std::forward<Args>(args)...);
    }

    template<typename InputIt, typename OutputIt, typename... Args>
    auto dispatch_adjacent_difference(std::false_type /*left*/,
                                      std::false_type /*in_place*/,
                                      void* const    temporary_storage,
                                      std::size_t&   storage_size,
                                      const InputIt  input,
                                      const OutputIt output,
                                      Args&&... args) const
    {
        return ::rocprim::adjacent_difference_right(temporary_storage,
                                                    storage_size,
                                                    input,
                                                    output,
                                                    std::forward<Args>(args)...);
    }

    template<typename InputIt, typename OutputIt, typename... Args>
    auto dispatch_adjacent_difference(std::true_type /*left*/,
                                      std::true_type /*in_place*/,
                                      void* const   temporary_storage,
                                      std::size_t&  storage_size,
                                      const InputIt input,
                                      const OutputIt /*output*/,
                                      Args&&... args) const
    {
        return ::rocprim::adjacent_difference_inplace<Config>(temporary_storage,
                                                              storage_size,
                                                              input,
                                                              std::forward<Args>(args)...);
    }

    template<typename InputIt, typename OutputIt, typename... Args>
    auto dispatch_adjacent_difference(std::false_type /*left*/,
                                      std::true_type /*in_place*/,
                                      void* const   temporary_storage,
                                      std::size_t&  storage_size,
                                      const InputIt input,
                                      const OutputIt /*output*/,
                                      Args&&... args) const
    {
        return ::rocprim::adjacent_difference_right_inplace<Config>(temporary_storage,
                                                                    storage_size,
                                                                    input,
                                                                    std::forward<Args>(args)...);
    }

    void run(benchmark::State& state,
             const std::size_t size,
             const hipStream_t stream) const override
    {
        using output_type = T;

        static constexpr bool debug_synchronous = false;

        // Generate data
        const std::vector<T> input = get_random_data<T>(size, 1, 100);

        T*           d_input;
        output_type* d_output = nullptr;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(input[0]),
                            hipMemcpyHostToDevice));

        if(!in_place)
        {
            HIP_CHECK(hipMalloc(&d_output, size * sizeof(output_type)));
        }

        static constexpr auto left_tag     = rocprim::detail::bool_constant<left>{};
        static constexpr auto in_place_tag = rocprim::detail::bool_constant<in_place>{};

        // Allocate temporary storage
        std::size_t temp_storage_size;
        void*       d_temp_storage = nullptr;

        const auto launch = [&]
        {
            return dispatch_adjacent_difference(left_tag,
                                                in_place_tag,
                                                d_temp_storage,
                                                temp_storage_size,
                                                d_input,
                                                d_output,
                                                size,
                                                rocprim::plus<>{},
                                                stream,
                                                debug_synchronous);
        };
        HIP_CHECK(launch());
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(launch());
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        for(auto _ : state)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(launch());
            }
            HIP_CHECK(hipStreamSynchronize(stream));

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds
                = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            state.SetIterationTime(elapsed_seconds.count());
        }
        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        hipFree(d_input);
        if(!in_place)
        {
            hipFree(d_output);
        }
        hipFree(d_temp_storage);
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_ADJACENT_DIFFERENCE_PARALLEL_HPP_
