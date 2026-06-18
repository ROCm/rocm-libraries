// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// CUB's implementation of DeviceRunLengthEncode has unused parameters,
// disable the warning because all warnings are threated as errors:

#include "benchmark_utils.hpp"

#include <hipcub/device/device_adjacent_difference.hpp>

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{

#ifndef DEFAULT_N
constexpr size_t DEFAULT_N = 128 * primbench::MiB;
#endif

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::true_type /*copy*/,
                                  void* const    temporary_storage,
                                  size_t&        storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeftCopy(temporary_storage,
                                                                storage_size,
                                                                input,
                                                                output,
                                                                std::forward<Args>(args)...);
}

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::true_type /*copy*/,
                                  void* const    temporary_storage,
                                  size_t&        storage_size,
                                  const InputIt  input,
                                  const OutputIt output,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRightCopy(temporary_storage,
                                                                 storage_size,
                                                                 input,
                                                                 output,
                                                                 std::forward<Args>(args)...);
}

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::true_type /*left*/,
                                  std::false_type /*copy*/,
                                  void* const   temporary_storage,
                                  size_t&       storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractLeft(temporary_storage,
                                                            storage_size,
                                                            input,
                                                            std::forward<Args>(args)...);
}

template<typename InputIt, typename OutputIt, typename... Args>
auto dispatch_adjacent_difference(std::false_type /*left*/,
                                  std::false_type /*copy*/,
                                  void* const   temporary_storage,
                                  size_t&       storage_size,
                                  const InputIt input,
                                  const OutputIt /*output*/,
                                  Args&&... args)
{
    return ::hipcub::DeviceAdjacentDifference::SubtractRight(temporary_storage,
                                                             storage_size,
                                                             input,
                                                             std::forward<Args>(args)...);
}

template<typename T, bool left, bool copy>
class device_adjacent_difference_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("algo", "device_adjacent_difference")
            .add("lvl", "device")
            .add("data_type", primbench::name<T>())
            .add("subalgo",
                 "subtract_" + std::string(left ? "left" : "right")
                     + std::string(copy ? "_copy" : ""));
    }

    void run(primbench::state& state) override
    {
        using output_type = T;

        const size_t size   = state.size;
        const auto&  stream = state.stream;

        // Generate data
        const std::vector<T> input = benchmark_utils::get_random_data<T>(size, 1, 100);

        T*           d_input;
        output_type* d_output = nullptr;
        HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(input[0])));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            input.size() * sizeof(input[0]),
                            hipMemcpyHostToDevice));

        if constexpr(copy)
        {
            HIP_CHECK(hipMalloc(&d_output, size * sizeof(output_type)));
        }

        static constexpr std::integral_constant<bool, left> left_tag;
        static constexpr std::integral_constant<bool, copy> copy_tag;

        // Allocate temporary storage
        size_t temp_storage_size{};
        void*  d_temp_storage = nullptr;

        const auto launch = [&]
        {
            return dispatch_adjacent_difference(left_tag,
                                                copy_tag,
                                                d_temp_storage,
                                                temp_storage_size,
                                                d_input,
                                                d_output,
                                                size,
                                                benchmark_utils::plus{},
                                                stream);
        };

        HIP_CHECK(launch());

        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));

        state.set_items(size);
        state.add_writes<T>(size);

        state.run(launch);

        HIP_CHECK(hipFree(d_input));
        if(copy)
        {
            HIP_CHECK(hipFree(d_output));
        }
        HIP_CHECK(hipFree(d_temp_storage));
    }
};

} // namespace

using namespace std::string_literals;

#define CREATE_BENCHMARK(T, left, copy) \
    executor.queue<device_adjacent_difference_benchmark<T, left, copy>>()

// clang-format off
#define CREATE_BENCHMARKS(T)           \
    CREATE_BENCHMARK(T, true,  false), \
    CREATE_BENCHMARK(T, true,  true),  \
    CREATE_BENCHMARK(T, false, false), \
    CREATE_BENCHMARK(T, false, true)
// clang-format on

int main(int argc, char* argv[])
{
    primbench::settings settings;
    settings.size                 = DEFAULT_N;
    settings.min_gpu_ms_per_batch = 100;

    primbench::executor executor(argc, argv, settings);

    CREATE_BENCHMARKS(int);
    CREATE_BENCHMARKS(int64_t);

    CREATE_BENCHMARKS(uint8_t);

    CREATE_BENCHMARKS(float);
    CREATE_BENCHMARKS(double);

    CREATE_BENCHMARKS(custom_float2);
    CREATE_BENCHMARKS(custom_double2);

    executor.run();
}
