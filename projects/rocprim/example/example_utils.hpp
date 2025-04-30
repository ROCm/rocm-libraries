// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_EXAMPLE_UTILS_HPP_
#define ROCPRIM_EXAMPLE_UTILS_HPP_

#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>

#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)         \
{                                    \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

#define OUTPUT_VALIDATION_CHECK(validation_result)               \
  {                                                              \
    if ( validation_result == false )                            \
    {                                                            \
        std::cout << "Output validation failed!" << std::endl;   \
        return;                                                  \
    }                                                            \
  }

template<class T>
inline auto get_random_data(size_t size, T min, T max)
    -> typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<T> distribution(min, max);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T>
inline void hip_read_device_memory(std::vector<T> &host_destination, T *device_source)
{
    HIP_CHECK(
        hipMemcpy(
            host_destination.data(), device_source,
            host_destination.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );
}

template<class T>
inline void hip_write_device_memory(T *device_destination, std::vector<T>& host_source)
{
    HIP_CHECK(
        hipMemcpy(
            device_destination, host_source.data(),
            host_source.size() * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
}

template<class T>
inline bool validate_device_output(const std::vector<T> &host_output, const std::vector<T> &expected_output)
{
    for(unsigned int index = 0; index < host_output.size(); index++)
    {
        if(host_output[index] != expected_output[index])
        {
            return false;
        }
    }
    return true;
}

// Generating expected output for block scan when using rocprim::plus as function
template<class T>
std::vector<T> get_expected_output(
    const std::vector<T> &host_input,
    const unsigned int block_size,
    const unsigned int items_per_thread = 1)
{
    unsigned int grid_size = host_input.size() / block_size;
    std::vector<T> host_expected_output(host_input.size());
    for(unsigned int block_index = 0; block_index < (grid_size / items_per_thread); block_index++)
    {
        host_expected_output[block_index * block_size] = host_input[block_index * block_size];
        for(unsigned int thread_index = 1; thread_index < (block_size * items_per_thread); thread_index++)
        {
            int index = block_index * block_size + thread_index;
            host_expected_output[index] = host_expected_output[index - 1] + host_input[index];
        }
    }
    return host_expected_output;
}

#endif // ROCPRIM_EXAMPLE_UTILS_HPP_
