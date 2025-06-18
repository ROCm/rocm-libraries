// MIT License
//
// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_COMMON_TEST_HEADER_HPP_
#define ROCPRIM_COMMON_TEST_HEADER_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/utils.hpp"

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#define HIP_CHECK_MEMORY(condition)                                                         \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error == hipErrorOutOfMemory)                                                    \
        {                                                                                   \
            (void) hipGetLastError();                                                       \
            std::cout << "Out of memory. Skipping size = " << size << std::endl;            \
            break;                                                                          \
        }                                                                                   \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ \
                      << std::endl;                                                         \
            exit(error);                                                                    \
        }                                                                                   \
    }

#ifndef ROCPRIM_HAS_INT128_SUPPORT
    #define ROCPRIM_HAS_INT128_SUPPORT 1
#endif

#define INSTANTIATE_TYPED_TEST_EXPANDED_1(line, test_suite_name, ...)         \
    namespace Id##line                                                        \
    {                                                                         \
        using test_type = __VA_ARGS__;                                        \
        INSTANTIATE_TYPED_TEST_SUITE_P(Id##line, test_suite_name, test_type); \
    }

#define INSTANTIATE_TYPED_TEST_EXPANDED(line, test_suite_name, ...) \
    INSTANTIATE_TYPED_TEST_EXPANDED_1(line, test_suite_name, __VA_ARGS__)

// Used in input file for rocprim_test_add_parallel.
// Instantiate a typed test suite with a unique name based on line number.
// Do not call this macro twice on the same line.
#define INSTANTIATE_TYPED_TEST(test_suite_name, ...) \
    INSTANTIATE_TYPED_TEST_EXPANDED(__LINE__, test_suite_name, __VA_ARGS__)

#include <cstdlib>
#include <string>
#include <cctype>

namespace test_common_utils
{

inline int obtain_device_from_ctest()
{
    static const std::string rg0    = "CTEST_RESOURCE_GROUP_0";
    char*                    env    = common::__get_env(rg0.c_str());
    int                      device = 0;
    if(env != nullptr)
    {
        std::string amdgpu_target(env);
        std::transform(
            amdgpu_target.cbegin(),
            amdgpu_target.cend(),
            amdgpu_target.begin(),
            // Feeding std::toupper plainly results in implicitly truncating conversions between int and char triggering warnings.
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        char*       env_reqs = common::__get_env((rg0 + "_" + amdgpu_target).c_str());
        std::string reqs(env_reqs);
        device = std::atoi(
            reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
        common::clean_env(env_reqs);
    }
    common::clean_env(env);
    return device;
}
}

#endif // ROCPRIM_COMMON_TEST_HEADER_HPP_
