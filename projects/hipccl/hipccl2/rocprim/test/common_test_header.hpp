// MIT License
//
// Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_COMMON_TEST_HEADER
#define ROCPRIM_COMMON_TEST_HEADER

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#ifndef __HIP_CPU_RT__
#include <hip/hip_ext.h>
#endif

// GoogleTest-compatible HIP_CHECK macro. FAIL is called to log the Google Test trace.
// The lambda is invoked immediately as assertions that generate a fatal failure can
// only be used in void-returning functions.
#ifndef HIP_CHECK
    #define HIP_CHECK(condition)                                                            \
        {                                                                                   \
            hipError_t error = condition;                                                   \
            if(error != hipSuccess)                                                         \
            {                                                                               \
                [error]()                                                                   \
                { FAIL() << "HIP error " << error << ": " << hipGetErrorString(error); }(); \
                exit(error);                                                                \
            }                                                                               \
        }
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
#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4996 ) // This function or variable may be unsafe. Consider using _dupenv_s instead.
#endif
    static const std::string rg0 = "CTEST_RESOURCE_GROUP_0";
    if (std::getenv(rg0.c_str()) != nullptr)
    {
        std::string amdgpu_target = std::getenv(rg0.c_str());
        std::transform(
            amdgpu_target.cbegin(),
            amdgpu_target.cend(),
            amdgpu_target.begin(),
            // Feeding std::toupper plainly results in implicitly truncating conversions between int and char triggering warnings.
            [](unsigned char c){ return static_cast<char>(std::toupper(c)); }
        );
        std::string reqs = std::getenv((rg0 + "_" + amdgpu_target).c_str());
        return std::atoi(reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
    }
    else
        return 0;
#ifdef _MSC_VER
#pragma warning( pop )
#endif
}

inline bool use_hmm()
{
    if (getenv("ROCPRIM_USE_HMM") == nullptr)
    {
        return false;
    }

    if (strcmp(getenv("ROCPRIM_USE_HMM"), "1") == 0)
    {
        return true;
    }
    return false;
}

// Helper for HMM allocations: HMM is requested through ROCPRIM_USE_HMM=1 environment variable
template <class T>
hipError_t hipMallocHelper(T** devPtr, size_t size)
{
    if (use_hmm())
    {
        return hipMallocManaged(reinterpret_cast<void**>(devPtr), size);
    }
    else
    {
        return hipMalloc(reinterpret_cast<void**>(devPtr), size);
    }
    return hipSuccess;
}

}

#endif
