// MIT License
//
// Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#if(defined(__GNUC__) || defined(__clang__)) && (defined(__GLIBCXX__) || defined(_LIBCPP_VERSION))
    #define ROCPRIM_HAS_INT128_SUPPORT 1
#else
    #define ROCPRIM_HAS_INT128_SUPPORT 0
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

inline char* __get_env(const char* name)
{
    char* env;
#ifdef _MSC_VER
    size_t  len;
    errno_t err = _dupenv_s(&env, &len, name);
    if(err)
    {
        return nullptr;
    }
#else
    env = std::getenv(name);
#endif
    return env;
}

inline void clean_env(char* name)
{
#ifdef _MSC_VER
    if(name != nullptr)
    {
        free(name);
    }
#endif
    (void)name;
}

inline int obtain_device_from_ctest()
{
    static const std::string rg0    = "CTEST_RESOURCE_GROUP_0";
    char*                    env    = __get_env(rg0.c_str());
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
        char*       env_reqs = __get_env((rg0 + "_" + amdgpu_target).c_str());
        std::string reqs(env_reqs);
        device = std::atoi(
            reqs.substr(reqs.find(':') + 1, reqs.find(',') - (reqs.find(':') + 1)).c_str());
        clean_env(env_reqs);
    }
    clean_env(env);
    return device;
}

inline bool use_hmm()
{

    char*      env = __get_env("ROCPRIM_USE_HMM");
    const bool hmm = (env != nullptr) && (strcmp(env, "1") == 0);
    clean_env(env);
    return hmm;
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
