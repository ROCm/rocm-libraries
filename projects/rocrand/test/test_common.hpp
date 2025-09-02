// Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_COMMON_HPP_
#define TEST_COMMON_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

// GoogleTest-compatible HIP_CHECK macro. FAIL is called to log the Google Test trace.
// The lambda is invoked immediately as assertions that generate a fatal failure can
// only be used in void-returning functions.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            [error]()                                                                       \
                { FAIL() << "HIP error " << error << ": " << hipGetErrorString(error); }(); \
            exit(error);                                                                    \
        }                                                                                   \
    }

#define HIP_CHECK_NON_VOID(condition)         \
{                                    \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

#ifdef __HIP_PLATFORM_NVCC__
    #include <cuda/std/cmath>
    #define POWF(...) cuda::std::powf(__VA_ARGS__)
#else
    #define POWF(...) std::powf(__VA_ARGS__)
#endif

#ifdef _MSC_VER
inline bool is_environment_variable_set_to_1(const char* name)
{
    char   buffer[2]{};
    size_t size;
    if(getenv_s(&size, buffer, name) != 0)
    {
        return false;
    }
    return strcmp(buffer, "1") == 0;
}
#else
inline bool is_environment_variable_set_to_1(const char* name)
{
    if(getenv(name) == nullptr)
    {
        return false;
    }

    if(strcmp(getenv(name), "1") == 0)
    {
        return true;
    }
    return false;
}
#endif

inline bool are_slow_tests_enabled()
{
    return is_environment_variable_set_to_1("RUN_SLOW_TESTS");
}

#define ROCRAND_SKIP_SLOW_TEST_IF_NOT_ENABLED()                                                   \
    do                                                                                            \
    {                                                                                             \
        if(!are_slow_tests_enabled())                                                             \
        {                                                                                         \
            GTEST_SKIP() << "This test can be enabled via environment variable RUN_SLOW_TESTS=1"; \
        }                                                                                         \
    }                                                                                             \
    while(0)

inline bool use_hmm()
{
    return is_environment_variable_set_to_1("ROCRAND_USE_HMM");
}

// Helper for HMM allocations: if HMM is requested through
// setting environment variable ROCRAND_USE_HMM=1
template <class T>
hipError_t hipMallocHelper(T** devPtr, size_t size)
{
    if (use_hmm())
    {
        return hipMallocManaged(devPtr, size);
    }
    else
    {
        return hipMalloc(devPtr, size);
    }
    return hipSuccess;
}

template<typename T>
T to_host(T x)
{
    return x;
}

inline float to_host(__half x)
{
    return static_cast<float>(x);
}

template<typename T>
void assert_eq(const std::vector<T>& a, const std::vector<T>& b)
{
    ASSERT_EQ(a.size(), b.size());
    for(size_t i = 0; i < a.size(); ++i)
    {
        ASSERT_EQ(to_host(a[i]), to_host(b[i])) << "where i = " << i;
    }
}

template<typename T>
void assert_near(const std::vector<T>& a, const std::vector<T>& b, double eps)
{
    ASSERT_EQ(a.size(), b.size());
    for(size_t i = 0; i < a.size(); ++i)
    {
        ASSERT_NEAR(to_host(a[i]), to_host(b[i]), eps)
            << "where i = " << i << ", a[i] = " << std::hexfloat << to_host(a[i])
            << ", b[i] = " << to_host(b[i]);
    }
}

template<typename T>
double get_mean(const std::vector<T>& values)
{
    double mean = 0.0f;
    for(auto v : values)
    {
        mean += static_cast<double>(v);
    }
    return mean / values.size();
}

template<typename T>
double get_variance(const std::vector<T>& values, double mean)
{
    double variance = 0.0f;
    for(auto v : values)
    {
        const double x = static_cast<double>(v) - mean;
        variance += x * x;
    }
    return variance / values.size();
}

#endif // TEST_COMMON_HPP_
