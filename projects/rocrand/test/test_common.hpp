// Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#define HIP_CHECK(condition)                                                                      \
    {                                                                                             \
        hipError_t error = condition;                                                             \
        if(error != hipSuccess)                                                                   \
        {                                                                                         \
            [error]() { FAIL() << "HIP error " << error << ": " << hipGetErrorString(error); }(); \
            exit(error);                                                                          \
        }                                                                                         \
    }

#define HIP_CHECK_NON_VOID(condition)                                                  \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
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
template<class T>
hipError_t hipMallocHelper(T** devPtr, size_t size)
{
    if(use_hmm())
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

// Helper function for implementing ASSERT_VEC_EQ.
// This function stops on the first mismatching element,
// printing its index and value.
template<typename T>
testing::AssertionResult verify_vec_eq(const char*           a_str,
                                       const char*           b_str,
                                       const std::vector<T>& a,
                                       const std::vector<T>& b)
{
    if(a.size() != b.size())
    {
        return testing::AssertionFailure() << "Expected equality of these values:\n"
                                           << "  " << a_str << ".size()\n"
                                           << "    Which is: " << a.size() << "\n"
                                           << "  " << b_str << ".size()\n"
                                           << "    Which is: " << b.size();
    }

    for(size_t i = 0; i < a.size(); ++i)
    {
        auto a_val = to_host(a[i]);
        auto b_val = to_host(b[i]);
        if(a_val != b_val)
        {
            return testing::AssertionFailure()
                   << "Expected equality of these values:" << "\n  " << a_str << "[" << i << "]"
                   << "\n    Which is: " << a_val << "\n  " << b_str << "[" << i << "]"
                   << "\n    Which is: " << b_val;
        }
    }

    return testing::AssertionSuccess();
}

#define ASSERT_VEC_EQ(a, b) ASSERT_PRED_FORMAT2(verify_vec_eq, a, b)

// Helper function for implementing ASSERT_VEC_NEAR.
// This function is based on Google Test's DoubleNearPredFormat function.
// This function stops on the first mismatching element,
// printing its index and value in decimal and hexadecimal.
testing::AssertionResult verify_near_element(
    const char* a_str, const char* b_str, double a, double b, double abs_error, size_t index)
{
    double diff = std::fabs(a - b);
    if(diff <= abs_error)
        return testing::AssertionSuccess();

    std::ostringstream a_hex, b_hex;
    a_hex << std::hexfloat << a;
    b_hex << std::hexfloat << b;

    const double min_abs = std::min(std::fabs(a), std::fabs(b));
    const double epsilon
        = std::nextafter(min_abs, std::numeric_limits<double>::infinity()) - min_abs;
    if(!(std::isnan)(a) && !(std::isnan)(b) && abs_error > 0 && abs_error < epsilon)
    {
        return testing::AssertionFailure()
               << "Expected " << a_str << "[" << index << "] and " << b_str << "[" << index
               << "] to be within epsilon " << abs_error << ", but difference is " << diff << ".\n"
               << "  This epsilon is smaller than the minimum representable spacing at this "
                  "magnitude,\n"
               << "  making this check nearly equivalent to an equality check.\n"
               << "  " << a_str << "[" << index << "]\n"
               << "    Which is: " << a << " (" << a_hex.str() << ")\n"
               << "  " << b_str << "[" << index << "]\n"
               << "    Which is: " << b << " (" << b_hex.str() << ")";
    }

    return testing::AssertionFailure()
           << "Expected " << a_str << "[" << index << "] and " << b_str << "[" << index
           << "] to be within epsilon " << abs_error << ", but difference is " << diff << ":"
           << "\n  " << a_str << "[" << index << "]" << "\n    Which is: " << a << " ("
           << a_hex.str() << ")" << "\n  " << b_str << "[" << index << "]"
           << "\n    Which is: " << b << " (" << b_hex.str() << ")";
}

template<typename T>
testing::AssertionResult verify_vec_near(const char*           a_str,
                                         const char*           b_str,
                                         const char*           abs_error_str,
                                         const std::vector<T>& a,
                                         const std::vector<T>& b,
                                         double                abs_error)
{
    (void)abs_error_str;

    if(a.size() != b.size())
    {
        return testing::AssertionFailure() << "Expected equality of these values:\n"
                                           << "  " << a_str << ".size()\n"
                                           << "    Which is: " << a.size() << "\n"
                                           << "  " << b_str << ".size()\n"
                                           << "    Which is: " << b.size();
    }

    for(size_t i = 0; i < a.size(); ++i)
    {
        auto result = verify_near_element(a_str, b_str, to_host(a[i]), to_host(b[i]), abs_error, i);
        if(!result)
            return result;
    }

    return testing::AssertionSuccess();
}

#define ASSERT_VEC_NEAR(a, b, abs_error) ASSERT_PRED_FORMAT3(verify_vec_near, a, b, abs_error)

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

// struct to represent an Emperical Distribution Function (EDF)
// of some sample
struct EDF
{
    std::vector<double> sample;
    double              n;
    EDF(const std::vector<double>& x)
    {
        sample = x;
        std::sort(sample.begin(), sample.end());
        n = static_cast<double>(sample.size());
    }

    double operator()(double x) const
    {
        auto   it  = std::upper_bound(sample.begin(), sample.end(), x);
        double pos = static_cast<double>(it - sample.begin());
        return pos / n;
    }
};

// Perform Two-Sample Kolmogorov-Smirnov Test
bool ks_test_2(const std::vector<double>& expected,
               const std::vector<double>& actual,
               double                     alpha = 0.1)
{
    EDF aEDF(expected);
    EDF eEDF(actual);

    double n = aEDF.n;
    double m = eEDF.n;

    double max_diff = std::numeric_limits<double>::min();

    // Calculate the statistical value: the maximum difference between the two EDF.
    for(const double& x : aEDF.sample)
    {
        max_diff = std::max(max_diff, std::abs(aEDF(x) - eEDF(x)));
    }

    for(const double& x : eEDF.sample)
    {
        max_diff = std::max(max_diff, std::abs(aEDF(x) - eEDF(x)));
    }

    // calculating the critical value
    double c_alpha = std::sqrt(-std::log(alpha / 2) * 0.5);
    double cv      = std::sqrt((n + m) / (n * m)) * c_alpha;

    return max_diff <= cv; // <= because we reject if d > cv
}

#endif // TEST_COMMON_HPP_
