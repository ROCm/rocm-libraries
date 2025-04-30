/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#include "../../library/src/include/handle.hpp"
#include "argument_model.hpp"
#include "client_omp.hpp"
#include "rocblas.h"
#include "rocblas_arguments.hpp"
#include "test_cleanup.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef WIN32
typedef long long ssize_t; /* x64 only supported */
#endif

#ifdef GOOGLE_TEST

#include "gtest_helpers.hpp"

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#define CHECK_DEVICE_ALLOCATION(ERROR)                                                          \
    do                                                                                          \
    {                                                                                           \
        /* Use error__ in case ERROR contains "error" */                                        \
        hipError_t error__ = (ERROR);                                                           \
        if(error__ != hipSuccess)                                                               \
        {                                                                                       \
            if(error__ == hipErrorOutOfMemory)                                                  \
            {                                                                                   \
                rocblas_cerr << "hip OutOfMemory at " << __FILE__ ":" << __LINE__ << std::endl; \
                GTEST_SKIP() << LIMITED_VRAM_STRING;                                            \
            }                                                                                   \
            else                                                                                \
                FAIL() << hipGetErrorString(error__);                                           \
            return;                                                                             \
        }                                                                                       \
    } while(0)

// This wraps the rocBLAS call with catch_signals_and_exceptions_as_failures().
// By placing it at the rocBLAS call site, memory resources are less likely to
// be leaked in the event of a caught signal.
#define EXPECT_ROCBLAS_STATUS(STATUS, EXPECT)               \
    do                                                      \
    {                                                       \
        volatile bool signal_or_exception = true;           \
        /* Use status__ in case STATUS contains "status" */ \
        rocblas_status status__;                            \
        catch_signals_and_exceptions_as_failures([&] {      \
            status__            = (STATUS);                 \
            signal_or_exception = false;                    \
        });                                                 \
        if(signal_or_exception)                             \
            return;                                         \
        { /* localize status for ASSERT_EQ message */       \
            rocblas_status status_ = status__;              \
            ASSERT_TRUE(status_match(EXPECT, status_));     \
        }                                                   \
    } while(0)

#define CHECK_ALLOC_QUERY(STATUS)                                  \
    do                                                             \
    {                                                              \
        auto status__ = (STATUS);                                  \
        ASSERT_TRUE(status__ == rocblas_status_size_increased      \
                    || status__ == rocblas_status_size_unchanged); \
    } while(0)

#else // GOOGLE_TEST

#define CHECK_ALLOC_QUERY(STATUS)                                                              \
    do                                                                                         \
    {                                                                                          \
        auto status__ = (STATUS);                                                              \
        if(!(status__ == rocblas_status_size_increased                                         \
             || status__ == rocblas_status_size_unchanged))                                    \
        {                                                                                      \
            rocblas_cerr << "rocBLAS status error: Expected rocblas_status_size_unchanged or " \
                            "rocblas_status_size_increase,\nreceived "                         \
                         << rocblas_status_to_string(status__) << std::endl;                   \
            rocblas_abort();                                                                   \
        }                                                                                      \
    } while(0)

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        rocblas_cerr << "rocBLAS status error: Expected " << rocblas_status_to_string(expect)
                     << ", received " << rocblas_status_to_string(status) << std::endl;
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(ERROR)                                                         \
    do                                                                                 \
    {                                                                                  \
        /* Use error__ in case ERROR contains "error" */                               \
        hipError_t error__ = (ERROR);                                                  \
        if(error__ != hipSuccess)                                                      \
        {                                                                              \
            rocblas_cerr << "error: " << hipGetErrorString(error__) << " (" << error__ \
                         << ") at " __FILE__ ":" << __LINE__ << std::endl;             \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while(0)

#define CHECK_DEVICE_ALLOCATION CHECK_HIP_ERROR

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#endif // GOOGLE_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)

#ifdef GOOGLE_TEST

/* ============================================================================================ */
// Function which matches Arguments with a category, accounting for arg.known_bug_platforms
bool match_test_category(const Arguments& arg, const char* category);

// The tests are instantiated by filtering through the RocBLAS_Data stream
// The filter is by category and by the type_filter() and function_filter()
// functions in the testclass
#define INSTANTIATE_TEST_CATEGORY(testclass, category)                                            \
    INSTANTIATE_TEST_SUITE_P(category,                                                            \
                             testclass,                                                           \
                             testing::ValuesIn(RocBLAS_TestData::begin([](const Arguments& arg) { \
                                                   return match_test_category(arg, #category)     \
                                                          && testclass::function_filter(arg)      \
                                                          && testclass::type_filter(arg);         \
                                               }),                                                \
                                               RocBLAS_TestData::end()),                          \
                             testclass::PrintToStringParamName());

#if defined(GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST)
#define ROCBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass) \
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(testclass);
#else
#define ROCBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass)
#endif

// Instantiate all test categories
#define INSTANTIATE_TEST_CATEGORIES(testclass)    \
    ROCBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass) \
    INSTANTIATE_TEST_CATEGORY(testclass, _)

// Category based instantiation requires pass of large yaml data for each category
// Using single '_' named category and category name is moved to test name prefix
// gtest_filter should be able to select same test subsets
// INSTANTIATE_TEST_CATEGORY(testclass, quick)       \
// INSTANTIATE_TEST_CATEGORY(testclass, pre_checkin) \
// INSTANTIATE_TEST_CATEGORY(testclass, nightly)     \
// INSTANTIATE_TEST_CATEGORY(testclass, multi_gpu)   \
// INSTANTIATE_TEST_CATEGORY(testclass, HMM)         \
// INSTANTIATE_TEST_CATEGORY(testclass, known_bug)

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda expression
#define CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(test) \
    catch_signals_and_exceptions_as_failures([&] { test; }, true)

// Function to catch signals and exceptions as failures
void launch_test_on_threads(std::function<void()> test,
                            size_t                numThreads,
                            size_t                numStreams,
                            size_t                numDevices);

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda expression
#define LAUNCH_TEST_ON_THREADS(test, threads, streams, devices) \
    launch_test_on_threads([&] { test; }, threads, streams, devices)

// Function to catch signals and exceptions as failures
void launch_test_on_streams(std::function<void()> test, size_t numStreams, size_t numDevices);

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda expression
#define LAUNCH_TEST_ON_STREAMS(test, streams, devices) \
    launch_test_on_streams([&] { test; }, streams, devices)

// Macro to run test across threads
#define RUN_TEST_ON_THREADS_STREAMS(test)                                                    \
    do                                                                                       \
    {                                                                                        \
        const auto& arg          = GetParam();                                               \
        size_t      threads      = arg.threads;                                              \
        size_t      streams      = arg.streams;                                              \
        size_t      devices      = arg.devices;                                              \
        int         availDevices = 0;                                                        \
        bool        HMM          = arg.HMM;                                                  \
        CHECK_HIP_ERROR(hipGetDeviceCount(&availDevices));                                   \
        if(devices > availDevices)                                                           \
        {                                                                                    \
            GTEST_SKIP() << TOO_FEW_DEVICES_PRESENT_STRING;                                  \
            return;                                                                          \
        }                                                                                    \
        else if(HMM)                                                                         \
        {                                                                                    \
            for(int i = 0; i < devices; i++)                                                 \
            {                                                                                \
                int flag = 0;                                                                \
                CHECK_HIP_ERROR(hipDeviceGetAttribute(                                       \
                    &flag, hipDeviceAttribute_t(hipDeviceAttributeManagedMemory), devices)); \
                if(!flag)                                                                    \
                {                                                                            \
                    GTEST_SKIP() << HMM_NOT_SUPPORTED_STRING;                                \
                    return;                                                                  \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        g_stream_pool.reset(devices, streams);                                               \
        if(threads)                                                                          \
        {                                                                                    \
            client_omp_manager manager(threads);                                             \
            LAUNCH_TEST_ON_THREADS(test, threads, streams, devices);                         \
        }                                                                                    \
        else                                                                                 \
            LAUNCH_TEST_ON_STREAMS(test, streams, devices);                                  \
    } while(0)

// Thread worker class
class thread_pool
{
    std::atomic_bool                                                 m_done{false};
    std::queue<std::pair<std::function<void()>, std::promise<void>>> m_work_queue;
    std::vector<std::thread>                                         m_threads;
    std::mutex                                                       m_mutex;
    std::condition_variable                                          m_cond;

    void worker_thread();

public:
    thread_pool();
    ~thread_pool();
    void submit(std::function<void()> func, std::promise<void> promise);
};

class stream_pool
{
    std::vector<std::vector<hipStream_t>> m_streams;

public:
    stream_pool() = default;

    void reset(size_t numDevices = 0, size_t numStreams = 0);

    ~stream_pool()
    {
        reset();
    }

    auto& operator[](size_t deviceId)
    {
        return m_streams.at(deviceId);
    }
};

extern stream_pool g_stream_pool;
extern thread_pool g_thread_pool;

/* ============================================================================================ */
/*! \brief  Normalized test name to conform to Google Tests */
// The template parameter is only used to generate multiple instantiations with distinct static local variables
template <typename>
class RocBLAS_TestName
{
    std::ostringstream m_str;

public:
    explicit RocBLAS_TestName(const char* name)
    {
        m_str << name << '_';
    }

    // Convert stream to normalized Google Test name
    // rvalue reference qualified so that it can only be called once
    // The name should only be generated once before the stream is destroyed
    operator std::string() &&
    {
        // This table is private to each instantiation of RocBLAS_TestName
        // Placed inside function to avoid dependency on initialization order
        static std::unordered_map<std::string, size_t>* table = test_cleanup::allocate(&table);
        std::string RocBLAS_TestName_to_string(std::unordered_map<std::string, size_t>&,
                                               const std::ostringstream&);
        return RocBLAS_TestName_to_string(*table, m_str);
    }

    // Stream output operations
    template <typename U> // Lvalue LHS
    friend RocBLAS_TestName& operator<<(RocBLAS_TestName& name, U&& obj)
    {
        name.m_str << std::forward<U>(obj);
        return name;
    }

    template <typename U> // Rvalue LHS
    friend RocBLAS_TestName&& operator<<(RocBLAS_TestName&& name, U&& obj)
    {
        name.m_str << std::forward<U>(obj);
        return std::move(name);
    }

    RocBLAS_TestName()                        = default;
    RocBLAS_TestName(const RocBLAS_TestName&) = delete;
    RocBLAS_TestName& operator=(const RocBLAS_TestName&) = delete;
};

bool rocblas_client_global_filters(const Arguments& args);

// ----------------------------------------------------------------------------
// RocBLAS_Test base class. All non-legacy rocBLAS Google tests derive from it.
// It defines a type_filter_functor() and a PrintToStringParamName class
// which calls name_suffix() in the derived class to form the test name suffix.
// ----------------------------------------------------------------------------
template <typename TEST, template <typename...> class FILTER>
class RocBLAS_Test : public testing::TestWithParam<Arguments>
{
protected:
    // This template functor returns true if the type arguments are valid.
    // It converts a FILTER specialization to bool to test type matching.
    template <typename... T>
    struct type_filter_functor
    {
        bool operator()(const Arguments& args)
        {
            // additional global filters applied first
            if(!rocblas_client_global_filters(args))
                return false;

            // type filters
            return static_cast<bool>(FILTER<T...>{});
        }
    };

public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            std::string name(info.param.category);
            name += "_";
            name += TEST::name_suffix(info.param);
            return name;
        }
    };
};

// Function to set up signal handlers
void rocblas_test_sigaction();

#endif // GOOGLE_TEST

// ----------------------------------------------------------------------------
// Normal tests which return true when converted to bool
// ----------------------------------------------------------------------------
struct rocblas_test_valid
{
    // Return true to indicate the type combination is valid, for filtering
    virtual explicit operator bool() final
    {
        return true;
    }

    // Require derived class to define functor which takes (const Arguments &)
    virtual void operator()(const Arguments&) = 0;

    virtual ~rocblas_test_valid() = default;
};

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct rocblas_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    virtual explicit operator bool() final
    {
        return false;
    }

    // If this specialization is actually called, print fatal error message
    virtual void operator()(const Arguments& arg) final
    {
        static constexpr char msg[] = "Internal error: Test called with invalid types";

#ifdef GOOGLE_TEST
        FAIL() << msg;
#else
        rocblas_cerr << msg << std::endl;
        rocblas_cerr << "function: " << arg.function << " types: "
                     << " a: " << rocblas_datatype2string(arg.a_type)
                     << " b: " << rocblas_datatype2string(arg.b_type)
                     << " c: " << rocblas_datatype2string(arg.c_type)
                     << " d: " << rocblas_datatype2string(arg.d_type)
                     << " compute:" << rocblas_datatype2string(arg.compute_type) << std::endl;
        rocblas_abort();
#endif
    }

    virtual ~rocblas_test_invalid() = default;
};
