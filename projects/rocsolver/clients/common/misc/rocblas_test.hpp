/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <rocblas/rocblas.h>

// Suppress warnings about hipMalloc(), hipFree() except in rocblas-test and
// rocblas-bench
#if !defined(ROCSOLVER_CLIENTS_TEST) && !defined(ROCBLAS_BENCH)
#undef hipMalloc
#undef hipFree
#endif

#ifdef ROCSOLVER_CLIENTS_TEST
#include <gtest/gtest.h>

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#define CHECK_DEVICE_ALLOCATION(ERROR)                                                     \
    do                                                                                     \
    {                                                                                      \
        auto error = ERROR;                                                                \
        if(error == hipErrorOutOfMemory)                                                   \
        {                                                                                  \
            SUCCEED() << LIMITED_MEMORY_STRING;                                            \
            return;                                                                        \
        }                                                                                  \
        else if(error != hipSuccess)                                                       \
        {                                                                                  \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, \
                    __FILE__, __LINE__);                                                   \
            return;                                                                        \
        }                                                                                  \
    } while(0)

#define CHECK_ALLOC_QUERY(STATUS)                                  \
    do                                                             \
    {                                                              \
        auto status__ = (STATUS);                                  \
        ASSERT_TRUE(status__ == rocblas_status_size_increased      \
                    || status__ == rocblas_status_size_unchanged); \
    } while(0)

#define EXPECT_ROCBLAS_STATUS ASSERT_EQ

#else // ROCSOLVER_CLIENTS_TEST

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        fmt::print(stderr, "rocBLAS status error: Expected {}, received {}\n",
                   rocblas_status_to_string(expect), rocblas_status_to_string(status));
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(ERROR)                                                        \
    do                                                                                \
    {                                                                                 \
        auto error = ERROR;                                                           \
        if(error != hipSuccess)                                                       \
        {                                                                             \
            fmt::print(stderr, "error: {} ({}) at {}:{}\n", hipGetErrorString(error), \
                       static_cast<int32_t>(error), __FILE__, __LINE__);              \
            rocblas_abort();                                                          \
        }                                                                             \
    } while(0)

#define CHECK_ALLOC_QUERY(STATUS)                                                                     \
    do                                                                                                \
    {                                                                                                 \
        auto status__ = (STATUS);                                                                     \
        if(!(status__ == rocblas_status_size_increased || status__ == rocblas_status_size_unchanged)) \
        {                                                                                             \
            fmt::print(stderr,                                                                        \
                       "rocBLAS status error: Expected rocblas_status_size_unchanged or "             \
                       "rocblas_status_size_increase,\nreceived {}\n",                                \
                       rocblas_status_to_string(status__));                                           \
            rocblas_abort();                                                                          \
        }                                                                                             \
    } while(0)

#define CHECK_DEVICE_ALLOCATION CHECK_HIP_ERROR

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

// The info provided to EXPECT macros is used in rocsolver-test, but
// in rocsolver-bench, the information is just discarded.
struct rocsolver_info_discarder
{
    template <typename T>
    rocsolver_info_discarder& operator<<(T&&)
    {
        return *this;
    }
};

#define EXPECT_EQ(v1, v2) rocsolver_info_discarder()
#define EXPECT_NE(v1, v2) rocsolver_info_discarder()
#define EXPECT_LT(v1, v2) rocsolver_info_discarder()
#define EXPECT_LE(v1, v2) rocsolver_info_discarder()
#define EXPECT_GT(v1, v2) rocsolver_info_discarder()
#define EXPECT_GE(v1, v2) rocsolver_info_discarder()

#endif // ROCSOLVER_CLIENTS_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)
