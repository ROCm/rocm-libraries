// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Fatal reports must return from the caller, including callers inside loops and lambdas.
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>

#define HIPBLASLT_FAIL(message) FAIL() << message
#define HIPBLASLT_SKIP(message) GTEST_SKIP() << message
#define HIPBLASLT_REPORT_FAILURE(message) ADD_FAILURE() << message

#else
#include "hipblaslt_ostream.hpp"

#define HIPBLASLT_REPORT_FAILURE(message)       \
    do                                          \
    {                                           \
        hipblaslt_cerr << message << std::endl; \
    } while(0)

#define HIPBLASLT_FAIL(message)            \
    do                                     \
    {                                      \
        HIPBLASLT_REPORT_FAILURE(message); \
        return;                            \
    } while(0)

#define HIPBLASLT_SKIP(message)                 \
    do                                          \
    {                                           \
        hipblaslt_cout << message << std::endl; \
        return;                                 \
    } while(0)
#endif
