// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_test.hpp" // gtest under GOOGLE_TEST, hipblaslt_cerr
#include "utility.hpp" // get_time_us_sync

#include <cstdlib>
#include <hipblaslt_benchmark/benchmark_timing.hpp>

// The clients' hooks policy for the library's measurement code. A failed HIP call is
// handled as CHECK_HIP_ERROR handles one, as a fatal test failure under gtest and
// otherwise by exiting, and the CPU clock is get_time_us_sync.
struct ClientBenchmarkHooks
{
    static bool check(hipError_t status, const char* expr, const char* file, int line)
    {
        if(status == hipSuccess)
            return true;
#ifdef GOOGLE_TEST
        [&] { GTEST_FAIL_AT(file, line) << expr << ": " << hipGetErrorString(status); }();
#else
        hipblaslt_cerr << "error: " << hipGetErrorString(status) << " (" << status << ") at "
                       << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
#endif
        return false;
    }

    static bool time_us_sync(hipStream_t stream, double& us)
    {
        us = get_time_us_sync(stream);
        return true;
    }
};
