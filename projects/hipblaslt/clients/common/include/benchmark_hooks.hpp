// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The library's measurement engine, with a failed HIP call and the CPU timer going
// through the clients' CHECK_HIP_ERROR and get_time_us_sync. The clients include the
// engine only through this header and candidate_search.hpp, so every client
// translation unit sees the same hooks.
#ifdef HIPBLASLT_BENCHMARK_CHECK_HIP
#error "include benchmark_hooks.hpp before hipblaslt_benchmark/benchmark_timing.hpp"
#endif

#include "hipblaslt_test.hpp" // CHECK_HIP_ERROR
#include "utility.hpp" // get_time_us_sync

#define HIPBLASLT_BENCHMARK_CHECK_HIP(expr) CHECK_HIP_ERROR(expr)
#define HIPBLASLT_BENCHMARK_TIME_US_SYNC(stream) get_time_us_sync(stream)

#include <hipblaslt_benchmark/benchmark_timing.hpp>
