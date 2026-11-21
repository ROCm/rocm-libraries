// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// For tests that need a HIP device. Without one (count 0 or any hipGetDeviceCount error)
// the test fails, unless TENSILE_SKIP_IF_NO_DEVICE=1 (CPU-only Component CI),
// where it skips. A broken driver on a GPU runner therefore still fails loudly.
#define SKIP_IF_NO_DEVICE()                                                                 \
    do                                                                                      \
    {                                                                                       \
        int              deviceCount_ = 0;                                                  \
        hipError_t const err_         = hipGetDeviceCount(&deviceCount_);                   \
        if(err_ != hipSuccess || deviceCount_ < 1)                                          \
        {                                                                                   \
            char const* skip_ = std::getenv("TENSILE_SKIP_IF_NO_DEVICE");                   \
            if(skip_ != nullptr && std::strcmp(skip_, "1") == 0)                            \
                GTEST_SKIP() << "needs a HIP device (" << hipGetErrorName(err_) << ")";     \
            FAIL() << "no HIP device: hipGetDeviceCount returned " << hipGetErrorName(err_) \
                   << ", count " << deviceCount_                                            \
                   << "; set TENSILE_SKIP_IF_NO_DEVICE=1 to skip instead";                  \
        }                                                                                   \
    } while(0)
