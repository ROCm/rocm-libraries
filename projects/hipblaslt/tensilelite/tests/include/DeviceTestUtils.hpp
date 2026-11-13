// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// Skips the calling test when no HIP device is visible (CPU-only Component CI).
#define SKIP_IF_NO_DEVICE()                                                    \
    do                                                                         \
    {                                                                          \
        int deviceCount_ = 0;                                                  \
        if(hipGetDeviceCount(&deviceCount_) != hipSuccess || deviceCount_ < 1) \
            GTEST_SKIP() << "needs a HIP device";                              \
    } while(0)
