// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// One rule for tests that need a HIP device. "No device" means hipGetDeviceCount returned
// an error or a count of 0. That is a test failure, unless TENSILE_SKIP_IF_NO_DEVICE=1 (set
// by the CPU-only Component CI lane), so a broken driver on a GPU runner fails loudly
// instead of turning every device test into a skip.
namespace TensileLite::TestUtils
{
    inline bool skipWithoutDevice()
    {
        char const* value = std::getenv("TENSILE_SKIP_IF_NO_DEVICE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }

    inline constexpr char const* noDeviceFix
        = "on a GPU machine, check that rocminfo lists the GPU and that HIP_VISIBLE_DEVICES does"
          " not hide it; on a machine without one, set TENSILE_SKIP_IF_NO_DEVICE=1 to skip";

    // For tests that check both the no-device and the device behaviour: returns the device
    // count, or 0 for "no device", and records a failure for that case unless
    // TENSILE_SKIP_IF_NO_DEVICE=1.
    inline int deviceCountOrFail()
    {
        int              count = 0;
        hipError_t const error = hipGetDeviceCount(&count);
        if(error == hipSuccess && count > 0)
            return count;
        if(!skipWithoutDevice())
            ADD_FAILURE() << "no HIP device: hipGetDeviceCount returned " << hipGetErrorName(error)
                          << ", count " << count << "; " << noDeviceFix;
        return 0;
    }
}

// Skips when the machine has devices but fewer than minDevices (for example a two-device
// test on a one-GPU runner). With no device at all, follows the rule above.
#define SKIP_IF_FEWER_DEVICES(minDevices)                                                   \
    do                                                                                      \
    {                                                                                       \
        int              deviceCount_ = 0;                                                  \
        hipError_t const err_         = hipGetDeviceCount(&deviceCount_);                   \
        if(err_ != hipSuccess || deviceCount_ < 1)                                          \
        {                                                                                   \
            if(::TensileLite::TestUtils::skipWithoutDevice())                               \
                GTEST_SKIP() << "needs a HIP device (" << hipGetErrorName(err_) << ")";     \
            FAIL() << "no HIP device: hipGetDeviceCount returned " << hipGetErrorName(err_) \
                   << ", count " << deviceCount_ << "; "                                    \
                   << ::TensileLite::TestUtils::noDeviceFix;                                \
        }                                                                                   \
        if(deviceCount_ < (minDevices))                                                     \
            GTEST_SKIP() << "needs " << (minDevices) << " HIP devices, found "              \
                         << deviceCount_;                                                   \
    } while(0)

#define SKIP_IF_NO_DEVICE() SKIP_IF_FEWER_DEVICES(1)
