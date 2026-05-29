// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <string>

// Two preconditions the GPU-gated unit suite shares with the
// integration test: a HIP-visible device must exist, and that
// device's gcnArchName must contain "gfx950" (the DSL emits
// gfx950-only HSACO today; any other arch fails hipModuleLoadData
// with hipErrorNoBinaryForGpu, which is otherwise confusing).
//
// The helpers are macros rather than free functions so the embedded
// GTEST_SKIP() / ASSERT_EQ short-circuit out of the calling test's
// body -- a free function's return would only exit the helper itself.

#define CK_DSL_PROVIDER_SKIP_IF_NO_GPU(testName)                                           \
    do {                                                                                   \
        int _ckdsl_device_count = 0;                                                       \
        hipError_t _ckdsl_hip_err = hipGetDeviceCount(&_ckdsl_device_count);               \
        if (_ckdsl_hip_err != hipSuccess || _ckdsl_device_count == 0) {                    \
            GTEST_SKIP() << (testName)                                                     \
                         << ": no HIP-visible device (deviceCount=" << _ckdsl_device_count \
                         << ", hipError=" << static_cast<int>(_ckdsl_hip_err) << ")";      \
        }                                                                                  \
        ASSERT_EQ(hipSetDevice(0), hipSuccess);                                            \
    } while (0)

// Sibling of CK_DSL_PROVIDER_SKIP_IF_NO_GPU that also skips when the
// present device is not gfx950 (the only ISA the DSL emits HSACO for
// today). Use in tests that load a DSL-produced HSACO via
// hipModuleLoadData.
#define CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950(testName)                                          \
    do {                                                                                      \
        CK_DSL_PROVIDER_SKIP_IF_NO_GPU(testName);                                             \
        hipDeviceProp_t _ckdsl_props{};                                                       \
        ASSERT_EQ(hipGetDeviceProperties(&_ckdsl_props, 0), hipSuccess);                      \
        std::string _ckdsl_arch_name = _ckdsl_props.gcnArchName;                              \
        if (_ckdsl_arch_name.find("gfx950") == std::string::npos) {                           \
            GTEST_SKIP() << (testName) << ": requires gfx950 (DSL emits gfx950-only HSACO); " \
                         << "device 0 reports gcnArchName='" << _ckdsl_arch_name << "'";      \
        }                                                                                     \
    } while (0)
