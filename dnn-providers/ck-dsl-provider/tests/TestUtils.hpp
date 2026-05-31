// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>

#include "runtime/DeviceArch.hpp"

// Tests in this suite share a device gate:
//
//   CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH  -- for tests that compile a
//     kernel for whatever DSL-supported device is present and run it.
//     These work on gfx942, gfx950, and gfx1151 (any arch the DSL can
//     build for). On success the bare gfx token is written to the caller's
//     ``outArch`` lvalue so the test can pass it to arch-aware entry
//     points such as compileSmoke(arch) or compile(opKind, payload, arch).
//     The production conv plan-builder path uses this gate too: the
//     adapter selects a valid per-arch codegen config for the detected
//     device (applyArchCodegenConfig), so buildPlan runs on any of the
//     three supported arches.
//
// The helpers are macros rather than free functions so the embedded
// GTEST_SKIP() / ASSERT_* short-circuit out of the calling test's body --
// a free function's return would only exit the helper itself.

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

// The arches the CK DSL provider can build and run a kernel for. Single
// source of truth for the "supported device present" gate. The DSL's
// arch-aware compile path (build_implicit_gemm_conv / compile_kernel)
// targets exactly these tokens.
inline bool ckDslIsSupportedArch(std::string_view arch) {
    constexpr std::array<std::string_view, 3> kSupported{"gfx942", "gfx950", "gfx1151"};
    for (std::string_view candidate : kSupported) {
        if (arch == candidate) {
            return true;
        }
    }
    return false;
}

// Skip unless a HIP device is present AND its arch is one the DSL can
// build for (gfx942/gfx950/gfx1151). On success the device's bare gfx
// token is written to ``outArch`` (a std::string lvalue) so the test can
// pass it to the arch-aware compile entry points. Use for tests that
// compile a kernel for the present device and run it -- they work on any
// supported arch, not just gfx950.
#define CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH(testName, outArch)                          \
    do {                                                                                     \
        CK_DSL_PROVIDER_SKIP_IF_NO_GPU(testName);                                            \
        std::optional<std::string> _ckdsl_supported_arch =                                   \
            ck_dsl_provider::detectDeviceArch(nullptr);                                      \
        ASSERT_TRUE(_ckdsl_supported_arch.has_value())                                       \
            << (testName) << ": a HIP device is present but its arch could not be detected"; \
        if (!ckDslIsSupportedArch(*_ckdsl_supported_arch)) {                                 \
            GTEST_SKIP() << (testName) << ": device arch '" << *_ckdsl_supported_arch        \
                         << "' is outside the DSL-supported set (gfx942/gfx950/gfx1151)";    \
        }                                                                                    \
        (outArch) = *_ckdsl_supported_arch;                                                  \
    } while (0)
