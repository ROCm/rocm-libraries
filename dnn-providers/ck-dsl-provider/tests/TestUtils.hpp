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

// Tests in this suite share two kinds of device gate:
//
//   CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH  -- for tests that compile a
//     kernel for whatever DSL-supported device is present and run it.
//     These work on gfx942, gfx950, and gfx1151 (any arch the DSL can
//     build for). On success the bare gfx token is written to the caller's
//     ``outArch`` lvalue so the test can pass it to arch-aware entry
//     points such as compileSmoke(arch) or compile(opKind, payload, arch).
//
//   CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950  -- for tests that build the
//     gfx950-tuned production-default config (the conv plan-builder path).
//     The default adapter knobs (32x32x16 atom, wave64) are only valid on
//     gfx950; use this gate ONLY for those tests.
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

// Skip unless a HIP device is present AND that device is gfx950. Use
// ONLY for tests that build the gfx950-tuned production-default config:
// the conv plan-builder path whose adapter emits the DSL dataclass
// defaults (32x32x16 atom, wave64) that are valid only on gfx950.
// buildPlan with those knobs correctly declines or throws on other arches
// until the adapter is arch-aware in M2.
//
// This is NOT a DSL limitation -- the DSL compiles gfx942/gfx950/gfx1151.
// Tests that select a per-arch config (e.g. cross-arch example compile,
// compileSmoke) should use CK_DSL_PROVIDER_SKIP_IF_UNSUPPORTED_ARCH
// instead.
#define CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950(testName)                                          \
    do {                                                                                      \
        CK_DSL_PROVIDER_SKIP_IF_NO_GPU(testName);                                             \
        hipDeviceProp_t _ckdsl_props{};                                                       \
        ASSERT_EQ(hipGetDeviceProperties(&_ckdsl_props, 0), hipSuccess);                      \
        std::string _ckdsl_arch_name = _ckdsl_props.gcnArchName;                              \
        if (_ckdsl_arch_name.find("gfx950") == std::string::npos) {                           \
            GTEST_SKIP() << (testName) << ": requires gfx950 (production-default conv knobs " \
                         << "are gfx950-tuned); " << "device 0 reports gcnArchName='"         \
                         << _ckdsl_arch_name << "'";                                          \
        }                                                                                     \
    } while (0)
