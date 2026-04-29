// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Test-side workarounds for known MIOpen-provider issues. Mirrors the
// production-side macros in `Workarounds.hpp` (kept in a separate header so
// production TUs never pull in gtest). Each macro is keyed by the upstream
// issue number so it is easy to grep and remove once the fix lands.
//
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#5409 — see `Workarounds.hpp` for the full description.
// SKIP_IF_WORKAROUND_ISSUE_5409() emits GTEST_SKIP() in tests that exercise
// the MIOpen CBA path so CI stays green on gfx90a while the production-side
// REJECT keeps the engine inert.
//
// Arch detection is deliberately inlined (not pulled from MiopenUtils) so this
// header has no dependency on the provider's compiled object files — it can be
// used from integration_tests, which only links the plugin via dlopen.
//
// To remove after the fix: delete this file alongside `Workarounds.hpp`, drop
// includes, and remove all macro call sites.
// `git grep WORKAROUND_ISSUE_5409` finds them all.
// ----------------------------------------------------------------------------

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace test_common::workarounds_detail
{

// Strips the feature suffix from a gcnArchName (so "gfx90a:sramecc+:xnack-"
// becomes "gfx90a"). Returns an empty string when the input is empty or
// begins with ':'. Pure string logic; covered by TestWorkarounds.cpp.
inline std::string stripArchFeatureSuffix(std::string_view archName)
{
    return std::string{archName.substr(0, archName.find(':'))};
}

// Returns the gcnArchName of device 0, stripped of any feature suffix.
// Empty string on HIP failure.
//
// Device 0 is intentional: tests in this provider don't switch devices, and
// avoiding hipStreamGetDevice() keeps this header free of the MiopenUtils
// linkage dependency (integration_tests only dlopens the plugin). On a
// multi-GPU host where device 0 differs from the device under test, the SKIP
// macro will mis-trigger; that case is not exercised in current CI.
inline std::string queryCurrentDeviceArch()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return {};
    }
    return stripArchFeatureSuffix(props.gcnArchName);
}

} // namespace test_common::workarounds_detail

#define SKIP_IF_WORKAROUND_ISSUE_5409()                                                           \
    do                                                                                            \
    {                                                                                             \
        const auto issue_5409_arch = ::test_common::workarounds_detail::queryCurrentDeviceArch(); \
        if(issue_5409_arch.empty())                                                               \
        {                                                                                         \
            GTEST_SKIP() << "Skipping due to ROCm/rocm-libraries#5409 "                           \
                            "(arch query failed; failing closed)";                                \
        }                                                                                         \
        if(issue_5409_arch == "gfx90a")                                                           \
        {                                                                                         \
            GTEST_SKIP() << "Skipping due to ROCm/rocm-libraries#5409 "                           \
                            "(MIOpen CBA fusion disabled on gfx90a)";                             \
        }                                                                                         \
    } while(0)
