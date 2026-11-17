// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Production-side workarounds for known hipBLASLt-provider issues. Each macro is
// keyed by the upstream issue number so it is easy to grep and remove once the
// underlying problem is fixed. Test-side counterparts live in
// `tests/TestWorkarounds.hpp` (kept separate so production TUs never pull in
// gtest).
//
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#9962 — hipBLASLt crashes (0xc0000005 access violation)
// while building a GEMM plan for some problem configs on the gfx115x on Windows.
// It faults inside hipBLASLt's heuristic instead of returning an error, so
// probing support (isApplicable, which constructs a plan) crashes the caller.
// Until the upstream fix lands we early-return `false` from the matmul plan
// builders' isApplicable() so engine selection skips the hipBLASLt matmul path.
//
// REJECT_IF_WORKAROUND_ISSUE_9962(handle) must only be invoked from a function
// whose return type is `bool` (it contains a `return`). The fault is only
// observed on Windows, so it is compile-time gated to Windows builds; elsewhere
// it expands to a no-op. An arch-query failure is fail-closed: declining reports
// the graph unsupported, guessing "not affected" on a real gfx115x crashes.
//
// To remove after the fix: `git grep WORKAROUND_ISSUE_9962` finds the macro, its
// test-side counterpart, and both call sites.
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#10811 — on gfx950 a block-scaled MX GEMM with FP8 OCP
// (E4M3 or E5M2) on graph operand A and FP6 (E2M3 or E3M2) on graph operand B
// returns incorrect results.
//
// REJECT_IF_WORKAROUND_ISSUE_10811 must only be invoked from a function with
// return type `bool` (it contains a `return`).
//
// To remove after the fix: `git grep WORKAROUND_ISSUE_10811` finds the macro and its
// single call site. Also delete the test
// `TestGpuHipblasltMxMatmulPlanBuilder.IsApplicableRejectsFp8AWithFp6B` and add the
// four `MxMixedConfig<FP8_*, FP6_*>` pairings to `MxMixedConfigs` in that same file.
// ----------------------------------------------------------------------------

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltUtils.hpp"

#include <exception>

#ifdef _WIN32
#define REJECT_IF_WORKAROUND_ISSUE_9962(handle)                                                    \
    do                                                                                             \
    {                                                                                              \
        try                                                                                        \
        {                                                                                          \
            if(::hipdnn_plugin_sdk::archMatches(                                                   \
                   ::hipdnn_plugin_sdk::getDeviceArch((handle).getStream()),                       \
                   "gfx115",                                                                       \
                   ::hipdnn_plugin_sdk::ArchMatchMode::SUBSTRING))                                 \
            {                                                                                      \
                HIPDNN_PLUGIN_LOG_INFO(                                                            \
                    "[#9962] hipBLASLt matmul not applicable: GEMM crashes on gfx115x (Windows)"); \
                return false;                                                                      \
            }                                                                                      \
        }                                                                                          \
        catch(const std::exception& workaround_9962_e)                                             \
        {                                                                                          \
            HIPDNN_PLUGIN_LOG_INFO("[#9962] arch query failed; treating as not-applicable: "       \
                                   << workaround_9962_e.what());                                   \
            return false;                                                                          \
        }                                                                                          \
    } while(0)
#else
#define REJECT_IF_WORKAROUND_ISSUE_9962(handle) \
    do                                          \
    {                                           \
    } while(0)
#endif

#define REJECT_IF_WORKAROUND_ISSUE_10811(deqAttrA, deqAttrB, tensorMap)             \
    do                                                                              \
    {                                                                               \
        const auto tXA = ::hipblaslt_plugin::hipblaslt_utils::findTensorAttributes( \
            (tensorMap), (deqAttrA).x_tensor_uid());                                \
        const auto tXB = ::hipblaslt_plugin::hipblaslt_utils::findTensorAttributes( \
            (tensorMap), (deqAttrB).x_tensor_uid());                                \
        if(::hipblaslt_plugin::hipblaslt_utils::isTypeFp8Ocp(tXA.dataType())        \
           && ::hipblaslt_plugin::hipblaslt_utils::isTypeFp6Ocp(tXB.dataType()))    \
        {                                                                           \
            HIPDNN_PLUGIN_LOG_INFO(                                                 \
                "[#10811] MX matmul not applicable: FP8 OCP A with FP6 B returns "  \
                "incorrect results");                                               \
            return false;                                                           \
        }                                                                           \
    } while(0)
