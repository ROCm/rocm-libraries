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
// builders' isApplicable() so engine selection skips the hipBLASLt matmul path
// and hipDNN falls back.
//
// REJECT_IF_WORKAROUND_ISSUE_9962(handle) must only be invoked from a function
// whose return type is `bool` (it contains a `return`). The fault is only
// observed on Windows, so it is compile-time gated to Windows builds; elsewhere
// it expands to a no-op. An arch-query failure is treated as "not affected" so a
// healthy device is never suppressed.
//
// To remove after the fix: delete this file and `tests/TestWorkarounds.hpp`,
// drop their includes, and remove the call sites. `git grep WORKAROUND_ISSUE_9962`
// finds them all.
// ----------------------------------------------------------------------------

#include "HipblasltUtils.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <exception>

#ifdef _WIN32
#define REJECT_IF_WORKAROUND_ISSUE_9962(handle)                                                    \
    do                                                                                             \
    {                                                                                              \
        try                                                                                        \
        {                                                                                          \
            if(::hipblaslt_plugin::hipblaslt_utils::getDeviceArch((handle).getStream())            \
                   .rfind("gfx115", 0)                                                             \
               == 0)                                                                               \
            {                                                                                      \
                HIPDNN_PLUGIN_LOG_INFO(                                                            \
                    "[#9962] hipBLASLt matmul not applicable: GEMM crashes on gfx115x (Windows)"); \
                return false;                                                                      \
            }                                                                                      \
        }                                                                                          \
        catch(const std::exception& workaround_9962_e)                                             \
        {                                                                                          \
            HIPDNN_PLUGIN_LOG_INFO("[#9962] arch query failed; not applying workaround: "          \
                                   << workaround_9962_e.what());                                   \
        }                                                                                          \
    } while(0)
#else
#define REJECT_IF_WORKAROUND_ISSUE_9962(handle) \
    do                                          \
    {                                           \
    } while(0)
#endif
