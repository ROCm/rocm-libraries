// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Production-side workarounds for known MIOpen-provider issues. Each macro is
// keyed by the upstream issue number so it is easy to grep and remove once the
// underlying problem is fixed. Test-side counterparts live in
// `tests/common/TestWorkarounds.hpp` (kept separate so production TUs never
// pull in gtest).
//
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#5409 — MIOpen Conv-Bias-Activation (CBA) fusion produces
// incorrect/unsupported results on gfx90a (also covers 2-node CB and CA paths
// because all conv-fusion variants route through MiopenConvFwdBiasActivPlan-
// Builder). Until upstream fix lands, we early-return `false` from the plan
// builder's isApplicable() so engine selection skips the MIOpen CBA path on
// gfx90a.
//
// REJECT_IF_WORKAROUND_ISSUE_5409(handle) must only be invoked from a function
// whose return type is `bool` (or convertible from `false`) since it contains
// a `return`. CBA-specific test TUs observe the resulting "no engine applicable"
// outcome via the test-side issue 6979 helpers (see TestWorkarounds.hpp).
//
// Issue 5409 (this file) is gfx90a-specific. Issue 6979 (test-side) tracks the
// broader CBA capability gap across architectures — gfx90a (covered by 5409)
// plus gfx1151 / RDNA archs that lack CK fusion kernels. The two are decoupled:
// fixing 5409 only removes the gfx90a REJECT here; the 6979 test-side skip
// stays in place until CK fusion kernels exist for the remaining archs.
//
// To remove after the 5409 fix: delete this file, drop its includes, and
// remove the REJECT_IF_WORKAROUND_ISSUE_5409 call sites. The test-side
// 6979 helpers stay until that issue is independently resolved.
// `git grep WORKAROUND_ISSUE_5409` finds them all.
// ----------------------------------------------------------------------------

#include "MiopenUtils.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <exception>

#define REJECT_IF_WORKAROUND_ISSUE_5409(handle)                                                \
    do                                                                                         \
    {                                                                                          \
        try                                                                                    \
        {                                                                                      \
            if(::miopen_plugin::miopen_utils::getDeviceArch((handle).getStream()) == "gfx90a") \
            {                                                                                  \
                HIPDNN_PLUGIN_LOG_INFO("Plan builder disabled on gfx90a (issue 5409)");        \
                return false;                                                                  \
            }                                                                                  \
        }                                                                                      \
        catch(const std::exception& issue_5409_e)                                              \
        {                                                                                      \
            HIPDNN_PLUGIN_LOG_INFO(                                                            \
                "Arch query failed; treating as not-applicable: " << issue_5409_e.what());     \
            return false;                                                                      \
        }                                                                                      \
    } while(0)
