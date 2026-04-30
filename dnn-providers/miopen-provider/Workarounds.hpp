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
// a `return`. Pair with SKIP_IF_WORKAROUND_ISSUE_5409() in test TUs.
//
// To remove after the fix: delete this file (and tests/common/TestWorkarounds.
// hpp), drop includes, and remove all macro call sites.
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
