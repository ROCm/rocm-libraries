// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Test-side workarounds for known hipBLASLt-provider issues, keyed by the same
// upstream issue numbers as the production `Workarounds.hpp` so
// `git grep WORKAROUND_ISSUE_<n>` finds every site. Kept separate so production
// TUs never pull in gtest.
//
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#9962 — hipBLASLt crashes building a GEMM plan on the
// gfx115x on Windows (see Workarounds.hpp). The provider declines matmul there;
// SKIP_IF_WORKAROUND_ISSUE_9962(handle) skips lower-level plan tests that build a
// plan directly so they do not exercise the crashing path. Windows-only; a no-op
// elsewhere. Any HIP query failure surfaces as a normal gtest failure.
//
// To remove after the fix: delete this file alongside `Workarounds.hpp`, drop
// includes, and remove the call sites.
// ----------------------------------------------------------------------------

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>

#include <gtest/gtest.h>

#ifdef _WIN32
#define SKIP_IF_WORKAROUND_ISSUE_9962(handle)                                      \
    do                                                                             \
    {                                                                              \
        if(::hipdnn_plugin_sdk::archMatches(                                       \
               ::hipdnn_plugin_sdk::getDeviceArch((handle).getStream()),           \
               "gfx115",                                                           \
               ::hipdnn_plugin_sdk::ArchMatchMode::SUBSTRING))                     \
        {                                                                          \
            GTEST_SKIP() << "[#9962] hipBLASLt GEMM crashes on gfx115x (Windows)"; \
        }                                                                          \
    } while(0)
#else
#define SKIP_IF_WORKAROUND_ISSUE_9962(handle) \
    do                                        \
    {                                         \
    } while(0)
#endif
