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
// ROCm/rocm-libraries#13715 — on gfx1250, hipBLASLt has MX (VEC32_UE8M0) GEMM
// kernels only for transA=T, transB=N, and does not use rocRoller there. This
// provider always sends transA=N, transB=T: it requires opA=T/opB=N in hipDNN's
// row-major frame, then swaps operands (C^T = B^T * A^T). So the heuristic
// finds no algorithm, and isApplicable() declines every MX graph on gfx1250.
// There is no production counterpart in `Workarounds.hpp`; the heuristic
// already declines.
//
// SKIP_IF_WORKAROUND_ISSUE_13715() skips the positive MX tests that expect a plan.
// TestGpuHipblasltMxMatmulPlanBuilder.IsApplicableRejectsOnGfx1250 pins the
// decline. If that test starts failing, hipBLASLt now offers a kernel for this
// provider's problem. Check the scale layout before enabling MX on gfx1250:
// the gfx1250 TN kernels expect pre-swizzled scales and return wrong results
// with natural-layout ones.
//
// To remove after the fix: `git grep WORKAROUND_ISSUE_13715` finds the macro, its
// call sites (including the integration test) and the pinning test.
// ----------------------------------------------------------------------------

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>

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

#define SKIP_IF_WORKAROUND_ISSUE_13715()                                                       \
    do                                                                                         \
    {                                                                                          \
        if(::hipdnn_plugin_sdk::archMatches(::hipdnn_test_sdk::utilities::currentDeviceArch(), \
                                            "gfx1250",                                         \
                                            ::hipdnn_plugin_sdk::ArchMatchMode::PREFIX))       \
        {                                                                                      \
            GTEST_SKIP() << "[#13715] hipBLASLt has no MX GEMM kernel for this provider's "    \
                            "layout on gfx1250";                                               \
        }                                                                                      \
    } while(0)
