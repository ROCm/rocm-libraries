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
// kernels only for transA=T, transB=N, and does not use rocRoller there. The
// provider reaches that layout only when both of our operands are K-contiguous
// (A row-major, B column-major); it then swizzles the scales for the gfx1250
// kernels on device. The canonical test graph (A column-major, B row-major)
// reaches hipBLASLt as transA=N, transB=T, so isApplicable() declines it on
// gfx1250. There is no production counterpart in `Workarounds.hpp`; the provider
// declines non-TN layouts there itself.
//
// SKIP_IF_WORKAROUND_ISSUE_13715() skips the positive MX tests that use the
// canonical layout. TestGpuHipblasltMxMatmulPlanBuilder.IsApplicableRejectsOnGfx1250
// pins the decline, and IsApplicableAcceptsOnlyKContiguousOnGfx1250 covers the
// supported layout. If hipBLASLt ever ships non-TN gfx1250 MX kernels, check
// their scale layout before relaxing the provider's TN-only rule.
//
// To remove after the fix: `git grep WORKAROUND_ISSUE_13715` finds the macro, its
// call sites (including the integration test) and the pinning test.
// ----------------------------------------------------------------------------

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>

#include <gtest/gtest.h>

#include "engines/plans/MxArchSupport.hpp"

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

#define SKIP_IF_WORKAROUND_ISSUE_13715()                                              \
    do                                                                                \
    {                                                                                 \
        if(::hipblaslt_plugin_test::isGfx1250Target(                                  \
               ::hipdnn_test_sdk::utilities::currentDeviceArch()))                    \
        {                                                                             \
            GTEST_SKIP() << "[#13715] hipBLASLt has no MX GEMM kernel for transA=N, " \
                            "transB=T on gfx1250";                                    \
        }                                                                             \
    } while(0)
