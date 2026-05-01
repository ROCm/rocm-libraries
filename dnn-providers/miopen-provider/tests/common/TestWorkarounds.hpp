// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Test-side workarounds for known MIOpen-provider issues. Each macro is keyed
// by the upstream issue number so it is easy to grep and remove once the
// underlying problem is fixed.
//
// ----------------------------------------------------------------------------
// ROCm/rocm-libraries#6979 — MIOpen Conv-Bias-Activation (CBA) fusion has
// limited solver coverage (gfx90a returns wrong results, gfx1151 / RDNA archs
// have no CK fusion kernels). CBA-specific tests must skip when the engine
// search returns no applicable solution on the current device.
//
// IS_WORKAROUND_ISSUE_6979(result) — predicate that returns true when the
// `hipdnn_frontend::Error` from an engine-config-querying frontend call
// (`Graph::build()`, `get_ranked_engine_ids()`, `create_execution_plan*()`)
// indicates this issue. Use ONLY in CBA-specific tests; pair with
// `ASSERT_EQ(..., OK)` (or equivalent) so other failures still surface.
//
// Preferred call site is the `shouldSkipOnEngineConfigResult` override in a
// CBA-specific subclass of `IntegrationGraphVerificationHarness`. The
// override returns `WORKAROUND_ISSUE_6979_SKIP_MSG` (wrapped in
// `std::optional<std::string>`) when the predicate matches and
// `std::nullopt` otherwise; the harness body emits `GTEST_SKIP()` with that
// message. This keeps the base harness free of workaround logic.
//
// SKIP_IF_WORKAROUND_ISSUE_6979(result) — convenience wrapper that emits
// `GTEST_SKIP()` directly. Use only in test bodies that are NOT routed through
// the harness hook (e.g., `IntegrationGpuDeterministic.cpp`, where the test
// fixture inherits from a different base).
//
// Do NOT use in non-CBA tests — `GRAPH_NOT_SUPPORTED` outside the CBA path
// would indicate a real regression, not a known capability gap.
//
// To remove after the fix: delete this header alongside the production-side
// `Workarounds.hpp`, drop includes, and remove all macro call sites.
// `git grep WORKAROUND_ISSUE_6979` finds them all.
//
// `<hipdnn_frontend.hpp>` is header-only, so this header has no linkage
// dependency on the provider's compiled object files — it remains usable from
// integration_tests, which only loads the plugin via dlopen.
// ----------------------------------------------------------------------------

#include <hipdnn_frontend.hpp>

#include <gtest/gtest.h>

// Single source for the issue-6979 skip message — referenced by both SKIP
// macros so wording stays in sync.
#define WORKAROUND_ISSUE_6979_SKIP_MSG                                               \
    "[#6979] Skipping due to ROCm/rocm-libraries#6979 (no engine has an applicable " \
    "solution for ConvBiasActiv fusion on the current device)"

// Inline function (not a macro) so the argument is evaluated exactly once.
// Name kept in IS_WORKAROUND_ISSUE_* form so `git grep WORKAROUND_ISSUE_6979`
// finds every site.
// NOLINTNEXTLINE(readability-identifier-naming)
inline bool IS_WORKAROUND_ISSUE_6979(const hipdnn_frontend::Error& result)
{
    return result.get_code() == hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED;
}

#define SKIP_IF_WORKAROUND_ISSUE_6979(result)                      \
    do                                                             \
    {                                                              \
        const auto& _skip_result_6979 = (result);                  \
        if(IS_WORKAROUND_ISSUE_6979(_skip_result_6979))            \
        {                                                          \
            GTEST_SKIP() << WORKAROUND_ISSUE_6979_SKIP_MSG << ": " \
                         << _skip_result_6979.get_message();       \
        }                                                          \
    } while(0)

// SKIP_IF_NO_APPLICABLE_CBA_ENGINE(handle, planBuilder, graph) -- for unit
// tests that build a CBA plan directly. Queries `isApplicable()` on the
// caller's actual test graph and emits `GTEST_SKIP()` when no solver is
// available on the current device -- before any state is mutated by
// `buildPlan()` or the plan constructor.
//
// IMPORTANT: this macro assumes the caller passes a known-VALID CBA graph
// (typically built via `createValidConvFwdActivGraph` /
// `createValidConvFwdBiasActivGraph` from FlatbufferGraphTestUtils). An
// invalid graph also reports `isApplicable() == false` and would skip
// silently rather than fail. Validity is the caller's responsibility; the
// macro does not (and intentionally cannot) distinguish "device cannot run
// this CBA" from "this graph is malformed."
//
// In parametric tests where some cases expect non-applicability (e.g.
// "Unsupported layout WHCN"), guard the call with `if(<expected applicable>)`
// so negative cases still run and validate the rejection assertion.
#define SKIP_IF_NO_APPLICABLE_CBA_ENGINE(handle, planBuilder, graph) \
    do                                                               \
    {                                                                \
        if(!(planBuilder).isApplicable((handle), (graph)))           \
        {                                                            \
            GTEST_SKIP() << WORKAROUND_ISSUE_6979_SKIP_MSG;          \
        }                                                            \
    } while(0)
