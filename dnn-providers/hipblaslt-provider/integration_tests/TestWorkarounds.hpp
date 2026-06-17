// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Test-side workarounds for known hipBLASLt-provider issues. Each macro is
// keyed by the upstream issue so it is easy to grep and remove once the
// underlying problem is fixed.
//
// ----------------------------------------------------------------------------
// hipBLASLt gfx12 FP16 trans-trans + bias-fused epilogue gap.
// Tracking: https://github.com/ROCm/rocm-libraries/issues/8033
// `hipblasLtMatmulAlgoGetHeuristic` returns 0 algorithms on gfx12 (gfx1200,
// gfx1201, gfx1250, ...) for the cell:
//     I/O dtype = FP16
//   ∧ transA == OP_T AND transB == OP_T
//   ∧ epilogue has the BIAS bit set (BIAS, RELU_BIAS, GELU_BIAS,
//                                    SWISH_BIAS_EXT, CLAMP_BIAS_EXT)
// Every neighboring cell (no bias, BF16/FP32 with bias, FP16 with any other
// transpose pair) returns ≥1 algorithm. The provider's plan builder
// correctly reports the graph as not supported when the heuristic returns
// nothing, so the failure surfaces as ErrorCode::GRAPH_NOT_SUPPORTED from
// `graph.build()` in the integration tests.
//
// IS_HIPBLASLT_GFX12_FP16_TT_BIAS(result) — predicate that returns true when
// the current device's arch string contains "gfx12" AND the frontend Error
// is GRAPH_NOT_SUPPORTED. Wire it into the Fp16 typedef class of any fixture
// whose param vector contains trans-trans matmul cases with a bias-bearing
// epilogue. Do NOT wire it into Fp32 / Bf16 typedef classes — those don't
// hit the failure cell and should still fail loudly on regression.
//
// Preferred call site is the `shouldSkipOnEngineConfigResult` override in
// the Fp16 fixture subclass of `IntegrationGraphVerificationHarness`. The
// override returns `HIPBLASLT_GFX12_FP16_TT_BIAS_SKIP_MSG` (wrapped in
// `std::optional<std::string>`) when the predicate matches and
// `std::nullopt` otherwise; the harness emits `GTEST_SKIP()` with that
// message.
//
// To remove after the fix: delete this header, drop includes and overrides
// in the two Bias / BiasActiv test files, and remove the wiring from the
// harness. `git grep HIPBLASLT_GFX12_FP16_TT_BIAS` finds them all.
// ----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hipdnn_frontend.hpp>

#include <gtest/gtest.h>

#include <string>

// Single source for the skip message — referenced by the override and by the
// predicate tests so wording stays in sync.
#define HIPBLASLT_GFX12_FP16_TT_BIAS_SKIP_MSG                                    \
    "[hipBLASLt gfx12 FP16 T-T + bias] hipblasLtMatmulAlgoGetHeuristic returns " \
    "no solutions for FP16 with transA=T, transB=T and any bias-fused "          \
    "epilogue on gfx12 — see "                                                 \
    "https://github.com/ROCm/rocm-libraries/issues/8033"

namespace hipblaslt_plugin::test_utilities
{

// Returns the raw gcnArchName string for device 0 (e.g.
// "gfx1201:sramecc+:xnack-"). Returns an empty string if the device cannot be
// queried — callers treat empty as "no skip" so tests run unmodified.
//
// Header-only to avoid an extra translation unit; the function is tiny.
inline std::string currentDeviceArchRaw()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return {};
    }
    return {props.gcnArchName};
}

} // namespace hipblaslt_plugin::test_utilities

// Inline function (not a macro) so the argument is evaluated exactly once.
// Name kept in IS_HIPBLASLT_GFX12_* form so `git grep` finds every site.
// NOLINTNEXTLINE(readability-identifier-naming)
inline bool IS_HIPBLASLT_GFX12_FP16_TT_BIAS(const hipdnn_frontend::Error& result)
{
    if(result.get_code() != hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED)
    {
        return false;
    }
    const auto arch = hipblaslt_plugin::test_utilities::currentDeviceArchRaw();
    return arch.find("gfx12") != std::string::npos;
}
