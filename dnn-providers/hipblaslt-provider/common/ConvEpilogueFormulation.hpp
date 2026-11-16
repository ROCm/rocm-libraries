// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include "common/ConvGemmFormulation.hpp"

namespace hipblaslt_plugin
{

// hipblasLtEpilogue_t values (hipblaslt.h), restated so the backend-neutral formulation builds
// without the hipBLASLt headers; HipblasltGemmPlan.cpp checks them against the library's.
inline constexpr int NATIVE_EPILOGUE_RELU = 2;
inline constexpr int NATIVE_EPILOGUE_BIAS = 4;
inline constexpr int NATIVE_EPILOGUE_RELU_BIAS = 6;
inline constexpr int NATIVE_EPILOGUE_GELU = 32;
inline constexpr int NATIVE_EPILOGUE_GELU_BIAS = 36;
inline constexpr int NATIVE_EPILOGUE_SIGMOID = 1024;
inline constexpr int NATIVE_EPILOGUE_SWISH = 65536;
inline constexpr int NATIVE_EPILOGUE_SWISH_BIAS = 65540;
inline constexpr int NATIVE_EPILOGUE_CLAMP = 131072;
inline constexpr int NATIVE_EPILOGUE_CLAMP_BIAS = 131076;

/**
 * @brief The hipBLASLt epilogue that computes @p problem's pointwise chain exactly as the
 *        chain's own rules do, special values included, or nullopt when it has none: a
 *        per-channel bias of fp32 or y's type alone, or sigmoid alone. hipBLASLt's relu, clamp,
 *        swish and GELU epilogues differ from the chain on NaN or infinite inputs, so those
 *        patterns run in a pass.
 */
std::optional<NativeEpilogue> nativeEpilogueFor(const ConvProblem& problem);

/**
 * @brief @p plan with @p problem's epilogue fused in, or nullopt when the plan's writes of y
 *        take none; @p plan unchanged for a problem without an epilogue.
 *
 * Every form computes the convolution and the whole chain in fp32 and rounds once into y:
 * - NATIVE (with @p native, for hipBLASLt): every GEMM writing y has the output channels as its
 *   rows, one group, no accumulation, and the chain maps to a library epilogue
 *   (nativeEpilogueFor);
 * - FUSED_PASS: y is written by the staged channels-last copy; the GEMMs write the staged
 *   result in fp32 and the copy becomes a cast that applies the epilogue;
 * - EXTRA_PASS: otherwise. Every GEMM (and any partials sum) writing y writes an fp32 copy of y
 *   in the workspace instead, laid out like y, and a final cast applies the epilogue into y.
 */
std::optional<ConvGemmPlanCalls>
    fuseEpilogue(const ConvProblem& problem, ConvGemmPlanCalls plan, bool native);

} // namespace hipblaslt_plugin
