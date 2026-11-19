// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <utility>

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

/// What a backend's GEMMs apply of a fused epilogue themselves.
struct GemmEpilogueSupport
{
    /// hipBLASLt's epilogues: a per-channel bias along the GEMM's rows, and the activations
    /// that give the chain's result for every input, special values included (sigmoid for a
    /// 16-bit y). Without it (rocBLAS) only the bias as an addend (GemmCall::c).
    bool libraryEpilogue = false;
    /// Also hipBLASLt's relu, clamp, swish (beta 1) and tanh GELU, and its sigmoid for an fp32
    /// y. Their special values differ from the chain's -- relu maps NaN to 0, clamp NaN to its
    /// upper bound, swish and tanh GELU -inf to NaN -- and the sigmoid is not the few-ulp form
    /// an fp32 y takes. Finite inputs stay within the fused-epilogue tolerance. Only the opt-in
    /// HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE.
    bool libraryActivations = false;
};

/**
 * @brief The part of a fused epilogue the GEMMs that write y apply, and the in-place tail after
 *        them.
 *
 * Every form computes the convolution and the chain up to the tail in fp32 and rounds once into
 * y. The tail, a relu or clamp whose bounds y's type holds exactly, clamps the rounded result in
 * place, which gives the same values as clamping before the rounding (NaN kept).
 */
struct GemmEpilogue
{
    /// The leading per-channel bias: a hipBLASLt bias along the rows, or an addend (GemmCall::c)
    /// broadcast along the rows where the channels are the columns (y's type only).
    std::optional<hipdnn_plugin_sdk::conv_epilogue::Operand> bias;
    /// The hipBLASLt activation epilogue without its bias bit (NATIVE_EPILOGUE_RELU, ...), or 0
    /// for none.
    int activation = 0;
    float act0 = 0.0F;
    float act1 = 0.0F;
    /// The tail's (lower, upper) bounds.
    std::optional<std::pair<float, float>> clamp;
};

/**
 * @brief The GEMM form of @p problem's pointwise chain under @p support, or nullopt when it has
 *        none: an optional per-channel bias (fp32 or y's type), then at most one of a native
 *        activation (support) or a relu / clamp with bounds exact in y's type (the tail).
 */
std::optional<GemmEpilogue> gemmEpilogueFor(const ConvProblem& problem,
                                            const GemmEpilogueSupport& support);

/**
 * @brief Whether @p problem's chain maps to a library activation (GemmEpilogueSupport::
 *        libraryActivations) that the exact form does not use: the graphs the opt-in sibling
 *        engine serves.
 */
bool usesLibraryActivation(const ConvProblem& problem);

/**
 * @brief @p plan with @p problem's epilogue fused in, or nullopt when the plan's writes of y
 *        take none; @p plan unchanged for a problem without an epilogue.
 *
 * Every form computes the convolution and the whole chain in fp32 and rounds once into y:
 * - NATIVE (with @p gemm): every GEMM writing y applies gemmEpilogueFor's part itself -- a bias
 *   along its rows (hipBLASLt, the output channels its rows) or added from a broadcast bias
 *   matrix in the workspace (the output channels its columns), and the activation -- with one
 *   group, no accumulation, and the call whole; a relu / clamp tail then clamps y in place;
 * - FUSED_PASS: y is written by the staged channels-last copy; the GEMMs write the staged
 *   result in fp32 and the copy becomes a cast that applies the epilogue;
 * - EXTRA_PASS: otherwise. Every GEMM (and any partials sum) writing y writes an fp32 copy of y
 *   in the workspace instead, laid out like y, and a final cast applies the epilogue into y.
 */
std::optional<ConvGemmPlanCalls> fuseEpilogue(const ConvProblem& problem,
                                              ConvGemmPlanCalls plan,
                                              const std::optional<GemmEpilogueSupport>& gemm);

/**
 * @brief @p plan with every GEMM that writes an fp32 result from 16-bit operands (a fused plan's
 *        y copy or staging) cut into @p chunks consecutive reductions, each after the first
 *        adding to the fp32 result (beta 1), in order.
 *
 * The same fp32 sum, rounded once by the epilogue pass. For a backend without a workspace-free
 * algorithm for the long fp32-output GEMM (hipBLASLt's deterministic engines on small-n, long-k
 * layers): the shorter reductions have one. Deterministic: the chunks run one after another.
 */
ConvGemmPlanCalls splitFp32OutputReductions(ConvGemmPlanCalls plan, int64_t chunks);

} // namespace hipblaslt_plugin
