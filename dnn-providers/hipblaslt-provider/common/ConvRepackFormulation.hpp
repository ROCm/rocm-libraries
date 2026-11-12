// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <optional>

#include "common/ConvGemmFormulation.hpp"

namespace hipblaslt_plugin
{

/// The most workspace the repack steps of one plan take by default. A column matrix that needs
/// more is built a slice of images at a time; a single image that needs more is a poor fit for
/// explicit GEMM and is declined. To be calibrated from exit-suite data.
inline constexpr size_t MAX_REPACK_WORKSPACE_BYTES = static_cast<size_t>(1) << 30;

/// How a plan that has a choice lays out what it computes through: an IM2COL plan's column
/// matrix, and a 1x1 channels-first backward weights of several images (batched fp32 partials
/// per image, or x and dy staged channels-last for one GEMM).
enum class ColumnLayout
{
    /// The default: columns like y; for the 1x1 backward weights, whichever moves fewer bytes.
    FOLLOW_Y,
    /// Channels-first: columns like a channels-first y; the 1x1 backward weights as batched
    /// partials.
    CHANNELS_FIRST,
    /// Channels-last whatever y's layout; the 1x1 backward weights staged channels-last.
    CHANNELS_LAST,
};

/**
 * @brief The IM2COL plan for @p problem within @p maxWorkspaceBytes of repack workspace, or
 *        nullopt when explicit GEMM is a poor fit for it or cannot read its operands.
 *
 * im2col writes the column matrix (kernels/ConvRepackArgs.h) to the workspace, which turns the
 * convolution into a ONE_BY_ONE one with C * filterSize channels over the output positions;
 * directGemmCalls formulates that. Backward data runs the same GEMM into an fp32 column matrix
 * and col2im sums it into dx. Columns follow the filter's own layout (KCRS or KRSC), so w is
 * read in place; a filter that is neither is declined. When the GEMM cannot read y (dy) as
 * given -- channels-first backward weights over several images, say -- y is staged in the
 * workspace channels-last by a strided copy.
 *
 * When the column matrix of every image does not fit, the images are processed a slice at a
 * time, each slice a stage reusing the same workspace. Backward weights sums its slices with
 * accumulating GEMMs: into w for fp32, otherwise into an fp32 accumulator in the workspace that
 * a final stage rounds into w once. Declined when one image does not fit.
 */
std::optional<ConvGemmPlanCalls> im2colGemmCalls(const ConvProblem& problem,
                                                 size_t maxWorkspaceBytes
                                                 = MAX_REPACK_WORKSPACE_BYTES,
                                                 ColumnLayout layout = ColumnLayout::FOLLOW_Y);

/// The plan for @p problem: a direct row (directGemmCalls) when one applies; for a 1x1
/// channels-first backward weights of several images, batched partials or a channels-last
/// staging (@p layout); IM2COL otherwise; or nullopt when none does.
std::optional<ConvGemmPlanCalls> convGemmCalls(const ConvProblem& problem,
                                               size_t maxWorkspaceBytes
                                               = MAX_REPACK_WORKSPACE_BYTES,
                                               ColumnLayout layout = ColumnLayout::FOLLOW_Y);

/**
 * @brief The reduced-precision-reduction plan for @p problem, or nullopt when it has none: the
 *        plan of the ROCBLAS_CONV_RPR and HIPBLASLT_CONV_RPR engines (ConvReducedPrecision.hpp).
 *
 * A 1x1, unit-stride, unpadded fp16 backward weights of several channels-first images, which no
 * single GEMM reads, as one GEMM per image into dw, each after the first adding to it (beta 1):
 * MIOpen's GemmWrw1x1_stride1. Each image's sum is exact in fp32, but dw is rounded to fp16 once
 * per image (cuDNN's REDUCED_PRECISION_REDUCTION) where convGemmCalls rounds once. No workspace.
 * bf16 has no reduced form: cuDNN serves it with FLOAT compute only.
 */
std::optional<ConvGemmPlanCalls> reducedPrecisionGemmCalls(const ConvProblem& problem);

} // namespace hipblaslt_plugin
