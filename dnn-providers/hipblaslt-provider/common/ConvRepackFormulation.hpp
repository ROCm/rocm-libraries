// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <optional>

#include "common/ConvGemmFormulation.hpp"

namespace hipblaslt_plugin
{

/// The most workspace the repack steps of one plan may take. A convolution whose column matrix
/// needs more is a poor fit for explicit GEMM and is declined. Conservative to start with; it is
/// calibrated, or replaced by image chunking, from exit-suite data.
inline constexpr size_t MAX_REPACK_WORKSPACE_BYTES = static_cast<size_t>(1) << 30;

/**
 * @brief The IM2COL plan for @p problem, or nullopt when explicit GEMM is a poor fit for it or
 *        cannot read its operands.
 *
 * im2col writes the column matrix (kernels/ConvRepackArgs.h) to the workspace, which turns the
 * convolution into a ONE_BY_ONE one with C * filterSize channels over the output positions;
 * directGemmCalls formulates that. Backward data runs the same GEMM into an fp32 column matrix
 * and col2im sums it into dx. Columns follow the filter's own layout (KCRS or KRSC), so w is
 * read in place; a filter that is neither is declined. When the GEMM cannot read y (dy) as
 * given -- channels-first backward weights over several images, say -- y is staged in the
 * workspace channels-last by a strided copy. Declined when the workspace would exceed
 * MAX_REPACK_WORKSPACE_BYTES.
 */
std::optional<ConvGemmPlanCalls> im2colGemmCalls(const ConvProblem& problem);

/// The plan for @p problem: a direct row (directGemmCalls) when one applies, IM2COL otherwise,
/// or nullopt when neither does.
std::optional<ConvGemmPlanCalls> convGemmCalls(const ConvProblem& problem);

} // namespace hipblaslt_plugin
