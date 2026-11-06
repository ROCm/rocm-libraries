// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "common/ConvProblem.hpp"
#include "common/GemmCall.hpp"

namespace hipblaslt_plugin
{

/// The explicit-GEMM optimizations (RFC 0002's catalog rows) this layer can express.
enum class ConvGemmRow
{
    /// 1x1 filter, unit stride, no padding: the convolution is a GEMM over pixels and channels.
    ONE_BY_ONE,
    /// The filter covers the whole input with no padding: one output pixel, a GEMM over images.
    POINT_OUTPUT,
};

std::string_view toString(ConvGemmRow row);

/// The GEMM calls that compute a convolution, run in order, each writing its own part of the
/// output once. No repack and no workspace.
struct ConvGemmPlanCalls
{
    ConvGemmRow row = ConvGemmRow::ONE_BY_ONE;
    std::vector<GemmCall> calls;
};

/**
 * @brief The GEMM calls that compute @p problem directly on its tensors, or nullopt when no
 *        catalog row applies to it without a repack.
 *
 * Rows are tried in catalog order. Each row's predicate is the shape condition alone; whether
 * the operands' strides let a BLAS GEMM address them is decided by gemmForViews, so a shape that
 * matches a row but is laid out in a way it cannot read is declined rather than misread.
 */
std::optional<ConvGemmPlanCalls> directGemmCalls(const ConvProblem& problem);

/// @p problem's shape satisfies the ONE_BY_ONE row: every filter extent 1, every stride 1, no
/// padding (dilation is irrelevant to a 1x1 filter).
bool isOneByOne(const ConvProblem& problem);

/// @p problem's shape satisfies the POINT_OUTPUT row: the filter spans the whole input with no
/// padding and unit dilation, so every output spatial extent is 1.
bool isPointOutput(const ConvProblem& problem);

} // namespace hipblaslt_plugin
