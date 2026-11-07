// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "common/ConvProblem.hpp"
#include "common/GemmCall.hpp"
#include "kernels/ConvRepackArgs.h"

namespace hipblaslt_plugin
{

/// The explicit-GEMM optimizations (RFC 0002's catalog rows) this layer can express.
enum class ConvGemmRow
{
    /// 1x1 filter, unit stride, no padding: the convolution is a GEMM over pixels and channels.
    ONE_BY_ONE,
    /// The filter covers the whole input with no padding: one output pixel, a GEMM over images.
    POINT_OUTPUT,
    /// Anything else: im2col (col2im for backward data) turns the convolution into a ONE_BY_ONE
    /// one over a column matrix in the workspace.
    IM2COL,
};

std::string_view toString(ConvGemmRow row);

/// A region of the plan's workspace that the steps address as a tensor with a uid of its own.
/// Its uid is not one of the graph's, so a step resolves it the way it resolves a graph tensor.
struct WorkspaceTensor
{
    int64_t uid = 0;
    /// Bytes from the start of the workspace.
    size_t offset = 0;
    size_t bytes = 0;
};

/// One launch of a repack kernel (kernels/ConvRepack.cpp). The pointers in `args` are left null
/// and filled at execution from `srcUid` and `dstUid`, advanced by the byte offsets.
struct RepackStep
{
    std::variant<StridedCopyArgs, StridedCastArgs, Im2ColArgs, Col2ImArgs> args;
    /// The element type moved (strided copy, im2col) or written (cast, col2im).
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    int64_t srcUid = 0;
    int64_t dstUid = 0;
    int64_t srcByteOffset = 0;
    int64_t dstByteOffset = 0;
};

/// Repack steps, then GEMM calls in order, then more repack steps.
struct ConvGemmStage
{
    std::vector<RepackStep> before;
    std::vector<GemmCall> calls;
    std::vector<RepackStep> after;
};

/// How a convolution runs: its stages in order, sharing one workspace. The direct rows are one
/// stage of GEMM calls with no steps and no workspace; IM2COL has a stage per slice of images
/// the workspace holds at once, and backward weights a final stage that rounds its fp32 sum.
struct ConvGemmPlanCalls
{
    ConvGemmRow row = ConvGemmRow::ONE_BY_ONE;
    std::vector<ConvGemmStage> stages;
    std::vector<WorkspaceTensor> workspace;
    /// Bytes of workspace the steps and calls address, before any the GEMM backend needs.
    size_t workspaceBytes = 0;
    /// IM2COL: the column matrix is laid out channels-first (positions innermost).
    bool columnsFirst = false;

    bool hasRepack() const
    {
        for(const auto& stage : stages)
        {
            if(!stage.before.empty() || !stage.after.empty())
            {
                return true;
            }
        }
        return stages.size() > 1;
    }
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

/// Extent and innermost stride of @p axes of @p operand read as one index, outermost axis
/// first, or nullopt when they do not nest into a single stride. Axes of extent 1 are free.
std::optional<std::pair<int64_t, int64_t>> collapseAxes(const ConvOperand& operand,
                                                        const std::vector<size_t>& axes);

/// @p problem's shape satisfies the ONE_BY_ONE row: every filter extent 1, every stride 1, no
/// padding (dilation is irrelevant to a 1x1 filter).
bool isOneByOne(const ConvProblem& problem);

/// @p problem's shape satisfies the POINT_OUTPUT row: the filter spans the whole input with no
/// padding and unit dilation, so every output spatial extent is 1.
bool isPointOutput(const ConvProblem& problem);

} // namespace hipblaslt_plugin
