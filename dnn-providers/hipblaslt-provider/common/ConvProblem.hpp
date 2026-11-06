// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipblaslt_plugin
{

enum class ConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHTS,
};

/// One convolution operand: its uid and its (n or k, c, spatial...) dims and strides.
struct ConvOperand
{
    int64_t uid = 0;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

/**
 * @brief A single-node convolution graph, validated and read into forward roles: x is dx for
 *        backward data, w is dw for backward weights, y is dy for both backward directions.
 *
 * The explicit-GEMM engines' copy of the hip-kernel-provider naive engines' acceptance rules
 * (packs/ConvNaiveProblem.cpp there): the providers share no code yet, so the two are kept to
 * the same field-by-field checks by hand.
 */
struct ConvProblem
{
    ConvDirection direction = ConvDirection::FORWARD;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::ConvMode mode
        = hipdnn_flatbuffers_sdk::data_objects::ConvMode::UNSET;
    ConvOperand x;
    ConvOperand w;
    ConvOperand y;

    int64_t n = 0;
    int64_t c = 0;
    int64_t k = 0;
    int64_t groups = 1;
    std::vector<int64_t> input; // x spatial extents
    std::vector<int64_t> output; // y spatial extents
    std::vector<int64_t> filter; // w spatial extents
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;

    int64_t cPerGroup() const
    {
        return c / groups;
    }

    int64_t kPerGroup() const
    {
        return k / groups;
    }
};

/**
 * @brief The convolution a single-node graph of @p direction describes, or nullopt when the
 *        graph is not a well-formed convolution these engines read.
 *
 * Gates every field of the graph, the node and the operands: one node, no override shapes, an
 * fp32-or-narrower compute type, a set convolution mode, three distinct real device tensors of
 * one element type among FLOAT, HALF and BFLOAT16, rank 3 to 5, positive extents, non-negative
 * strides, matching channel and group counts, and y exactly the extent the parameters produce.
 */
std::optional<ConvProblem>
    parseConvProblem(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                     ConvDirection direction);

} // namespace hipblaslt_plugin
