// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The second engine, split from Pointwise by graph node type. One pack, so
/// `operationMatcher` is empty -- the graph matcher both admits the node type and
/// validates shape in one pass.
inline constexpr PackSymbols CONV_FWD{"hipkernel:ConvFwd",
                                      "hipkernel.conv_fwd.graph_match",
                                      "",
                                      "hipkernel.conv_fwd.kernel_match",
                                      "hipkernel.conv_fwd.score",
                                      "hipkernel.conv_fwd.dispatch",
                                      "conv_fwd.x.uid",
                                      "conv_fwd.w.uid",
                                      "conv_fwd.y.uid"};

/// @brief Row-major packed strides for @p dims -- the layout the conv kernel's flat
/// index arithmetic assumes, since it takes no stride arguments of its own.
inline std::vector<int64_t> packedRowMajorStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// Tensor uids buildConvFwdGraph() uses, in kernel argument order.
constexpr int64_t CONV_X_UID = 1;
constexpr int64_t CONV_W_UID = 2;
constexpr int64_t CONV_Y_UID = 3;

/**
 * @brief Builds a single-node conv-forward graph, parameterized on everything this
 *        pack's matcher gates: mode, stride, dilation, padding, and dtype. Defaults to
 *        the one shape the naive kernel can serve (unit stride/dilation, no padding,
 *        cross-correlation, packed NCHW/KCRS/NKPQ, uniform dtype); @p wDims and @p yDims
 *        default from @p xDims (P = H - R + 1, Q = W - S + 1) so a caller overriding
 *        only @p xDims for a refusal case need not keep w/y consistent by hand.
 *
 * @param wDataType Overrides w's dtype away from @p dataType, for the cross-operand
 *        dtype-mismatch refusal.
 * @param xStridesOverride Overrides x's strides away from packed row-major, for the
 *        non-packed-layout refusal; the kernel takes no strides of its own.
 */
inline flatbuffers::FlatBufferBuilder
    buildConvFwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                      = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                      hipdnn_flatbuffers_sdk::data_objects::ConvMode convMode
                      = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION,
                      const std::vector<int64_t>& stride = {1, 1},
                      const std::vector<int64_t>& dilation = {1, 1},
                      const std::vector<int64_t>& prePadding = {0, 0},
                      const std::vector<int64_t>& postPadding = {0, 0},
                      const std::vector<int64_t>& xDims = {1, 1, 3, 3},
                      const std::optional<std::vector<int64_t>>& wDims = std::nullopt,
                      const std::optional<std::vector<int64_t>>& yDims = std::nullopt,
                      std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> wDataType
                      = std::nullopt,
                      const std::optional<std::vector<int64_t>>& xStridesOverride = std::nullopt)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    // std::optional::value_or evaluates its argument unconditionally, so deriving the
    // defaults with value_or indexes xDims even when the caller supplied w/y dims. The
    // rank-3 refusal case passes a 3-element xDims, making xDims[3] an out-of-bounds
    // read. Derive a default only when one is needed.
    const auto resolvedWDims
        = wDims.has_value() ? *wDims : std::vector<int64_t>{1, xDims.at(1), 2, 2};
    const auto resolvedYDims = yDims.has_value()
                                   ? *yDims
                                   : std::vector<int64_t>{xDims.at(0),
                                                          resolvedWDims.at(0),
                                                          xDims.at(2) - resolvedWDims.at(2) + 1,
                                                          xDims.at(3) - resolvedWDims.at(3) + 1};
    const auto resolvedWDataType = wDataType.value_or(dataType);

    const auto xStrides = xStridesOverride.value_or(packedRowMajorStrides(xDims));
    const auto wStrides = packedRowMajorStrides(resolvedWDims);
    const auto yStrides = packedRowMajorStrides(resolvedYDims);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CONV_X_UID, nullptr, dataType, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CONV_W_UID, nullptr, resolvedWDataType, &wStrides, &resolvedWDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, CONV_Y_UID, nullptr, dataType, &yStrides, &resolvedYDims));

    auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                         CONV_X_UID,
                                                                         CONV_W_UID,
                                                                         CONV_Y_UID,
                                                                         &prePadding,
                                                                         &postPadding,
                                                                         &stride,
                                                                         &dilation,
                                                                         convMode);

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "conv_fwd",
                                       dataType,
                                       data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                       attributes.Union()));

    auto name = builder.CreateString("conv_fwd_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());

    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
