// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file Gfx1250ConvFwdAotTestGraphs.hpp
 * @brief Single-node conv_fwd graphs for hipkernel:Gfx1250ConvFwdAot host tests, built the
 *        way the engine's graph matcher reads them: x, w and y as uids 1, 2 and 3, packed
 *        channels-last by default, on a gfx1250 device. No device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx1250_conv_fwd_aot
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

inline constexpr int64_t X_UID = 1;
inline constexpr int64_t W_UID = 2;
inline constexpr int64_t Y_UID = 3;

/// A 2-D forward convolution as the graph spells it.
struct ConvShape
{
    int64_t n = 2;
    int64_t c = 64;
    int64_t k = 64;
    int64_t h = 14;
    int64_t w = 14;
    int64_t y = 3;
    int64_t x = 3;
    int64_t groups = 1;
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    std::vector<int64_t> prePadding = {1, 1};
    std::vector<int64_t> postPadding = {1, 1};
    data_objects::DataType dataType = data_objects::DataType::HALF;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    std::optional<data_objects::DataType> wDataType;
    std::optional<data_objects::DataType> yDataType;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
    /// Strides packed channels-first (NCHW) instead of channels-last.
    bool channelsFirst = false;
    /// Overrides y's logical dims away from the convolution's output.
    std::optional<std::array<int64_t, 4>> yDims;
    /// Overrides x's strides away from packed.
    std::optional<std::vector<int64_t>> xStrides;
    bool virtualY = false;
    /// The node is repeated this many times.
    unsigned int nodeCount = 1;
    bool overrideShape = false;
    /// The uid the convolution names as its output.
    int64_t yUid = Y_UID;

    int64_t ho() const
    {
        return (h + prePadding[0] + postPadding[0] - dilation[0] * (y - 1) - 1) / stride[0] + 1;
    }
    int64_t wo() const
    {
        return (w + prePadding[1] + postPadding[1] - dilation[1] * (x - 1) - 1) / stride[1] + 1;
    }
};

/// Strides of logical (N, C, H, W) @p dims packed in @p order (axes slowest to fastest).
inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims,
                                          const std::array<size_t, 4>& order)
{
    std::vector<int64_t> strides(4, 1);
    int64_t stride = 1;
    for(size_t i = 4; i-- > 0;)
    {
        strides[order[i]] = stride;
        stride *= dims[order[i]];
    }
    return strides;
}

/// A gfx1250 device, the one the engine serves.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties gfx1250Device()
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = "gfx1250";
    properties.warpSize = 32;
    return properties;
}

inline GraphFixture graphOf(const ConvShape& shape,
                            hipdnn_plugin_sdk::ingestor::DeviceProperties device = gfx1250Device())
{
    const std::array<size_t, 4> order = shape.channelsFirst ? std::array<size_t, 4>{0, 1, 2, 3}
                                                            : std::array<size_t, 4>{0, 2, 3, 1};
    const std::vector<int64_t> xDims = {shape.n, shape.c, shape.h, shape.w};
    const std::vector<int64_t> wDims = {shape.k, shape.c / shape.groups, shape.y, shape.x};
    // Not value_or: ho() divides by the stride, which a declined shape may set to 0.
    const auto yArray = shape.yDims
                            ? *shape.yDims
                            : std::array<int64_t, 4>{shape.n, shape.k, shape.ho(), shape.wo()};
    const std::vector<int64_t> yDims(yArray.begin(), yArray.end());
    const auto xStrides = shape.xStrides.value_or(packedStrides(xDims, order));
    const auto wStrides = packedStrides(wDims, order);
    const auto yStrides = packedStrides(yDims, order);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, X_UID, nullptr, shape.dataType, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, W_UID, nullptr, shape.wDataType.value_or(shape.dataType), &wStrides, &wDims));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder,
                                                   Y_UID,
                                                   nullptr,
                                                   shape.yDataType.value_or(shape.dataType),
                                                   &yStrides,
                                                   &yDims,
                                                   shape.virtualY));
    const auto attributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                               X_UID,
                                                                               W_UID,
                                                                               shape.yUid,
                                                                               &shape.prePadding,
                                                                               &shape.postPadding,
                                                                               &shape.stride,
                                                                               &shape.dilation,
                                                                               shape.mode);
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.reserve(shape.nodeCount);
    for(unsigned int i = 0; i < shape.nodeCount; ++i)
    {
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "conv_fwd",
                                           shape.computeType,
                                           data_objects::NodeAttributes::ConvolutionFwdAttributes,
                                           attributes.Union()));
    }
    const auto name = builder.CreateString("gfx1250_conv_fwd_aot_test");
    const auto tensorsVector = builder.CreateVector(tensors);
    const auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(shape.overrideShape);
    builder.Finish(graphBuilder.Finish());
    return GraphFixture(std::move(builder), std::move(device));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx1250_conv_fwd_aot

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
