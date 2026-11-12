// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "common/ConvProblem.hpp"

namespace hipblaslt_plugin::test
{

namespace conv_data_objects = hipdnn_flatbuffers_sdk::data_objects;

inline constexpr int64_t CONV_X_UID = 1;
inline constexpr int64_t CONV_W_UID = 2;
inline constexpr int64_t CONV_Y_UID = 3;

/// A convolution problem, from which a consistent graph is built.
struct ConvGemmShape
{
    int64_t n = 2;
    int64_t c = 4;
    int64_t k = 6;
    int64_t groups = 1;
    std::vector<int64_t> input = {5, 7};
    std::vector<int64_t> filter = {1, 1};
    std::vector<int64_t> prePadding = {0, 0};
    std::vector<int64_t> postPadding = {0, 0};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    bool xChannelsLast = false;
    /// The filter's own layout: KCRS (false) or KRSC (true).
    bool wChannelsLast = false;
    bool yChannelsLast = false;
    /// Unused elements between consecutive images of x and y, so their images do not nest
    /// into one stride with the rest of the tensor.
    int64_t imagePadding = 0;
    conv_data_objects::DataType dataType = conv_data_objects::DataType::FLOAT;
    conv_data_objects::ConvMode mode = conv_data_objects::ConvMode::CROSS_CORRELATION;
    conv_data_objects::DataType computeType = conv_data_objects::DataType::FLOAT;

    std::vector<int64_t> output() const
    {
        std::vector<int64_t> extents;
        for(size_t axis = 0; axis < input.size(); ++axis)
        {
            const int64_t span = input[axis] + prePadding[axis] + postPadding[axis]
                                 - dilation[axis] * (filter[axis] - 1) - 1;
            extents.push_back(span / stride[axis] + 1);
        }
        return extents;
    }

    std::vector<int64_t> xDims() const
    {
        std::vector<int64_t> dims{n, c};
        dims.insert(dims.end(), input.begin(), input.end());
        return dims;
    }

    std::vector<int64_t> wDims() const
    {
        std::vector<int64_t> dims{k, c / groups};
        dims.insert(dims.end(), filter.begin(), filter.end());
        return dims;
    }

    std::vector<int64_t> yDims() const
    {
        std::vector<int64_t> dims{n, k};
        const auto out = output();
        dims.insert(dims.end(), out.begin(), out.end());
        return dims;
    }
};

/// Packed strides for (n or k, c, spatial...) @p dims, channels second or last.
inline std::vector<int64_t> convPackedStrides(const std::vector<int64_t>& dims, bool channelsLast)
{
    std::vector<size_t> order{0};
    if(!channelsLast)
    {
        order.push_back(1);
    }
    for(size_t axis = 2; axis < dims.size(); ++axis)
    {
        order.push_back(axis);
    }
    if(channelsLast)
    {
        order.push_back(1);
    }
    std::vector<int64_t> strides(dims.size(), 0);
    int64_t running = 1;
    for(size_t i = order.size(); i-- > 0;)
    {
        strides[order[i]] = running;
        running *= dims[order[i]];
    }
    return strides;
}

inline flatbuffers::FlatBufferBuilder buildConvGemmGraph(const ConvGemmShape& shape,
                                                         ConvDirection direction)
{
    const auto xDims = shape.xDims();
    const auto wDims = shape.wDims();
    const auto yDims = shape.yDims();
    auto xStrides = convPackedStrides(xDims, shape.xChannelsLast);
    const auto wStrides = convPackedStrides(wDims, shape.wChannelsLast);
    auto yStrides = convPackedStrides(yDims, shape.yChannelsLast);
    xStrides[0] += shape.imagePadding;
    yStrides[0] += shape.imagePadding;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<conv_data_objects::TensorAttributes>> tensors{
        conv_data_objects::CreateTensorAttributesDirect(
            builder, CONV_X_UID, nullptr, shape.dataType, &xStrides, &xDims),
        conv_data_objects::CreateTensorAttributesDirect(
            builder, CONV_W_UID, nullptr, shape.dataType, &wStrides, &wDims),
        conv_data_objects::CreateTensorAttributesDirect(
            builder, CONV_Y_UID, nullptr, shape.dataType, &yStrides, &yDims)};

    auto type = conv_data_objects::NodeAttributes::NONE;
    flatbuffers::Offset<void> attributes;
    switch(direction)
    {
    case ConvDirection::FORWARD:
        type = conv_data_objects::NodeAttributes::ConvolutionFwdAttributes;
        attributes = conv_data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                             CONV_X_UID,
                                                                             CONV_W_UID,
                                                                             CONV_Y_UID,
                                                                             &shape.prePadding,
                                                                             &shape.postPadding,
                                                                             &shape.stride,
                                                                             &shape.dilation,
                                                                             shape.mode)
                         .Union();
        break;
    case ConvDirection::BACKWARD_DATA:
        type = conv_data_objects::NodeAttributes::ConvolutionBwdAttributes;
        attributes = conv_data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                             CONV_Y_UID,
                                                                             CONV_W_UID,
                                                                             CONV_X_UID,
                                                                             &shape.prePadding,
                                                                             &shape.postPadding,
                                                                             &shape.stride,
                                                                             &shape.dilation,
                                                                             shape.mode)
                         .Union();
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        type = conv_data_objects::NodeAttributes::ConvolutionWrwAttributes;
        attributes = conv_data_objects::CreateConvolutionWrwAttributesDirect(builder,
                                                                             CONV_X_UID,
                                                                             CONV_Y_UID,
                                                                             CONV_W_UID,
                                                                             &shape.prePadding,
                                                                             &shape.postPadding,
                                                                             &shape.stride,
                                                                             &shape.dilation,
                                                                             shape.mode)
                         .Union();
        break;
    default:
        break;
    }

    std::vector<flatbuffers::Offset<conv_data_objects::Node>> nodes{
        conv_data_objects::CreateNodeDirect(builder, "conv", shape.computeType, type, attributes)};
    builder.Finish(conv_data_objects::CreateGraphDirect(builder,
                                                        "conv_gemm_test",
                                                        conv_data_objects::DataType::UNSET,
                                                        conv_data_objects::DataType::UNSET,
                                                        conv_data_objects::DataType::UNSET,
                                                        &tensors,
                                                        &nodes));
    return builder;
}

/// Owns a built graph and reads it the way the provider does.
class ConvGemmGraph
{
public:
    ConvGemmGraph(const ConvGemmShape& shape, ConvDirection direction)
        : _builder(buildConvGemmGraph(shape, direction))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph() const
    {
        return _graph;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
};

} // namespace hipblaslt_plugin::test
