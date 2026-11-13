// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

namespace conv_naive_test
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

/// Uids of the operands in forward roles: x (dx), w (dw), y (dy).
inline constexpr int64_t X_UID = 1;
inline constexpr int64_t W_UID = 2;
inline constexpr int64_t Y_UID = 3;

/// One operand, field for field as a TensorAttributes table carries it, so a refusal case
/// changes exactly one of them.
struct TensorSpec
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    data_objects::DataType dataType = data_objects::DataType::FLOAT;
    bool isVirtual = false;
    bool runtimePassByValue = false;
    std::optional<int64_t> raggedOffsetUid;
    int64_t raggedOffsetMultiplier = 1;
    int64_t alignment = 16;
};

/// A single-node convolution graph, field for field.
struct ConvGraphSpec
{
    ConvDirection direction = ConvDirection::FORWARD;
    TensorSpec x;
    TensorSpec w;
    TensorSpec y;
    int64_t xUid = X_UID;
    int64_t wUid = W_UID;
    int64_t yUid = Y_UID;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    bool overrideShapeEnabled = false;
    /// Leaves w out of the graph's tensor list while the node still names it.
    bool omitW = false;
};

/// Packed strides for @p dims (n, c, spatial...) in NCHW-style order, or channels-last.
inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims, bool channelsLast)
{
    std::vector<size_t> order; // slowest to fastest
    order.push_back(0);
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

/// The problem a consistent spec is built from.
struct ConvShape
{
    int64_t n = 2;
    int64_t c = 4;
    int64_t k = 6;
    int64_t groups = 1;
    std::vector<int64_t> input = {7, 6}; // spatial extents of x
    std::vector<int64_t> filter = {3, 2};
    std::vector<int64_t> prePadding = {0, 0};
    std::vector<int64_t> postPadding = {0, 0};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    bool channelsLast = false;
    data_objects::DataType dataType = data_objects::DataType::FLOAT;
    data_objects::ConvMode mode = data_objects::ConvMode::CROSS_CORRELATION;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;

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
};

/// A consistent spec for @p shape: every operand packed, every dtype the shape's.
inline ConvGraphSpec specFor(const ConvShape& shape, ConvDirection direction)
{
    ConvGraphSpec spec;
    spec.direction = direction;

    spec.x.dims = {shape.n, shape.c};
    spec.x.dims.insert(spec.x.dims.end(), shape.input.begin(), shape.input.end());
    spec.w.dims = {shape.k, shape.c / shape.groups};
    spec.w.dims.insert(spec.w.dims.end(), shape.filter.begin(), shape.filter.end());
    const auto output = shape.output();
    spec.y.dims = {shape.n, shape.k};
    spec.y.dims.insert(spec.y.dims.end(), output.begin(), output.end());

    for(auto* tensor : {&spec.x, &spec.w, &spec.y})
    {
        tensor->strides = packedStrides(tensor->dims, shape.channelsLast);
        tensor->dataType = shape.dataType;
    }

    spec.prePadding = shape.prePadding;
    spec.postPadding = shape.postPadding;
    spec.stride = shape.stride;
    spec.dilation = shape.dilation;
    spec.mode = shape.mode;
    spec.computeType = shape.computeType;
    return spec;
}

inline ::flatbuffers::Offset<data_objects::TensorAttributes>
    serializeTensor(flatbuffers::FlatBufferBuilder& builder, int64_t uid, const TensorSpec& tensor)
{
    return data_objects::CreateTensorAttributesDirect(
        builder,
        uid,
        nullptr,
        tensor.dataType,
        &tensor.strides,
        &tensor.dims,
        tensor.isVirtual,
        data_objects::TensorValue::NONE,
        0,
        tensor.runtimePassByValue,
        tensor.raggedOffsetUid.has_value()
            ? ::flatbuffers::Optional<int64_t>(*tensor.raggedOffsetUid)
            : ::flatbuffers::nullopt,
        tensor.alignment,
        tensor.raggedOffsetMultiplier);
}

inline flatbuffers::FlatBufferBuilder buildGraph(const ConvGraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(serializeTensor(builder, spec.xUid, spec.x));
    if(spec.wUid != spec.xUid && !spec.omitW)
    {
        tensors.push_back(serializeTensor(builder, spec.wUid, spec.w));
    }
    if(spec.yUid != spec.xUid && spec.yUid != spec.wUid)
    {
        tensors.push_back(serializeTensor(builder, spec.yUid, spec.y));
    }

    data_objects::NodeAttributes type = data_objects::NodeAttributes::NONE;
    ::flatbuffers::Offset<void> attributes;
    switch(spec.direction)
    {
    case ConvDirection::FORWARD:
        type = data_objects::NodeAttributes::ConvolutionFwdAttributes;
        attributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                        spec.xUid,
                                                                        spec.wUid,
                                                                        spec.yUid,
                                                                        &spec.prePadding,
                                                                        &spec.postPadding,
                                                                        &spec.stride,
                                                                        &spec.dilation,
                                                                        spec.mode)
                         .Union();
        break;
    case ConvDirection::BACKWARD_DATA:
        type = data_objects::NodeAttributes::ConvolutionBwdAttributes;
        attributes = data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                        spec.yUid,
                                                                        spec.wUid,
                                                                        spec.xUid,
                                                                        &spec.prePadding,
                                                                        &spec.postPadding,
                                                                        &spec.stride,
                                                                        &spec.dilation,
                                                                        spec.mode)
                         .Union();
        break;
    case ConvDirection::BACKWARD_WEIGHTS:
        type = data_objects::NodeAttributes::ConvolutionWrwAttributes;
        attributes = data_objects::CreateConvolutionWrwAttributesDirect(builder,
                                                                        spec.xUid,
                                                                        spec.yUid,
                                                                        spec.wUid,
                                                                        &spec.prePadding,
                                                                        &spec.postPadding,
                                                                        &spec.stride,
                                                                        &spec.dilation,
                                                                        spec.mode)
                         .Union();
        break;
    default:
        break;
    }

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder, "conv", spec.computeType, type, attributes));

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "conv_naive_test",
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   &tensors,
                                                   &nodes,
                                                   ::flatbuffers::nullopt,
                                                   spec.overrideShapeEnabled));
    return builder;
}

/// cuDNN's convolution compute types for @p element (cudnn-frontend docs/developer/graph-api.mdx,
/// "Compute Data Type"): FLOAT for every element type, HALF also for HALF; UNSET reads as FLOAT.
/// Restated here rather than calling the provider's convServesComputeType, so a change to
/// either one fails the tests below.
inline bool cudnnServesComputeType(data_objects::DataType compute, data_objects::DataType element)
{
    using data_objects::DataType;
    return compute == DataType::UNSET || compute == DataType::FLOAT
           || (compute == DataType::HALF && element == DataType::HALF);
}

/**
 * @brief For each of @p elements, @p accepts(shape) holds for exactly the compute types cuDNN
 *        serves for that element type.
 *
 * @p shape with a FLOAT compute type must be accepted for every one of @p elements, so each
 * refusal can only come from the compute type.
 */
template <typename Accepts>
void expectCudnnComputeTypes(ConvShape shape,
                             std::initializer_list<data_objects::DataType> elements,
                             Accepts accepts)
{
    using data_objects::DataType;
    for(const auto element : elements)
    {
        shape.dataType = element;
        shape.computeType = DataType::FLOAT;
        ASSERT_TRUE(accepts(shape)) << data_objects::EnumNameDataType(element);
        for(const auto compute : {DataType::UNSET,
                                  DataType::FLOAT,
                                  DataType::HALF,
                                  DataType::BFLOAT16,
                                  DataType::DOUBLE,
                                  DataType::INT32,
                                  DataType::INT8,
                                  DataType::FP8_E4M3})
        {
            shape.computeType = compute;
            EXPECT_EQ(accepts(shape), cudnnServesComputeType(compute, element))
                << data_objects::EnumNameDataType(element) << " elements, "
                << data_objects::EnumNameDataType(compute) << " compute";
        }
    }
}

} // namespace conv_naive_test

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
