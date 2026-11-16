// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"

/**
 * @file ConvEpilogueTestGraphs.hpp
 * @brief Forward convolution graphs with a fused pointwise chain, field for field, so a refusal
 *        case changes exactly one field; and the chain's reference semantics on the host.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_epilogue_test
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive_test::ConvGraphSpec;
using conv_naive_test::ConvShape;
using conv_naive_test::TensorSpec;
using data_objects::DataType;
using data_objects::PointwiseMode;

/// One tensor beyond the convolution's x, w and y.
struct TensorEntry
{
    int64_t uid = 0;
    TensorSpec tensor;
    /// A compile-time constant (TensorAttributes.value), in the tensor's own data type.
    std::optional<float> value;
};

/// One node after the convolution, field for field.
struct NodeSpec
{
    /// false: a node with no attributes, which no epilogue serves.
    bool pointwise = true;
    PointwiseMode mode = PointwiseMode::RELU_FWD;
    int64_t in0 = 0;
    std::optional<int64_t> in1;
    std::optional<int64_t> in2;
    std::optional<int64_t> axis;
    int64_t out = 0;
    std::optional<float> lowerClip;
    std::optional<float> upperClip;
    std::optional<float> slope;
    std::optional<float> swishBeta;
    std::optional<float> eluAlpha;
    std::optional<float> softplusBeta;
    DataType compute = DataType::FLOAT;
};

struct FusedGraphSpec
{
    /// The convolution; its y is the chain's virtual head.
    ConvGraphSpec conv;
    std::vector<TensorEntry> tensors;
    std::vector<NodeSpec> nodes;
};

/// How a binary step's other input is supplied.
enum class OperandKind
{
    NONE,
    PER_CHANNEL, ///< device [1, K, 1, ...]
    DEVICE_SCALAR, ///< device [1, 1, 1, ...]
    CONSTANT, ///< a compile-time constant scalar
    RUNTIME, ///< a runtime pass-by-value scalar the caller supplies at execute
};

/// A step of a chain to build, in the convenient form.
struct Step
{
    PointwiseMode mode = PointwiseMode::RELU_FWD;
    OperandKind operand = OperandKind::NONE;
    DataType operandType = DataType::FLOAT;
    /// The running value enters as in_1 (the operand as in_0).
    bool operandFirst = false;
    /// CONSTANT and RUNTIME: the value.
    float value = 0.0F;
    std::optional<float> lowerClip;
    std::optional<float> upperClip;
    std::optional<float> slope;
    std::optional<float> swishBeta;
    std::optional<float> eluAlpha;
    std::optional<float> softplusBeta;
    DataType compute = DataType::FLOAT;
};

inline Step bias(DataType type = DataType::FLOAT)
{
    Step step;
    step.mode = PointwiseMode::ADD;
    step.operand = OperandKind::PER_CHANNEL;
    step.operandType = type;
    return step;
}

inline Step scale(DataType type = DataType::FLOAT)
{
    Step step;
    step.mode = PointwiseMode::MUL;
    step.operand = OperandKind::PER_CHANNEL;
    step.operandType = type;
    return step;
}

inline Step unary(PointwiseMode mode)
{
    Step step;
    step.mode = mode;
    return step;
}

inline Step relu(std::optional<float> lower = std::nullopt,
                 std::optional<float> upper = std::nullopt,
                 std::optional<float> slope = std::nullopt)
{
    Step step;
    step.mode = PointwiseMode::RELU_FWD;
    step.lowerClip = lower;
    step.upperClip = upper;
    step.slope = slope;
    return step;
}

/// Uids: the conv's y is conv_naive_test::Y_UID; step i writes OUT_UID + i, reads OPERAND_UID + i.
inline constexpr int64_t OUT_UID = 100;
inline constexpr int64_t OPERAND_UID = 200;

/// The uid the chain of @p steps writes.
inline int64_t outputUid(size_t steps)
{
    return OUT_UID + static_cast<int64_t>(steps) - 1;
}

/// A consistent fused graph: the convolution of @p shape (forward), its y virtual of
/// @p intermediate type, then @p steps; every intermediate virtual, the last output real.
inline FusedGraphSpec fusedSpecFor(const ConvShape& shape,
                                   const std::vector<Step>& steps,
                                   DataType intermediate = DataType::FLOAT)
{
    FusedGraphSpec spec;
    spec.conv = conv_naive_test::specFor(shape, conv_naive::ConvDirection::FORWARD);
    const TensorSpec output = spec.conv.y;
    spec.conv.y.isVirtual = true;
    spec.conv.y.dataType = intermediate;

    int64_t current = spec.conv.yUid;
    for(size_t i = 0; i < steps.size(); ++i)
    {
        const auto& step = steps[i];
        NodeSpec node;
        node.mode = step.mode;
        node.lowerClip = step.lowerClip;
        node.upperClip = step.upperClip;
        node.slope = step.slope;
        node.swishBeta = step.swishBeta;
        node.eluAlpha = step.eluAlpha;
        node.softplusBeta = step.softplusBeta;
        node.compute = step.compute;
        node.in0 = current;
        if(step.operand != OperandKind::NONE)
        {
            const int64_t operandUid = OPERAND_UID + static_cast<int64_t>(i);
            TensorEntry entry;
            entry.uid = operandUid;
            entry.tensor.dataType = step.operandType;
            entry.tensor.dims.assign(output.dims.size(), 1);
            if(step.operand == OperandKind::PER_CHANNEL)
            {
                entry.tensor.dims[1] = shape.k;
            }
            entry.tensor.strides
                = conv_naive_test::packedStrides(entry.tensor.dims, shape.channelsLast);
            if(step.operand == OperandKind::CONSTANT)
            {
                entry.value = step.value;
            }
            if(step.operand == OperandKind::RUNTIME)
            {
                entry.tensor.runtimePassByValue = true;
            }
            spec.tensors.push_back(entry);
            node.in0 = step.operandFirst ? operandUid : current;
            node.in1 = step.operandFirst ? current : operandUid;
        }
        node.out = OUT_UID + static_cast<int64_t>(i);
        TensorEntry out;
        out.uid = node.out;
        out.tensor = output;
        if(i + 1 < steps.size())
        {
            out.tensor.isVirtual = true;
            out.tensor.dataType = intermediate;
        }
        spec.tensors.push_back(out);
        spec.nodes.push_back(node);
        current = node.out;
    }
    return spec;
}

/// The entry of @p uid in @p spec, for a refusal case to change one field.
inline TensorEntry& entryOf(FusedGraphSpec& spec, int64_t uid)
{
    for(auto& entry : spec.tensors)
    {
        if(entry.uid == uid)
        {
            return entry;
        }
    }
    throw std::out_of_range("no tensor " + std::to_string(uid));
}

inline ::flatbuffers::Offset<data_objects::TensorAttributes>
    serializeEntry(flatbuffers::FlatBufferBuilder& builder, const TensorEntry& entry)
{
    const auto& t = entry.tensor;
    auto valueType = data_objects::TensorValue::NONE;
    ::flatbuffers::Offset<void> value = 0;
    if(entry.value.has_value())
    {
        switch(t.dataType)
        {
        case DataType::HALF:
            valueType = data_objects::TensorValue::Float16Value;
            value = builder.CreateStruct(data_objects::Float16Value(*entry.value)).Union();
            break;
        case DataType::BFLOAT16:
            valueType = data_objects::TensorValue::BFloat16Value;
            value = builder.CreateStruct(data_objects::BFloat16Value(*entry.value)).Union();
            break;
        case DataType::DOUBLE:
            valueType = data_objects::TensorValue::Float64Value;
            value = builder
                        .CreateStruct(data_objects::Float64Value(static_cast<double>(*entry.value)))
                        .Union();
            break;
        case DataType::INT32:
            valueType = data_objects::TensorValue::Int32Value;
            value = builder.CreateStruct(data_objects::Int32Value(static_cast<int>(*entry.value)))
                        .Union();
            break;
        default:
            valueType = data_objects::TensorValue::Float32Value;
            value = builder.CreateStruct(data_objects::Float32Value(*entry.value)).Union();
            break;
        }
    }
    return data_objects::CreateTensorAttributesDirect(builder,
                                                      entry.uid,
                                                      nullptr,
                                                      t.dataType,
                                                      &t.strides,
                                                      &t.dims,
                                                      t.isVirtual,
                                                      valueType,
                                                      value,
                                                      t.runtimePassByValue,
                                                      t.raggedOffsetUid,
                                                      t.alignment,
                                                      t.raggedOffsetMultiplier);
}

template <typename T>
::flatbuffers::Optional<T> fbOptional(const std::optional<T>& value)
{
    return value.has_value() ? ::flatbuffers::Optional<T>(value.value_or(T{}))
                             : ::flatbuffers::nullopt;
}

inline flatbuffers::FlatBufferBuilder buildFusedGraph(const FusedGraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto& conv = spec.conv;
    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(conv_naive_test::serializeTensor(builder, conv.xUid, conv.x));
    tensors.push_back(conv_naive_test::serializeTensor(builder, conv.wUid, conv.w));
    tensors.push_back(conv_naive_test::serializeTensor(builder, conv.yUid, conv.y));
    for(const auto& entry : spec.tensors)
    {
        tensors.push_back(serializeEntry(builder, entry));
    }

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    // The convolution node of the spec's direction: a backward node followed by pointwise nodes
    // is a graph every engine declines (fusion is forward only).
    auto convType = data_objects::NodeAttributes::ConvolutionFwdAttributes;
    ::flatbuffers::Offset<void> convAttributes;
    switch(conv.direction)
    {
    case conv_naive::ConvDirection::BACKWARD_DATA:
        convType = data_objects::NodeAttributes::ConvolutionBwdAttributes;
        convAttributes = data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                            conv.yUid,
                                                                            conv.wUid,
                                                                            conv.xUid,
                                                                            &conv.prePadding,
                                                                            &conv.postPadding,
                                                                            &conv.stride,
                                                                            &conv.dilation,
                                                                            conv.mode)
                             .Union();
        break;
    case conv_naive::ConvDirection::BACKWARD_WEIGHTS:
        convType = data_objects::NodeAttributes::ConvolutionWrwAttributes;
        convAttributes = data_objects::CreateConvolutionWrwAttributesDirect(builder,
                                                                            conv.xUid,
                                                                            conv.yUid,
                                                                            conv.wUid,
                                                                            &conv.prePadding,
                                                                            &conv.postPadding,
                                                                            &conv.stride,
                                                                            &conv.dilation,
                                                                            conv.mode)
                             .Union();
        break;
    default:
        convAttributes = data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                            conv.xUid,
                                                                            conv.wUid,
                                                                            conv.yUid,
                                                                            &conv.prePadding,
                                                                            &conv.postPadding,
                                                                            &conv.stride,
                                                                            &conv.dilation,
                                                                            conv.mode)
                             .Union();
        break;
    }
    nodes.push_back(data_objects::CreateNodeDirect(
        builder, "conv", conv.computeType, convType, convAttributes));
    for(const auto& node : spec.nodes)
    {
        if(!node.pointwise)
        {
            nodes.push_back(data_objects::CreateNodeDirect(
                builder, "other", node.compute, data_objects::NodeAttributes::NONE, 0));
            continue;
        }
        const auto attributes
            = data_objects::CreatePointwiseAttributes(builder,
                                                      node.mode,
                                                      fbOptional(node.lowerClip),
                                                      fbOptional(node.upperClip),
                                                      fbOptional(node.slope),
                                                      fbOptional(node.axis),
                                                      node.in0,
                                                      fbOptional(node.in1),
                                                      fbOptional(node.in2),
                                                      node.out,
                                                      fbOptional(node.swishBeta),
                                                      fbOptional(node.eluAlpha),
                                                      fbOptional(node.softplusBeta))
                  .Union();
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "pointwise",
                                           node.compute,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           attributes));
    }

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "conv_epilogue_test",
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   data_objects::DataType::UNSET,
                                                   &tensors,
                                                   &nodes,
                                                   ::flatbuffers::nullopt,
                                                   conv.overrideShapeEnabled));
    return builder;
}

/// @p value widened, or @p fallback when unset.
inline double valueOr(const std::optional<float>& value, double fallback)
{
    return value.has_value() ? static_cast<double>(*value) : fallback;
}

/// The value a step's other input contributes for channel @p k: the operand data for a tensor,
/// the step's value for a scalar.
struct StepInput
{
    std::vector<double> perChannel;
    double scalar = 0.0;
};

/// One step on the host, in double, with the epilogue's NaN and infinity rules (PyTorch's):
/// relu propagates NaN; a gate that saturates to 0 (silu, gelu at -inf) gives 0.
inline double applyStep(const Step& step, double v, double input)
{
    switch(step.mode)
    {
    case PointwiseMode::ADD:
        return v + input;
    case PointwiseMode::MUL:
        return v * input;
    case PointwiseMode::RELU_FWD:
    {
        const double lower = valueOr(step.lowerClip, 0.0);
        const double upper = valueOr(step.upperClip, std::numeric_limits<double>::infinity());
        const double slope = valueOr(step.slope, 0.0);
        if(v <= lower)
        {
            return slope == 0.0 ? lower : lower + slope * (v - lower);
        }
        return v >= upper ? upper : v;
    }
    case PointwiseMode::GELU_FWD:
    {
        const double gate = 0.5 * std::erfc(-v / std::sqrt(2.0));
        return gate == 0.0 ? 0.0 : v * gate;
    }
    case PointwiseMode::GELU_APPROX_TANH_FWD:
    {
        const double gate
            = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (v + 0.044715 * v * v * v)));
        return gate == 0.0 ? 0.0 : v * gate;
    }
    case PointwiseMode::SWISH_FWD:
    {
        const double gate = 1.0 / (1.0 + std::exp(-valueOr(step.swishBeta, 1.0) * v));
        return gate == 0.0 ? 0.0 : v * gate;
    }
    case PointwiseMode::SIGMOID_FWD:
        return 1.0 / (1.0 + std::exp(-v));
    case PointwiseMode::TANH_FWD:
        return std::tanh(v);
    case PointwiseMode::ELU_FWD:
        return v > 0.0 ? v : valueOr(step.eluAlpha, 1.0) * std::expm1(v);
    case PointwiseMode::ABS:
        return std::fabs(v);
    case PointwiseMode::SOFTPLUS_FWD:
    {
        const double beta = valueOr(step.softplusBeta, 1.0);
        return beta * v > 20.0 ? v : std::log1p(std::exp(beta * v)) / beta;
    }
    default:
        break;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/// The largest factor a step can scale an error in its input by (its slope's bound).
inline double stepGain(const Step& step, double inputMagnitude)
{
    switch(step.mode)
    {
    case PointwiseMode::MUL:
        return std::max(1.0, inputMagnitude);
    case PointwiseMode::RELU_FWD:
        return std::max(1.0, std::fabs(valueOr(step.slope, 0.0)));
    case PointwiseMode::GELU_FWD:
    case PointwiseMode::GELU_APPROX_TANH_FWD:
        return 1.2;
    case PointwiseMode::SWISH_FWD:
        return 1.2 * std::max(1.0, std::fabs(valueOr(step.swishBeta, 1.0)));
    case PointwiseMode::ELU_FWD:
        return std::max(1.0, std::fabs(valueOr(step.eluAlpha, 1.0)));
    default:
        return 1.0;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_epilogue_test

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
