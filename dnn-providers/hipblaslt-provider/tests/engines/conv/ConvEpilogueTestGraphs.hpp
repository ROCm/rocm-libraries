// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "engines/conv/ConvGemmTestGraphs.hpp"

/**
 * @file ConvEpilogueTestGraphs.hpp
 * @brief Forward convolution graphs followed by a fused pointwise chain, and the chain's
 *        reference semantics on the host: the explicit-GEMM engines' copy of the
 *        hip-kernel-provider's test helper (the providers' test trees are separate).
 */
namespace hipblaslt_plugin::test
{

using conv_data_objects::PointwiseMode;

/// How a binary step's other input is supplied.
enum class OperandKind
{
    NONE,
    PER_CHANNEL, ///< device [1, K, 1, ...]
    DEVICE_SCALAR, ///< device [1, 1, 1, ...]
    CONSTANT, ///< a compile-time constant scalar
    RUNTIME, ///< a runtime pass-by-value scalar the caller supplies at execute
};

/// One step of a chain.
struct Step
{
    PointwiseMode mode = PointwiseMode::RELU_FWD;
    OperandKind operand = OperandKind::NONE;
    conv_data_objects::DataType operandType = conv_data_objects::DataType::FLOAT;
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
    conv_data_objects::DataType compute = conv_data_objects::DataType::FLOAT;
};

inline Step bias(conv_data_objects::DataType type = conv_data_objects::DataType::FLOAT)
{
    Step step;
    step.mode = PointwiseMode::ADD;
    step.operand = OperandKind::PER_CHANNEL;
    step.operandType = type;
    return step;
}

inline Step scale(conv_data_objects::DataType type = conv_data_objects::DataType::FLOAT)
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

/// Uids: the conv's y is CONV_Y_UID; step i writes EPI_OUT_UID + i and reads EPI_OPERAND_UID + i.
inline constexpr int64_t EPI_OUT_UID = 100;
inline constexpr int64_t EPI_OPERAND_UID = 200;

inline int64_t epilogueOutputUid(size_t steps)
{
    return EPI_OUT_UID + static_cast<int64_t>(steps) - 1;
}

template <typename T>
::flatbuffers::Optional<T> fbOptional(const std::optional<T>& value)
{
    return value.has_value() ? ::flatbuffers::Optional<T>(value.value_or(T{}))
                             : ::flatbuffers::nullopt;
}

/// The forward convolution of @p shape, its y virtual of @p intermediate type, then @p steps.
inline flatbuffers::FlatBufferBuilder buildFusedConvGraph(const ConvGemmShape& shape,
                                                          const std::vector<Step>& steps,
                                                          conv_data_objects::DataType intermediate
                                                          = conv_data_objects::DataType::FLOAT,
                                                          ConvDirection direction
                                                          = ConvDirection::FORWARD)
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
            builder, CONV_Y_UID, nullptr, intermediate, &yStrides, &yDims, true)};

    std::vector<flatbuffers::Offset<conv_data_objects::Node>> nodes;
    flatbuffers::Offset<void> convAttributes;
    auto convType = conv_data_objects::NodeAttributes::ConvolutionFwdAttributes;
    if(direction == ConvDirection::FORWARD)
    {
        convAttributes = conv_data_objects::CreateConvolutionFwdAttributesDirect(builder,
                                                                                 CONV_X_UID,
                                                                                 CONV_W_UID,
                                                                                 CONV_Y_UID,
                                                                                 &shape.prePadding,
                                                                                 &shape.postPadding,
                                                                                 &shape.stride,
                                                                                 &shape.dilation,
                                                                                 shape.mode)
                             .Union();
    }
    else
    {
        // Backward data with its dx virtual and a chain after it: a pattern no engine fuses.
        convType = conv_data_objects::NodeAttributes::ConvolutionBwdAttributes;
        convAttributes = conv_data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                                 CONV_X_UID,
                                                                                 CONV_W_UID,
                                                                                 CONV_Y_UID,
                                                                                 &shape.prePadding,
                                                                                 &shape.postPadding,
                                                                                 &shape.stride,
                                                                                 &shape.dilation,
                                                                                 shape.mode)
                             .Union();
    }
    nodes.push_back(conv_data_objects::CreateNodeDirect(
        builder, "conv", shape.computeType, convType, convAttributes));

    int64_t current = CONV_Y_UID;
    for(size_t i = 0; i < steps.size(); ++i)
    {
        const auto& step = steps[i];
        std::optional<int64_t> in1;
        int64_t in0 = current;
        if(step.operand != OperandKind::NONE)
        {
            const int64_t uid = EPI_OPERAND_UID + static_cast<int64_t>(i);
            std::vector<int64_t> dims(yDims.size(), 1);
            if(step.operand == OperandKind::PER_CHANNEL)
            {
                dims[1] = shape.k;
            }
            const auto strides = convPackedStrides(dims, shape.yChannelsLast);
            auto valueType = conv_data_objects::TensorValue::NONE;
            flatbuffers::Offset<void> value = 0;
            if(step.operand == OperandKind::CONSTANT)
            {
                valueType = conv_data_objects::TensorValue::Float32Value;
                value = builder.CreateStruct(conv_data_objects::Float32Value(step.value)).Union();
            }
            tensors.push_back(conv_data_objects::CreateTensorAttributesDirect(
                builder,
                uid,
                nullptr,
                step.operand == OperandKind::CONSTANT || step.operand == OperandKind::RUNTIME
                    ? conv_data_objects::DataType::FLOAT
                    : step.operandType,
                &strides,
                &dims,
                false,
                valueType,
                value,
                step.operand == OperandKind::RUNTIME));
            in0 = step.operandFirst ? uid : current;
            in1 = step.operandFirst ? current : uid;
        }
        const int64_t out = EPI_OUT_UID + static_cast<int64_t>(i);
        const bool last = i + 1 == steps.size();
        tensors.push_back(conv_data_objects::CreateTensorAttributesDirect(
            builder, out, nullptr, last ? shape.dataType : intermediate, &yStrides, &yDims, !last));
        const auto attributes
            = conv_data_objects::CreatePointwiseAttributes(builder,
                                                           step.mode,
                                                           fbOptional(step.lowerClip),
                                                           fbOptional(step.upperClip),
                                                           fbOptional(step.slope),
                                                           ::flatbuffers::nullopt,
                                                           in0,
                                                           fbOptional(in1),
                                                           ::flatbuffers::nullopt,
                                                           out,
                                                           fbOptional(step.swishBeta),
                                                           fbOptional(step.eluAlpha),
                                                           fbOptional(step.softplusBeta))
                  .Union();
        nodes.push_back(conv_data_objects::CreateNodeDirect(
            builder,
            "pointwise",
            step.compute,
            conv_data_objects::NodeAttributes::PointwiseAttributes,
            attributes));
        current = out;
    }
    builder.Finish(conv_data_objects::CreateGraphDirect(builder,
                                                        "conv_epilogue_test",
                                                        conv_data_objects::DataType::UNSET,
                                                        conv_data_objects::DataType::UNSET,
                                                        conv_data_objects::DataType::UNSET,
                                                        &tensors,
                                                        &nodes));
    return builder;
}

/// Owns a built fused graph and reads it the way the provider does.
class FusedConvGraph
{
public:
    FusedConvGraph(const ConvGemmShape& shape,
                   const std::vector<Step>& steps,
                   conv_data_objects::DataType intermediate = conv_data_objects::DataType::FLOAT,
                   ConvDirection direction = ConvDirection::FORWARD)
        : _builder(buildFusedConvGraph(shape, steps, intermediate, direction))
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

inline double valueOr(const std::optional<float>& value, double fallback)
{
    return value.has_value() ? static_cast<double>(*value) : fallback;
}

/// One step on the host, in double, with the epilogue's NaN and infinity rules: relu and clamp
/// keep NaN; a gate that saturates to exactly 0 (silu, gelu at -inf) gives 0.
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

/// The largest factor a step can scale an error in its input by.
inline double stepGain(const Step& step, double operandMagnitude)
{
    switch(step.mode)
    {
    case PointwiseMode::MUL:
        return std::max(1.0, operandMagnitude);
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

} // namespace hipblaslt_plugin::test
