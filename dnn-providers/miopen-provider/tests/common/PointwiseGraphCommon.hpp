// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace test_pointwise_graph_common
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// Every knob the unary and binary applicability checks look at, so a single graph factory
// covers both. Defaults describe the canonical valid unary graph: a single non-virtual fp32
// NCHW RELU_FWD node with no second or third input (in_0 uid 1, out_0 uid 2).
//
// The binary-specific fields below default to "absent", which keeps a default-constructed
// spec producing byte-identical output to the unary-only factory this was extracted from.
struct PointwiseGraphSpec
{
    PointwiseMode mode = PointwiseMode::RELU_FWD;
    DataType ioDataType = DataType::FLOAT;
    DataType computeDataType = DataType::FLOAT;
    std::vector<int64_t> inputDims{1, 3, 4, 4};
    std::vector<int64_t> outputDims{1, 3, 4, 4};
    // std::nullopt emits a null strides vector, which is what the applicability check's
    // "tensor dims or strides are null" guard looks for. A plain empty vector would not do:
    // CreateTensorAttributesDirect only omits the field when handed a null pointer.
    std::optional<std::vector<int64_t>> inputStrides{{48, 16, 4, 1}};
    std::optional<std::vector<int64_t>> outputStrides{{48, 16, 4, 1}};
    bool virtualInput = false;
    bool virtualOutput = false;
    bool overrideShapeEnabled = false;
    flatbuffers::Optional<float> reluLowerClip = flatbuffers::nullopt;
    flatbuffers::Optional<float> reluUpperClip = flatbuffers::nullopt;
    flatbuffers::Optional<float> reluLowerClipSlope = flatbuffers::nullopt;

    // Binary-specific knobs. A second input (uid 3, "in_1") is only added to the graph when
    // this has a value; leaving it nullopt reproduces the original unary-only graph exactly.
    std::optional<std::vector<int64_t>> secondInputDims;
    std::optional<std::vector<int64_t>> secondInputStrides{{48, 16, 4, 1}};
    bool virtualSecondInput = false;
    std::optional<DataType> secondInputDataType; // defaults to ioDataType when unset

    // A third input (uid 4, "in_2") makes the node ternary. Independent of secondInputDims so
    // tests can exercise "in_2 present, in_1 absent" as well as "in_2 present, in_1 present".
    bool addThirdInput = false;
};

inline flatbuffers::FlatBufferBuilder createPointwiseGraph(const PointwiseGraphSpec& spec)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;

    const std::vector<int64_t>* inputStrides
        = spec.inputStrides ? &spec.inputStrides.value() : nullptr;
    const std::vector<int64_t>* outputStrides
        = spec.outputStrides ? &spec.outputStrides.value() : nullptr;

    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, 1, "input", spec.ioDataType, inputStrides, &spec.inputDims, spec.virtualInput));

    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                          2,
                                                                          "output",
                                                                          spec.ioDataType,
                                                                          outputStrides,
                                                                          &spec.outputDims,
                                                                          spec.virtualOutput));

    flatbuffers::Optional<int64_t> in1Uid = flatbuffers::nullopt;
    if(spec.secondInputDims.has_value())
    {
        const std::vector<int64_t>* secondInputStrides
            = spec.secondInputStrides ? &spec.secondInputStrides.value() : nullptr;
        tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            3,
            "second_input",
            spec.secondInputDataType.value_or(spec.ioDataType),
            secondInputStrides,
            &spec.secondInputDims.value(),
            spec.virtualSecondInput));
        in1Uid = 3;
    }

    flatbuffers::Optional<int64_t> in2Uid = flatbuffers::nullopt;
    if(spec.addThirdInput)
    {
        // The exact shape of the third input does not matter for the tests that use it: they
        // only assert that its mere *presence* is declined.
        tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
            builder, 4, "third_input", spec.ioDataType, inputStrides, &spec.inputDims, false));
        in2Uid = 4;
    }

    auto pwAttr = data_objects::CreatePointwiseAttributes(builder,
                                                          spec.mode,
                                                          spec.reluLowerClip,
                                                          spec.reluUpperClip,
                                                          spec.reluLowerClipSlope,
                                                          flatbuffers::nullopt,
                                                          1,
                                                          in1Uid,
                                                          in2Uid,
                                                          2);

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "pointwise",
                                       spec.computeDataType,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       pwAttr.Union()));

    auto graphOffset = data_objects::CreateGraphDirect(builder,
                                                       "test",
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       &tensorAttributes,
                                                       &nodes,
                                                       flatbuffers::nullopt,
                                                       spec.overrideShapeEnabled);
    builder.Finish(graphOffset);

    return builder;
}

} // namespace test_pointwise_graph_common
