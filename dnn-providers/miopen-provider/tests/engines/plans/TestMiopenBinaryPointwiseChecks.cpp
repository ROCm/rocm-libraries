// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

#include "common/PointwiseCommon.hpp"
#include "engines/plans/MiopenBinaryPointwiseChecks.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace pointwise_common;

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

class TestMiopenBinaryPointwiseChecksModes : public ::testing::TestWithParam<ModeCase>
{
};

TEST_P(TestMiopenBinaryPointwiseChecksModes, IsSupportedTrueForValidGraph)
{
    auto spec = validBinarySpec();
    spec.mode = GetParam().mode;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(binary_pointwise_applicability::isSupported(graph));
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestMiopenBinaryPointwiseChecksModes,
                         ::testing::ValuesIn(getBinaryModeCases()),
                         [](const ::testing::TestParamInfo<ModeCase>& info) {
                             return std::string(info.param.name);
                         });

// Mode-independent decline cases.

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));
    // The node-count guard must reject the graph before anything else is inspected: if it were
    // deleted or weakened, hasOnlySupportedAttributes() would be reached next, and this
    // expectation would fail the test instead of silently passing via gmock's default action.
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_)).Times(0);

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(mockGraph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForUnsupportedAttributes)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(mockGraph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForNonFloatComputeType)
{
    auto spec = validBinarySpec();
    spec.computeDataType = DataType::HALF;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForMissingSecondInput)
{
    // A node with no in_1_tensor_uid is a unary node, not this provider's concern.
    PointwiseGraphSpec spec;
    spec.mode = PointwiseMode::ADD;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForThirdInputPresent)
{
    auto spec = validBinarySpec();
    spec.addThirdInput = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForVirtualFirstInput)
{
    auto spec = validBinarySpec();
    spec.virtualInput = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForVirtualSecondInput)
{
    auto spec = validBinarySpec();
    spec.virtualSecondInput = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForVirtualOutput)
{
    auto spec = validBinarySpec();
    spec.virtualOutput = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForBfloat16Dtype)
{
    auto spec = validBinarySpec();
    spec.ioDataType = DataType::BFLOAT16;
    spec.secondInputDataType = DataType::BFLOAT16;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedTrueForHalfDtype)
{
    auto spec = validBinarySpec();
    spec.ioDataType = DataType::HALF;
    spec.secondInputDataType = DataType::HALF;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForMismatchedDtypes)
{
    auto spec = validBinarySpec();
    spec.secondInputDataType = DataType::HALF;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForNullSecondInputStrides)
{
    auto spec = validBinarySpec();
    spec.secondInputStrides = std::nullopt;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForOutputRankBelowThree)
{
    auto spec = validBinarySpec();
    spec.inputDims = {3, 4};
    spec.inputStrides = {4, 1};
    spec.outputDims = {3, 4};
    spec.outputStrides = {4, 1};
    spec.secondInputDims = {3, 4};
    spec.secondInputStrides = {4, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForOutputRankAboveFive)
{
    auto spec = validBinarySpec();
    spec.inputDims = {1, 2, 3, 4, 5, 1};
    spec.inputStrides = {120, 60, 20, 5, 1, 1};
    spec.outputDims = spec.inputDims;
    spec.outputStrides = spec.inputStrides;
    spec.secondInputDims = spec.inputDims;
    spec.secondInputStrides = spec.inputStrides;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForSecondInputRankMismatch)
{
    auto spec = validBinarySpec();
    spec.secondInputDims = {3, 1, 1};
    spec.secondInputStrides = {1, 1, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForNonPositiveDim)
{
    auto spec = validBinarySpec();
    spec.secondInputDims = {1, 3, 1, 0};
    spec.secondInputStrides = {3, 1, 1, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForNonPackedFirstInput)
{
    auto spec = validBinarySpec();
    // Channel stride does not equal the product of trailing dims (4*4=16, not 8).
    spec.inputStrides = {48, 8, 4, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForFirstInputBroadcasting)
{
    // A must equal the output shape exactly; MIOpen's tensorOp cannot broadcast its first
    // operand and this provider does not swap operands to make it fit.
    auto spec = validBinarySpec();
    spec.inputDims = {1, 1, 4, 4};
    spec.inputStrides = {16, 16, 4, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForSecondInputNotBroadcastable)
{
    // b's dim at an axis must be either 1 or equal to c's -- 2 is neither.
    auto spec = validBinarySpec();
    spec.secondInputDims = {1, 2, 1, 1};
    spec.secondInputStrides = {2, 1, 1, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedTrueForFullSizeSecondInputNoBroadcast)
{
    auto spec = validBinarySpec();
    spec.secondInputDims = {1, 3, 4, 4};
    spec.secondInputStrides = {48, 16, 4, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedTrueRegardlessOfOverrideShape)
{
    // isApplicable (in the plan builder) declines override-shape graphs; the applicability
    // resolver tested here has no opinion on that flag at all.
    auto spec = validBinarySpec();
    spec.overrideShapeEnabled = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(binary_pointwise_applicability::isSupported(graph));
}

// Modes this provider doesn't map to a miopenTensorOp_t.

class TestMiopenBinaryPointwiseChecksUnsupportedModes : public ::testing::TestWithParam<ModeCase>
{
};

TEST_P(TestMiopenBinaryPointwiseChecksUnsupportedModes, IsSupportedFalseForUnsupportedMode)
{
    auto spec = validBinarySpec();
    spec.mode = GetParam().mode;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestMiopenBinaryPointwiseChecksUnsupportedModes,
                         ::testing::Values(ModeCase{PointwiseMode::DIV, "Div"},
                                           ModeCase{PointwiseMode::CMP_GT, "CmpGt"},
                                           ModeCase{PointwiseMode::CMP_EQ, "CmpEq"},
                                           ModeCase{PointwiseMode::LOGICAL_AND, "LogicalAnd"},
                                           ModeCase{PointwiseMode::ADD_SQUARE, "AddSquare"},
                                           ModeCase{PointwiseMode::RELU_BWD, "ReluBwd"},
                                           ModeCase{PointwiseMode::SIGMOID_BWD, "SigmoidBwd"},
                                           ModeCase{PointwiseMode::TANH_BWD, "TanhBwd"},
                                           ModeCase{PointwiseMode::BINARY_SELECT, "BinarySelect"}),
                         [](const ::testing::TestParamInfo<ModeCase>& info) {
                             return std::string(info.param.name);
                         });

namespace
{

flatbuffers::FlatBufferBuilder buildGraphWithUnresolvableSecondInputUid()
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder fbb;

    const std::vector<int64_t> dims{1, 3, 4, 4};
    const std::vector<int64_t> strides{48, 16, 4, 1};

    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        fbb, 1, "input", DataType::FLOAT, &strides, &dims, false));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        fbb, 2, "output", DataType::FLOAT, &strides, &dims, false));
    // Deliberately no tensor with uid 3 -- in_1_tensor_uid below dangles.

    auto pwAttr = data_objects::CreatePointwiseAttributes(fbb,
                                                          PointwiseMode::ADD,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          1,
                                                          3,
                                                          flatbuffers::nullopt,
                                                          2);

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(fbb,
                                       "pointwise",
                                       DataType::FLOAT,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       pwAttr.Union()));

    auto graphOffset = data_objects::CreateGraphDirect(
        fbb, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensorAttributes, &nodes);
    fbb.Finish(graphOffset);
    return fbb;
}

// Builds the canonical valid binary graph, except in_1 (uid 3) carries the given tweak so the
// caller can flip one flag (pass-by-value / ragged) without hand-rolling the whole graph.
flatbuffers::FlatBufferBuilder
    buildGraphWithTweakedSecondInput(bool isRuntimePassByValue,
                                     flatbuffers::Optional<int64_t> raggedOffsetTensorUid)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder fbb;

    const std::vector<int64_t> dims{1, 3, 4, 4};
    const std::vector<int64_t> strides{48, 16, 4, 1};

    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        fbb, 1, "input", DataType::FLOAT, &strides, &dims, false));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        fbb, 2, "output", DataType::FLOAT, &strides, &dims, false));
    tensorAttributes.push_back(
        data_objects::CreateTensorAttributesDirect(fbb,
                                                   3,
                                                   "second_input",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   false,
                                                   data_objects::TensorValue::NONE,
                                                   0,
                                                   isRuntimePassByValue,
                                                   raggedOffsetTensorUid));

    auto pwAttr = data_objects::CreatePointwiseAttributes(fbb,
                                                          PointwiseMode::ADD,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          1,
                                                          3,
                                                          flatbuffers::nullopt,
                                                          2);

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(fbb,
                                       "pointwise",
                                       DataType::FLOAT,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       pwAttr.Union()));

    auto graphOffset = data_objects::CreateGraphDirect(
        fbb, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensorAttributes, &nodes);
    fbb.Finish(graphOffset);
    return fbb;
}

flatbuffers::FlatBufferBuilder buildGraphWithInPlaceOutput(int64_t outUid)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder fbb;

    const std::vector<int64_t> dims{1, 3, 4, 4};
    const std::vector<int64_t> strides{48, 16, 4, 1};

    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        fbb, 1, "input", DataType::FLOAT, &strides, &dims, false));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        fbb, 3, "second_input", DataType::FLOAT, &strides, &dims, false));

    auto pwAttr = data_objects::CreatePointwiseAttributes(fbb,
                                                          PointwiseMode::ADD,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
                                                          1,
                                                          3,
                                                          flatbuffers::nullopt,
                                                          outUid); // in-place: aliases in_0 or in_1

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(fbb,
                                       "pointwise",
                                       DataType::FLOAT,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       pwAttr.Union()));

    auto graphOffset = data_objects::CreateGraphDirect(
        fbb, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensorAttributes, &nodes);
    fbb.Finish(graphOffset);
    return fbb;
}

} // namespace

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForUnresolvableSecondInputUid)
{
    auto builder = buildGraphWithUnresolvableSecondInputUid();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForDimsStridesSizeMismatch)
{
    auto spec = validBinarySpec();
    spec.secondInputDims = {1, 3, 1, 1};
    spec.secondInputStrides = {3, 1, 1}; // one short of dims' length
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForPassByValueSecondInput)
{
    auto builder = buildGraphWithTweakedSecondInput(true, flatbuffers::nullopt);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForRaggedSecondInput)
{
    auto builder = buildGraphWithTweakedSecondInput(false, 7);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForInPlaceOutputEqualsFirstInput)
{
    auto builder = buildGraphWithInPlaceOutput(1);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForInPlaceOutputEqualsSecondInput)
{
    auto builder = buildGraphWithInPlaceOutput(3);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForOutputRankZero)
{
    auto spec = validBinarySpec();
    spec.inputDims = {};
    spec.inputStrides = std::vector<int64_t>{};
    spec.outputDims = {};
    spec.outputStrides = std::vector<int64_t>{};
    // secondInputDims is std::optional<std::vector<int64_t>>: assigning `{}` would reset it to
    // nullopt (no second input at all) rather than an empty, present vector. Wrap explicitly to
    // keep in_1 present with rank 0.
    spec.secondInputDims = std::vector<int64_t>{};
    spec.secondInputStrides = std::vector<int64_t>{};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForFirstInputRankMismatch)
{
    auto spec = validBinarySpec();
    spec.inputDims = {3, 4, 4};
    spec.inputStrides = {16, 4, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForNegativeStride)
{
    auto spec = validBinarySpec();
    spec.secondInputDims = {1, 3, 1, 1};
    spec.secondInputStrides = {3, 1, 1, -1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}

TEST(TestMiopenBinaryPointwiseChecks, IsSupportedFalseForOutputElementCountExceedsInt32Max)
{
    auto spec = validBinarySpec();
    // 2 * 40000 * 40000 * 1 > INT32_MAX, and stays packed/channels-first.
    spec.inputDims = {2, 40000, 40000, 1};
    spec.inputStrides = {1600000000, 40000, 1, 1};
    spec.outputDims = spec.inputDims;
    spec.outputStrides = spec.inputStrides;
    spec.secondInputDims = spec.inputDims;
    spec.secondInputStrides = spec.inputStrides;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(binary_pointwise_applicability::isSupported(graph));
}
