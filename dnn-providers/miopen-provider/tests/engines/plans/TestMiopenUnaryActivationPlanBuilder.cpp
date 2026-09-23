// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

namespace
{

// Every knob the applicability checks look at, so a single graph factory covers all cases.
// Defaults describe the canonical valid graph: a single non-virtual fp32 NCHW RELU_FWD node.
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
};

flatbuffers::FlatBufferBuilder createPointwiseGraph(const PointwiseGraphSpec& spec)
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

    auto pwAttr = data_objects::CreatePointwiseAttributes(builder,
                                                          spec.mode,
                                                          spec.reluLowerClip,
                                                          spec.reluUpperClip,
                                                          spec.reluLowerClipSlope,
                                                          flatbuffers::nullopt,
                                                          1,
                                                          flatbuffers::nullopt,
                                                          flatbuffers::nullopt,
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

// The pointwise modes handled by MiopenUnaryActivationPlanBuilder. Every check that is not
// ReLU-parameter specific must behave identically for all of them.
struct ActivationCase
{
    PointwiseMode mode;
    const char* name;
};

const std::vector<ActivationCase>& getActivationCases()
{
    static const std::vector<ActivationCase> s_cases = {{PointwiseMode::RELU_FWD, "ReluFwd"},
                                                        {PointwiseMode::SIGMOID_FWD, "SigmoidFwd"},
                                                        {PointwiseMode::TANH_FWD, "TanhFwd"}};
    return s_cases;
}

// Shared state for both fixtures below. A single builder handles every unary activation, so
// there is nothing per-activation to configure here.
class UnaryActivationPlanBuilderFixture
{
protected:
    MiopenUnaryActivationPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
    MockEngineConfig _mockEngineConfig;
};

} // namespace

// Mode-independent behavior, and the ReLU-specific parameter combinations.
class TestMiopenUnaryActivationPlanBuilder : public ::testing::Test,
                                             protected UnaryActivationPlanBuilderFixture
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }
};

// Behavior that must hold identically for every supported activation mode.
class TestMiopenUnaryActivationPlanBuilderModes : public ::testing::TestWithParam<ActivationCase>,
                                                  protected UnaryActivationPlanBuilderFixture
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    static PointwiseGraphSpec validSpec()
    {
        PointwiseGraphSpec spec;
        spec.mode = GetParam().mode;
        return spec;
    }
};

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestMiopenUnaryActivationPlanBuilderModes,
                         ::testing::ValuesIn(getActivationCases()),
                         [](const ::testing::TestParamInfo<ActivationCase>& info) {
                             return std::string(info.param.name);
                         });

// ============================================================================
// Per-mode behavior
// ============================================================================

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsTrueForValidGraph)
{
    auto builder = createPointwiseGraph(validSpec());
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes,
       IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto spec = validSpec();
    spec.overrideShapeEnabled = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForNonFloatComputeType)
{
    auto spec = validSpec();
    spec.computeDataType = DataType::HALF;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForVirtualInputTensor)
{
    auto spec = validSpec();
    spec.virtualInput = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForVirtualOutputTensor)
{
    auto spec = validSpec();
    spec.virtualOutput = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsTrueForHalfIoDtype)
{
    auto spec = validSpec();
    spec.ioDataType = DataType::HALF;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForBfloat16IoDtype)
{
    auto spec = validSpec();
    spec.ioDataType = DataType::BFLOAT16;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsTrueForRank1Tensor)
{
    auto spec = validSpec();
    spec.inputDims = {16};
    spec.inputStrides = {1};
    spec.outputDims = {16};
    spec.outputStrides = {1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForRank5Tensor)
{
    auto spec = validSpec();
    spec.inputDims = {1, 2, 3, 4, 5};
    spec.inputStrides = {120, 60, 20, 5, 1};
    spec.outputDims = {1, 2, 3, 4, 5};
    spec.outputStrides = {120, 60, 20, 5, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForMismatchedElementCount)
{
    auto spec = validSpec();
    spec.outputDims = {1, 3, 4, 8};
    spec.outputStrides = {96, 32, 8, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = createPointwiseGraph(validSpec());
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const HipdnnMiopenSettings settings;
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings), 0u);
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, GetCustomKnobsReturnsEmpty)
{
    auto builder = createPointwiseGraph(validSpec());
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(*_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, BuildPlanDoesNotThrowForValidGraph)
{
    auto builder = createPointwiseGraph(validSpec());
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
}

// ============================================================================
// Mode-independent graph shape checks
// ============================================================================

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, mockGraph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForUnsupportedAttributes)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
    EXPECT_CALL(mockGraph, hasOnlySupportedAttributes(::testing::_))
        .WillOnce(::testing::Return(false));

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, mockGraph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForUnsupportedMode)
{
    PointwiseGraphSpec spec;
    spec.mode = PointwiseMode::ADD;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForBackwardModes)
{
    // mapPointwiseModeToMiopenActivation maps the *_BWD modes, but the plan always calls
    // miopenActivationForward, so the backward modes must be declined here rather than
    // silently computing the forward activation.
    for(const auto mode :
        {PointwiseMode::RELU_BWD, PointwiseMode::SIGMOID_BWD, PointwiseMode::TANH_BWD})
    {
        PointwiseGraphSpec spec;
        spec.mode = mode;
        auto builder = createPointwiseGraph(spec);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph))
            << "mode: " << hipdnn_flatbuffers_sdk::data_objects::EnumNamePointwiseMode(mode);
    }
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForNullStrides)
{
    // The plan reads strides straight out of the flatbuffer when building the MIOpen tensor
    // descriptor, so a tensor with no strides at all must be declined rather than dereferenced.
    for(const bool nullInput : {true, false})
    {
        PointwiseGraphSpec spec;
        (nullInput ? spec.inputStrides : spec.outputStrides) = std::nullopt;
        auto builder = createPointwiseGraph(spec);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph))
            << (nullInput ? "null input strides" : "null output strides");
    }
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForDimsStridesSizeMismatch)
{
    // Dims and strides are indexed in lockstep when the descriptor is built; a rank-4 dims array
    // paired with a rank-3 strides array would read past the end of the shorter one.
    PointwiseGraphSpec spec;
    spec.inputStrides = {16, 4, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// ReLU parameter combinations
//
// Only the ReLU family carries parameters; Sigmoid and Tanh ignore them entirely.
// ============================================================================

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsTrueForReluWithUpperClip)
{
    PointwiseGraphSpec spec;
    spec.reluUpperClip = 1.0f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsTrueForReluWithLowerClipSlope)
{
    PointwiseGraphSpec spec;
    spec.reluLowerClipSlope = 0.1f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithNonZeroLowerClipAndSlope)
{
    // MIOpen's LEAKYRELU is slope-only (knee fixed at 0) and cannot represent a non-zero
    // lower_clip; accepting this would silently drop the lower_clip and miscompute the op.
    PointwiseGraphSpec spec;
    spec.reluLowerClip = 0.3f;
    spec.reluLowerClipSlope = 0.01f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithZeroLowerClipAndSlope)
{
    // A zero lower_clip is a no-op knee, so slope-only leaky ReLU is faithfully representable.
    PointwiseGraphSpec spec;
    spec.reluLowerClip = 0.0f;
    spec.reluLowerClipSlope = 0.01f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsTrueForReluWithLowerAndUpperClip)
{
    PointwiseGraphSpec spec;
    spec.reluLowerClip = -1.0f;
    spec.reluUpperClip = 1.0f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithNonZeroLowerClipOnly)
{
    PointwiseGraphSpec spec;
    spec.reluLowerClip = 0.5f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithZeroLowerAndUpperClip)
{
    PointwiseGraphSpec spec;
    spec.reluLowerClip = 0.0f;
    spec.reluUpperClip = 1.0f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithNegativeLowerClipAndNoUpperClipOrSlope)
{
    PointwiseGraphSpec spec;
    spec.reluLowerClip = -1.0f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsTrueForReluWithZeroLowerClipOnly)
{
    PointwiseGraphSpec spec;
    spec.reluLowerClip = 0.0f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForReluWithUpperClipAndSlope)
{
    // The reference computes a leaky ramp below the knee and clips above it. MIOpen's
    // CLIPPEDRELU is flat below the knee and has no slope parameter, and the mapping reaches it
    // before its leaky branch, so accepting this would silently drop the slope.
    PointwiseGraphSpec spec;
    spec.reluUpperClip = 6.0f;
    spec.reluLowerClipSlope = 0.01f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithLowerAndUpperClipAndSlope)
{
    // Same reasoning for CLAMP, which floors at lower_clip instead of following the slope.
    PointwiseGraphSpec spec;
    spec.reluLowerClip = -1.0f;
    spec.reluUpperClip = 1.0f;
    spec.reluLowerClipSlope = 0.01f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithUpperClipAndZeroSlope)
{
    // A zero slope is a no-op ramp: CLIPPEDRELU already computes zero below the knee, so the
    // combination is representable exactly and must not be declined.
    PointwiseGraphSpec spec;
    spec.reluUpperClip = 6.0f;
    spec.reluLowerClipSlope = 0.0f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithLowerAndUpperClipAndZeroSlope)
{
    // Likewise CLAMP floors at lower_clip, which is what a zero slope asks for.
    PointwiseGraphSpec spec;
    spec.reluLowerClip = -1.0f;
    spec.reluUpperClip = 1.0f;
    spec.reluLowerClipSlope = 0.0f;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}
