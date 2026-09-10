// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "common/PointwiseCommon.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace pointwise_common;

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

namespace
{

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

    static flatbuffers::FlatBufferBuilder validGraph()
    {
        return createUnaryPointwiseGraph(GetParam().mode);
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
    auto builder = validGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes,
       IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             false,
                                             false,
                                             /*overrideShapeEnabled=*/true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForNonFloatComputeType)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::HALF);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForVirtualInputTensor)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             /*virtualInput=*/true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForVirtualOutputTensor)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             false,
                                             true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsTrueForHalfIoDtype)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::HALF);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForBfloat16IoDtype)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::BFLOAT16);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsTrueForRank1Tensor)
{
    auto builder = createUnaryPointwiseGraph(
        GetParam().mode, {16}, std::vector<int64_t>{1}, {16}, std::vector<int64_t>{1});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForRank5Tensor)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 2, 3, 4, 5},
                                             std::vector<int64_t>{120, 60, 20, 5, 1},
                                             {1, 2, 3, 4, 5},
                                             std::vector<int64_t>{120, 60, 20, 5, 1});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, IsApplicableReturnsFalseForMismatchedElementCount)
{
    auto builder = createUnaryPointwiseGraph(GetParam().mode,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 8},
                                             std::vector<int64_t>{96, 32, 8, 1});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = validGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const HipdnnMiopenSettings settings;
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings), 0u);
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, GetCustomKnobsReturnsEmpty)
{
    auto builder = validGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(*_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_P(TestMiopenUnaryActivationPlanBuilderModes, BuildPlanDoesNotThrowForValidGraph)
{
    auto builder = validGraph();
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
    auto builder = createUnaryPointwiseGraph(PointwiseMode::ADD);
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
        auto builder = createUnaryPointwiseGraph(mode);
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
        std::optional<std::vector<int64_t>> inputStrides = std::vector<int64_t>{48, 16, 4, 1};
        std::optional<std::vector<int64_t>> outputStrides = std::vector<int64_t>{48, 16, 4, 1};
        (nullInput ? inputStrides : outputStrides) = std::nullopt;

        auto builder = createUnaryPointwiseGraph(
            PointwiseMode::RELU_FWD, {1, 3, 4, 4}, inputStrides, {1, 3, 4, 4}, outputStrides);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph))
            << (nullInput ? "null input strides" : "null output strides");
    }
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForDimsStridesSizeMismatch)
{
    // Dims and strides are indexed in lockstep when the descriptor is built; a rank-4 dims array
    // paired with a rank-3 strides array would read past the end of the shorter one.
    auto builder = createUnaryPointwiseGraph(
        PointwiseMode::RELU_FWD, {1, 3, 4, 4}, std::vector<int64_t>{16, 4, 1}, {1, 3, 4, 4});
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
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             std::nullopt,
                                             /*reluUpperClip=*/1.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsTrueForReluWithLowerClipSlope)
{
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             std::nullopt,
                                             std::nullopt,
                                             /*reluLowerClipSlope=*/0.1f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithNonZeroLowerClipAndSlope)
{
    // MIOpen's LEAKYRELU is slope-only (knee fixed at 0) and cannot represent a non-zero
    // lower_clip; accepting this would silently drop the lower_clip and miscompute the op.
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/0.3f,
                                             std::nullopt,
                                             /*reluLowerClipSlope=*/0.01f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithZeroLowerClipAndSlope)
{
    // A zero lower_clip is a no-op knee, so slope-only leaky ReLU is faithfully representable.
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/0.0f,
                                             std::nullopt,
                                             /*reluLowerClipSlope=*/0.01f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsTrueForReluWithLowerAndUpperClip)
{
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/-1.0f,
                                             /*reluUpperClip=*/1.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithNonZeroLowerClipOnly)
{
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/0.5f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithZeroLowerAndUpperClip)
{
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/0.0f,
                                             /*reluUpperClip=*/1.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithNegativeLowerClipAndNoUpperClipOrSlope)
{
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/-1.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsTrueForReluWithZeroLowerClipOnly)
{
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/0.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, IsApplicableReturnsFalseForReluWithUpperClipAndSlope)
{
    // The reference computes a leaky ramp below the knee and clips above it. MIOpen's
    // CLIPPEDRELU is flat below the knee and has no slope parameter, and the mapping reaches it
    // before its leaky branch, so accepting this would silently drop the slope.
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             std::nullopt,
                                             /*reluUpperClip=*/6.0f,
                                             /*reluLowerClipSlope=*/0.01f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsFalseForReluWithLowerAndUpperClipAndSlope)
{
    // Same reasoning for CLAMP, which floors at lower_clip instead of following the slope.
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/-1.0f,
                                             /*reluUpperClip=*/1.0f,
                                             /*reluLowerClipSlope=*/0.01f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithUpperClipAndZeroSlope)
{
    // A zero slope is a no-op ramp: CLIPPEDRELU already computes zero below the knee, so the
    // combination is representable exactly and must not be declined.
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             std::nullopt,
                                             /*reluUpperClip=*/6.0f,
                                             /*reluLowerClipSlope=*/0.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder,
       IsApplicableReturnsTrueForReluWithLowerAndUpperClipAndZeroSlope)
{
    // Likewise CLAMP floors at lower_clip, which is what a zero slope asks for.
    auto builder = createUnaryPointwiseGraph(PointwiseMode::RELU_FWD,
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             {1, 3, 4, 4},
                                             std::vector<int64_t>{48, 16, 4, 1},
                                             DataType::FLOAT,
                                             DataType::FLOAT,
                                             /*reluLowerClip=*/-1.0f,
                                             /*reluUpperClip=*/1.0f,
                                             /*reluLowerClipSlope=*/0.0f);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

// ============================================================================
// Binary-node hardening (see MiopenBinaryPointwiseChecks.cpp for the counterpart)
// ============================================================================

TEST_F(TestMiopenUnaryActivationPlanBuilder, UnaryBuilderDeclinesNodeWithSecondInput)
{
    // A node carrying in_1_tensor_uid is a binary node; the unary builder must not treat it as
    // a unary RELU/SIGMOID/TANH node purely because in_2 is absent.
    auto builder = createPointwiseGraph(PointwiseMode::RELU_FWD,
                                        {1, 3, 4, 4},
                                        std::vector<int64_t>{48, 16, 4, 1},
                                        {1, 3, 4, 4},
                                        std::vector<int64_t>{48, 16, 4, 1},
                                        std::vector<int64_t>{1, 3, 4, 4});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, UnaryBuilderDeclinesNodeWithThirdInput)
{
    // A node carrying in_2_tensor_uid is a ternary node (e.g. BINARY_SELECT); it must be
    // declined regardless of whether in_1 is also present.
    auto builder = createPointwiseGraph(PointwiseMode::RELU_FWD,
                                        {1, 3, 4, 4},
                                        std::vector<int64_t>{48, 16, 4, 1},
                                        {1, 3, 4, 4},
                                        std::vector<int64_t>{48, 16, 4, 1},
                                        std::nullopt,
                                        std::vector<int64_t>{48, 16, 4, 1},
                                        std::vector<int64_t>{1, 3, 4, 4});
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestMiopenUnaryActivationPlanBuilder, BinaryModesAreDeclinedByUnaryBuilder)
{
    // A unary regression should not be diagnosed under a binary test name: this belongs here,
    // exercising the mode side of the split rather than the tensor-arity side above.
    for(const auto mode : {PointwiseMode::ADD,
                           PointwiseMode::SUB,
                           PointwiseMode::MUL,
                           PointwiseMode::MAX_OP,
                           PointwiseMode::MIN_OP})
    {
        auto builder = createBinaryPointwiseGraph(mode,
                                                  {1, 3, 4, 4},
                                                  std::vector<int64_t>{48, 16, 4, 1},
                                                  {1, 3, 4, 4},
                                                  std::vector<int64_t>{48, 16, 4, 1},
                                                  std::vector<int64_t>{1, 3, 4, 4},
                                                  std::vector<int64_t>{48, 16, 4, 1});
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph))
            << "mode: " << hipdnn_flatbuffers_sdk::data_objects::EnumNamePointwiseMode(mode);
    }
}
