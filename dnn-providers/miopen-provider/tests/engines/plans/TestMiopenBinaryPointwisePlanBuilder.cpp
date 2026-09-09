// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "common/PointwiseExecutionCommon.hpp"
#include "common/PointwiseGraphCommon.hpp"
#include "engines/plans/MiopenBinaryPointwisePlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace test_pointwise_graph_common;
using namespace test_pointwise_execution_common;

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

namespace
{

PointwiseGraphSpec validBinarySpec()
{
    PointwiseGraphSpec spec;
    spec.mode = PointwiseMode::ADD;
    spec.secondInputDims = {1, 3, 1, 1};
    spec.secondInputStrides = {3, 1, 1, 1};
    return spec;
}

struct BinaryModeCase
{
    PointwiseMode mode;
    const char* name;
};

const std::vector<BinaryModeCase>& getBinaryModeCases()
{
    static const std::vector<BinaryModeCase> s_cases = {{PointwiseMode::ADD, "Add"},
                                                        {PointwiseMode::SUB, "Sub"},
                                                        {PointwiseMode::MUL, "Mul"},
                                                        {PointwiseMode::MAX_OP, "MaxOp"},
                                                        {PointwiseMode::MIN_OP, "MinOp"}};
    return s_cases;
}

// Shared state for both fixtures below.
class BinaryPointwisePlanBuilderFixture
{
protected:
    MiopenBinaryPointwisePlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
    MockEngineConfig _mockEngineConfig;
};

// Every shape the spec's accepting-case list calls out for TestGpuMiopenBinaryPointwisePlanBuilderModes:
// canonical fp32, all-HALF, rank 3/4/5, both B-broadcast stride variants at [1,4,1,1], and the
// fully-scalar [1,1,1,1] case (safe here because this is a unit test bypassing engine ranking).
struct ShapeCase
{
    DataType dtype;
    TensorShape a;
    TensorShape b;
    TensorShape c;
    const char* name;
};

const std::vector<ShapeCase>& getShapeCases()
{
    static const std::vector<ShapeCase> s_cases = {
        // Canonical fp32: rank 4, B broadcasting on the last two axes.
        {DataType::FLOAT,
         {{1, 3, 4, 4}, {48, 16, 4, 1}},
         {{1, 3, 1, 1}, {3, 1, 1, 1}},
         {{1, 3, 4, 4}, {48, 16, 4, 1}},
         "CanonicalFp32"},
        // All-HALF, same shape as canonical.
        {DataType::HALF,
         {{1, 3, 4, 4}, {48, 16, 4, 1}},
         {{1, 3, 1, 1}, {3, 1, 1, 1}},
         {{1, 3, 4, 4}, {48, 16, 4, 1}},
         "AllHalf"},
        // Rank 3, B fully broadcasting.
        {DataType::FLOAT,
         {{4, 4, 4}, {16, 4, 1}},
         {{1, 1, 1}, {1, 1, 1}},
         {{4, 4, 4}, {16, 4, 1}},
         "Rank3"},
        // Rank 4, B full-size (no broadcast).
        {DataType::FLOAT,
         {{1, 3, 4, 4}, {48, 16, 4, 1}},
         {{1, 3, 4, 4}, {48, 16, 4, 1}},
         {{1, 3, 4, 4}, {48, 16, 4, 1}},
         "Rank4NoBroadcast"},
        // Rank 5, B fully broadcasting.
        {DataType::FLOAT,
         {{1, 2, 3, 4, 4}, {96, 48, 16, 4, 1}},
         {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
         {{1, 2, 3, 4, 4}, {96, 48, 16, 4, 1}},
         "Rank5"},
        // B-broadcast [1,4,1,1] with strides [1,1,1,1].
        {DataType::FLOAT,
         {{1, 4, 8, 8}, {256, 64, 8, 1}},
         {{1, 4, 1, 1}, {1, 1, 1, 1}},
         {{1, 4, 8, 8}, {256, 64, 8, 1}},
         "BBroadcastOnesStrides"},
        // B-broadcast [1,4,1,1] with strides [4,1,1,1] -- the leading axis has dim 1, so its
        // stride is never dereferenced; this exercises check 17's per-axis exemption directly.
        {DataType::FLOAT,
         {{1, 4, 8, 8}, {256, 64, 8, 1}},
         {{1, 4, 1, 1}, {4, 1, 1, 1}},
         {{1, 4, 8, 8}, {256, 64, 8, 1}},
         "BBroadcastLeadingStrideFour"},
        // Fully scalar: [1,1,1,1] op [1,1,1,1] -> [1,1,1,1].
        {DataType::FLOAT,
         {{1, 1, 1, 1}, {1, 1, 1, 1}},
         {{1, 1, 1, 1}, {1, 1, 1, 1}},
         {{1, 1, 1, 1}, {1, 1, 1, 1}},
         "FullyScalar"},
    };
    return s_cases;
}

} // namespace

// Behavior that must hold identically for every supported binary pointwise mode.
class TestGpuMiopenBinaryPointwisePlanBuilderModes
    : public ::testing::TestWithParam<BinaryModeCase>,
      protected BinaryPointwisePlanBuilderFixture
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    static PointwiseGraphSpec validSpec()
    {
        auto spec = validBinarySpec();
        spec.mode = GetParam().mode;
        return spec;
    }
};

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestGpuMiopenBinaryPointwisePlanBuilderModes,
                         ::testing::ValuesIn(getBinaryModeCases()),
                         [](const ::testing::TestParamInfo<BinaryModeCase>& info) {
                             return std::string(info.param.name);
                         });

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, IsApplicableReturnsTrueForValidGraph)
{
    auto builder = createPointwiseGraph(validSpec());
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes,
       IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto spec = validSpec();
    spec.overrideShapeEnabled = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = createPointwiseGraph(validSpec());
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const HipdnnMiopenSettings settings;
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings), 0u);
}

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, GetCustomKnobsReturnsEmpty)
{
    auto builder = createPointwiseGraph(validSpec());
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(*_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

// Every accepting shape must actually execute and assert numerics, not just EXPECT_NO_THROW:
// constraints 5, 6 and 12 (see MiopenBinaryPointwiseChecks.cpp) are all silent-wrong-answer
// classes that a no-throw build+execute cannot see.
TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, BuildPlanExecutesAndProducesCorrectResults)
{
    for(const auto& shape : getShapeCases())
    {
        SCOPED_TRACE(shape.name);

        auto spec = validSpec();
        spec.ioDataType = shape.dtype;
        spec.secondInputDataType = shape.dtype;
        spec.inputDims = shape.a.dims;
        spec.inputStrides = shape.a.strides;
        spec.secondInputDims = shape.b.dims;
        spec.secondInputStrides = shape.b.strides;
        spec.outputDims = shape.c.dims;
        spec.outputStrides = shape.c.strides;

        auto builder = createPointwiseGraph(spec);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        HipdnnMiopenContext ctx;
        ASSERT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));

        if(shape.dtype == DataType::HALF)
        {
            executeAndVerify<hipdnn_data_sdk::types::half>(
                ctx.plan(), *_dummyHandle, GetParam().mode, shape.dtype, shape.a, shape.b, shape.c);
        }
        else
        {
            executeAndVerify<float>(
                ctx.plan(), *_dummyHandle, GetParam().mode, shape.dtype, shape.a, shape.b, shape.c);
        }
    }
}

// ============================================================================
// Mode-independent graph shape checks. These mirror the check order in
// MiopenBinaryPointwiseChecks.cpp; the exhaustive per-check coverage lives in
// TestMiopenBinaryPointwiseChecks.cpp, which does not require a handle or GPU.
// ============================================================================

class TestGpuMiopenBinaryPointwisePlanBuilder : public ::testing::Test,
                                                protected BinaryPointwisePlanBuilderFixture
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }
};

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForMultiNodeGraph)
{
    const MockGraph mockGraph;
    EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(2));

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, mockGraph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForMissingSecondInput)
{
    PointwiseGraphSpec spec;
    spec.mode = PointwiseMode::ADD;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForThirdInputPresent)
{
    auto spec = validBinarySpec();
    spec.addThirdInput = true;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForUnsupportedMode)
{
    auto spec = validBinarySpec();
    spec.mode = PointwiseMode::DIV;
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForFirstInputBroadcasting)
{
    auto spec = validBinarySpec();
    spec.inputDims = {1, 1, 4, 4};
    spec.inputStrides = {16, 16, 4, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder,
       IsApplicableReturnsFalseForSecondInputNotBroadcastable)
{
    auto spec = validBinarySpec();
    spec.secondInputDims = {1, 2, 1, 1};
    spec.secondInputStrides = {2, 1, 1, 1};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, UnaryModesAreDeclinedByBinaryBuilder)
{
    // A binary regression should not be diagnosed under a unary test name: exercises the mode
    // side of the unary/binary split from this side.
    for(const auto mode :
        {PointwiseMode::RELU_FWD, PointwiseMode::SIGMOID_FWD, PointwiseMode::TANH_FWD})
    {
        auto spec = validBinarySpec();
        spec.mode = mode;
        auto builder = createPointwiseGraph(spec);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph))
            << "mode: " << hipdnn_flatbuffers_sdk::data_objects::EnumNamePointwiseMode(mode);
    }
}
