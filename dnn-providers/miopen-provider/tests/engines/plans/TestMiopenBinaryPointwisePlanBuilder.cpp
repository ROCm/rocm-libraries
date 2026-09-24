// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "common/PointwiseCommon.hpp"
#include "engines/plans/MiopenBinaryPointwisePlanBuilder.hpp"

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace pointwise_common;

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

namespace
{

// Shared state for both fixtures below.
class BinaryPointwisePlanBuilderFixture
{
protected:
    MiopenBinaryPointwisePlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _dummyHandle;
    MockEngineConfig _mockEngineConfig;
};

} // namespace

// Behavior that must hold identically for every supported binary pointwise mode.
class TestGpuMiopenBinaryPointwisePlanBuilderModes : public ::testing::TestWithParam<ModeCase>,
                                                     protected BinaryPointwisePlanBuilderFixture
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _dummyHandle = std::make_unique<HipdnnMiopenHandle>();
    }

    static flatbuffers::FlatBufferBuilder validGraph()
    {
        return createPointwiseGraph(PointwiseGraphSpec::binary(GetParam().mode));
    }
};

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, IsApplicableReturnsTrueForValidGraph)
{
    auto builder = validGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes,
       IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder
        = createPointwiseGraph(PointwiseGraphSpec::binary(GetParam().mode,
                                                          {1, 3, 4, 4},
                                                          std::vector<int64_t>{48, 16, 4, 1},
                                                          {1, 3, 4, 4},
                                                          std::vector<int64_t>{48, 16, 4, 1},
                                                          std::vector<int64_t>{1, 3, 1, 1},
                                                          std::vector<int64_t>{3, 1, 1, 1},
                                                          DataType::FLOAT,
                                                          DataType::FLOAT,
                                                          std::nullopt,
                                                          false,
                                                          false,
                                                          false,
                                                          /*overrideShapeEnabled=*/true));
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = validGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const HipdnnMiopenSettings settings;
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(*_dummyHandle, graph, settings), 0u);
}

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, GetCustomKnobsReturnsEmpty)
{
    auto builder = validGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    auto knobs = _planBuilder.getCustomKnobs(*_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

TEST_P(TestGpuMiopenBinaryPointwisePlanBuilderModes, BuildPlanSetsPlanForValidGraph)
{
    auto builder = validGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    HipdnnMiopenContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(*_dummyHandle, graph, _mockEngineConfig, ctx));
    ASSERT_TRUE(ctx.hasValidPlan());
    EXPECT_EQ(ctx.plan().getWorkspaceSize(*_dummyHandle), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestGpuMiopenBinaryPointwisePlanBuilderModes,
                         ::testing::ValuesIn(getBinaryModeCases()),
                         [](const ::testing::TestParamInfo<ModeCase>& info) {
                             return std::string(info.param.name);
                         });

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
    auto builder = createPointwiseGraph(PointwiseGraphSpec::unary(PointwiseMode::ADD));
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForThirdInputPresent)
{
    auto spec = PointwiseGraphSpec::binary();
    spec.thirdInputDims = std::vector<int64_t>{1, 3, 4, 4};
    auto builder = createPointwiseGraph(spec);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForUnsupportedMode)
{
    auto builder = createPointwiseGraph(PointwiseGraphSpec::binary(PointwiseMode::DIV));
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder, IsApplicableReturnsFalseForFirstInputBroadcasting)
{
    auto builder
        = createPointwiseGraph(PointwiseGraphSpec::binary(PointwiseMode::ADD,
                                                          {1, 1, 4, 4},
                                                          std::vector<int64_t>{16, 16, 4, 1},
                                                          {1, 3, 4, 4},
                                                          std::vector<int64_t>{48, 16, 4, 1},
                                                          std::vector<int64_t>{1, 3, 1, 1},
                                                          std::vector<int64_t>{3, 1, 1, 1}));
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}

TEST_F(TestGpuMiopenBinaryPointwisePlanBuilder,
       IsApplicableReturnsFalseForSecondInputNotBroadcastable)
{
    auto builder
        = createPointwiseGraph(PointwiseGraphSpec::binary(PointwiseMode::ADD,
                                                          {1, 3, 4, 4},
                                                          std::vector<int64_t>{48, 16, 4, 1},
                                                          {1, 3, 4, 4},
                                                          std::vector<int64_t>{48, 16, 4, 1},
                                                          std::vector<int64_t>{1, 2, 1, 1},
                                                          std::vector<int64_t>{2, 1, 1, 1}));
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(*_dummyHandle, graph));
}
