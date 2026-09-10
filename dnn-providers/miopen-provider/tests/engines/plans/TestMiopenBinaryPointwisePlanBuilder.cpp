// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
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

    static PointwiseGraphSpec validSpec()
    {
        auto spec = validBinarySpec();
        spec.mode = GetParam().mode;
        return spec;
    }
};

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
