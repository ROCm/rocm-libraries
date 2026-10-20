// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>
#include <ostream>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "common/ReductionCommon.hpp"
#include "engines/plans/MiopenReductionPlanBuilder.hpp"

using namespace miopen_plugin;
using namespace reduction_common;

using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::MockEngineConfig;

namespace
{

// Owns the flatbuffer a GraphWrapper views, so the two cannot be separated.
class ReductionGraph
{
public:
    explicit ReductionGraph(const ReductionGraphSpec& spec)
        : _builder(createReductionGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
    }

    const GraphWrapper& graph() const
    {
        return _graph;
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    GraphWrapper _graph;
};

// Shared state for both fixtures below.
class ReductionPlanBuilderFixture
{
protected:
    MiopenReductionPlanBuilder _planBuilder;
    std::unique_ptr<HipdnnMiopenHandle> _handle;
    MockEngineConfig _mockEngineConfig;
};

struct WorkspaceCase
{
    ReductionGraphSpec spec;
    const char* name;
};

std::ostream& operator<<(std::ostream& os, const WorkspaceCase& workspaceCase)
{
    return os << workspaceCase.name;
}

} // namespace

class TestGpuMiopenReductionPlanBuilder : public ::testing::Test,
                                          protected ReductionPlanBuilderFixture
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }
};

TEST_F(TestGpuMiopenReductionPlanBuilder, IsApplicableReturnsTrueForValidGraph)
{
    const ReductionGraph graph{ReductionGraphSpec{}};

    EXPECT_TRUE(_planBuilder.isApplicable(*_handle, graph.graph()));
}

TEST_F(TestGpuMiopenReductionPlanBuilder, IsApplicableReturnsFalseForMulNoZeros)
{
    ReductionGraphSpec spec;
    spec.mode = ReductionMode::MUL_NO_ZEROS;
    const ReductionGraph graph{spec};

    EXPECT_FALSE(_planBuilder.isApplicable(*_handle, graph.graph()));
}

TEST_F(TestGpuMiopenReductionPlanBuilder, IsApplicableReturnsFalseForBfloat16)
{
    ReductionGraphSpec spec;
    spec.dataType = DataType::BFLOAT16;
    const ReductionGraph graph{spec};

    EXPECT_FALSE(_planBuilder.isApplicable(*_handle, graph.graph()));
}

TEST_F(TestGpuMiopenReductionPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabled)
{
    ReductionGraphSpec spec;
    spec.overrideShapeEnabled = true;
    const ReductionGraph graph{spec};

    EXPECT_FALSE(_planBuilder.isApplicable(*_handle, graph.graph()));
}

TEST_F(TestGpuMiopenReductionPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    const ReductionGraph graph{ReductionGraphSpec{}};

    EXPECT_TRUE(_planBuilder.getCustomKnobs(*_handle, graph.graph()).empty());
}

TEST_F(TestGpuMiopenReductionPlanBuilder, InitializeExecutionSettingsLeavesSettingsUnchanged)
{
    const ReductionGraph graph{ReductionGraphSpec{}};
    HipdnnMiopenSettings settings;

    _planBuilder.initializeExecutionSettings(*_handle, graph.graph(), _mockEngineConfig, settings);

    EXPECT_FALSE(settings.benchmarkingEnabled());
    EXPECT_FALSE(settings.workspaceSizeLimit().has_value());
    EXPECT_FALSE(settings.defaultWorkspaceSize().has_value());
}

// The workspace depends on the reduction method MIOpen picks for the shape, so cover a
// single-pass shape (only MIOpen's descriptor page) and a multi-block one (page plus partials).
class TestGpuMiopenReductionPlanBuilderWorkspace : public ::testing::TestWithParam<WorkspaceCase>,
                                                   protected ReductionPlanBuilderFixture
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    static size_t miopenWorkspaceSize(const HipdnnMiopenHandle& handle)
    {
        return queryMiopenReductionWorkspaceSize(
            handle.miopenHandle, MIOPEN_REDUCE_TENSOR_ADD, GetParam().spec);
    }
};

TEST_P(TestGpuMiopenReductionPlanBuilderWorkspace, GetMaxWorkspaceSizeMatchesMiopenQuery)
{
    const ReductionGraph graph{GetParam().spec};
    const HipdnnMiopenSettings settings;

    const auto workspaceSize = _planBuilder.getMaxWorkspaceSize(*_handle, graph.graph(), settings);

    EXPECT_GT(workspaceSize, 0u);
    EXPECT_EQ(workspaceSize, miopenWorkspaceSize(*_handle));
}

TEST_P(TestGpuMiopenReductionPlanBuilderWorkspace, BuildPlanSetsPlanWithQueriedWorkspaceSize)
{
    const ReductionGraph graph{GetParam().spec};
    HipdnnMiopenContext context;

    _planBuilder.buildPlan(*_handle, graph.graph(), _mockEngineConfig, context);

    ASSERT_TRUE(context.hasValidPlan());
    EXPECT_EQ(context.plan().getWorkspaceSize(*_handle), miopenWorkspaceSize(*_handle));
}

INSTANTIATE_TEST_SUITE_P(
    AllCases,
    TestGpuMiopenReductionPlanBuilderWorkspace,
    ::testing::Values(
        WorkspaceCase{ReductionGraphSpec{}, "SinglePassFloat"},
        WorkspaceCase{ReductionGraphSpec::packed(ReductionMode::ADD, {2, 4096}, {2, 1}),
                      "MultiBlockFloat"},
        WorkspaceCase{
            ReductionGraphSpec::packed(ReductionMode::ADD, {2, 4096}, {2, 1}, DataType::HALF),
            "MultiBlockHalf"}),
    [](const ::testing::TestParamInfo<WorkspaceCase>& info) {
        return std::string(info.param.name);
    });
