// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenPlanBuilderBase.hpp"

#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

using namespace miopen_plugin;
using namespace hipdnn_test_sdk::utilities;

namespace
{
// Minimal concrete plan builder that doesn't override getMaxWorkspaceSize() or
// getWorkspaceSizeRange() to test the default implementations
class SimplePlanBuilder : public MiopenPlanBuilderBase
{
public:
    bool isApplicable([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                      [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override
    {
        return true;
    }

    void buildPlan([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                   [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                   [[maybe_unused]] HipdnnEnginePluginExecutionContext& executionContext) const override
    {
    }
};

} // namespace

class TestMiopenPlanBuilderBase : public ::testing::Test
{
protected:
    SimplePlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

TEST_F(TestMiopenPlanBuilderBase, DefaultGetMaxWorkspaceSizeReturnsZero)
{
    MockGraph mockGraph;

    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_dummyHandle, mockGraph);

    EXPECT_EQ(workspaceSize, 0u);
}

TEST_F(TestMiopenPlanBuilderBase, DefaultGetWorkspaceSizeRangeReturnsZero)
{
    MockGraph mockGraph;

    auto workspaceSizeRange = _planBuilder.getWorkspaceSizeRange(_dummyHandle, mockGraph);

    EXPECT_EQ(workspaceSizeRange.min, 0u);
    EXPECT_EQ(workspaceSizeRange.max, 0u);
}
