// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/PlanBuilderInterface.hpp"

#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

using namespace miopen_legacy_plugin;
using namespace hipdnn_test_sdk::utilities;

namespace
{
// Minimal concrete plan builder that doesn't override getMaxWorkspaceSize()
// to test the default implementation
class SimplePlanBuilder : public IPlanBuilder
{
public:
    bool isApplicable([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                      [[maybe_unused]] const hipdnn_plugin_sdk::IGraph& opGraph) const override
    {
        return true;
    }

    void buildPlan([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                   [[maybe_unused]] const hipdnn_plugin_sdk::IGraph& opGraph,
                   [[maybe_unused]] HipdnnEnginePluginExecutionContext& executionContext) const override
    {
    }

    // getMaxWorkspaceSize() NOT overridden - uses default from IPlanBuilder
};
} // namespace

class TestIPlanBuilder : public ::testing::Test
{
protected:
    SimplePlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

TEST_F(TestIPlanBuilder, DefaultGetMaxWorkspaceSizeReturnsZero)
{
    MockGraph mockGraph;

    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_dummyHandle, mockGraph);

    EXPECT_EQ(workspaceSize, 0u);
}
