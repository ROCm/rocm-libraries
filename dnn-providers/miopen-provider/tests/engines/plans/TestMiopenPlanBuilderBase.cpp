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
// Minimal concrete plan builder that doesn't override getMaxWorkspaceSize()
// to test the default implementation
class SimplePlanBuilder : public MiopenPlanBuilderBase
{
public:
    bool isApplicable([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                      [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph)
        const override
    {
        return true;
    }

    void buildPlan(
        [[maybe_unused]] const HipdnnEnginePluginHandle& handle,
        [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        [[maybe_unused]] HipdnnEnginePluginExecutionContext& executionContext) const override
    {
    }

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                       [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IGraph&
                           opGraph) const override
    {
        return {};
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
