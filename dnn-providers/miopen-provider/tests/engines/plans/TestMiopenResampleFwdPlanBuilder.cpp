// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "engines/plans/MiopenResampleFwdPlanBuilder.hpp"

using namespace miopen_plugin;

using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using hipdnn_test_sdk::utilities::createResampleFwdGraph;
using hipdnn_test_sdk::utilities::MockEngineConfig;
using hipdnn_test_sdk::utilities::ResampleFwdGraphSpec;

namespace
{

// A spec the checks decline: asymmetric padding.
ResampleFwdGraphSpec unsupportedSpec()
{
    ResampleFwdGraphSpec spec;
    spec.postPadding = std::vector<int64_t>{0, 0};
    return spec;
}

} // namespace

class TestGpuMiopenResampleFwdPlanBuilder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _handle = std::make_unique<HipdnnMiopenHandle>();
    }

    bool isApplicable(const ResampleFwdGraphSpec& spec) const
    {
        auto fbb = createResampleFwdGraph(spec);
        const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());
        return _planBuilder.isApplicable(*_handle, graph);
    }

    size_t maxWorkspaceSize(const ResampleFwdGraphSpec& spec) const
    {
        auto fbb = createResampleFwdGraph(spec);
        const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());
        const HipdnnMiopenSettings settings;
        return _planBuilder.getMaxWorkspaceSize(*_handle, graph, settings);
    }

    MiopenResampleFwdPlanBuilder _planBuilder;
    MockEngineConfig _engineConfig;
    std::unique_ptr<HipdnnMiopenHandle> _handle;
};

// ============================================================================
// isApplicable
// ============================================================================

TEST_F(TestGpuMiopenResampleFwdPlanBuilder, IsApplicableDefersToTheChecks)
{
    EXPECT_TRUE(isApplicable(ResampleFwdGraphSpec{}));
    EXPECT_FALSE(isApplicable(unsupportedSpec()));
}

TEST_F(TestGpuMiopenResampleFwdPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    ResampleFwdGraphSpec spec;
    spec.overrideShapeEnabled = true;
    EXPECT_FALSE(isApplicable(spec));
}

// ============================================================================
// getMaxWorkspaceSize and buildPlan
// ============================================================================

TEST_F(TestGpuMiopenResampleFwdPlanBuilder, BuildPlanWorkspaceMatchesMaxWorkspaceSize)
{
    // Channels-last max pooling: MIOpen's workspace holds the index region and the transposed X
    // and Y, so the builder and the plan must size it from the same resolved configuration.
    ResampleFwdGraphSpec spec;
    spec.makeChannelsLast();
    auto fbb = createResampleFwdGraph(spec);
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());
    const auto expected = maxWorkspaceSize(spec);
    ASSERT_GT(expected, 0u);

    HipdnnMiopenContext context;
    ASSERT_NO_THROW(_planBuilder.buildPlan(*_handle, graph, _engineConfig, context));
    ASSERT_TRUE(context.hasValidPlan());
    EXPECT_EQ(context.plan().getWorkspaceSize(*_handle), expected);
}

TEST_F(TestGpuMiopenResampleFwdPlanBuilder, MaxWorkspaceSizeThrowsBadParamForUnsupportedGraph)
{
    try
    {
        maxWorkspaceSize(unsupportedSpec());
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    }
}

TEST_F(TestGpuMiopenResampleFwdPlanBuilder, BuildPlanThrowsBadParamForUnsupportedGraph)
{
    auto fbb = createResampleFwdGraph(unsupportedSpec());
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());

    HipdnnMiopenContext context;
    try
    {
        _planBuilder.buildPlan(*_handle, graph, _engineConfig, context);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    }
    EXPECT_FALSE(context.hasValidPlan());
}

TEST_F(TestGpuMiopenResampleFwdPlanBuilder, InitializeExecutionSettingsLeavesSettingsUntouched)
{
    auto fbb = createResampleFwdGraph(ResampleFwdGraphSpec{});
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());

    HipdnnMiopenSettings settings;
    EXPECT_NO_THROW(
        _planBuilder.initializeExecutionSettings(*_handle, graph, _engineConfig, settings));
    EXPECT_FALSE(settings.workspaceSizeLimit().has_value());
    EXPECT_FALSE(settings.defaultWorkspaceSize().has_value());
    EXPECT_FALSE(settings.benchmarkingEnabled());
}

TEST_F(TestGpuMiopenResampleFwdPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    auto fbb = createResampleFwdGraph(ResampleFwdGraphSpec{});
    const GraphWrapper graph(fbb.GetBufferPointer(), fbb.GetSize());

    EXPECT_TRUE(_planBuilder.getCustomKnobs(*_handle, graph).empty());
}
