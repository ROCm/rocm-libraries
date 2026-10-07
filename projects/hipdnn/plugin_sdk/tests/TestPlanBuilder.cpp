// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>

using namespace hipdnn_plugin_sdk;

namespace
{

struct CatalogHandle
{
};

struct CatalogSettings
{
};

struct CatalogContext
{
};

using BasePlanBuilder = IPlanBuilder<CatalogHandle, CatalogSettings, CatalogContext>;

/// Implements only what IPlanBuilder requires, so getKnobCatalog() is the default.
class MinimalPlanBuilder : public BasePlanBuilder
{
public:
    bool isApplicable(
        const CatalogHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const override
    {
        return true;
    }

    size_t
        getMaxWorkspaceSize(const CatalogHandle& /*handle*/,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                            const CatalogSettings& /*executionSettings*/) const override
    {
        return 0;
    }

    void initializeExecutionSettings(
        const CatalogHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
        CatalogSettings& /*executionSettings*/) const override
    {
    }

    void buildPlan(
        const CatalogHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
        CatalogContext& /*executionContext*/) const override
    {
    }

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const CatalogHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const override
    {
        return {};
    }
};

} // namespace

// The status is what lets the host tell "no catalog here" from a failure and apply its fallback.
TEST(TestPlanBuilder, DefaultKnobCatalogDeclinesWithNotApplicable)
{
    const MinimalPlanBuilder builder;
    const hipdnn_test_sdk::utilities::MockGraph graph;

    try
    {
        static_cast<void>(builder.getKnobCatalog(CatalogHandle{}, graph));
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE);
    }
}
