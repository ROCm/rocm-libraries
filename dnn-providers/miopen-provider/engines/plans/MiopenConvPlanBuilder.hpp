// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace miopen_legacy_plugin
{

class MiopenConvPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder
{
public:
    MiopenConvPlanBuilder() = default;
    ~MiopenConvPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenConvPlanBuilder(const MiopenConvPlanBuilder&) = delete;
    MiopenConvPlanBuilder& operator=(const MiopenConvPlanBuilder&) = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_plugin_sdk::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                               const hipdnn_plugin_sdk::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_plugin_sdk::IGraph& opGraph,
                   [[maybe_unused]] const hipdnn_plugin_sdk::IEngineConfig& engineConfig,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnEnginePluginHandle& handle,
                       const hipdnn_plugin_sdk::IGraph& opGraph) const override;
};

}
