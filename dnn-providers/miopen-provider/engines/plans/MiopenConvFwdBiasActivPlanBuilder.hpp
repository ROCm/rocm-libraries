// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlanBuilderInterface.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace miopen_plugin
{

class MiopenConvFwdBiasActivPlanBuilder : public IPlanBuilder
{
public:
    MiopenConvFwdBiasActivPlanBuilder() = default;
    ~MiopenConvFwdBiasActivPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenConvFwdBiasActivPlanBuilder(const MiopenConvFwdBiasActivPlanBuilder&) = delete;
    MiopenConvFwdBiasActivPlanBuilder& operator=(const MiopenConvFwdBiasActivPlanBuilder&) = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_plugin_sdk::IGraph& opGraph) const override;
    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            const hipdnn_plugin_sdk::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_plugin_sdk::IGraph& opGraph,
                   const hipdnn_plugin_sdk::IEngineConfig& engineConfig,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnEnginePluginHandle& handle,
                       const hipdnn_plugin_sdk::IGraph& opGraph) const override;
};

}
