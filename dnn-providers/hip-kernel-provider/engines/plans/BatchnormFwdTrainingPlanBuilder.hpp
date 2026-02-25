// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PlanBuilderInterface.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hip_kernel_plugin
{

class BatchnormFwdTrainingPlanBuilder : public IPlanBuilder
{
public:
    BatchnormFwdTrainingPlanBuilder() = default;
    ~BatchnormFwdTrainingPlanBuilder() override = default;

    // Disallow copy and assignment
    BatchnormFwdTrainingPlanBuilder(const BatchnormFwdTrainingPlanBuilder&) = delete;
    BatchnormFwdTrainingPlanBuilder& operator=(const BatchnormFwdTrainingPlanBuilder&) = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getWorkspaceSize(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    void buildPlan(const HipdnnEnginePluginHandle& handle,
                   const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   HipdnnEnginePluginExecutionContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnEnginePluginHandle& handle,
                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

}
