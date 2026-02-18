// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <vector>

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "HipdnnEngineSpecificSettings.hpp"

namespace miopen_plugin
{

class MiopenBatchnormPlanBuilder : public hipdnn_plugin_sdk::IPlanBuilder
{
public:
    MiopenBatchnormPlanBuilder() = default;
    ~MiopenBatchnormPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenBatchnormPlanBuilder(const MiopenBatchnormPlanBuilder&) = delete;
    MiopenBatchnormPlanBuilder& operator=(const MiopenBatchnormPlanBuilder&) = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    size_t
        getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                            const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const HipdnnEngineSpecificSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipdnnEngineSpecificSettings& executionSettings) const override;

    void buildPlan(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipdnnEnginePluginExecutionContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnEnginePluginHandle& handle,
                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

}
