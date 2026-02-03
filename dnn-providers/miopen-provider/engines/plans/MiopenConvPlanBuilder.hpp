// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MiopenPlanBuilderBase.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace miopen_plugin
{

class MiopenConvPlanBuilder : public MiopenPlanBuilderBase
{
public:
    MiopenConvPlanBuilder() = default;
    ~MiopenConvPlanBuilder() override = default;

    // Disallow copy and assignment
    MiopenConvPlanBuilder(const MiopenConvPlanBuilder&) = delete;
    MiopenConvPlanBuilder& operator=(const MiopenConvPlanBuilder&) = delete;

    bool isApplicable(const HipdnnEnginePluginHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    WorkspaceSizeRange getWorkspaceSizeRange(const HipdnnEnginePluginHandle& handle,
                                             const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
    size_t getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
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
