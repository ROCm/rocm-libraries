// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>
#include <vector>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenExecutionSettings.hpp"

namespace miopen_plugin
{

class IPlanBuilder
{
public:
    virtual ~IPlanBuilder() = default;

    virtual bool isApplicable(const HipdnnEnginePluginHandle& handle,
                              const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;

    virtual size_t getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                                       const MiopenExecutionSettings& executionSettings) const
        = 0;

    virtual void initializeExecutionSettings(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        MiopenExecutionSettings& executionSettings) const
        = 0;

    virtual void buildPlan(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                           HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;

    virtual std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnEnginePluginHandle& handle,
                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;
};
}
