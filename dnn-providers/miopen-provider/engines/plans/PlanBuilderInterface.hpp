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

namespace miopen_plugin
{

class IPlanBuilder
{
public:
    virtual ~IPlanBuilder() = default;

    virtual bool isApplicable(const HipdnnEnginePluginHandle& handle,
                              const hipdnn_plugin_sdk::IGraph& opGraph) const
        = 0;

    virtual size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin_sdk::IGraph& opGraph) const
        = 0;

    virtual void buildPlan(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_plugin_sdk::IGraph& opGraph,
                           [[maybe_unused]] const hipdnn_plugin_sdk::IEngineConfig& engineConfig,
                           HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;

    virtual std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnEnginePluginHandle& handle,
                       const hipdnn_plugin_sdk::IGraph& opGraph) const
        = 0;
};
}
