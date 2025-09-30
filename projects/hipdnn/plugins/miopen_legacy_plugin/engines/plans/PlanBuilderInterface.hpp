// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace miopen_legacy_plugin
{

class IPlanBuilder
{
public:
    virtual ~IPlanBuilder() = default;

    virtual bool isApplicable(const HipdnnEnginePluginHandle& handle,
                              const hipdnn_plugin::IGraph& opGraph) const
        = 0;

    virtual size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                    const hipdnn_plugin::IGraph& opGraph) const
        = 0;

    virtual void buildPlan(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_plugin::IGraph& opGraph,
                           HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;
};
}
