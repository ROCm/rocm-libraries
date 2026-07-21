// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace hipblaslt_plugin
{

class IPlanBuilder
{
public:
    virtual ~IPlanBuilder() = default;

    virtual bool
        isApplicable(const HipdnnEnginePluginHandle& handle,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;

    virtual size_t
        getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                         const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;

    virtual void buildPlan(const HipdnnEnginePluginHandle& handle,
                           const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                           HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;
};
}
