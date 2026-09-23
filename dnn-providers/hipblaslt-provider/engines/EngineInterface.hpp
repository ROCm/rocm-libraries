// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipblaslt_plugin
{

class IEngine
{
public:
    virtual ~IEngine() = default;

    virtual int64_t id() const = 0;

    virtual bool
        isApplicable(HipdnnEnginePluginHandle& handle,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;
    virtual void getDetails(HipdnnEnginePluginHandle& handle,
                            hipdnnPluginConstData_t& detailsOut) const
        = 0;

    virtual size_t
        getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                         const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;

    virtual void initializeExecutionContext(
        const HipdnnEnginePluginHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;
};

} // namespace hipblaslt_plugin
