// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

namespace hipdnn_plugin
{
inline hipdnnPluginConstData_t
    createValidConstDataGraph(flatbuffers::DetachedBuffer& serializedGraph)
{
    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = serializedGraph.data();
    opGraph.size = serializedGraph.size();
    return opGraph;
}

inline hipdnnPluginConstData_t
    createValidConstDataEngineDetails(flatbuffers::DetachedBuffer& serializedEngineDetails)
{
    hipdnnPluginConstData_t engineDetails;
    engineDetails.ptr = serializedEngineDetails.data();
    engineDetails.size = serializedEngineDetails.size();
    return engineDetails;
}

inline hipdnnPluginConstData_t
    createValidConstDataEngineConfig(flatbuffers::DetachedBuffer& serializedEngineConfig)
{
    hipdnnPluginConstData_t engineConfig;
    engineConfig.ptr = serializedEngineConfig.data();
    engineConfig.size = serializedEngineConfig.size();
    return engineConfig;
}

}
