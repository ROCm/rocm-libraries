// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_plugin_sdk
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
