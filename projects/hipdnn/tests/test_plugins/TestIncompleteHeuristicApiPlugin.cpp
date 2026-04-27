// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

/**
 * This plugin intentionally OMITS required symbols to test error handling
 * in HeuristicPlugin::resolveSymbols().
 *
 * Missing symbols:
 * - hipdnnPluginGetName (required)
 * - hipdnnHeuristicPolicyFinalize (required)
 * - hipdnnHeuristicPolicyGetSortedEngineIds (required)
 *
 * This should cause GET_REQUIRED_SYMBOL to throw HipdnnException with a
 * detailed error message about incomplete ABI implementation.
 */

// NOLINTBEGIN(readability-identifier-naming)
extern "C" {

// ========== Module Metadata (Partial) ==========

hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *version = "0.1.0-incomplete";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    (void)callback;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(error_str != nullptr)
    {
        *error_str = "Incomplete plugin - missing required symbols";
    }
}

// ========== Handle Lifecycle (Partial) ==========

hipdnnPluginStatus_t hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle)
{
    if(out_handle == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    *out_handle = reinterpret_cast<hipdnnHeuristicHandle_t>(0x1234);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    (void)handle;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicHandleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                             const hipdnnPluginConstData_t* devicePropsSerialized)
{
    (void)handle;
    (void)devicePropsSerialized;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ========== Policy Descriptor Lifecycle (Partial) ==========

hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorCreate(hipdnnHeuristicHandle_t handle,
                                          hipdnnHeuristicPolicyDescriptor_t* out_descriptor)
{
    (void)handle;
    if(out_descriptor == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    *out_descriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0x5678);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t descriptor)
{
    (void)descriptor;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ========== Policy Inputs (Partial) ==========

hipdnnPluginStatus_t hipdnnHeuristicPolicySetEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                                       const int64_t* engine_ids,
                                                       size_t engine_id_count)
{
    (void)descriptor;
    (void)engine_ids;
    (void)engine_id_count;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                            const hipdnnPluginConstData_t* serialized_graph)
{
    (void)descriptor;
    (void)serialized_graph;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ========== MISSING REQUIRED SYMBOLS ==========
// hipdnnPluginGetName - NOT IMPLEMENTED
// hipdnnHeuristicPolicyFinalize - NOT IMPLEMENTED
// hipdnnHeuristicPolicyGetSortedEngineIds - NOT IMPLEMENTED

} // extern "C"
// NOLINTEND(readability-identifier-naming)
