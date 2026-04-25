// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <cstdint>

/**
 * This plugin implements ALL required symbols but NONE of the optional symbols.
 * Used to test optional symbol handling via tryAssignSymbol().
 *
 * Missing optional symbols:
 * - hipdnnPluginSetLogLevel (optional)
 *
 * This tests the code paths where _funcSetLogLevel is nullptr.
 */

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

namespace
{
// NOLINTBEGIN(readability-identifier-naming)
const char* POLICY_NAME = "TestNoOptionalHeuristicPolicy";
// NOLINTEND(readability-identifier-naming)
} // anonymous namespace

// NOLINTBEGIN(readability-identifier-naming)
extern "C" {

// ========== Module Metadata (Required Only) ==========

hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    if(version == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "version pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetName(const char** policy_name)
{
    if(policy_name == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "policy_name pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *policy_name = POLICY_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "version pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *version = "1.0.0-no-optional";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "type pointer is null");
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

// hipdnnPluginSetLogLevel - OPTIONAL, NOT IMPLEMENTED

void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(error_str != nullptr)
    {
        *error_str = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();
    }
}

// ========== Handle Lifecycle ==========

hipdnnPluginStatus_t hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle)
{
    if(out_handle == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "out_handle pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    *out_handle = reinterpret_cast<hipdnnHeuristicHandle_t>(0xABCD);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    if(handle == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "handle is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicHandleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                             const hipdnnPluginConstData_t* devicePropsSerialized)
{
    if(handle == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "handle is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    if(devicePropsSerialized == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE, "devicePropsSerialized pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ========== Policy Descriptor Lifecycle ==========

hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorCreate(hipdnnHeuristicHandle_t handle,
                                          hipdnnHeuristicPolicyDescriptor_t* out_descriptor)
{
    if(handle == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "handle is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    if(out_descriptor == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "out_descriptor pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    *out_descriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(0xDCBA);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t descriptor)
{
    if(descriptor == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "descriptor is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ========== Policy Inputs ==========

hipdnnPluginStatus_t hipdnnHeuristicPolicySetEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                                       const int64_t* engine_ids,
                                                       size_t engine_id_count)
{
    if(descriptor == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "descriptor is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    if(engine_ids == nullptr && engine_id_count > 0)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "engine_ids is null but count > 0");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                            const hipdnnPluginConstData_t* serialized_graph)
{
    if(descriptor == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "descriptor is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    if(serialized_graph == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "serialized_graph pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ========== Selection Execution ==========

hipdnnPluginStatus_t hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                                   int32_t* applied)
{
    if(descriptor == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "descriptor is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    if(applied == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "applied pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *applied = 0; // Policy declined to apply
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                            [[maybe_unused]] int64_t* engine_ids,
                                            size_t* count)
{
    if(descriptor == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "descriptor is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    if(count == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "count pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }

    // Return empty list
    *count = 0;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // extern "C"
// NOLINTEND(readability-identifier-naming)
