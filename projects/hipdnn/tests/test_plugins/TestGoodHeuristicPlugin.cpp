// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

namespace
{
// NOLINTBEGIN(readability-identifier-naming)
const char* POLICY_NAME = "TestGoodHeuristicPolicy";
const char* PLUGIN_VERSION = "1.0.0";

hipdnnCallback_t g_loggingCallback = nullptr;
hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO;
// NOLINTEND(readability-identifier-naming)

// Simple handle implementation
struct HeuristicHandleImpl
{
    int handleId;
    bool devicePropsSet{false};
};

// Simple policy descriptor implementation
struct PolicyDescriptorImpl
{
    std::vector<int64_t> inputEngineIds;
    std::vector<uint8_t> serializedGraph;
    std::vector<int64_t> sortedEngineIds;
    bool finalized{false};
};

} // anonymous namespace

// NOLINTBEGIN(readability-identifier-naming)
extern "C" {

// ========== Base Plugin API (PluginApi.h) ==========

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                                "name pointer is null");
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *name = POLICY_NAME;
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
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

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
    g_loggingCallback = callback;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

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

    auto* handle = new HeuristicHandleImpl{42, false};
    *out_handle = reinterpret_cast<hipdnnHeuristicHandle_t>(handle);
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

    auto* impl = reinterpret_cast<HeuristicHandleImpl*>(handle);
    delete impl;
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

    auto* impl = reinterpret_cast<HeuristicHandleImpl*>(handle);
    impl->devicePropsSet = true;
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

    auto* desc = new PolicyDescriptorImpl{};
    *out_descriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc);
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

    auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);
    delete impl;
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

    auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);
    impl->inputEngineIds.assign(engine_ids, engine_ids + engine_id_count);
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

    auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);
    const auto* bytes = static_cast<const uint8_t*>(serialized_graph->ptr);
    impl->serializedGraph.assign(bytes, bytes + serialized_graph->size);
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

    auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);

    // Simple policy: reverse the input order
    impl->sortedEngineIds = impl->inputEngineIds;
    std::reverse(impl->sortedEngineIds.begin(), impl->sortedEngineIds.end());

    impl->finalized = true;
    *applied = 1; // Policy applied

    const hipdnnSeverity_t level = g_logLevel;
    if(g_loggingCallback != nullptr)
    {
        g_loggingCallback(level, "TestGoodHeuristicPlugin: policy finalized");
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicPolicyGetSortedEngineIds(
    hipdnnHeuristicPolicyDescriptor_t descriptor, int64_t* engine_ids, size_t* count)
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

    auto* impl = reinterpret_cast<PolicyDescriptorImpl*>(descriptor);

    if(engine_ids == nullptr)
    {
        // Query mode: return count only
        *count = impl->sortedEngineIds.size();
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    // Retrieve mode: copy IDs
    const size_t numToCopy = std::min(*count, impl->sortedEngineIds.size());
    std::memcpy(engine_ids, impl->sortedEngineIds.data(), numToCopy * sizeof(int64_t));
    *count = numToCopy;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // extern "C"
// NOLINTEND(readability-identifier-naming)
