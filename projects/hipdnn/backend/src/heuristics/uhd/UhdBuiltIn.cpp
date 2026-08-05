// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file UhdBuiltIn.cpp
 * @brief Backend-internal implementation of the SelectionHeuristic::UHD policy.
 *
 * The Universal Heuristic Descriptor (UHD) policy is a data-driven kernel
 * selection model that ranks engines based on learned models (GBDT). It uses
 * features extracted from device properties, kernel metadata, and query
 * attributes to predict performance and sort engines by predicted score.
 *
 * This implementation follows the same C ABI pattern as StaticOrderingBuiltIn.
 * When RFC 0017 (UKD/UED) is implemented, this will integrate with the kernel
 * descriptor system to get per-engine UHD metadata.
 */

#include "UhdBuiltIn.hpp"

#include "EngineRegistry.hpp"
#include "FeatureExtractor.hpp"
#include "SelectionEngine.hpp"
#include "UhdModelCache.hpp"
#include "adapters/IUhdAdapter.hpp"
#include "adapters/StaticOrderAdapter.hpp"
#include "adapters/TreeDataAdapter.hpp"
#include "heuristics/BuiltInLogging.hpp"
#include "logging/Logging.hpp"

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/uhd_generated.h>
#include <hipdnn_plugin_sdk/HeuristicValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* PLUGIN_NAME = "BuiltInUHDHeuristic";
constexpr const char* PLUGIN_VERSION = "1.0.0";
constexpr const char* POLICY_NAME = "SelectionHeuristic::UHD";

hipdnnCallback_t gLoggingCallback = nullptr;
hipdnnSeverity_t gLogLevel = HIPDNN_SEV_INFO;

#define UHD_LOG(severity, ...) \
    HIPDNN_BUILTIN_HEURISTIC_LOG(gLoggingCallback, gLogLevel, severity, "[BuiltInUHD] ", __VA_ARGS__)

int64_t policyId()
{
    static const int64_t s_id = hipdnn_data_sdk::utilities::policyNameToId(POLICY_NAME);
    return s_id;
}

// Per-handle state. Stores parsed device properties for feature extraction.
struct Handle
{
    std::vector<uint8_t> devicePropertiesBuffer;
    std::unique_ptr<fb::DevicePropertiesT> deviceProperties;
    bool devicePropertiesSet = false;
};

// Per-policy-descriptor state.
struct PolicyDescriptor
{
    Handle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
    std::vector<uint8_t> serializedGraph;
    std::vector<int64_t> sortedEngineIds;
    bool finalized = false;

    explicit PolicyDescriptor(Handle* h)
        : handle(h)
    {
    }
};

// ---- Base plugin metadata --------------------------------------------------

hipdnnPluginStatus_t getName(const char** name)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, UHD_LOG, "getName: null output pointer");
    *name = PLUGIN_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(version, UHD_LOG, "getVersion: null output pointer");
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getApiVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(version, UHD_LOG, "getApiVersion: null output pointer");
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getType(hipdnnPluginType_t* type)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(type, UHD_LOG, "getType: null output pointer");
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback)
{
    gLoggingCallback = callback;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level)
{
    gLogLevel = level;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void getLastErrorString(const char** errorStr)
{
    if(errorStr == nullptr)
    {
        return;
    }
    *errorStr = "No error information available";
}

// ---- Policy enumeration ----------------------------------------------------

hipdnnPluginStatus_t
    getAllPolicyIds(int64_t* policyIds, uint32_t maxPolicies, uint32_t* numPolicies)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(numPolicies, UHD_LOG, "getAllPolicyIds: null num_policies");

    constexpr uint32_t TOTAL_POLICIES = 1;
    *numPolicies = TOTAL_POLICIES;
    if(policyIds == nullptr || maxPolicies == 0)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    if(maxPolicies < TOTAL_POLICIES)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    policyIds[0] = policyId();
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getPolicyName(int64_t id, const char** name)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, UHD_LOG, "getPolicyName: null output pointer");
    if(id != policyId())
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "getPolicyName: unknown policy ID");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = POLICY_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Handle lifecycle ------------------------------------------------------

hipdnnPluginStatus_t handleCreate(hipdnnHeuristicHandle_t* outHandle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(outHandle, UHD_LOG, "handleCreate: null output pointer");
    try
    {
        auto h = std::make_unique<Handle>();
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(h.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "handleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t handleDestroy(hipdnnHeuristicHandle_t handle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, UHD_LOG, "handleDestroy: null handle");
    delete reinterpret_cast<Handle*>(handle);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t handleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                               const hipdnnPluginConstData_t* devicePropsSerialized)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, UHD_LOG, "handleSetDeviceProperties: null handle");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(
        devicePropsSerialized, true, UHD_LOG, "handleSetDeviceProperties: invalid buffer");
    try
    {
        auto* h = reinterpret_cast<Handle*>(handle);
        const auto* data = reinterpret_cast<const uint8_t*>(devicePropsSerialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + devicePropsSerialized->size);

        // Verify and parse device properties
        flatbuffers::Verifier verifier(h->devicePropertiesBuffer.data(),
                                       h->devicePropertiesBuffer.size());
        if(!fb::VerifyDevicePropertiesBuffer(verifier))
        {
            UHD_LOG(HIPDNN_SEV_ERROR, "handleSetDeviceProperties: invalid DeviceProperties buffer");
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
        }

        const auto* props = fb::GetDeviceProperties(h->devicePropertiesBuffer.data());
        h->deviceProperties = std::unique_ptr<fb::DevicePropertiesT>(props->UnPack());
        h->devicePropertiesSet = true;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "handleSetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Policy descriptor lifecycle ------------------------------------------

hipdnnPluginStatus_t policyDescriptorCreate(hipdnnHeuristicHandle_t pluginHandle,
                                            int64_t id,
                                            hipdnnHeuristicPolicyDescriptor_t* outDesc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(pluginHandle, UHD_LOG, "policyDescriptorCreate: null handle");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(outDesc, UHD_LOG, "policyDescriptorCreate: null output pointer");
    if(id != policyId())
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate: unknown policy ID");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    try
    {
        auto desc = std::make_unique<PolicyDescriptor>(reinterpret_cast<Handle*>(pluginHandle));
        *outDesc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, UHD_LOG, "policyDescriptorDestroy: null descriptor");
    delete reinterpret_cast<PolicyDescriptor*>(desc);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Policy inputs ---------------------------------------------------------

hipdnnPluginStatus_t policySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                        const int64_t* engineIds,
                                        size_t engineIdCount)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, UHD_LOG, "policySetEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_ARRAY(
        engineIds, engineIdCount, UHD_LOG, "policySetEngineIds: null engine_ids with count > 0");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engineIds, engineIds + engineIdCount);
        d->finalized = false;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "policySetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                              const hipdnnPluginConstData_t* serializedGraph)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, UHD_LOG, "policySetSerializedGraph: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(
        serializedGraph, false, UHD_LOG, "policySetSerializedGraph: invalid graph buffer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(serializedGraph != nullptr && serializedGraph->ptr != nullptr && serializedGraph->size > 0)
        {
            const auto* data = reinterpret_cast<const uint8_t*>(serializedGraph->ptr);
            d->serializedGraph.assign(data, data + serializedGraph->size);
        }
        else
        {
            d->serializedGraph.clear();
        }
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "policySetSerializedGraph failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Selection -------------------------------------------------------------

/// Extract device variables from parsed DevicePropertiesT.
FeatureExtractionContext::ValueMap extractDeviceVars(const fb::DevicePropertiesT* props)
{
    FeatureExtractionContext::ValueMap deviceVars;
    if(props == nullptr)
    {
        return deviceVars;
    }

    // Map actual schema fields to feature names
    deviceVars["device_id"] = static_cast<int64_t>(props->device_id);
    deviceVars["cu_count"] = static_cast<int64_t>(props->multi_processor_count);
    deviceVars["multi_processor_count"] = static_cast<int64_t>(props->multi_processor_count);
    deviceVars["total_global_mem"] = static_cast<int64_t>(props->total_global_mem);
    if(!props->architecture_name.empty())
    {
        deviceVars["architecture_name"] = props->architecture_name;
    }

    return deviceVars;
}

hipdnnPluginStatus_t policyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, UHD_LOG, "policyFinalize: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(outApplied, UHD_LOG, "policyFinalize: null output pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(d->candidateEngineIds.empty())
        {
            UHD_LOG(HIPDNN_SEV_WARN, "policyFinalize: no candidate engines");
            *outApplied = 0;
            d->finalized = true;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Extract device properties from handle
        FeatureExtractionContext::ValueMap deviceVars;
        if(d->handle != nullptr && d->handle->devicePropertiesSet)
        {
            deviceVars = extractDeviceVars(d->handle->deviceProperties.get());
        }

        // TODO(RFC-0017): Extract query vars from serialized graph.
        // For now, use empty query vars — the graph parsing will come with UMD.
        const FeatureExtractionContext::ValueMap queryVars;

        // Check if any candidate engines are registered in the mock registry.
        // This allows the selection flow to work when engines are registered,
        // while falling back gracefully when they are not.
        auto& registry = EngineRegistry::instance();
        bool anyEngineRegistered = false;
        for(const auto engineId : d->candidateEngineIds)
        {
            if(registry.hasEngine(engineId))
            {
                anyEngineRegistered = true;
                break;
            }
        }

        if(!anyEngineRegistered)
        {
            // No engines in registry — decline so StaticOrdering handles it.
            // This is the expected path until RFC 0017 populates the registry.
            UHD_LOG(HIPDNN_SEV_INFO,
                    "policyFinalize: no candidate engines in UHD registry, declining");
            *outApplied = 0;
            d->finalized = true;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Run selection for each registered engine and collect results.
        // For kernel-level selection, we rank kernels within each engine.
        // For engine-level selection (RFC §12), we would compare best scores
        // across engines — that's a future enhancement.
        d->sortedEngineIds.clear();
        bool anyApplied = false;

        for(const auto engineId : d->candidateEngineIds)
        {
            if(!registry.hasEngine(engineId))
            {
                // Engine not in registry — skip (will be handled by fallback)
                continue;
            }

            auto result = SelectionEngine::select(engineId, deviceVars, queryVars);

            if(result.applied)
            {
                anyApplied = true;
                // For now, add the engine ID if selection succeeded.
                // The sorted kernel IDs are stored in the result but we
                // don't have a place to return them in this API yet.
                // TODO(RFC-0017): Extend API to return per-engine kernel ranking.
                d->sortedEngineIds.push_back(engineId);

                UHD_LOG(HIPDNN_SEV_INFO,
                        "policyFinalize: engine %lld selection applied, %zu kernels ranked",
                        static_cast<long long>(engineId),
                        result.sortedKernelIds.size());

                if(result.bestScore.has_value())
                {
                    UHD_LOG(HIPDNN_SEV_INFO,
                            "policyFinalize: engine %lld best kernel=%lld score=%f",
                            static_cast<long long>(engineId),
                            static_cast<long long>(*result.bestKernelId),
                            *result.bestScore);
                }
            }
            else
            {
                UHD_LOG(HIPDNN_SEV_WARN,
                        "policyFinalize: engine %lld selection failed: %s",
                        static_cast<long long>(engineId),
                        result.fallbackReason.c_str());
            }
        }

        *outApplied = anyApplied ? 1 : 0;
        d->finalized = true;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "policyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                              int64_t* engineIds,
                                              size_t* numEngines)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, UHD_LOG, "policyGetSortedEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numEngines, UHD_LOG, "policyGetSortedEngineIds: null num_engines pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(!d->finalized)
        {
            UHD_LOG(HIPDNN_SEV_ERROR, "policyGetSortedEngineIds: descriptor not finalized");
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }
        if(engineIds == nullptr)
        {
            *numEngines = d->sortedEngineIds.size();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
        *numEngines = std::min(*numEngines, d->sortedEngineIds.size());
        std::copy_n(d->sortedEngineIds.begin(), *numEngines, engineIds);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "policyGetSortedEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // namespace

hipdnn_backend::plugin::HeuristicPluginFunctionTable populateFunctionTable()
{
    hipdnn_backend::plugin::HeuristicPluginFunctionTable funcs{};
    funcs.getName = &getName;
    funcs.getVersion = &getVersion;
    funcs.getApiVersion = &getApiVersion;
    funcs.getType = &getType;
    funcs.setLoggingCallback = &setLoggingCallback;
    funcs.setLogLevel = &setLogLevel;
    funcs.getLastErrorString = &getLastErrorString;
    funcs.getAllPolicyIds = &getAllPolicyIds;
    funcs.getPolicyName = &getPolicyName;
    funcs.handleCreate = &handleCreate;
    funcs.handleDestroy = &handleDestroy;
    funcs.handleSetDeviceProperties = &handleSetDeviceProperties;
    funcs.policyDescriptorCreate = &policyDescriptorCreate;
    funcs.policyDescriptorDestroy = &policyDescriptorDestroy;
    funcs.policySetEngineIds = &policySetEngineIds;
    funcs.policySetSerializedGraph = &policySetSerializedGraph;
    funcs.policyFinalize = &policyFinalize;
    funcs.policyGetSortedEngineIds = &policyGetSortedEngineIds;
    return funcs;
}

} // namespace hipdnn_backend::heuristics::uhd
