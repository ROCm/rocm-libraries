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
#include "adapters/IUhdAdapter.hpp"
#include "adapters/TreeDataAdapter.hpp"
#include "heuristics/BuiltInLogging.hpp"
#include "logging/Logging.hpp"

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/uhd_generated.h>

#include <nlohmann/json.hpp>
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

#define UHD_LOG(severity, ...)    \
    HIPDNN_BUILTIN_HEURISTIC_LOG( \
        gLoggingCallback, gLogLevel, severity, "[BuiltInUHD] ", __VA_ARGS__)

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

    // RFC 0019 §13: Selection traces per engine for observability
    std::unordered_map<int64_t, SelectionTrace> traces;
    std::unordered_map<int64_t, std::string> traceJsonCache; // Serialized JSON per engine

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
        if(serializedGraph != nullptr && serializedGraph->ptr != nullptr
           && serializedGraph->size > 0)
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
        deviceVars["arch"] = props->architecture_name;
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

        // Rank kernels within each registered engine. Engine ORDER is deliberately
        // left alone: RFC 0019 §2 puts engine selection in RFC 0007's scope, not the
        // UHD's, and this policy computes no engine-level ordering information.
        //
        // Every candidate engine is echoed back in input order, including engines
        // absent from the registry and engines whose selection produced no ranking.
        // Emitting a subset would silently delete engines from the plan:
        // SelectionHeuristic::getSortedEngineIds accepts any subset of the input, and
        // EngineHeuristicDescriptor::finalize adopts whatever comes back as the whole
        // candidate list, so a dropped engine never reaches execution.
        d->sortedEngineIds.clear();
        d->sortedEngineIds.reserve(d->candidateEngineIds.size());

        for(const auto engineId : d->candidateEngineIds)
        {
            d->sortedEngineIds.push_back(engineId);

            if(!registry.hasEngine(engineId))
            {
                // No UHD for this engine; it keeps its place and is ranked by
                // whichever policy owns engine order.
                continue;
            }

            auto result = SelectionEngine::select(engineId, deviceVars, queryVars);

            // Store trace for retrieval (RFC 0019 §13)
            d->traces[engineId] = result.trace;

            // RFC 0019 §6 step 6 requires failing *open*. hasOrdering() distinguishes
            // "selection produced a kernel ranking" from "selection produced nothing";
            // neither outcome may affect the engine's presence above.
            if(result.hasOrdering())
            {
                // The kernel ranking is computed but has nowhere to go: the heuristic
                // plugin ABI carries engine IDs only.
                // TODO(RFC-0017): Extend API to return per-engine kernel ranking.

                // Report on trace.usedModel, not `applied`. An engine with no
                // candidates completes with applied=true but never builds an adapter,
                // so reporting it as a model application would log an empty
                // model_version alongside a zero-kernel ranking.
                if(result.trace.usedModel)
                {
                    UHD_LOG(HIPDNN_SEV_INFO,
                            "policyFinalize: engine %lld selection applied, %zu kernels ranked "
                            "(uhd=%s model_version=%s adapter=%s)",
                            static_cast<long long>(engineId),
                            result.sortedKernelIds.size(),
                            result.trace.uhdId.c_str(),
                            result.trace.modelVersion.c_str(),
                            result.trace.adapterType.c_str());

                    if(result.bestScore.has_value())
                    {
                        UHD_LOG(HIPDNN_SEV_INFO,
                                "policyFinalize: engine %lld best kernel=%lld score=%f",
                                static_cast<long long>(engineId),
                                static_cast<long long>(*result.bestKernelId),
                                *result.bestScore);
                    }
                }
                else if(result.applied)
                {
                    UHD_LOG(HIPDNN_SEV_INFO,
                            "policyFinalize: engine %lld had nothing to rank (uhd=%s): %s",
                            static_cast<long long>(engineId),
                            result.trace.uhdId.c_str(),
                            result.fallbackReason.c_str());
                }
                else
                {
                    UHD_LOG(HIPDNN_SEV_WARN,
                            "policyFinalize: engine %lld degraded to static order, %zu kernels "
                            "ranked (uhd=%s adapter=%s arch=%s): %s",
                            static_cast<long long>(engineId),
                            result.sortedKernelIds.size(),
                            result.trace.uhdId.c_str(),
                            result.trace.adapterType.c_str(),
                            result.trace.deviceArch.c_str(),
                            result.fallbackReason.c_str());
                }
            }
            else
            {
                UHD_LOG(HIPDNN_SEV_WARN,
                        "policyFinalize: engine %lld produced no ranking (uhd=%s): %s",
                        static_cast<long long>(engineId),
                        result.trace.uhdId.c_str(),
                        result.fallbackReason.c_str());
            }
        }

        // Decline the engine-ordering decision.
        //
        // Returning applied=1 would make EngineHeuristicDescriptor::finalize adopt
        // this policy's output and `break` the chain, so StaticOrdering would never
        // run and its vendor precedence would be replaced by raw input order — a
        // worse ordering, asserted by a policy that computed no ordering at all.
        //
        // The kernel ranking above is the UHD's actual product, and the plugin ABI
        // cannot carry it yet. Until it can, the honest answer to "did you order these
        // engines?" is no. Flip this once the ABI returns per-engine kernel rankings
        // (TODO(RFC-0017)); the engine list is already populated correctly for it.
        *outApplied = 0;
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

hipdnnPluginStatus_t policyGetTrace(hipdnnHeuristicPolicyDescriptor_t desc,
                                    int64_t engineId,
                                    const char** traceJson)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, UHD_LOG, "policyGetTrace: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(traceJson, UHD_LOG, "policyGetTrace: null output pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(!d->finalized)
        {
            UHD_LOG(HIPDNN_SEV_ERROR, "policyGetTrace: descriptor not finalized");
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }

        // Check if trace exists for this engine
        auto traceIt = d->traces.find(engineId);
        if(traceIt == d->traces.end())
        {
            UHD_LOG(HIPDNN_SEV_WARN,
                    "policyGetTrace: no trace available for engine %lld",
                    static_cast<long long>(engineId));
            return HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE;
        }

        const SelectionTrace& trace = traceIt->second;

        // Check if JSON is already cached
        auto jsonIt = d->traceJsonCache.find(engineId);
        if(jsonIt != d->traceJsonCache.end())
        {
            *traceJson = jsonIt->second.c_str();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // Build JSON representation
        nlohmann::json j;
        if(!trace.uhdId.empty())
        {
            j["uhd_id"] = trace.uhdId;
        }
        if(!trace.modelVersion.empty())
        {
            j["model_version"] = trace.modelVersion;
        }
        if(!trace.trainingArches.empty())
        {
            j["training_arches"] = trace.trainingArches;
        }
        if(!trace.adapterType.empty())
        {
            j["adapter_type"] = trace.adapterType;
        }
        j["used_model"] = trace.usedModel;
        if(!trace.fallbackReason.empty())
        {
            j["fallback_reason"] = trace.fallbackReason;
        }
        j["arch_was_trained"] = trace.archWasTrained;
        if(!trace.deviceArch.empty())
        {
            j["device_arch"] = trace.deviceArch;
        }
        if(!trace.featuresHashModel.empty())
        {
            j["features_hash_model"] = trace.featuresHashModel;
        }
        if(!trace.featuresHashConfig.empty())
        {
            j["features_hash_config"] = trace.featuresHashConfig;
        }
        j["features_hash_match"] = trace.featuresHashMatch;

        // Cache the JSON string
        d->traceJsonCache[engineId] = j.dump();
        *traceJson                  = d->traceJsonCache[engineId].c_str();

        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        UHD_LOG(HIPDNN_SEV_ERROR, "policyGetTrace failed: %s", e.what());
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
    funcs.policyGetTrace = &policyGetTrace;
    return funcs;
}

} // namespace hipdnn_backend::heuristics::uhd
