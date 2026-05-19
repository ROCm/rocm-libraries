// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file StaticOrderingBuiltIn.cpp
 * @brief Backend-internal implementation of the
 *        SelectionHeuristic::StaticOrdering policy.
 *
 * The policy wraps utilities::sortEngineIds — the legacy MIOPEN_ENGINE-first /
 * MIOPEN_ENGINE_DETERMINISTIC-last fallback ordering — and exposes it through
 * the heuristic plugin C ABI shape. Functions live in an unnamed namespace
 * inside a backend translation unit: they are *not* exported as
 * hipdnnHeuristic* symbols, but their signatures match the C ABI exactly so a
 * HeuristicPluginFunctionTable can dispatch through them just like a loaded
 * plugin's dlsym'd entry points. The wrapper layer (HeuristicPlugin) does not
 * distinguish dlopen plugins from built-ins.
 */

#include "StaticOrderingBuiltIn.hpp"

#include "heuristics/BuiltInLogging.hpp"
#include "logging/Logging.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/HeuristicValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace hipdnn_backend::heuristics::static_ordering
{
namespace
{

constexpr const char* PLUGIN_NAME = "BuiltInStaticOrderingHeuristic";
constexpr const char* PLUGIN_VERSION = "1.0.0";
constexpr const char* POLICY_NAME = "SelectionHeuristic::StaticOrdering";

// File-scope logging callback / level, set via the C-ABI-shaped
// SetLoggingCallback / SetLogLevel below. The backend supplies its own
// callback when registering the built-in (see PluginManagerBase::registerPlugin)
// so log lines from this module flow through the backend logger.
//
// Identity contract: the built-in is statically linked into the backend, so
// these globals live in the same process image as the caller. The last writer
// wins — if multiple HeuristicPluginManager instances register the built-in
// they overwrite each other's callback. This is intentional: registerPlugin()
// hands in a callback that forwards to the backend logger, which is itself a
// process-wide sink, so the identity of the "current" callback does not matter
// as long as one is installed. Do not assume per-instance scoping here.
hipdnnCallback_t g_loggingCallback = nullptr; // NOLINT(readability-identifier-naming)
hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO; // NOLINT(readability-identifier-naming)

#define STATIC_ORDERING_LOG(severity, ...) \
    HIPDNN_BUILTIN_HEURISTIC_LOG(          \
        g_loggingCallback, g_logLevel, severity, "[BuiltInStaticOrdering] ", __VA_ARGS__)

int64_t policyId()
{
    static const int64_t s_id = hipdnn_data_sdk::utilities::policyNameToId(POLICY_NAME);
    return s_id;
}

constexpr const char* FALLBACK_ORDERING_ENV = "HIPDNN_HEUR_FALLBACK_ENGINE_ORDER";

/// Parse HIPDNN_HEUR_FALLBACK_ENGINE_ORDER (comma-separated engine names) into a
/// list of engine IDs in the order the user wrote them. Empty / unset env →
/// empty vector (caller falls back to the legacy sortEngineIds ordering).
/// Blank tokens are skipped; unknown engine names hash to a deterministic ID
/// via engineNameToId — the caller filters against the candidate list, so a
/// typo'd name simply won't match anything.
std::vector<int64_t> parseFallbackOrderingEnv()
{
    const std::string raw = hipdnn_data_sdk::utilities::getEnv(FALLBACK_ORDERING_ENV, "");
    if(hipdnn_data_sdk::utilities::trim(raw).empty())
    {
        return {};
    }

    std::vector<int64_t> ids;
    std::stringstream stream(raw);
    std::string token;
    while(std::getline(stream, token, ','))
    {
        const std::string name = hipdnn_data_sdk::utilities::trim(token);
        if(name.empty())
        {
            continue;
        }
        ids.push_back(hipdnn_data_sdk::utilities::engineNameToId(name));
    }
    return ids;
}

/// Restrict @p candidates to engines named in @p envOrder, preserving the env
/// order. Engines not listed in the env are dropped — when the operator sets
/// HIPDNN_HEUR_FALLBACK_ENGINE_ORDER they are explicitly opting out of every
/// other engine. Names in the env that are not in @p candidates are silently
/// skipped (the policy loop only sees engines the rest of the stack already
/// filtered down to).
std::vector<int64_t> applyFallbackOrdering(const std::vector<int64_t>& candidates,
                                           const std::vector<int64_t>& envOrder)
{
    const std::unordered_set<int64_t> candidateSet(candidates.begin(), candidates.end());
    std::vector<int64_t> out;
    out.reserve(envOrder.size());
    for(const int64_t id : envOrder)
    {
        if(candidateSet.count(id) != 0U)
        {
            out.push_back(id);
        }
    }
    return out;
}

// Per-handle state. StaticOrdering does not consume device properties, but the
// C ABI requires a working SetDeviceProperties entry point.
struct Handle
{
    std::vector<uint8_t> devicePropertiesBuffer;
    bool devicePropertiesSet = false;
};

// Per-policy-descriptor state.
struct PolicyDescriptor
{
    Handle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
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
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, STATIC_ORDERING_LOG, "getName: null output pointer");
    *name = PLUGIN_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(version, STATIC_ORDERING_LOG, "getVersion: null output pointer");
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getApiVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        version, STATIC_ORDERING_LOG, "getApiVersion: null output pointer");
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getType(hipdnnPluginType_t* type)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(type, STATIC_ORDERING_LOG, "getType: null output pointer");
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback)
{
    g_loggingCallback = callback;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
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
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numPolicies, STATIC_ORDERING_LOG, "getAllPolicyIds: null num_policies");

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
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, STATIC_ORDERING_LOG, "getPolicyName: null output pointer");
    if(id != policyId())
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "getPolicyName: unknown policy ID");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = POLICY_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Handle lifecycle ------------------------------------------------------

hipdnnPluginStatus_t handleCreate(hipdnnHeuristicHandle_t* outHandle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outHandle, STATIC_ORDERING_LOG, "handleCreate: null output pointer");
    try
    {
        auto h = std::make_unique<Handle>();
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(h.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "handleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t handleDestroy(hipdnnHeuristicHandle_t handle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, STATIC_ORDERING_LOG, "handleDestroy: null handle");
    delete reinterpret_cast<Handle*>(handle);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t handleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                               const hipdnnPluginConstData_t* devicePropsSerialized)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        handle, STATIC_ORDERING_LOG, "handleSetDeviceProperties: null handle");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(devicePropsSerialized,
                                     true,
                                     STATIC_ORDERING_LOG,
                                     "handleSetDeviceProperties: invalid buffer");
    try
    {
        auto* h = reinterpret_cast<Handle*>(handle);
        const auto* data = reinterpret_cast<const uint8_t*>(devicePropsSerialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + devicePropsSerialized->size);
        h->devicePropertiesSet = true;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "handleSetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Policy descriptor lifecycle ------------------------------------------

hipdnnPluginStatus_t policyDescriptorCreate(hipdnnHeuristicHandle_t pluginHandle,
                                            int64_t id,
                                            hipdnnHeuristicPolicyDescriptor_t* outDesc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        pluginHandle, STATIC_ORDERING_LOG, "policyDescriptorCreate: null handle");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outDesc, STATIC_ORDERING_LOG, "policyDescriptorCreate: null output pointer");
    if(id != policyId())
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate: unknown policy ID");
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
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, STATIC_ORDERING_LOG, "policyDescriptorDestroy: null descriptor");
    delete reinterpret_cast<PolicyDescriptor*>(desc);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Policy inputs ---------------------------------------------------------

hipdnnPluginStatus_t policySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                        const int64_t* engineIds,
                                        size_t engineIdCount)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, STATIC_ORDERING_LOG, "policySetEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_ARRAY(engineIds,
                                engineIdCount,
                                STATIC_ORDERING_LOG,
                                "policySetEngineIds: null engine_ids with count > 0");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engineIds, engineIds + engineIdCount);
        d->finalized = false;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "policySetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                              const hipdnnPluginConstData_t* serializedGraph)
{
    // StaticOrdering ignores the serialized graph; only validate args.
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, STATIC_ORDERING_LOG, "policySetSerializedGraph: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(serializedGraph,
                                     false,
                                     STATIC_ORDERING_LOG,
                                     "policySetSerializedGraph: invalid graph buffer");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Selection -------------------------------------------------------------

hipdnnPluginStatus_t policyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, STATIC_ORDERING_LOG, "policyFinalize: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outApplied, STATIC_ORDERING_LOG, "policyFinalize: null output pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(d->candidateEngineIds.empty())
        {
            STATIC_ORDERING_LOG(HIPDNN_SEV_WARN, "policyFinalize: no candidate engines");
            *outApplied = 0;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        // HIPDNN_HEUR_FALLBACK_ENGINE_ORDER, when set, replaces the legacy
        // MIOPEN-first / DETERMINISTIC-last ordering. Only engines named in
        // the env are eligible — operators use this to constrain selection
        // to a known-good shortlist. If the env is set but no listed engine
        // is among the candidates the policy declines (outApplied = 0) so
        // the policy loop can try the next plugin.
        const auto envOrder = parseFallbackOrderingEnv();
        if(!envOrder.empty())
        {
            d->sortedEngineIds = applyFallbackOrdering(d->candidateEngineIds, envOrder);
            if(d->sortedEngineIds.empty())
            {
                STATIC_ORDERING_LOG(HIPDNN_SEV_WARN,
                                    "policyFinalize: HIPDNN_HEUR_FALLBACK_ENGINE_ORDER listed no "
                                    "engines that are candidates; declining.");
                *outApplied = 0;
                return HIPDNN_PLUGIN_STATUS_SUCCESS;
            }
            d->finalized = true;
            *outApplied = 1;
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        d->sortedEngineIds = d->candidateEngineIds;
        hipdnn_data_sdk::utilities::sortEngineIds(d->sortedEngineIds);
        d->finalized = true;
        *outApplied = 1;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "policyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                              int64_t* engineIds,
                                              size_t* numEngines)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, STATIC_ORDERING_LOG, "policyGetSortedEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numEngines, STATIC_ORDERING_LOG, "policyGetSortedEngineIds: null num_engines pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(!d->finalized)
        {
            STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR,
                                "policyGetSortedEngineIds: descriptor not finalized");
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
        STATIC_ORDERING_LOG(HIPDNN_SEV_ERROR, "policyGetSortedEngineIds failed: %s", e.what());
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

} // namespace hipdnn_backend::heuristics::static_ordering
