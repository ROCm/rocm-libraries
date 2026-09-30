// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HeuristicPlugin.hpp"

#include "HipdnnException.hpp"
#include "logging/Logging.hpp"
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>

#include <algorithm>
#include <utility>

namespace hipdnn_backend::plugin
{

namespace
{

// std::string_view{nullptr} is UB. Plugin code is untrusted: an out-param
// const char** may be left null on a "successful" status return. Funnel every
// plugin-supplied C string through this helper before constructing a view.
std::string_view safeStringView(const char* str) noexcept
{
    return (str != nullptr) ? std::string_view{str} : std::string_view{};
}

} // anonymous namespace

HeuristicPlugin::HeuristicPlugin(SharedLibrary&& lib)
    : _lib(std::move(lib))
    , _sourceLabel(_lib.libraryPath().string())
{
    resolveSymbols();
    validateFunctionTable();
    validatePluginMetadata(*this);
#ifndef NDEBUG
    _initialized = true;
#endif
}

HeuristicPlugin::HeuristicPlugin(HeuristicPluginFunctionTable funcs, std::string sourceLabel)
    : _sourceLabel(std::move(sourceLabel))
    , _funcs(funcs)
{
    validateFunctionTable();
    validatePluginMetadata(*this);
#ifndef NDEBUG
    _initialized = true;
#endif
}

HeuristicPlugin::HeuristicPlugin() = default;

std::shared_ptr<HeuristicPlugin> HeuristicPlugin::createBuiltIn(HeuristicPluginFunctionTable funcs,
                                                                std::string sourceLabel)
{
    return std::shared_ptr<HeuristicPlugin>(new HeuristicPlugin(funcs, std::move(sourceLabel)));
}

std::string_view HeuristicPlugin::sourceLabel() const noexcept
{
    return _sourceLabel;
}

void HeuristicPlugin::resolveSymbols()
{
    // NOLINTBEGIN(bugprone-macro-parentheses, cppcoreguidelines-macro-usage)
    // Helper macro to provide clearer error messages when required symbols are missing
#define GET_REQUIRED_SYMBOL(funcPtr, symbolName)                                                \
    do                                                                                          \
    {                                                                                           \
        try                                                                                     \
        {                                                                                       \
            (funcPtr) = _lib.getSymbol<decltype(funcPtr)>(symbolName);                          \
        }                                                                                       \
        catch(const HipdnnException& e)                                                         \
        {                                                                                       \
            throw HipdnnException(                                                              \
                HIPDNN_STATUS_PLUGIN_ERROR,                                                     \
                std::string("ERROR: HEURISTIC PLUGIN ABI INCOMPLETE\n") + "Plugin: "            \
                    + _sourceLabel + "\n" + "Missing required symbol: " symbolName "\n"         \
                    + "This plugin does not implement the complete heuristic plugin C ABI.\n"   \
                    + "See plugin_sdk/include/hipdnn_plugin_sdk/HeuristicsPluginApi.h for the " \
                      "full API.\n"                                                             \
                    + "Original error: " + e.what());                                           \
        }                                                                                       \
    } while(0)
    // NOLINTEND(bugprone-macro-parentheses, cppcoreguidelines-macro-usage)

    // Required base plugin symbols (from PluginApi.h)
    GET_REQUIRED_SYMBOL(_funcs.getName, "hipdnnPluginGetName");
    GET_REQUIRED_SYMBOL(_funcs.getVersion, "hipdnnPluginGetVersion");
    GET_REQUIRED_SYMBOL(_funcs.getApiVersion, "hipdnnPluginGetApiVersion");
    GET_REQUIRED_SYMBOL(_funcs.getType, "hipdnnPluginGetType");
    GET_REQUIRED_SYMBOL(_funcs.setLoggingCallback, "hipdnnPluginSetLoggingCallback");
    GET_REQUIRED_SYMBOL(_funcs.getLastErrorString, "hipdnnPluginGetLastErrorString");

    // Optional base plugin symbols
    tryAssignSymbol(_funcs.setLogLevel, "hipdnnPluginSetLogLevel");

    // Required policy enumeration symbols
    GET_REQUIRED_SYMBOL(_funcs.getAllPolicyIds, "hipdnnHeuristicPluginGetAllPolicyIds");
    GET_REQUIRED_SYMBOL(_funcs.getPolicyName, "hipdnnHeuristicPluginGetPolicyName");

    // Required handle lifecycle symbols
    GET_REQUIRED_SYMBOL(_funcs.handleCreate, "hipdnnHeuristicHandleCreate");
    GET_REQUIRED_SYMBOL(_funcs.handleDestroy, "hipdnnHeuristicHandleDestroy");
    GET_REQUIRED_SYMBOL(_funcs.handleSetDeviceProperties,
                        "hipdnnHeuristicHandleSetDeviceProperties");

    // Required policy descriptor lifecycle symbols
    GET_REQUIRED_SYMBOL(_funcs.policyDescriptorCreate, "hipdnnHeuristicPolicyDescriptorCreate");
    GET_REQUIRED_SYMBOL(_funcs.policyDescriptorDestroy, "hipdnnHeuristicPolicyDescriptorDestroy");

    // Required policy input symbols
    GET_REQUIRED_SYMBOL(_funcs.policySetEngineIds, "hipdnnHeuristicPolicySetEngineIds");
    GET_REQUIRED_SYMBOL(_funcs.policySetSerializedGraph, "hipdnnHeuristicPolicySetSerializedGraph");

    // Required selection symbols
    GET_REQUIRED_SYMBOL(_funcs.policyFinalize, "hipdnnHeuristicPolicyFinalize");
    GET_REQUIRED_SYMBOL(_funcs.policyGetSortedEngineIds, "hipdnnHeuristicPolicyGetSortedEngineIds");

#undef GET_REQUIRED_SYMBOL
}

void HeuristicPlugin::validateFunctionTable() const
{
    // Every required entry point must be populated. This mirrors the dlsym
    // checks in resolveSymbols and catches built-ins that forget to wire
    // something up. setLogLevel is optional and may remain null.
    auto require = [&](const void* ptr, const char* name) {
        if(ptr == nullptr)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  std::string("ERROR: HEURISTIC PLUGIN ABI INCOMPLETE\n")
                                      + "Plugin: " + _sourceLabel + "\n"
                                      + "Missing required entry point: " + name);
        }
    };
    require(reinterpret_cast<const void*>(_funcs.getName), "hipdnnPluginGetName");
    require(reinterpret_cast<const void*>(_funcs.getVersion), "hipdnnPluginGetVersion");
    require(reinterpret_cast<const void*>(_funcs.getApiVersion), "hipdnnPluginGetApiVersion");
    require(reinterpret_cast<const void*>(_funcs.getType), "hipdnnPluginGetType");
    require(reinterpret_cast<const void*>(_funcs.setLoggingCallback),
            "hipdnnPluginSetLoggingCallback");
    require(reinterpret_cast<const void*>(_funcs.getLastErrorString),
            "hipdnnPluginGetLastErrorString");
    require(reinterpret_cast<const void*>(_funcs.getAllPolicyIds),
            "hipdnnHeuristicPluginGetAllPolicyIds");
    require(reinterpret_cast<const void*>(_funcs.getPolicyName),
            "hipdnnHeuristicPluginGetPolicyName");
    require(reinterpret_cast<const void*>(_funcs.handleCreate), "hipdnnHeuristicHandleCreate");
    require(reinterpret_cast<const void*>(_funcs.handleDestroy), "hipdnnHeuristicHandleDestroy");
    require(reinterpret_cast<const void*>(_funcs.handleSetDeviceProperties),
            "hipdnnHeuristicHandleSetDeviceProperties");
    require(reinterpret_cast<const void*>(_funcs.policyDescriptorCreate),
            "hipdnnHeuristicPolicyDescriptorCreate");
    require(reinterpret_cast<const void*>(_funcs.policyDescriptorDestroy),
            "hipdnnHeuristicPolicyDescriptorDestroy");
    require(reinterpret_cast<const void*>(_funcs.policySetEngineIds),
            "hipdnnHeuristicPolicySetEngineIds");
    require(reinterpret_cast<const void*>(_funcs.policySetSerializedGraph),
            "hipdnnHeuristicPolicySetSerializedGraph");
    require(reinterpret_cast<const void*>(_funcs.policyFinalize), "hipdnnHeuristicPolicyFinalize");
    require(reinterpret_cast<const void*>(_funcs.policyGetSortedEngineIds),
            "hipdnnHeuristicPolicyGetSortedEngineIds");
}

void HeuristicPlugin::validatePluginMetadata(const HeuristicPlugin& plugin)
{
    auto pluginType = plugin.type();
    if(pluginType != HIPDNN_PLUGIN_TYPE_HEURISTIC)
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              "Plugin type mismatch: expected HIPDNN_PLUGIN_TYPE_HEURISTIC, got "
                                  + std::to_string(pluginType));
    }

    // Verify the plugin reports a non-empty library name (used purely for diagnostics now;
    // policy identity flows through the policy IDs enumerated below).
    if(plugin.name().empty())
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              "Cannot load heuristic plugin: plugin name is empty");
    }

    // Eagerly enumerate policies and validate that each policy ID matches the FNV-1a hash of
    // its canonical name. Mismatches indicate a malformed plugin and cause rejection at load.
    const auto policyIds = plugin.getAllPolicyIds();
    for(const int64_t policyId : policyIds)
    {
        const auto policyName = plugin.getPolicyName(policyId);
        if(policyName.empty())
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "Heuristic plugin returned empty name for policy ID "
                                      + std::to_string(policyId));
        }
        const int64_t expectedId
            = hipdnn_data_sdk::utilities::policyNameToId(std::string(policyName));
        if(expectedId != policyId)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  "Policy ID/name mismatch: plugin reported policy '"
                                      + std::string(policyName) + "' with ID "
                                      + std::to_string(policyId) + " but policyNameToId yields "
                                      + std::to_string(expectedId));
        }
    }
}

std::string_view HeuristicPlugin::apiVersion() const
{
    const char* version = nullptr;
    invokeHeuristicFunction("get API version", _funcs.getApiVersion, &version);
    return safeStringView(version);
}

std::string_view HeuristicPlugin::name() const
{
    const char* name = nullptr;
    invokeHeuristicFunction("get plugin name", _funcs.getName, &name);
    return safeStringView(name);
}

std::string_view HeuristicPlugin::version() const
{
    const char* version = nullptr;
    invokeHeuristicFunction("get plugin version", _funcs.getVersion, &version);
    return safeStringView(version);
}

hipdnnPluginType_t HeuristicPlugin::type() const
{
    hipdnnPluginType_t pluginType = HIPDNN_PLUGIN_TYPE_UNSPECIFIED;
    invokeHeuristicFunction("get plugin type", _funcs.getType, &pluginType);
    return pluginType;
}

std::vector<int64_t> HeuristicPlugin::getAllPolicyIds() const
{
    if(!_allPolicyIds.empty())
    {
        return _allPolicyIds;
    }

    uint32_t expectedCount = 0;
    invokeHeuristicFunction(
        "get number of policies", _funcs.getAllPolicyIds, nullptr, 0u, &expectedCount);

    std::vector<int64_t> policyIds(expectedCount);
    uint32_t actualCount = expectedCount;
    if(expectedCount > 0)
    {
        invokeHeuristicFunction("get all policy IDs",
                                _funcs.getAllPolicyIds,
                                policyIds.data(),
                                expectedCount,
                                &actualCount);
    }

    validatePolicyIdsBuffer(expectedCount, actualCount, policyIds);

    _allPolicyIds = policyIds;
    return policyIds;
}

void HeuristicPlugin::validatePolicyIdsBuffer(uint32_t expectedCount,
                                              uint32_t actualCount,
                                              std::vector<int64_t>& policyIds)
{
    if(expectedCount == 0)
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "No policies found in the plugin");
    }

    if(actualCount != expectedCount)
    {
        throw HipdnnException(
            HIPDNN_STATUS_PLUGIN_ERROR,
            "Number of policies returned does not match the number reported by the plugin");
    }

    std::sort(policyIds.begin(), policyIds.end());
    if(std::adjacent_find(policyIds.begin(), policyIds.end()) != policyIds.end())
    {
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR, "Duplicate policy IDs found");
    }
}

std::string_view HeuristicPlugin::getPolicyName(int64_t policyId) const
{
    const char* name = nullptr;
    invokeHeuristicFunction("get policy name", _funcs.getPolicyName, policyId, &name);
    return safeStringView(name);
}

hipdnnPluginStatus_t HeuristicPlugin::setLoggingCallback(hipdnnCallback_t callback) const
{
    return _funcs.setLoggingCallback(callback);
}

hipdnnPluginStatus_t HeuristicPlugin::setLogLevel(hipdnnSeverity_t level) const
{
    if(_funcs.setLogLevel == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS; // Optional function not implemented
    }
    return _funcs.setLogLevel(level);
}

hipdnnHeuristicHandle_t HeuristicPlugin::createHandle() const
{
    hipdnnHeuristicHandle_t handle = nullptr;
    invokeHeuristicFunction("create handle", _funcs.handleCreate, &handle);
    return handle;
}

void HeuristicPlugin::destroyHandle(hipdnnHeuristicHandle_t handle) const
{
    invokeHeuristicFunction("destroy handle", _funcs.handleDestroy, handle);
}

void HeuristicPlugin::setDeviceProperties(
    hipdnnHeuristicHandle_t handle, const hipdnnPluginConstData_t* devicePropsSerialized) const
{
    invokeHeuristicFunction(
        "set device properties", _funcs.handleSetDeviceProperties, handle, devicePropsSerialized);
}

hipdnnHeuristicPolicyDescriptor_t
    HeuristicPlugin::createPolicyDescriptor(hipdnnHeuristicHandle_t pluginHandle,
                                            int64_t policyId) const
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    invokeHeuristicFunction(
        "create policy descriptor", _funcs.policyDescriptorCreate, pluginHandle, policyId, &desc);
    return desc;
}

void HeuristicPlugin::destroyPolicyDescriptor(hipdnnHeuristicPolicyDescriptor_t desc) const
{
    invokeHeuristicFunction("destroy policy descriptor", _funcs.policyDescriptorDestroy, desc);
}

void HeuristicPlugin::setEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                   const int64_t* engineIds,
                                   size_t engineIdCount) const
{
    invokeHeuristicFunction(
        "set engine IDs", _funcs.policySetEngineIds, desc, engineIds, engineIdCount);
}

void HeuristicPlugin::setSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                         const hipdnnPluginConstData_t* serializedGraph) const
{
    invokeHeuristicFunction(
        "set serialized graph", _funcs.policySetSerializedGraph, desc, serializedGraph);
}

bool HeuristicPlugin::finalize(hipdnnHeuristicPolicyDescriptor_t desc) const
{
    int32_t applied = 0;
    invokeHeuristicFunction("finalize policy", _funcs.policyFinalize, desc, &applied);
    return applied != 0;
}

std::vector<int64_t>
    HeuristicPlugin::getSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc) const
{
    // Query the count first (pass nullptr for engine_ids)
    size_t count = 0;
    invokeHeuristicFunction(
        "get sorted engine IDs count", _funcs.policyGetSortedEngineIds, desc, nullptr, &count);

    if(count == 0)
    {
        return {};
    }

    // Retrieve the actual IDs
    std::vector<int64_t> ids(count);
    size_t actualCount = count;
    invokeHeuristicFunction(
        "get sorted engine IDs", _funcs.policyGetSortedEngineIds, desc, ids.data(), &actualCount);

    ids.resize(actualCount);
    return ids;
}

std::string_view HeuristicPlugin::getLastErrorString() const noexcept
{
    const char* error = nullptr;
    _funcs.getLastErrorString(&error);
    return safeStringView(error);
}

} // namespace hipdnn_backend::plugin
