// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

#include "PluginCore.hpp"
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

namespace hipdnn_backend::plugin
{

/**
 * @brief Function-pointer table for the heuristic plugin C ABI.
 *
 * Holds every entry point HeuristicPlugin needs to drive a plugin. Populated
 * either by dlsym from a loaded shared library (the `SharedLibrary` ctor below)
 * or by a backend-internal "built-in" that supplies the table directly without
 * a `.so` (see `HeuristicPluginManager::registerBuiltIn`). The downstream
 * wrapper code does not distinguish the two cases.
 */
struct HeuristicPluginFunctionTable
{
    // Base plugin metadata (PluginApi.h)
    hipdnnPluginStatus_t (*getName)(const char**) = nullptr;
    hipdnnPluginStatus_t (*getVersion)(const char**) = nullptr;
    hipdnnPluginStatus_t (*getApiVersion)(const char**) = nullptr;
    hipdnnPluginStatus_t (*getType)(hipdnnPluginType_t*) = nullptr;
    hipdnnPluginStatus_t (*setLoggingCallback)(hipdnnCallback_t) = nullptr;
    hipdnnPluginStatus_t (*setLogLevel)(hipdnnSeverity_t) = nullptr; // optional
    void (*getLastErrorString)(const char**) = nullptr;

    // Policy enumeration
    hipdnnPluginStatus_t (*getAllPolicyIds)(int64_t*, uint32_t, uint32_t*) = nullptr;
    hipdnnPluginStatus_t (*getPolicyName)(int64_t, const char**) = nullptr;

    // Handle lifecycle
    hipdnnPluginStatus_t (*handleCreate)(hipdnnHeuristicHandle_t*) = nullptr;
    hipdnnPluginStatus_t (*handleDestroy)(hipdnnHeuristicHandle_t) = nullptr;
    hipdnnPluginStatus_t (*handleSetDeviceProperties)(hipdnnHeuristicHandle_t,
                                                      const hipdnnPluginConstData_t*)
        = nullptr;

    // Policy descriptor lifecycle
    hipdnnPluginStatus_t (*policyDescriptorCreate)(hipdnnHeuristicHandle_t,
                                                   int64_t,
                                                   hipdnnHeuristicPolicyDescriptor_t*)
        = nullptr;
    hipdnnPluginStatus_t (*policyDescriptorDestroy)(hipdnnHeuristicPolicyDescriptor_t) = nullptr;

    // Policy inputs
    hipdnnPluginStatus_t (*policySetEngineIds)(hipdnnHeuristicPolicyDescriptor_t,
                                               const int64_t*,
                                               size_t)
        = nullptr;
    hipdnnPluginStatus_t (*policySetSerializedGraph)(hipdnnHeuristicPolicyDescriptor_t,
                                                     const hipdnnPluginConstData_t*)
        = nullptr;

    // Selection
    hipdnnPluginStatus_t (*policyFinalize)(hipdnnHeuristicPolicyDescriptor_t, int32_t*) = nullptr;
    hipdnnPluginStatus_t (*policyGetSortedEngineIds)(hipdnnHeuristicPolicyDescriptor_t,
                                                     int64_t*,
                                                     size_t*)
        = nullptr;
};

/**
 * @brief Wrapper for a heuristic plugin (shared library or backend built-in).
 *
 * Provides a C++ interface over the heuristic plugin C ABI defined in
 * HeuristicsPluginApi.h. Two construction paths populate the same function
 * table:
 *   - `HeuristicPlugin(SharedLibrary&&)` resolves every symbol via dlsym from
 *     the loaded `.so` — used by `HeuristicPluginManager::loadPlugins`.
 *   - `HeuristicPlugin(HeuristicPluginFunctionTable, std::string sourceLabel)`
 *     accepts a pre-populated table from a backend built-in module — used by
 *     `HeuristicPluginManager::registerBuiltIn`. `sourceLabel` is shown in
 *     diagnostics in place of a library path.
 *
 * Validation (`validatePluginMetadata`, `validatePolicyIdsBuffer`) runs
 * identically for both paths.
 */
class HeuristicPlugin : public PluginBase
{
protected:
    // Shared-library ctor: populates the function table via dlsym.
    explicit HeuristicPlugin(SharedLibrary&& lib);

    // Built-in ctor: caller hands over a fully populated function table.
    // `sourceLabel` is a human-readable identifier used in error/diagnostic
    // messages (e.g. "built-in:SelectionHeuristic::StaticOrdering").
    HeuristicPlugin(HeuristicPluginFunctionTable funcs, std::string sourceLabel);

    // For mocking in tests
    HeuristicPlugin();

public:
    // Virtual destructor for polymorphic class
    ~HeuristicPlugin() override = default;

    // Base plugin metadata (from PluginApi.h)
    std::string_view apiVersion() const override;
    std::string_view name() const override; // Plugin (library) name (via hipdnnPluginGetName)
    std::string_view
        version() const override; // Returns plugin version (via hipdnnPluginGetVersion)
    hipdnnPluginType_t
        type() const override; // Returns HIPDNN_PLUGIN_TYPE_HEURISTIC (via hipdnnPluginGetType)

    // Heuristic-specific metadata: a single plugin may expose multiple policies.
    // getAllPolicyIds() is cached after first invocation; getPolicyName() is
    // queried on demand and returns the canonical name reported by the plugin.
    virtual std::vector<int64_t> getAllPolicyIds() const;
    virtual std::string_view getPolicyName(int64_t policyId) const;

    // Plugin type - heuristic plugins return HEURISTIC
    static hipdnnPluginType_t getPluginType()
    {
        return HIPDNN_PLUGIN_TYPE_HEURISTIC;
    }

    // Logging setup (called at module load time)
    hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback) const;

    hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level) const;

    // Plugin handle lifecycle (one handle per loaded plugin, shared across policies)
    virtual hipdnnHeuristicHandle_t createHandle() const;
    virtual void destroyHandle(hipdnnHeuristicHandle_t handle) const;
    virtual void setDeviceProperties(hipdnnHeuristicHandle_t handle,
                                     const hipdnnPluginConstData_t* devicePropsSerialized) const;

    // Policy descriptor lifecycle (one descriptor per policy slot)
    virtual hipdnnHeuristicPolicyDescriptor_t
        createPolicyDescriptor(hipdnnHeuristicHandle_t pluginHandle, int64_t policyId) const;
    virtual void destroyPolicyDescriptor(hipdnnHeuristicPolicyDescriptor_t desc) const;

    // Policy inputs
    virtual void setEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                              const int64_t* engineIds,
                              size_t engineIdCount) const;
    virtual void setSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                    const hipdnnPluginConstData_t* serializedGraph) const;

    // Selection execution
    virtual bool finalize(hipdnnHeuristicPolicyDescriptor_t desc) const;
    virtual std::vector<int64_t> getSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc) const;

    // Validation helpers shared between resolveSymbols() (run at load time) and
    // unit tests. Each helper throws HipdnnException on failure.
    //
    // validatePluginMetadata: checks plugin type is HEURISTIC, plugin library
    // name is non-empty, and every reported policy has a non-empty name whose
    // FNV-1a hash matches its policy ID. Operates entirely through virtual
    // accessors so a NiceMock can drive each rejection path.
    static void validatePluginMetadata(const HeuristicPlugin& plugin);

    // validatePolicyIdsBuffer: checks the raw policy ID buffer returned by a
    // plugin: actual count matches the expected count from the prior count
    // query, and the buffer contains no intra-plugin duplicates. Sorts
    // policyIds in place.
    static void validatePolicyIdsBuffer(uint32_t expectedCount,
                                        uint32_t actualCount,
                                        std::vector<int64_t>& policyIds);

    // Factory for backend built-in heuristics. Wraps a fully populated function
    // table in a HeuristicPlugin without going through dlopen. Validates the
    // table is complete and the metadata matches the same rules as a loaded
    // plugin (HeuristicPlugin::validatePluginMetadata).
    static std::shared_ptr<HeuristicPlugin> createBuiltIn(HeuristicPluginFunctionTable funcs,
                                                          std::string sourceLabel);

    // Source identifier used in diagnostics (library path for dlopen plugins,
    // "built-in:<name>" for built-ins).
    std::string_view sourceLabel() const noexcept;

protected:
    // Error handling helper (must not throw, used during error handling)
    std::string_view getLastErrorString() const noexcept;

    template <typename Callable, typename... Args>
    void invokeHeuristicFunction(const char* description, Callable&& func, Args&&... args) const
    {
        auto status = func(std::forward<Args>(args)...);
        if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  std::string("Heuristic plugin failed to ") + description
                                      + ". Status: " + std::to_string(status)
                                      + ", Error: " + std::string(getLastErrorString()));
        }
    }

    template <class F>
    bool tryAssignSymbol(F& functionPtr, const char* symbolName)
    {
        try
        {
            functionPtr = _lib.getSymbol<F>(symbolName);
            return true;
        }
        catch(const HipdnnException&)
        {
            functionPtr = nullptr;
            return false;
        }
    }

    SharedLibrary _lib;

    // For diagnostics — either the library path string or a built-in label.
    std::string _sourceLabel;

    // Function-pointer table for the heuristic C ABI. Populated by
    // resolveSymbols() in the dlopen ctor or by the caller in the built-in ctor.
    HeuristicPluginFunctionTable _funcs;

private:
    void resolveSymbols();
    void validateFunctionTable() const;

#ifndef NDEBUG
    bool _initialized = false;
#endif

    // Cached policy IDs (lazily populated by getAllPolicyIds and validated in
    // resolveSymbols). Mutable so the const accessor can fill the cache.
    mutable std::vector<int64_t> _allPolicyIds;

    friend class PluginManagerBase<HeuristicPlugin>;
};

} // namespace hipdnn_backend::plugin
