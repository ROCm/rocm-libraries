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
 * @brief Wrapper for a heuristic plugin shared library.
 *
 * This class provides a C++ interface to the heuristic plugin C ABI defined in
 * HeuristicsPluginApi.h. It manages symbol resolution and provides type-safe
 * wrappers around the C function pointers.
 *
 * Heuristic plugins implement base PluginApi.h functions PLUS HeuristicsPluginApi.h extensions.
 */
class HeuristicPlugin : public PluginBase
{
protected:
    // Protected constructor to prevent direct instantiation
    explicit HeuristicPlugin(SharedLibrary&& lib);

    // For mocking in tests
    HeuristicPlugin();

public:
    // Virtual destructor for polymorphic class
    ~HeuristicPlugin() override = default;

    // Base plugin metadata (from PluginApi.h)
    std::string_view apiVersion() const override;
    std::string_view name() const override; // Returns policy name (via hipdnnPluginGetName)
    std::string_view
        version() const override; // Returns plugin version (via hipdnnPluginGetVersion)
    hipdnnPluginType_t
        type() const override; // Returns HIPDNN_PLUGIN_TYPE_HEURISTIC (via hipdnnPluginGetType)

    // Heuristic-specific metadata
    virtual int64_t policyId() const; // Computed from name via engineNameToId

    // Plugin type - heuristic plugins return HEURISTIC
    static hipdnnPluginType_t getPluginType()
    {
        return HIPDNN_PLUGIN_TYPE_HEURISTIC;
    }

    // Logging setup (called at module load time)
    hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback) const;

    hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level) const;

    // Plugin handle lifecycle
    virtual hipdnnHeuristicHandle_t createHandle() const;
    virtual void destroyHandle(hipdnnHeuristicHandle_t handle) const;
    virtual void setDeviceProperties(hipdnnHeuristicHandle_t handle,
                                     const hipdnnPluginConstData_t* devicePropsSerialized) const;

    // Policy descriptor lifecycle
    virtual hipdnnHeuristicPolicyDescriptor_t
        createPolicyDescriptor(hipdnnHeuristicHandle_t pluginHandle) const;
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

private:
    void resolveSymbols();

#ifndef NDEBUG
    bool _initialized = false;
#endif

    // Cached metadata (eagerly initialized in resolveSymbols)
    int64_t _policyId = -1;

    // Base plugin function pointers (from PluginApi.h)
    hipdnnPluginStatus_t (*_funcGetName)(const char**);
    hipdnnPluginStatus_t (*_funcGetVersion)(const char**);
    hipdnnPluginStatus_t (*_funcGetApiVersion)(const char**);
    hipdnnPluginStatus_t (*_funcGetType)(hipdnnPluginType_t*);
    hipdnnPluginStatus_t (*_funcSetLoggingCallback)(hipdnnCallback_t);
    hipdnnPluginStatus_t (*_funcSetLogLevel)(hipdnnSeverity_t);
    void (*_funcGetLastErrorString)(const char**);

    // Handle lifecycle function pointers
    hipdnnPluginStatus_t (*_funcHandleCreate)(hipdnnHeuristicHandle_t*);
    hipdnnPluginStatus_t (*_funcHandleDestroy)(hipdnnHeuristicHandle_t);
    hipdnnPluginStatus_t (*_funcHandleSetDeviceProperties)(hipdnnHeuristicHandle_t,
                                                           const hipdnnPluginConstData_t*);

    // Policy descriptor lifecycle function pointers
    hipdnnPluginStatus_t (*_funcPolicyDescriptorCreate)(hipdnnHeuristicHandle_t,
                                                        hipdnnHeuristicPolicyDescriptor_t*);
    hipdnnPluginStatus_t (*_funcPolicyDescriptorDestroy)(hipdnnHeuristicPolicyDescriptor_t);

    // Policy input function pointers
    hipdnnPluginStatus_t (*_funcPolicySetEngineIds)(hipdnnHeuristicPolicyDescriptor_t,
                                                    const int64_t*,
                                                    size_t);
    hipdnnPluginStatus_t (*_funcPolicySetSerializedGraph)(hipdnnHeuristicPolicyDescriptor_t,
                                                          const hipdnnPluginConstData_t*);

    // Selection function pointers
    hipdnnPluginStatus_t (*_funcPolicyFinalize)(hipdnnHeuristicPolicyDescriptor_t, int32_t*);
    hipdnnPluginStatus_t (*_funcPolicyGetSortedEngineIds)(hipdnnHeuristicPolicyDescriptor_t,
                                                          int64_t*,
                                                          size_t*);

    friend class PluginManagerBase<HeuristicPlugin>;
};

} // namespace hipdnn_backend::plugin
