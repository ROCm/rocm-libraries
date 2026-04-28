// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "hipdnn_backend.h"
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

// Forward declarations
namespace hipdnn_backend::plugin
{
class HeuristicPlugin;
class HeuristicPluginManager;
} // namespace hipdnn_backend::plugin

// Include base template (manager types will be complete when base methods are used in .cpp)
#include "PluginResourceManagerBase.hpp"

// Include complete manager type for template instantiation
#include "HeuristicPluginManager.hpp"

namespace hipdnn_backend::plugin
{

class HeuristicPlugin;
class HeuristicPluginManager;

/**
 * @brief Metadata for a loaded heuristic policy plugin.
 */
struct HeuristicPolicyInfo
{
    std::string policyName; ///< Canonical policy name (UTF-8)
    int64_t policyId; ///< Stable policy ID (engineNameToId hash)
    std::string pluginVersion; ///< Plugin implementation version
    std::string apiVersion; ///< Heuristic C ABI version
};

/**
 * @brief Per-handle resource manager for heuristic plugins.
 *
 * This class manages heuristic plugin handles and provides high-level operations
 * for the backend. It follows the same pattern as EnginePluginResourceManager:
 * - Owns plugin handles per hipdnnHandle
 * - Provides lookup by policy ID
 * - Manages device properties serialization (stubbed for now)
 * - Exposes plugin enumeration for diagnostics
 *
 * MT-safety: Instance methods are NOT thread-safe. Each hipdnnHandle should have
 * its own HeuristicPluginResourceManager instance.
 */
class HeuristicPluginResourceManager
    : public PluginResourceManagerBase<HeuristicPluginResourceManager,
                                       HeuristicPluginManager,
                                       HeuristicPlugin>
{
    // Allow base class to access private static accessors
    friend class PluginResourceManagerBase<HeuristicPluginResourceManager,
                                           HeuristicPluginManager,
                                           HeuristicPlugin>;

private:
    // Static accessors for CRTP base class
    static std::mutex& getMutex();
    static PluginLoadingConfig& getConfig();
    static std::weak_ptr<HeuristicPluginManager>& getWeakPtr();
    static std::shared_ptr<HeuristicPluginManager>& getPersistentPtr();
    static std::atomic<bool>& getShutdownFlag();
    static const char* getPluginTypeName();

protected:
    // Protected constructor for mock testing
    HeuristicPluginResourceManager();

public:
    // MT-safe static functions (inherited from base, re-declared for documentation)
    using PluginResourceManagerBase::getPluginPaths;
    using PluginResourceManagerBase::setPluginLogLevel;
    using PluginResourceManagerBase::setPluginPaths;
    using PluginResourceManagerBase::setPluginUnloadingMode;

    // Compatibility aliases for heuristic-specific naming
    static void setHeuristicPluginPaths(const std::vector<std::filesystem::path>& pluginPaths,
                                        hipdnnPluginLoadingMode_ext_t loadingMode)
    {
        setPluginPaths(pluginPaths, loadingMode);
    }

    static std::set<std::filesystem::path> getHeuristicPluginPaths()
    {
        return getPluginPaths();
    }

    // Factory method
    static std::shared_ptr<HeuristicPluginResourceManager> create();

    HeuristicPluginResourceManager(std::shared_ptr<HeuristicPluginManager> pm);
    ~HeuristicPluginResourceManager() override;

    // Prevent copying
    HeuristicPluginResourceManager(const HeuristicPluginResourceManager&) = delete;
    HeuristicPluginResourceManager& operator=(const HeuristicPluginResourceManager&) = delete;

    // Allow moving
    HeuristicPluginResourceManager(HeuristicPluginResourceManager&& other) noexcept;
    HeuristicPluginResourceManager& operator=(HeuristicPluginResourceManager&& other) noexcept;

    // Instance methods (MT-unsafe - one per hipdnnHandle)

    /**
     * @brief Get the plugin handle for a given policy ID.
     *
     * Returns the hipdnnHeuristicHandle_t created for the plugin that implements
     * the given policy ID. Returns nullptr if no plugin with that policy ID is loaded.
     *
     * @param policyId The policy ID (int64_t from engineNameToId)
     * @return The plugin handle, or nullptr if not found
     */
    virtual hipdnnHeuristicHandle_t getHeuristicHandleForPolicyId(int64_t policyId) const;

    /**
     * @brief Get the HeuristicPlugin pointer for a given policy ID.
     *
     * Returns the HeuristicPlugin* that implements the given policy ID.
     * Returns nullptr if no plugin with that policy ID is loaded.
     *
     * @param policyId The policy ID (int64_t from engineNameToId)
     * @return The plugin pointer, or nullptr if not found
     */
    virtual const HeuristicPlugin* getPluginForPolicyId(int64_t policyId) const;

    /**
     * @brief Set device properties on all loaded heuristic plugin handles.
     *
     * Calls hipdnnHeuristicHandleSetDeviceProperties on each loaded plugin handle
     * with the serialized device properties buffer.
     *
     * @param devicePropsSerialized Pointer to hipdnnPluginConstData_t containing
     *                              FlatBuffer-serialized device properties
     */
    virtual void
        setDevicePropertiesOnAllHandles(const hipdnnPluginConstData_t* devicePropsSerialized) const;

    /**
     * @brief Get information about all loaded heuristic policies.
     *
     * Returns metadata for all heuristic plugins that passed validation and were
     * successfully loaded.
     *
     * @return Vector of policy metadata
     */
    virtual std::vector<HeuristicPolicyInfo> getHeuristicPolicyInfos() const;

    // Inherited from base: getLoadedPluginFiles()
    using PluginResourceManagerBase::getLoadedPluginFiles;

    /**
     * @brief Get a string representation of the resource manager state.
     *
     * @return String describing loaded plugins and handles
     */
    virtual std::string toString() const;

protected:
    // Note: _pm member is inherited from PluginResourceManagerBase

private:
    std::unordered_map<hipdnnHeuristicHandle_t, const HeuristicPlugin*> _handleToPlugin;
    std::unordered_map<int64_t, hipdnnHeuristicHandle_t> _policyIdToHandle;
    mutable std::optional<std::vector<HeuristicPolicyInfo>> _cachedPolicyInfos;
};

} // namespace hipdnn_backend::plugin
