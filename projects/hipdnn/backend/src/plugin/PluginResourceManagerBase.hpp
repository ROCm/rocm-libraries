// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "HipdnnException.hpp"
#include "hipdnn_backend.h"
#include "logging/Logging.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_backend::plugin
{

/**
 * @brief Configuration for plugin loading behavior.
 *
 * Stores the search paths and loading/unloading modes for a plugin type.
 */
struct PluginLoadingConfig
{
    std::set<std::filesystem::path> paths;
    hipdnnPluginLoadingMode_ext_t mode = HIPDNN_DEFAULT_PLUGIN_LOADING_MODE;
    hipdnnPluginUnloadingMode_ext_t unloadingMode = HIPDNN_DEFAULT_PLUGIN_UNLOADING_MODE;
};

/**
 * @brief Base class for plugin resource managers using CRTP.
 *
 * Provides common functionality for managing plugin loading, unloading, path configuration,
 * and plugin handle lifecycle. This class uses the Curiously Recurring Template Pattern (CRTP)
 * to allow sharing of static method implementations while maintaining separate static storage
 * for each plugin type (engine vs heuristic).
 *
 * Template parameters:
 * - Derived: The derived class type (for CRTP)
 * - PluginManagerType: The type of plugin manager (EnginePluginManager or HeuristicPluginManager)
 * - PluginType: The type of plugin (EnginePlugin or HeuristicPlugin)
 *
 * Derived classes must provide static accessor methods:
 * - static std::mutex& getMutex()
 * - static PluginLoadingConfig& getConfig()
 * - static std::weak_ptr<PluginManagerType>& getWeakPtr()
 * - static std::shared_ptr<PluginManagerType>& getPersistentPtr()
 * - static std::atomic<bool>& getShutdownFlag()
 * - static const char* getPluginTypeName()
 *
 * MT-safety: Static methods are thread-safe. Instance methods are NOT thread-safe
 * (each hipdnnHandle should have its own resource manager instance).
 */
template <typename Derived, typename PluginManagerType, typename PluginType>
class PluginResourceManagerBase
{
    // Allow derived class to access private constructors
    friend Derived;

private:
    // Private constructors for CRTP (only accessible via Derived)
    PluginResourceManagerBase() = default;

    explicit PluginResourceManagerBase(std::shared_ptr<PluginManagerType> pm)
        : _pm(std::move(pm))
    {
    }

    // Allow moving (private, only accessible via Derived)
    PluginResourceManagerBase(PluginResourceManagerBase&& other) noexcept
        : _pm(std::move(other._pm))
    {
    }

    PluginResourceManagerBase& operator=(PluginResourceManagerBase&& other) noexcept
    {
        if(this != &other)
        {
            _pm = std::move(other._pm);
        }
        return *this;
    }

    // Virtual destructor (protected to prevent direct instantiation via base pointer)
    virtual ~PluginResourceManagerBase() = default;

public:
    // Prevent copying (public per modernize-use-equals-delete, NOLINT for false-positive CRTP warning)
    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
    PluginResourceManagerBase(const PluginResourceManagerBase&) = delete;
    PluginResourceManagerBase& operator=(const PluginResourceManagerBase&) = delete;

    /**
     * @brief Set plugin search paths (static, MT-safe).
     *
     * @param pluginPaths Vector of filesystem paths to search for plugins
     * @param loadingMode ABSOLUTE (replace) or ADDITIVE (append to existing paths)
     */
    static void setPluginPaths(const std::vector<std::filesystem::path>& pluginPaths,
                               hipdnnPluginLoadingMode_ext_t loadingMode)
    {
        const std::lock_guard<std::mutex> lock(Derived::getMutex());

        auto newPathsSet = std::set<std::filesystem::path>{pluginPaths.begin(), pluginPaths.end()};
        auto& config = Derived::getConfig();

        if(config.paths == newPathsSet && config.mode == loadingMode)
        {
            return;
        }

        // Clear persistent pointer first to allow lazy mode check to work correctly.
        // If only persistentPtr is keeping plugins alive (no active handles),
        // then weakPtr will expire after this reset.
        Derived::getPersistentPtr().reset();

        THROW_IF_FALSE(Derived::getWeakPtr().expired(),
                       HIPDNN_STATUS_NOT_SUPPORTED,
                       std::string("Cannot set ") + Derived::getPluginTypeName()
                           + " plugin paths with an active handle.");

        config.mode = loadingMode;

        if(loadingMode == HIPDNN_PLUGIN_LOADING_ABSOLUTE)
        {
            config.paths = {pluginPaths.begin(), pluginPaths.end()};
        }
        else // HIPDNN_PLUGIN_LOADING_ADDITIVE
        {
            config.paths.insert(pluginPaths.begin(), pluginPaths.end());
        }
    }

    /**
     * @brief Get currently configured plugin search paths (static, MT-safe).
     */
    static std::set<std::filesystem::path> getPluginPaths()
    {
        const std::lock_guard<std::mutex> lock(Derived::getMutex());
        return Derived::getConfig().paths;
    }

    /**
     * @brief Set plugin unloading mode (static, MT-safe).
     *
     * @param mode EAGER (unload when no handles) or LAZY (keep loaded until path change/exit)
     */
    static void setPluginUnloadingMode(hipdnnPluginUnloadingMode_ext_t mode)
    {
        const std::lock_guard<std::mutex> lock(Derived::getMutex());

        auto& config = Derived::getConfig();

        switch(mode)
        {
        case HIPDNN_PLUGIN_UNLOAD_EAGER:
            // Clear persistent pointer - if no handles exist, plugins will be unloaded
            Derived::getPersistentPtr().reset();
            break;

        case HIPDNN_PLUGIN_UNLOAD_LAZY:
            // If plugins are already loaded, keep them alive by storing in persistent pointer
            if(auto pm = Derived::getWeakPtr().lock())
            {
                Derived::getPersistentPtr() = pm;
            }
            // If no plugins loaded yet, persistentPtr will be set when create() is called
            break;

        default:
            throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                                  std::string("Invalid plugin unloading mode for ")
                                      + Derived::getPluginTypeName()
                                      + " plugins: " + std::to_string(mode));
        }

        config.unloadingMode = mode;
    }

    /**
     * @brief Set log level on all loaded plugins (static, MT-safe).
     *
     * Implementation defined below class after manager types are complete.
     */
    static void setPluginLogLevel(hipdnnSeverity_t level);

    /**
     * @brief Get the file paths of all loaded plugin libraries.
     *
     * This method supports two usage patterns:
     * 1. Query mode: Pass pluginPaths = nullptr to get count and max string length
     * 2. Retrieve mode: Pass non-null pluginPaths with sufficient capacity
     *
     * @param numPlugins Output: number of loaded plugins
     * @param pluginPaths Output: array of plugin file paths (caller-allocated), or nullptr to query
     * @param maxStringLen Output: maximum string length needed (including null terminator)
     *
     * Implementation defined below class after manager types are complete.
     */
    void getLoadedPluginFiles(size_t* numPlugins, char** pluginPaths, size_t* maxStringLen) const;

protected:
    /**
     * @brief Get or create the shared plugin manager (static, MT-safe).
     *
     * This is a helper for the derived class's create() method.
     * Implementation defined below class after manager types are complete.
     */
    static std::shared_ptr<PluginManagerType> getOrCreatePluginManager();

    std::shared_ptr<PluginManagerType> _pm;
};

// Template method implementations (defined after class so derived classes can provide complete types)

template <typename Derived, typename PluginManagerType, typename PluginType>
void PluginResourceManagerBase<Derived, PluginManagerType, PluginType>::setPluginLogLevel(
    hipdnnSeverity_t level)
{
    const std::lock_guard<std::mutex> lock(Derived::getMutex());
    auto pm = Derived::getWeakPtr().lock();
    if(!pm)
    {
        return; // No plugins loaded yet
    }

    const auto& plugins = pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        auto status = plugin->setLogLevel(level);
        if(status != HIPDNN_PLUGIN_STATUS_SUCCESS && status != HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
        {
            HIPDNN_BACKEND_LOG_WARN("Failed to set log level for {} plugin '{}': status {}",
                                    Derived::getPluginTypeName(),
                                    plugin->name(),
                                    static_cast<int>(status));
        }
    }
}

template <typename Derived, typename PluginManagerType, typename PluginType>
std::shared_ptr<PluginManagerType>
    PluginResourceManagerBase<Derived, PluginManagerType, PluginType>::getOrCreatePluginManager()
{
    // Return nullptr if we're in static destruction to avoid accessing destroyed objects
    if(Derived::getShutdownFlag().load(std::memory_order_acquire))
    {
        return nullptr;
    }

    const std::lock_guard<std::mutex> lock(Derived::getMutex());

    auto pm = Derived::getWeakPtr().lock();
    if(!pm)
    {
        pm = std::make_shared<PluginManagerType>();
        auto& config = Derived::getConfig();
        pm->loadPlugins(config.paths, config.mode);
        Derived::getWeakPtr() = pm;

        // In lazy mode, keep the plugin manager alive by storing in persistent pointer
        if(config.unloadingMode == HIPDNN_PLUGIN_UNLOAD_LAZY)
        {
            Derived::getPersistentPtr() = pm;
        }
    }

    return pm;
}

template <typename Derived, typename PluginManagerType, typename PluginType>
void PluginResourceManagerBase<Derived, PluginManagerType, PluginType>::getLoadedPluginFiles(
    size_t* numPlugins, char** pluginPaths, size_t* maxStringLen) const
{
    THROW_IF_FALSE(
        numPlugins != nullptr, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "numPlugins is null");

    if(!_pm)
    {
        *numPlugins = 0;
        if(maxStringLen != nullptr)
        {
            *maxStringLen = 0;
        }
        return;
    }

    const auto& pathSet = _pm->getLoadedPluginFiles();

    size_t requiredLen = 0;
    for(const auto& path : pathSet)
    {
        requiredLen = std::max(requiredLen, path.string().length() + 1);
    }

    if(pluginPaths == nullptr)
    {
        // Query mode: return count and max length
        *numPlugins = pathSet.size();
        if(maxStringLen != nullptr)
        {
            *maxStringLen = requiredLen;
        }
        return;
    }

    // Retrieve mode: validate buffer capacity
    if(*numPlugins < pathSet.size() || (maxStringLen != nullptr && *maxStringLen < requiredLen))
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Insufficient buffer space provided.");
    }

    // Convert set to vector for indexed access
    std::vector<std::filesystem::path> pathsVec;
    pathsVec.reserve(pathSet.size());
    pathsVec.assign(pathSet.begin(), pathSet.end());

    // Copy paths to caller's buffers
    for(size_t i = 0; i < pathsVec.size(); ++i)
    {
        if(pluginPaths[i] == nullptr)
        {
            throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "A plugin path string buffer is null.");
        }
        const size_t copyLen = maxStringLen != nullptr ? *maxStringLen : requiredLen;
        hipdnn_data_sdk::utilities::copyMaxSizeWithNullTerminator(
            pluginPaths[i], pathsVec[i].string().c_str(), copyLen);
    }

    *numPlugins = pathsVec.size();
}

} // namespace hipdnn_backend::plugin
