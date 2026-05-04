// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HeuristicPluginResourceManager.hpp"

#include "HeuristicPlugin.hpp"
#include "HeuristicPluginManager.hpp"
#include "HipdnnBackendPluginUnloadingMode.h"
#include "HipdnnException.hpp"
#include "logging/Logging.hpp"
#include <mutex>
#include <sstream>

namespace hipdnn_backend::plugin
{

namespace
{

// Static storage for heuristic plugin configuration
std::mutex gHeuristicMutex;
PluginLoadingConfig gHeuristicConfig;
std::weak_ptr<HeuristicPluginManager> gHeuristicWeakPtr;
std::shared_ptr<HeuristicPluginManager> gHeuristicPersistentPtr;
std::atomic<bool> gHeuristicShutdownFlag{false};

// Register atexit handler to set shutdown flag
struct HeuristicPluginShutdownRegistrar
{
    HeuristicPluginShutdownRegistrar()
    {
        std::atexit([]() { gHeuristicShutdownFlag.store(true, std::memory_order_release); });
    }
};

HeuristicPluginShutdownRegistrar gHeuristicShutdownRegistrar;

} // anonymous namespace

// Static accessor implementations for CRTP base class
std::mutex& HeuristicPluginResourceManager::getMutex()
{
    return gHeuristicMutex;
}

PluginLoadingConfig& HeuristicPluginResourceManager::getConfig()
{
    return gHeuristicConfig;
}

std::weak_ptr<HeuristicPluginManager>& HeuristicPluginResourceManager::getWeakPtr()
{
    return gHeuristicWeakPtr;
}

std::shared_ptr<HeuristicPluginManager>& HeuristicPluginResourceManager::getPersistentPtr()
{
    return gHeuristicPersistentPtr;
}

std::atomic<bool>& HeuristicPluginResourceManager::getShutdownFlag()
{
    return gHeuristicShutdownFlag;
}

const char* HeuristicPluginResourceManager::getPluginTypeName()
{
    return "heuristic";
}

HeuristicPluginResourceManager::HeuristicPluginResourceManager() = default;

HeuristicPluginResourceManager::HeuristicPluginResourceManager(
    std::shared_ptr<HeuristicPluginManager> pm)
    : PluginResourceManagerBase(std::move(pm))
{
    // During static destruction, pm may be null - this is expected and safe
    if(!_pm)
    {
        return; // No plugins to initialize
    }

    // Create plugin handles for all loaded heuristic plugins
    const auto& plugins = _pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        // Set logging callback before creating handle
        auto logStatus
            = plugin->setLoggingCallback(hipdnn_backend::logging::backendLoggingCallback);
        if(logStatus != HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Failed to set logging callback on heuristic plugin with policy ID {}",
                plugin->policyId());
            continue;
        }

        // Set log level (optional)
        hipdnnSeverity_t level = HIPDNN_SEV_INFO;
        hipdnn_backend::logging::getGlobalLogLevel(level);
        plugin->setLogLevel(level);

        // Create plugin handle
        hipdnnHeuristicHandle_t handle = nullptr;
        try
        {
            handle = plugin->createHandle();
            if(handle == nullptr)
            {
                HIPDNN_BACKEND_LOG_ERROR("Plugin with policy ID {} ({}) returned null handle "
                                         "despite reporting success. Plugin will be unavailable.",
                                         plugin->policyId(),
                                         plugin->name());
                continue;
            }
        }
        catch(const HipdnnException& e)
        {
            HIPDNN_BACKEND_LOG_ERROR("Failed to create handle for heuristic plugin with policy ID "
                                     "{} ({}): {}. Plugin will be unavailable.",
                                     plugin->policyId(),
                                     plugin->name(),
                                     e.what());
            continue;
        }

        _handleToPlugin[handle] = plugin.get();
        _policyIdToHandle[plugin->policyId()] = handle;

        HIPDNN_BACKEND_LOG_INFO("Created heuristic plugin handle for policy ID {} ({})",
                                plugin->policyId(),
                                plugin->name());
    }
}

HeuristicPluginResourceManager::~HeuristicPluginResourceManager()
{
    // Lambda to safely destroy a handle, catching all errors
    auto safeDestroyHandle = [](const HeuristicPlugin* plugin, hipdnnHeuristicHandle_t handle) {
        try
        {
            plugin->destroyHandle(handle);
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_WARN("Failed to destroy handle for heuristic plugin '{}' (policy ID "
                                    "{}) during cleanup: {}",
                                    plugin->name(),
                                    plugin->policyId(),
                                    e.what());
        }
        catch(...)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Failed to destroy handle for heuristic plugin '{}' (policy ID {}) during cleanup: "
                "unknown error",
                plugin->name(),
                plugin->policyId());
        }
    };

    // Destroy all plugin handles
    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        safeDestroyHandle(plugin, handle);
    }

    _handleToPlugin.clear();
    _policyIdToHandle.clear();
}

HeuristicPluginResourceManager::HeuristicPluginResourceManager(
    HeuristicPluginResourceManager&& other) noexcept
    : _handleToPlugin(std::move(other._handleToPlugin))
    , _policyIdToHandle(std::move(other._policyIdToHandle))
    , _cachedPolicyInfos(std::move(other._cachedPolicyInfos))
{
    // Move base class member explicitly
    _pm = std::move(other._pm);
}

HeuristicPluginResourceManager&
    HeuristicPluginResourceManager::operator=(HeuristicPluginResourceManager&& other) noexcept
{
    if(this != &other)
    {
        _handleToPlugin = std::move(other._handleToPlugin);
        _policyIdToHandle = std::move(other._policyIdToHandle);
        _cachedPolicyInfos = std::move(other._cachedPolicyInfos);
        _pm = std::move(other._pm);
    }
    return *this;
}

std::shared_ptr<HeuristicPluginResourceManager> HeuristicPluginResourceManager::create()
{
    auto pm = getOrCreatePluginManager();
    return std::make_shared<HeuristicPluginResourceManager>(pm);
}

hipdnnHeuristicHandle_t
    HeuristicPluginResourceManager::getHeuristicHandleForPolicyId(int64_t policyId) const
{
    auto it = _policyIdToHandle.find(policyId);
    if(it == _policyIdToHandle.end())
    {
        return nullptr;
    }
    return it->second;
}

const HeuristicPlugin* HeuristicPluginResourceManager::getPluginForPolicyId(int64_t policyId) const
{
    auto handle = getHeuristicHandleForPolicyId(policyId);
    if(handle == nullptr)
    {
        return nullptr;
    }

    auto it = _handleToPlugin.find(handle);
    if(it == _handleToPlugin.end())
    {
        return nullptr;
    }
    return it->second;
}

void HeuristicPluginResourceManager::setDevicePropertiesOnAllHandles(
    const hipdnnPluginConstData_t* devicePropsSerialized) const
{
    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        try
        {
            plugin->setDeviceProperties(handle, devicePropsSerialized);
        }
        catch(const HipdnnException& e)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Failed to set device properties on heuristic plugin with policy ID {}: {}",
                plugin->policyId(),
                e.what());
            // Continue with other plugins
        }
    }
}

std::vector<HeuristicPolicyInfo> HeuristicPluginResourceManager::getHeuristicPolicyInfos() const
{
    if(_cachedPolicyInfos)
    {
        return *_cachedPolicyInfos;
    }

    std::vector<HeuristicPolicyInfo> infos;
    infos.reserve(_handleToPlugin.size());

    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        HeuristicPolicyInfo info;
        info.policyId = plugin->policyId();
        info.policyName = std::string(plugin->name());
        info.pluginVersion = std::string(plugin->version());
        info.apiVersion = std::string(plugin->apiVersion());
        infos.push_back(info);
    }

    _cachedPolicyInfos = infos;
    return infos;
}

std::string HeuristicPluginResourceManager::toString() const
{
    std::ostringstream oss;
    oss << "HeuristicPluginResourceManager {\n";
    oss << "  Loaded plugins: " << _handleToPlugin.size() << "\n";

    // Add loaded plugin paths
    if(_pm)
    {
        auto loadedPlugins = _pm->getLoadedPluginFiles();
        oss << "  Loaded plugin paths: [";
        bool first = true;
        for(const auto& path : loadedPlugins)
        {
            if(!first)
            {
                oss << ", ";
            }
            oss << path.string();
            first = false;
        }
        oss << "]\n";
    }

    auto infos = getHeuristicPolicyInfos();
    for(const auto& info : infos)
    {
        oss << "    Policy ID: " << info.policyId;
        if(!info.policyName.empty())
        {
            oss << " (" << info.policyName << ")";
        }
        oss << ", Plugin Version: " << info.pluginVersion;
        oss << ", API Version: " << info.apiVersion << "\n";
    }

    oss << "}";
    return oss.str();
}

} // namespace hipdnn_backend::plugin
