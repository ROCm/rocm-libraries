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

// Best-effort handle destruction used by both the constructor (rollback) and
// the destructor. Plugin code is untrusted and may throw arbitrary types; we
// must swallow everything because we are either mid-construction-failure or
// mid-destruction.
void safeDestroyHandle(const HeuristicPlugin* plugin, hipdnnHeuristicHandle_t handle) noexcept
{
    try
    {
        plugin->destroyHandle(handle);
    }
    catch(const std::exception& e)
    {
        HIPDNN_BACKEND_LOG_WARN(
            "Failed to destroy handle for heuristic plugin '{}': {}", plugin->name(), e.what());
    }
    catch(...)
    {
        HIPDNN_BACKEND_LOG_WARN(
            "Failed to destroy handle for heuristic plugin '{}' (unknown exception)",
            plugin->name());
    }
}

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

    // Create plugin handles for all loaded heuristic plugins. Each plugin has a single
    // handle, but a plugin may expose multiple policies; every policy ID maps to the
    // same handle (N:1).
    //
    // Per-plugin setup is wrapped in try/catch so that a single failing plugin does
    // not leave the resource manager partially constructed. If any step after
    // createHandle() throws, the constructor would otherwise exit with an active
    // plugin handle recorded only in _handleToPlugin — and since the destructor
    // does NOT run for an object whose constructor threw, that handle would leak.
    // The rollback path below mirrors what the destructor would do for the
    // half-initialized plugin.
    const auto& plugins = _pm->getPlugins();
    for(const auto& plugin : plugins)
    {
        // Set logging callback before creating handle. This is a status-returning
        // call (not throwing) so it stays outside the try block.
        auto logStatus
            = plugin->setLoggingCallback(hipdnn_backend::logging::backendLoggingCallback);
        if(logStatus != HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            HIPDNN_BACKEND_LOG_WARN("Failed to set logging callback on heuristic plugin '{}'",
                                    plugin->name());
            continue;
        }

        hipdnnHeuristicHandle_t handle = nullptr;
        std::vector<int64_t> registeredPolicyIds;

        try
        {
            // Set log level (optional). May throw from untrusted plugin code.
            hipdnnSeverity_t level = HIPDNN_SEV_INFO;
            hipdnn_backend::logging::getGlobalLogLevel(level);
            plugin->setLogLevel(level);

            // Create plugin handle (one per plugin, shared across all its policies)
            handle = plugin->createHandle();
            if(handle == nullptr)
            {
                HIPDNN_BACKEND_LOG_ERROR("Heuristic plugin '{}' returned null handle despite "
                                         "reporting success. Plugin will be unavailable.",
                                         plugin->name());
                continue;
            }

            _handleToPlugin[handle] = plugin.get();

            const auto policyIds = plugin->getAllPolicyIds();
            for(const int64_t policyId : policyIds)
            {
                _policyIdToHandle[policyId] = handle;
                registeredPolicyIds.push_back(policyId);
                HIPDNN_BACKEND_LOG_INFO("Registered heuristic policy ID {} ({}) from plugin '{}'",
                                        policyId,
                                        std::string(plugin->getPolicyName(policyId)),
                                        plugin->name());
            }

            continue; // success — skip the rollback block below
        }
        catch(const std::exception& e)
        {
            HIPDNN_BACKEND_LOG_ERROR("Failed to initialize heuristic plugin '{}': {}. "
                                     "Plugin will be unavailable.",
                                     plugin->name(),
                                     e.what());
        }
        catch(...)
        {
            HIPDNN_BACKEND_LOG_ERROR(
                "Failed to initialize heuristic plugin '{}' (unknown exception). "
                "Plugin will be unavailable.",
                plugin->name());
        }

        // Reached only via catch: undo any partial state for this plugin so the
        // resource manager remains consistent (and free the handle if we got
        // far enough to create it).
        for(const int64_t policyId : registeredPolicyIds)
        {
            _policyIdToHandle.erase(policyId);
        }
        if(handle != nullptr)
        {
            _handleToPlugin.erase(handle);
            safeDestroyHandle(plugin.get(), handle);
        }
    }
}

HeuristicPluginResourceManager::~HeuristicPluginResourceManager()
{
    // Destroy all plugin handles. safeDestroyHandle (anon namespace above)
    // swallows exceptions; the constructor uses the same helper for its
    // partial-failure rollback.
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
        // Destroy any handles we currently own before overwriting the map —
        // move-assigning into _handleToPlugin would otherwise silently drop
        // them and leak. Mirrors the destructor's loop.
        for(const auto& [handle, plugin] : _handleToPlugin)
        {
            safeDestroyHandle(plugin, handle);
        }

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
            HIPDNN_BACKEND_LOG_WARN("Failed to set device properties on heuristic plugin '{}': {}",
                                    plugin->name(),
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
    infos.reserve(_policyIdToHandle.size());

    for(const auto& [handle, plugin] : _handleToPlugin)
    {
        const auto policyIds = plugin->getAllPolicyIds();
        for(const int64_t policyId : policyIds)
        {
            HeuristicPolicyInfo info;
            info.policyId = policyId;
            info.policyName = std::string(plugin->getPolicyName(policyId));
            info.pluginName = std::string(plugin->name());
            info.pluginVersion = std::string(plugin->version());
            info.apiVersion = std::string(plugin->apiVersion());
            infos.push_back(info);
        }
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
        oss << ", Plugin: " << info.pluginName;
        oss << ", Plugin Version: " << info.pluginVersion;
        oss << ", API Version: " << info.apiVersion << "\n";
    }

    oss << "}";
    return oss.str();
}

} // namespace hipdnn_backend::plugin
