// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <cstdlib>
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

#if defined(__GLIBC__)
// The Itanium C++ ABI exit-handler interface, provided by glibc. libstdc++ declares both in
// <cxxabi.h>, which a translation unit may already include; libc++abi's <cxxabi.h> leaves them to
// the C library and does not declare them. The names are fixed by the ABI.
// NOLINTBEGIN(bugprone-reserved-identifier,readability-identifier-naming,readability-redundant-declaration)
extern "C" {
int __cxa_atexit(void (*func)(void*), void* arg, void* dsoHandle) noexcept;
void __cxa_finalize(void* dsoHandle);
}
// NOLINTEND(bugprone-reserved-identifier,readability-identifier-naming,readability-redundant-declaration)
#endif

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

namespace detail
{

#if defined(__GLIBC__)
/**
 * @brief One registration of the exit handler that sets a plugin type's shutdown flag.
 *
 * Its address is both the handler's argument and the identity the handler is registered
 * under, so retiring one registration never runs or removes another one's handler.
 */
struct ExitHookRegistration
{
    std::atomic<bool>* shutdownFlag = nullptr;
    std::atomic<bool> armed{false};
    bool registered = false;
};

// The signature is fixed by __cxa_atexit.
// NOLINTNEXTLINE(readability-non-const-parameter)
inline void setShutdownFlagIfArmed(void* registration) noexcept
{
    const auto* hook = static_cast<const ExitHookRegistration*>(registration);
    if(hook->armed.load(std::memory_order_acquire))
    {
        hook->shutdownFlag->store(true, std::memory_order_release);
    }
}

inline void registerExitHook(ExitHookRegistration& registration) noexcept
{
    registration.armed.store(true, std::memory_order_release);
    registration.registered
        = __cxa_atexit(&setShutdownFlagIfArmed, &registration, &registration) == 0;
}

/**
 * @brief Retires and frees @p registration once the plugins loaded before it are unloaded.
 *
 * The handler is disarmed first, so __cxa_finalize runs it as a no-op while freeing its
 * exit-list slot. Once exit has begun the handler may still be due to run with the
 * registration as its argument, so both are left in place.
 */
inline void retireExitHook(ExitHookRegistration* registration) noexcept
{
    if(registration == nullptr
       || (registration->registered && registration->shutdownFlag->load(std::memory_order_acquire)))
    {
        return;
    }

    const std::unique_ptr<ExitHookRegistration> owned(registration);
    if(owned->registered)
    {
        owned->armed.store(false, std::memory_order_release);
        __cxa_finalize(owned.get());
    }
}

/// Plugin manager deleter: unloads the plugins, then retires the exit handler registered after
/// they were loaded, which the deleter owns.
template <typename PluginManagerType>
struct ExitHookRetiringDelete
{
    ExitHookRegistration* registration = nullptr;

    void operator()(PluginManagerType* manager) const noexcept
    {
        std::default_delete<PluginManagerType>()(manager);
        retireExitHook(registration);
    }
};
#endif

} // namespace detail

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
 * - static std::atomic<bool>& getShutdownFlag() (constant-initialized; set at process exit)
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

        const auto newPathsSet
            = std::set<std::filesystem::path>{pluginPaths.begin(), pluginPaths.end()};
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
            if(const auto pm = Derived::getWeakPtr().lock())
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

    /**
     * @brief True once process exit has begun (static, MT-safe).
     *
     * Set by an exit handler registered after plugins are loaded (see
     * createLoadedPluginManager()). Exit handlers run in reverse registration order, and a plugin
     * registers its static destructors when it is loaded, so on Linux the handler registered
     * after the latest load runs before those destructors and before the destructors of every
     * object created earlier, including the program's namespace-scope objects. A handle or
     * descriptor released by such a destructor finds it set and is left to the operating system
     * instead of calling into plugin state that may already be torn down. On Windows the handler
     * belongs to the backend DLL and runs when that DLL detaches, after the executable's static
     * destructors, so those destructors find it unset and release normally while the plugins
     * are still loaded.
     */
    static bool isShuttingDown()
    {
        return Derived::getShutdownFlag().load(std::memory_order_acquire);
    }

protected:
    /**
     * @brief Get or create the shared plugin manager (static, MT-safe).
     *
     * This is a helper for the derived class's create() method.
     * Implementation defined below class after manager types are complete.
     */
    static std::shared_ptr<PluginManagerType> getOrCreatePluginManager();

    std::shared_ptr<PluginManagerType> _pm;

private:
    /// Creates a plugin manager, loads the configured plugins and registers the exit handler
    /// that sets the shutdown flag. Called with Derived::getMutex() held.
    static std::shared_ptr<PluginManagerType>
        createLoadedPluginManager(const PluginLoadingConfig& config);
};

// Template method implementations (defined after class so derived classes can provide complete types)

template <typename Derived, typename PluginManagerType, typename PluginType>
void PluginResourceManagerBase<Derived, PluginManagerType, PluginType>::setPluginLogLevel(
    hipdnnSeverity_t level)
{
    const std::lock_guard<std::mutex> lock(Derived::getMutex());
    const auto pm = Derived::getWeakPtr().lock();
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
        const auto& config = Derived::getConfig();
        pm = createLoadedPluginManager(config);
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
std::shared_ptr<PluginManagerType>
    PluginResourceManagerBase<Derived, PluginManagerType, PluginType>::createLoadedPluginManager(
        const PluginLoadingConfig& config)
{
    // Handlers must be registered after the plugins are loaded: exit handlers run in reverse
    // registration order, and a plugin, including one reloaded under HIPDNN_PLUGIN_UNLOAD_EAGER,
    // registers its static destructors when it is loaded.
    const auto setShutdownFlag
        = [] { Derived::getShutdownFlag().store(true, std::memory_order_release); };

#if defined(__GLIBC__)
    // Each load registers its own handler, which the manager's deleter retires once the
    // plugins are unloaded, so a handler registered after the latest load is live while any
    // plugin is loaded. Its exit-list slot is then reused by the next registration, because
    // glibc reuses freed slots at the top of the list and the unloaded plugins' slots above it
    // are freed with them.
    auto registration = std::make_unique<detail::ExitHookRegistration>();
    registration->shutdownFlag = &Derived::getShutdownFlag();
    std::unique_ptr<PluginManagerType, detail::ExitHookRetiringDelete<PluginManagerType>> manager(
        new PluginManagerType());
    manager.get_deleter().registration = registration.release();
    manager->loadPlugins(config.paths, config.mode);
    detail::registerExitHook(*manager.get_deleter().registration);
#else
    auto manager = std::make_shared<PluginManagerType>();
    manager->loadPlugins(config.paths, config.mode);
#endif

#if defined(__GLIBC__) || defined(_WIN32)
    // One handler registered at the first load and never retired, so the flag is also set
    // during exit when no plugin manager is alive, and a manager requested then is empty. On
    // Windows it is the only one: it runs when the backend DLL detaches, after the executable's
    // static destructors, so the number of registrations does not change when it runs.
    static bool s_permanentHandlerRegistered = false;
    if(!s_permanentHandlerRegistered)
    {
        std::atexit(setShutdownFlag);
        s_permanentHandlerRegistered = true;
    }
#else
    // Without a way to retire a handler, each load registers one that is never retired, so
    // each reload adds one exit handler. Setting the flag is idempotent.
    std::atexit(setShutdownFlag);
#endif

    return std::shared_ptr<PluginManagerType>(std::move(manager));
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
