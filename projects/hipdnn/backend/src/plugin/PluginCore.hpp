// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "Helpers.hpp"
#include "HipdnnBackendPluginLoadingMode.h"
#include "PlatformUtils.hpp"
#include "logging/Logging.hpp"
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDataTypeHelpers.hpp>

#include "HipdnnException.hpp"
#include "SharedLibrary.hpp"

namespace hipdnn_backend::plugin
{

// The PluginBase is the base class for all plugins.
class PluginBase
{
protected:
    // The constructor is protected to prevent direct instantiation of the class.
    PluginBase(SharedLibrary&& lib);

    // This constructor is used for mocking purposes in tests.
    PluginBase();

public:
    // Prevent copying
    PluginBase(const PluginBase&) = delete;
    PluginBase& operator=(const PluginBase&) = delete;

    // Allow moving
    PluginBase(PluginBase&& other) = default;
    PluginBase& operator=(PluginBase&& other) = default;

    virtual ~PluginBase() = default;

    virtual std::string_view name() const;
    virtual std::string_view version() const;
    virtual std::string_view apiVersion() const;
    virtual hipdnnPluginType_t type() const;

    /**
     * @brief Returns the plugin's API version parsed into a structured
     *        `Version` object, or `std::nullopt` if the version string is
     *        malformed.
     *
     * Plugin managers validate this at load time and reject malformed
     * plugin API versions before dispatch.
     */
    std::optional<hipdnn_data_sdk::utilities::Version> parsedApiVersion() const;

    /**
     * @brief Returns the plugin's name captured during plugin load.
     */
    const std::string& cachedName() const;

    static hipdnnPluginType_t getPluginType();

    hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback) const;

    /**
     * @brief Sets the log level for the plugin.
     *
     * This method calls the plugin's hipdnnPluginSetLogLevel function if available.
     * If the plugin does not implement this function, the call is a no-op and returns success.
     *
     * @param level The log level to set.
     * @return hipdnnPluginStatus_t indicating success or failure.
     */
    hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level) const;

protected:
    // This function must not throw as it is used during error handling.
    std::string_view getLastErrorString() const noexcept;

    template <typename Callable, typename... Args>
    void invokePluginFunction(const char* description, Callable&& func, Args&&... args) const
    {
        auto status = func(std::forward<Args>(args)...);
        if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                  std::string("Failed to ") + description + ". Status: "
                                      + toString(status) + "(" + std::to_string(status) + ")"
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
    hipdnnPluginStatus_t (*_funcGetName)(const char**);
    hipdnnPluginStatus_t (*_funcGetVersion)(const char**);
    hipdnnPluginStatus_t (*_funcGetApiVersion)(const char**);
    hipdnnPluginStatus_t (*_funcGetType)(hipdnnPluginType_t*);
    void (*_funcGetLastErrorStr)(const char**);
    hipdnnPluginStatus_t (*_funcSetLoggingCallback)(hipdnnCallback_t);
    hipdnnPluginStatus_t (*_funcSetLogLevel)(hipdnnSeverity_t);

    // The plugin name is captured during construction. Mock/default
    // construction uses a deterministic fallback so cachedName() remains safe.
    std::string _name = "uninitialized_plugin";
};

// The PluginManagerBase is responsible for loading and unloading plugins. This class is the base class for all plugin managers.
template <class Plugin>
class PluginManagerBase
{
    static_assert(std::is_base_of_v<PluginBase, Plugin>, "Plugin must be derived from PluginBase");

protected:
    explicit PluginManagerBase(std::set<std::filesystem::path> defaultPaths)
        : _defaultPluginPaths(std::move(defaultPaths))
    {
    }

    // This function is called before adding a plugin to the plugin list.
    // The function must throw Hipdnn_exception if the plugin is not valid.
    virtual void validateBeforeAdding([[maybe_unused]] const Plugin& plugin) {}

    // This function is called after the plugin is added to the plugin list.
    virtual void actionAfterAdding([[maybe_unused]] const Plugin& plugin) {}

    // For cases where tests need to override the default plugin search paths
    static std::set<std::filesystem::path>
        getPluginSearchPaths(const char* envVarName,
                             const std::set<std::filesystem::path>& defaultPaths)
    {
        const auto envPath = hipdnn_data_sdk::utilities::getEnv(envVarName);
        if(!envPath.empty())
        {
            // Could make this take multiple dirs
            return {std::filesystem::path(envPath)};
        }
        return defaultPaths;
    }

public:
    virtual ~PluginManagerBase() = default;

    virtual void loadPlugins(const std::set<std::filesystem::path>& customPaths,
                             hipdnnPluginLoadingMode_ext_t mode)
    {
        std::set<std::filesystem::path> pathsToProcess;

        if(mode == HIPDNN_PLUGIN_LOADING_ADDITIVE)
        {
            pathsToProcess.insert(_defaultPluginPaths.begin(), _defaultPluginPaths.end());
        }

        pathsToProcess.insert(customPaths.begin(), customPaths.end());

        if(mode == HIPDNN_PLUGIN_LOADING_ABSOLUTE)
        {
            clearPlugins();
        }

        std::set<std::filesystem::path> filesToLoad;
        for(const auto& path : pathsToProcess)
        {
            try
            {
                auto resolvedPath = path;
                if(path.is_relative())
                {
                    // If the path is relative, resolve it to our current module directory instead of allowing
                    // std::filesystem to resolve it relative to the current working directory.
                    resolvedPath = hipdnn_backend::platform_utilities::getCurrentModuleDirectory()
                                   / resolvedPath;
                }

                resolvedPath = std::filesystem::weakly_canonical(resolvedPath);

                if(std::filesystem::is_directory(resolvedPath))
                {
                    scanDirectoryForPlugins(resolvedPath, filesToLoad);
                }
                // Cannot necessarily check that a custom file path exists, because it can be platform opaque
                else if(!resolvedPath.filename().empty())
                {
                    filesToLoad.insert(resolvedPath);
                }
                else
                {
                    HIPDNN_BACKEND_LOG_WARN(
                        "Plugin path '{}' is invalid - expected either a directory "
                        "containing plugins or a path to a plugin file",
                        path.string());
                }
            }
            catch(const std::filesystem::filesystem_error& e)
            {
                HIPDNN_BACKEND_LOG_ERROR(
                    "Error resolving plugin path '{}': {}", path.string(), e.what());
            }
        }

        for(const auto& filePath : filesToLoad)
        {
            loadPluginFromFile(filePath);
        }
    }

    virtual const std::vector<std::shared_ptr<Plugin>>& getPlugins() const
    {
        return _plugins;
    }

    virtual const std::set<std::filesystem::path>& getLoadedPluginFiles() const
    {
        return _loadedPluginFiles;
    }

private:
    void clearPlugins()
    {
        _plugins.clear();
        _loadedPluginFiles.clear();
    }

    void scanDirectoryForPlugins(const std::filesystem::path& dirPath,
                                 std::set<std::filesystem::path>& pathsToLoad) const
    {
        try
        {
            for(const auto& entry : std::filesystem::directory_iterator(dirPath))
            {
                const auto& path = entry.path();
                if(entry.is_regular_file()
                   && path.extension() == hipdnn_data_sdk::utilities::SHARED_LIB_EXT)
                {
                    pathsToLoad.insert(std::filesystem::weakly_canonical(path));
                }
            }
        }
        catch(const std::filesystem::filesystem_error& e)
        {
            HIPDNN_BACKEND_LOG_WARN(
                "Error scanning plugin directory {}: {}", dirPath.string(), e.what());
        }
    }

    void loadPluginFromFile(const std::filesystem::path& filePath)
    {

        HIPDNN_BACKEND_LOG_INFO("Attempting to load plugin from [{}]", filePath.string());

        hipdnn_backend::tryCatch(
            [&]() {
                SharedLibrary lib(filePath);
                const auto libraryPath = lib.libraryPath();

                // Shared library ensures an injective, weakly canonical mapping to a path
                if(_loadedPluginFiles.find(libraryPath) != _loadedPluginFiles.end())
                {
                    return;
                }

                std::shared_ptr<Plugin> plugin
                    = std::shared_ptr<Plugin>(new Plugin(std::move(lib)));

                const auto name = plugin->name();
                const auto version = plugin->version();
                const auto type = plugin->type();

                // For now only use engine or unspecified plugin types
                if(type != Plugin::getPluginType())
                {
                    throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                                          std::string("Plugin type mismatch: expected ")
                                              + toString(Plugin::getPluginType()) + ", got "
                                              + toString(type));
                }

                plugin->setLoggingCallback(logging::backendLoggingCallback);

                // Synchronize plugin log level with backend's global log level
                hipdnnSeverity_t currentLogLevel{};
                logging::getGlobalLogLevel(currentLogLevel);
                plugin->setLogLevel(currentLogLevel);

                validateBeforeAdding(*plugin);

                _plugins.emplace_back(std::move(plugin));
                _loadedPluginFiles.insert(libraryPath);

                HIPDNN_BACKEND_LOG_INFO("Plugin loaded successfully: {}", filePath.string());
                HIPDNN_BACKEND_LOG_INFO("Plugin info: name={}, version={}, type={}({})",
                                        name,
                                        version,
                                        toString(type),
                                        static_cast<int>(type));

                actionAfterAdding(*_plugins.back());
            },
            fmt::format("Error loading plugin from [{}]: ", filePath.string()));
    }

    std::vector<std::shared_ptr<Plugin>> _plugins;
    std::set<std::filesystem::path> _loadedPluginFiles;
    std::set<std::filesystem::path> _defaultPluginPaths;
};

} // namespace hipdnn_backend::plugin
