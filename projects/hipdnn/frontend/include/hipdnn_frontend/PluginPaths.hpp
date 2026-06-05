// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file PluginPaths.hpp
 * @brief Frontend API for configuring engine plugin search paths
 *
 * Provides a type-safe C++ wrapper around the backend's
 * hipdnnSetEnginePluginPaths_ext() function. Accepts any iterable
 * container whose elements are convertible to std::filesystem::path
 * (e.g., std::filesystem::path, std::string, or const char*).
 *
 * @code{.cpp}
 * #include <hipdnn_frontend.hpp>
 *
 * // Load plugins from specific folders only (absolute mode)
 * std::vector<std::filesystem::path> paths = {"./plugins/engine_a.so", "./plugins/engine_b.so"};
 * auto err = hipdnn_frontend::setEnginePluginPaths(paths, PluginLoadingMode::MODE_ABSOLUTE);
 * if (err.is_bad()) {
 *     // handle error
 * }
 *
 * // Add additional folders to load plugins from (additive mode)
 * std::vector<std::string> paths2 = {"./plugins/engine_a.so", "./plugins/engine_b.so"};
 * auto err2 = hipdnn_frontend::setEnginePluginPaths(paths2, PluginLoadingMode::MODE_ADDITIVE);
 * @endcode
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend
{

/**
 * @enum PluginLoadingMode
 * @brief Specifies the plugin loading mode for hipDNN
 *
 * Controls how user-specified plugins interact with default plugins.
 */
enum class PluginLoadingMode
{
    MODE_ADDITIVE, ///< Load user-specified plugins alongside defaults
    MODE_ABSOLUTE, ///< Load only user-specified plugins
};
typedef PluginLoadingMode PluginLoadingMode_t; ///< @brief Type alias for PluginLoadingMode

namespace detail
{

// Convert frontend PluginLoadingMode to backend hipdnnPluginLoadingMode_ext_t.
// Returns std::nullopt for unrecognized mode values.
inline std::optional<hipdnnPluginLoadingMode_ext_t>
    toBackendPluginLoadingMode(PluginLoadingMode mode)
{
    switch(mode)
    {
    case PluginLoadingMode::MODE_ADDITIVE:
        return HIPDNN_PLUGIN_LOADING_ADDITIVE;
    case PluginLoadingMode::MODE_ABSOLUTE:
        return HIPDNN_PLUGIN_LOADING_ABSOLUTE;
    default:
        return std::nullopt;
    }
}

} // namespace detail

/**
 * @brief Configure the engine plugin search paths (container overload)
 *
 * Sets the list of directories or file paths that hipDNN will search
 * when loading engine plugins. Must be called before creating a handle.
 *
 * Accepts any iterable container whose elements are convertible to
 * std::filesystem::path (e.g., std::filesystem::path, std::string,
 * or const char*).
 *
 * @tparam Container An iterable container with size() and range-for support
 * @param pluginPaths Container of paths to plugin directories or files
 * @param mode Loading mode: PluginLoadingMode::MODE_ADDITIVE loads user paths
 *             alongside defaults; PluginLoadingMode::MODE_ABSOLUTE loads only
 *             user-specified paths.
 * @return Error indicating success or failure
 */
template <typename Container>
inline Error setEnginePluginPaths(const Container& pluginPaths, PluginLoadingMode mode)
{
    static_assert(std::is_constructible_v<
                      std::filesystem::path,
                      std::decay_t<decltype(*std::begin(std::declval<const Container&>()))>>,
                  "Container elements must be convertible to std::filesystem::path "
                  "(e.g., std::filesystem::path, std::string, or const char*)");

    auto backendMode = detail::toBackendPluginLoadingMode(mode);
    if(!backendMode)
    {
        return {ErrorCode::INVALID_VALUE,
                "Invalid PluginLoadingMode value: " + std::to_string(static_cast<int>(mode))};
    }

    std::vector<std::string> pathStrings;
    pathStrings.reserve(pluginPaths.size());
    for(const auto& p : pluginPaths)
    {
        pathStrings.push_back(std::filesystem::path(p).string());
    }

    std::vector<const char*> cPaths;
    cPaths.reserve(pathStrings.size());
    for(const auto& s : pathStrings)
    {
        cPaths.push_back(s.c_str());
    }

    auto status = detail::hipdnnBackend()->setEnginePluginPathsExt(
        cPaths.size(), cPaths.data(), *backendMode);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to set engine plugin paths");
    return {};
}

/**
 * @brief Retrieve the list of engine plugin paths loaded by a hipDNN handle
 *
 * Queries the backend for all engine plugin paths that were loaded when
 * the handle was created.
 *
 * @param handle A valid hipDNN handle created with hipdnnCreate
 * @param[out] paths Populated with the loaded plugin paths on success;
 *                   cleared on failure or when no plugins are loaded
 * @return Error indicating success or failure
 */
inline Error getLoadedEnginePluginPaths(hipdnnHandle_t handle,
                                        std::vector<std::filesystem::path>& paths)
{
    // Query call: retrieve the number of loaded plugins and the max path length
    size_t numPlugins = 0;
    size_t maxPathLength = 0;
    auto status = detail::hipdnnBackend()->getLoadedEnginePluginPathsExt(
        handle, &numPlugins, nullptr, &maxPathLength);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to query loaded engine plugin count");

    if(numPlugins == 0)
    {
        paths.clear();
        return {};
    }

    // Allocate buffers for each plugin path string
    std::vector<std::vector<char>> pathBuffers(numPlugins, std::vector<char>(maxPathLength));
    std::vector<char*> pluginPathsC(numPlugins);
    for(size_t i = 0; i < numPlugins; ++i)
    {
        pluginPathsC[i] = pathBuffers[i].data();
    }

    // Retrieve call: populate the buffers with actual path strings
    status = detail::hipdnnBackend()->getLoadedEnginePluginPathsExt(
        handle, &numPlugins, pluginPathsC.data(), &maxPathLength);
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Failed to retrieve loaded engine plugin paths");

    // Convert C strings to std::filesystem::path
    paths.clear();
    paths.reserve(numPlugins);
    for(size_t i = 0; i < numPlugins; ++i)
    {
        paths.emplace_back(pluginPathsC[i]);
    }
    return {};
}

} // namespace hipdnn_frontend
