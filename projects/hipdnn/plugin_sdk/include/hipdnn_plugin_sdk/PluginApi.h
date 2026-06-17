// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

/**
 * @file PluginApi.h
 * @brief The hipDNN Plugin API - base interface for all plugins
 *
 * This file defines the base plugin API that ALL hipDNN plugins must implement,
 * regardless of plugin type (engine, heuristic, etc.).
 *
 * All plugins must export these functions:
 * - hipdnnPluginGetName - Returns the plugin name
 * - hipdnnPluginGetVersion - Returns the plugin version
 * - hipdnnPluginGetApiVersion - Returns the API version
 * - hipdnnPluginGetType - Returns the plugin type
 * - hipdnnPluginGetLastErrorString - Returns per-thread error messages
 * - hipdnnPluginSetLoggingCallback - Sets the logging callback
 * - hipdnnPluginSetLogLevel - Sets the log level (optional)
 */

/**
 * @defgroup PluginCommonInfrastructure Common Plugin Infrastructure
 * @brief Platform-specific macros and constants used across all plugin types.
 * @{
 */

#ifdef _WIN32
#ifdef HIPDNN_PLUGIN_STATIC_DEFINE
#define HIPDNN_PLUGIN_EXPORT
#else
#define HIPDNN_PLUGIN_EXPORT __declspec(dllexport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HIPDNN_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#error "Unsupported platform or compiler"
#endif

/**
 * @brief Nodiscard attribute for plugin functions.
 *
 * Marks plugin functions whose return values should not be ignored.
 */
#ifdef __cplusplus
#define HIPDNN_PLUGIN_NODISCARD [[nodiscard]]
#else
#define HIPDNN_PLUGIN_NODISCARD
#endif

/**
 * @brief The maximum length for plugin error strings.
 *
 * Plugins are recommended to adhere to this value for error messages.
 * The length includes the null-terminating character.
 * This is the recommended size for internal per-thread error buffers.
 */
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH 2048

/** @} */ // End of PluginCommonInfrastructure group

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup PluginFunctions Plugin API Functions
 * @brief Functions that define the plugin API. Most are required; optional functions
 * are marked in their documentation.
 * @{
 */

/**
 * @brief Retrieves the name of the plugin.
 *
 * @param[out] name A pointer to a pointer where the address of the plugin name will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginGetName(const char** name);

/**
 * @brief Retrieves the version of the plugin.
 *
 * @param[out] version A pointer to a pointer where the address of the plugin version will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginGetVersion(const char** version);

/**
 * @brief Returns the API version that the plugin supports
 *
 * @param[out] version A pointer to a pointer where the address of the plugin api version will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation
 *
 * @note The version returned by this function must be a version for which all plugin API functions have been
 * implemented for your class of plugin:
 * - Engine plugins need to implement PluginApi.h and EnginePluginApi.h
 * - Heuristic plugins need to implement PluginApi.h and HeuristicsPluginApi.h (once it exists)
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginGetApiVersion(const char** version);

/**
 * @brief Retrieves the type of the plugin.
 *
 * @param[out] type A pointer to a `hipdnnPluginType_t` where the plugin type will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginGetType(hipdnnPluginType_t* type);

/**
 * @brief Retrieves the last error string from the plugin.
 *
 * @param[out] error_str A pointer to a constant character pointer where the address of the last error string will
 *                       be stored.
 *
 * @note Plugins must store the entire error string internally and maintain it on a per-thread basis. This function
 *       should only be called after receiving an error status from the plugin API in the same thread. The pointer
 *       returned by this function should not be stored, as its sole purpose is to retrieve the string and
 *       immediately use it to form a complete error message. If the user passes a null pointer, this function does
 *       nothing.
 */
HIPDNN_PLUGIN_EXPORT void hipdnnPluginGetLastErrorString(const char** error_str);

/**
 * @brief Sets the logging callback function for the plugin.
 *
 * @param[in] callback The logging callback function to use.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback);

/**
 * @brief Sets the log level for the plugin.
 *
 * This function synchronizes the plugin's log level with the backend's global log level.
 * The backend calls this function when the global log level changes or when a plugin is loaded.
 *
 * @param[in] level The log level to set.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note This function is optional for backwards compatibility with plugins built against
 *       earlier SDK versions. Plugins that do not implement this function will continue
 *       to work but will not receive log level updates from the backend.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetLogLevel(hipdnnSeverity_t level);

/** @} */ // End of PluginFunctions group

#ifdef __cplusplus
}
#endif
// NOLINTEND
