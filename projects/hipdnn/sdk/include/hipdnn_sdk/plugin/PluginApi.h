// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/logging/CallbackTypes.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

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

#ifdef __cplusplus
#define HIPDNN_PLUGIN_NODISCARD [[nodiscard]]
#else
#define HIPDNN_PLUGIN_NODISCARD
#endif

/**
 * @file PluginApi.h
 * @brief The hipDNN Plugin API
 *
 * This file contains the definitions and declarations for the hipDNN Plugin API.
 * The API allows users to create and manage custom plugins for hipDNN.
 */

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup PluginFunctions Plugin API Functions
 * @brief Functions that every plugin must implement.
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
 * @brief The maximum length for plugin error strings.
 *
 * Plugins are recommended to adhere to this value.
 * The length includes the null-terminating character.
 * This is the recommended size for the internal per-thread buffer.
 */
#define HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH 2048

/**
 * @brief Sets the logging callback function for the plugin.
 *
 * @param[in] callback The logging callback function to use.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback);

/** @} */ // End of PluginFunctions group

#ifdef __cplusplus
}
#endif
// NOLINTEND
