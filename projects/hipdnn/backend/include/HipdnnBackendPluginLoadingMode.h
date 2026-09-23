// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnBackendPluginLoadingMode.h
 * @brief Plugin loading mode configuration for hipDNN
 *
 * This file defines the plugin loading modes that control how hipDNN
 * loads and manages execution engine plugins. Plugins provide the
 * actual GPU kernel implementations for deep learning operations.
 */

#pragma once

/**
 * @enum hipdnnPluginLoadingMode_ext_t
 * @brief Specifies the plugin loading mode for hipDNN (extension API)
 *
 * This enumeration defines how plugins are loaded into hipDNN.
 * The loading mode affects which plugins are available for engine
 * discovery and execution.
 *
 * @see hipdnnRegisterEnginePlugin_ext()
 */
typedef enum
{
    /**
     * @brief Additive loading mode (default)
     *
     * Loads all user-specified plugins in addition to the default plugins.
     * This is useful when you want to extend hipDNN with custom plugins
     * while retaining access to the built-in implementations.
     */
    HIPDNN_PLUGIN_LOADING_ADDITIVE,

    /**
     * @brief Absolute loading mode
     *
     * Loads only the user-specified plugin from the most recent
     * registration call, ignoring all default plugins. This is useful
     * when you want complete control over which plugins are active.
     */
    HIPDNN_PLUGIN_LOADING_ABSOLUTE,

} hipdnnPluginLoadingMode_ext_t;

/**
 * @brief Default plugin loading mode
 *
 * The default mode is HIPDNN_PLUGIN_LOADING_ADDITIVE, which loads
 * user plugins alongside the built-in default plugins.
 */
constexpr hipdnnPluginLoadingMode_ext_t HIPDNN_DEFAULT_PLUGIN_LOADING_MODE
    = HIPDNN_PLUGIN_LOADING_ADDITIVE;
