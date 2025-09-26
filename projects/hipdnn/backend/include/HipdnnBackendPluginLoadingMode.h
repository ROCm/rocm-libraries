// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @enum hipdnnPluginLoadingMode_ext_t
 * @brief Specifies the plugin loading mode for hipDNN.
 *
 * This enumeration defines how plugins are loaded into hipDNN:
 * - HIPDNN_PLUGIN_LOADING_ADDITIVE: Loads all user-specified plugins in addition to the default plugins.
 * - HIPDNN_PLUGIN_LOADING_ABSOLUTE: Loads only the user-specified plugin from the most-recent function call, ignoring the defaults.
 */
typedef enum
{
    HIPDNN_PLUGIN_LOADING_ADDITIVE,
    HIPDNN_PLUGIN_LOADING_ABSOLUTE,
} hipdnnPluginLoadingMode_ext_t;

constexpr hipdnnPluginLoadingMode_ext_t HIPDNN_DEFAULT_PLUGIN_LOADING_MODE
    = HIPDNN_PLUGIN_LOADING_ADDITIVE;
