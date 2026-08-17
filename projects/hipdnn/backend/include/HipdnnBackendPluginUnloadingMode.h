// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @enum hipdnnPluginUnloadingMode_ext_t
 * @brief Specifies the plugin unloading mode for hipDNN.
 *
 * This enumeration defines when plugins are unloaded from memory:
 * - HIPDNN_PLUGIN_UNLOAD_LAZY: Keeps plugins loaded until application exit or when
 *   hipdnnSetEnginePluginPaths_ext is called. This avoids expensive plugin reloading
 *   when handles are frequently created and destroyed.
 * - HIPDNN_PLUGIN_UNLOAD_EAGER: Unloads plugins immediately when all handles are released.
 */
typedef enum
{
    HIPDNN_PLUGIN_UNLOAD_LAZY,
    HIPDNN_PLUGIN_UNLOAD_EAGER,
} hipdnnPluginUnloadingMode_ext_t;

constexpr hipdnnPluginUnloadingMode_ext_t HIPDNN_DEFAULT_PLUGIN_UNLOADING_MODE
    = HIPDNN_PLUGIN_UNLOAD_LAZY;
