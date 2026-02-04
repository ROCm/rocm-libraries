// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file PluginLogging.hpp
 * @brief Dual-mode logging header for plugins
 *
 * This header provides logging macros that work in two modes:
 *
 * 1. HIPDNN_PLUGIN_USE_SPDLOG defined:
 *    - Uses spdlog/fmt style logging with format strings
 *    - Usage: HIPDNN_LOG_INFO("Value: {}", value);
 *    - Requires linking against spdlog
 *    - Used by production plugins (miopen-provider, hipblaslt-provider)
 *
 * 2. HIPDNN_PLUGIN_USE_SPDLOG not defined (default):
 *    - Uses stream-style logging with operator<<
 *    - Usage: HIPDNN_LOG_INFO("Value: " << value);
 *    - No spdlog dependency required
 *    - Used by test plugins and lightweight integrations
 *
 * Before including this header, define COMPONENT_NAME to identify the plugin:
 *   #define COMPONENT_NAME "my_plugin"
 */

#include <hipdnn_data_sdk/logging/Logger.hpp>
