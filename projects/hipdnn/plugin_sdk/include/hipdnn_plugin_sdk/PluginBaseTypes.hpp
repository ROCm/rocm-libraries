// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file PluginBaseTypes.hpp
 * @brief Base struct definitions for plugin opaque handle types.
 *
 * These empty structs provide the type names referenced by the opaque pointers
 * in PluginApiDataTypes.h. Plugin implementations inherit from these structs
 * and add their plugin-specific members.
 *
 * The EnginePluginImpl.inl file casts from these base types to the plugin's
 * derived types using the HIPDNN_PLUGIN_HANDLE_TYPE and HIPDNN_PLUGIN_CONTEXT_TYPE
 * macros.
 */

/**
 * @brief Base struct for engine plugin handles.
 *
 * Plugin implementations should inherit from this struct.
 *
 * Example:
 * @code
 * struct HipdnnMiopenHandle : HipdnnEnginePluginHandle
 * {
 *     miopenHandle_t miopenHandle;
 *     void setStream(hipStream_t stream) { ... }
 *     EngineManager<...>& getEngineManager() { ... }
 *     // ... plugin-specific members
 * };
 * @endcode
 */
struct HipdnnEnginePluginHandle
{
    virtual ~HipdnnEnginePluginHandle() = default;
};

/**
 * @brief Base struct for engine execution contexts.
 *
 * Plugin implementations should inherit from ExecutionContextBase<THandle, TSettings>
 * which provides plan storage and settings management.
 *
 * Example:
 * @code
 * struct HipdnnMiopenContext :
 *     HipdnnEnginePluginExecutionContext,
 *     ExecutionContextBase<HipdnnMiopenHandle, HipdnnMiopenSettings>
 * {};
 * @endcode
 */
struct HipdnnEnginePluginExecutionContext
{
    virtual ~HipdnnEnginePluginExecutionContext() = default;
};
