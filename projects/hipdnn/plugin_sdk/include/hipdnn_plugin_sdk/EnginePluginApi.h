// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginApi.h>

/**
 * @file EnginePluginApi.h
 * @brief hipDNN Engine Plugin API
 *
 * This file contains the definitions and declarations for the hipDNN Engine Plugin API.
 * The API allows users to create and manage custom plugins for hipDNN.
 */

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup EnginePluginFunctions Engine Plugin API Functions
 * @brief Required and optional functions that engine plugins may implement.
 * @{
 */

/**
 * @brief Retrieves the IDs of all engines available in the engine plugin.
 *
 * @param[out] engine_ids A pointer to an array where the IDs of available engines will be stored. The array must
 *                        have a size of at least `max_engines`.
 * @param[in] max_engines The maximum number of engine IDs that can be stored in the `engine_ids` array.
 * @param[out] num_engines A pointer to a variable where the total number of available engines will be stored.
 *                         When `max_engines` is zero, `num_engines` must be upated with the total number of engines
 *                         the plugin can return. When `max_engines` is non-zero, `num_engines` is set to the number of
 *                         engines stored in the `engine_ids` array.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The caller is responsible for ensuring that the `engine_ids` array is large enough to hold up to
 *       `max_engines` IDs. If the number of available engines exceeds `max_engines`, only the first
 *       `max_engines` IDs will be returned, and `num_engines` is set to the number of engines stored in
 *       the `engine_ids` array.
 * @note This function will be called with `max_engines = 0` (in this case, `engine_ids` may be `NULL`)
 *       to retrieve the total count of available engines without returning their IDs. In this situation
 *       the plugin must set `num_engines` to the total number of engines available in the plugin.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(
    int64_t* engine_ids, uint32_t max_engines, uint32_t* num_engines);

/**
 * @brief Creates a new engine plugin handle.
 *
 * @param[out] handle A pointer to a variable where the created engine plugin handle will be stored. The handle is
 *                    used for subsequent operations on the engine plugin.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The caller is responsible for ensuring that the handle is destroyed properly to avoid resource leaks.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle);

/**
 * @brief Destroys an engine plugin handle and releases associated resources.
 *
 * @param[in] handle An engine plugin handle to be destroyed.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The handle becomes invalid after this function is called.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle);

/**
 * @brief Sets the HIP stream for the specified engine plugin handle.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] stream A HIP stream to be associated with the engine plugin handle.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The caller must ensure that the provided stream remains valid for the duration of operations performed
 *       using the engine plugin handle.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle, hipStream_t stream);

/**
 * @brief Retrieves the IDs of the applicable engines for a given operation graph.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] op_graph A pointer to a structure where the serialized `Graph` from `graph.fbs` is stored.
 * @param[out] engine_ids A pointer to an array where the IDs of the applicable engines will be stored. The array
 *                        must have a size of at least `max_engines`.
 * @param[in] max_engines The maximum number of engine IDs that can be stored in the `engine_ids` array.
 * @param[out] num_engines A pointer to a variable where the total number of applicable engines will be stored.
 *                         This value may exceed `max_engines` if more engines are applicable than the array can
 *                         accommodate.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The caller is responsible for ensuring that the `engine_ids` array is large enough to hold up to
 *       `max_engines` IDs. If the number of applicable engines exceeds `max_engines`, only the first
 *       `max_engines` IDs will be returned, and the total count will be stored in `num_engines`.
 *       The function can be called with `max_engines = 0` (in this case `engine_ids` may be `NULL`)
 *       to retrieve the total count of applicable engines without returning their IDs.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* op_graph,
                                             int64_t* engine_ids,
                                             uint32_t max_engines,
                                             uint32_t* num_engines);

/**
 * @brief Retrieves the details of a specific engine using its ID and the operation graph.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] engine_id The ID of the engine whose details are to be retrieved.
 * @param[in] op_graph A pointer to a structure where the serialized `Graph` from `graph.fbs` is stored.
 * @param[in,out] engine_details A pointer to a structure where the serialized `EngineDetails` from
 *                               `engine_details.fbs` will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The `engine_details` structure is provided by the user, while the function fills in its fields, including
 *       allocating the buffer for the serialized `EngineDetails`. After use, this memory must be freed using
 *       hipdnnEnginePluginDestroyEngineDetails().
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engine_id,
                                       const hipdnnPluginConstData_t* op_graph,
                                       hipdnnPluginConstData_t* engine_details);

/**
 * @brief Destroys the `engine_details` object and releases the associated resources.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in,out] engine_details A pointer to a structure where the serialized `EngineDetails` from
 *                               `engine_details.fbs` is stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note This function takes a structure as input, deallocates the buffer, and sets all fields to 0.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                           hipdnnPluginConstData_t* engine_details);

/**
 * @brief Retrieves the required workspace size for a specific engine configuration and an operation graph.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] engine_config A pointer to a structure where the serialized `EngineConfig` from `engine_config.fbs`
 *                          is stored.
 * @param[in] op_graph A pointer to a structure where the serialized `Graph` from `graph.fbs` is stored.
 * @param[out] workspace_size A pointer to a variable where the required workspace size (in bytes) will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engine_config,
                                       const hipdnnPluginConstData_t* op_graph,
                                       size_t* workspace_size);

/**
 * @brief Creates an execution context for a specific engine configuration and an operation graph.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] engine_config A pointer to a structure where the serialized `EngineConfig` from
 *                          `engine_config.fbs` is stored.
 * @param[in] op_graph A pointer to a structure where the serialized `Graph` from `graph.fbs` is stored.
 * @param[out] execution_context A pointer to a variable where the created execution context will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note After calling this function, the engine configuration and the operation graph are no longer needed, as
 *       the execution context stores all the required information internally. Internal resources are allocated
 *       for the execution context, and the user is responsible for releasing these resources by calling
 *       `hipdnnEnginePluginDestroyExecutionContext()` when the execution context is no longer needed.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(
        hipdnnEnginePluginHandle_t handle,
        const hipdnnPluginConstData_t* engine_config,
        const hipdnnPluginConstData_t* op_graph,
        hipdnnEnginePluginExecutionContext_t* execution_context);

/**
 * @brief Destroys an execution context and releases the associated resources.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] execution_context The execution context to be destroyed.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The execution context becomes invalid after this function is called.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyExecutionContext(
        hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context);

/**
 * @brief Serializes an execution context into plugin-owned opaque bytes.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] execution_context The execution context to serialize.
 * @param[in,out] serialized_context A pointer to a structure where the plugin-specific serialized execution
 *                                  context bytes will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note This function is optional. Plugins that do not export it do not support compiled execution plan
 *       serialization.
 * @note The serialized bytes are plugin-specific and are treated as opaque by hipDNN.
 * @note The plugin owns the returned buffer. hipDNN must release it by calling
 *       hipdnnEnginePluginDestroySerializedExecutionContext().
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginSerializeExecutionContext(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t execution_context,
        hipdnnPluginConstData_t* serialized_context);

/**
 * @brief Destroys plugin-owned serialized execution context bytes.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in,out] serialized_context A pointer to the serialized execution context bytes returned by
 *                                  hipdnnEnginePluginSerializeExecutionContext().
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note This function is optional. Plugins that export hipdnnEnginePluginSerializeExecutionContext must also
 *       export this function.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroySerializedExecutionContext(
        hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* serialized_context);

/**
 * @brief Creates an execution context from plugin-specific serialized bytes.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] serialized_context A pointer to the plugin-specific serialized execution context bytes.
 * @param[out] execution_context A pointer to a variable where the created execution context will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note This function is optional. Plugins that export hipdnnEnginePluginSerializeExecutionContext must also
 *       export this function.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContextFromSerialized(
        hipdnnEnginePluginHandle_t handle,
        const hipdnnPluginConstData_t* serialized_context,
        hipdnnEnginePluginExecutionContext_t* execution_context);

/**
 * @brief Retrieves the required workspace size for a given execution context.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] execution_context The execution context that encapsulates the operation graph and the engine
 *                             configuration.
 * @param[out] workspace_size A pointer to a variable where the required workspace size (in bytes) will be stored.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t execution_context,
        size_t* workspace_size);

/**
 * @brief Executes an operation graph using a specified execution context.
 *
 * @param[in] handle The engine plugin handle.
 * @param[in] execution_context The execution context that encapsulates the operation graph and the engine
 *                              configuration to be executed.
 * @param[in] workspace A pointer to the workspace memory allocated by the user. The workspace must be large
 *                      enough to meet the requirements of the operation graph.
 * @param[in] device_buffers A pointer to an array of device buffers.
 * @param[in] num_device_buffers The number of device buffers provided in the `device_buffers` array.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t execution_context,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* device_buffers,
                                     uint32_t num_device_buffers);

/**
 * @brief Optional override-aware execute entry point for engine plugins.
 *
 * The host selects this entry only when the plugin reports a new enough API
 * version and exports this symbol. If either requirement is missing, the
 * plugin is ineligible for override execution. A plugin that exports this
 * symbol but cannot support overrides for a particular graph should reject
 * that graph during applicability; the execute implementation may then be an
 * empty stub.
 *
 * @param[in] handle             The engine plugin handle.
 * @param[in] execution_context  Execution context produced for this engine
 *                               configuration.
 * @param[in] workspace          User-provided workspace memory; may be NULL
 *                               when the workspace size is zero.
 * @param[in] device_buffers     Tensor unique-id-keyed device pointer table
 *                               (same layout as the existing execute entry).
 * @param[in] num_device_buffers Number of entries in `device_buffers`.
 * @param[in] num_overrides      Number of tensor unique-ids in the override
 *                               selectors. The current host dispatches empty
 *                               override sets through the existing execute
 *                               entry instead of this function.
 * @param[in] override_unique_ids Pointer to an array of length `num_overrides`
 *                                identifying which graph tensors carry an
 *                                override. Each unique id must also appear in
 *                                `device_buffers`.
 * @param[in] override_lengths   Pointer to an array of length `num_overrides`
 *                                holding the rank (number of dimensions) of
 *                                each override tensor's shape and stride
 *                                vectors.
 * @param[in] override_shapes    Pointer to an array of length `num_overrides`
 *                                where the i-th entry points to a buffer of
 *                                `override_lengths[i]` int64 values: the
 *                                runtime shape for the i-th override tensor.
 * @param[in] override_strides   Pointer to an array of length `num_overrides`
 *                                where the i-th entry points to a buffer of
 *                                `override_lengths[i]` int64 values: the
 *                                runtime strides for the i-th override tensor.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the
 *         operation.
 *
 * @note **Pointer-lifetime invariant.** All pointers passed via
 *       `override_unique_ids`, `override_lengths`, `override_shapes`,
 *       `override_strides`, and the inner per-tensor buffers reachable
 *       through them are owned by the host and are valid only for the
 *       duration of this call. Plugins must not retain any of these pointers
 *       past return; copy any data that must outlive the call. This matches
 *       the lifetime of `device_buffers` for the existing execute entry.
 *
 * @note The host resolves this symbol through the loader's optional-symbol
 *       mechanism for backward compatibility with pre-1.1 plugins.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraphWithOverrides(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t execution_context,
        void* workspace,
        const hipdnnPluginDeviceBuffer_t* device_buffers,
        uint32_t num_device_buffers,
        uint32_t num_overrides,
        const int64_t* override_unique_ids,
        const uint32_t* override_lengths,
        const int64_t* const* override_shapes,
        const int64_t* const* override_strides);

/** @} */ // End of EnginePluginFunctions group

#ifdef __cplusplus
}
#endif
// NOLINTEND
