// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdint.h>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/PluginApi.h>

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
 * @brief Functions that each engine plugin must implement.
 * @{
 */

/**
 * @brief Retrieves the IDs of all engines available in the engine plugin.
 *
 * @param[out] engine_ids A pointer to an array where the IDs of available engines will be stored. The array must
 *                        have a size of at least `max_engines`.
 * @param[in] max_engines The maximum number of engine IDs that can be stored in the `engine_ids` array.
 * @param[out] num_engines A pointer to a variable where the total number of available engines will be stored.
 *                         This value may exceed `max_engines` if more engines are available than the array can
 *                         accommodate.
 *
 * @return A value of type `hipdnnPluginStatus_t` indicating the status of the operation.
 *
 * @note The caller is responsible for ensuring that the `engine_ids` array is large enough to hold up to
 *       `max_engines` IDs. If the number of available engines exceeds `max_engines`, only the first
 *       `max_engines` IDs will be returned, and the total count will be stored in `num_engines`.
 *       The function can be called with `max_engines = 0` (in this case, `engine_ids` may be `NULL`)
 *       to retrieve the total count of available engines without returning their IDs.
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

/** @} */ // End of EnginePluginFunctions group

#ifdef __cplusplus
}
#endif
// NOLINTEND
