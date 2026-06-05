// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stddef.h>
#include <stdint.h>

// Heuristic plugins implement the base PluginApi.h functions PLUS heuristic-specific extensions
#include <hipdnn_plugin_sdk/PluginApi.h>

/**
 * @file HeuristicsPluginApi.h
 * @brief hipDNN Heuristics Plugin API
 *
 * This file defines extensions to the base plugin API (PluginApi.h) for heuristic/selection
 * policy plugins that control engine ordering.
 *
 * IMPORTANT: Heuristic plugins must implement ALL base plugin functions from PluginApi.h:
 * - hipdnnPluginGetName - Returns the plugin (shared library) name, used for diagnostics
 * - hipdnnPluginGetVersion - Returns the plugin implementation version
 * - hipdnnPluginGetApiVersion - Returns the API version this plugin supports
 * - hipdnnPluginGetType - Returns HIPDNN_PLUGIN_TYPE_HEURISTIC
 * - hipdnnPluginGetLastErrorString - Returns per-thread error messages
 * - hipdnnPluginSetLoggingCallback - Sets the logging callback
 * - hipdnnPluginSetLogLevel - Sets the log level (optional)
 *
 * PLUS the heuristic-specific functions defined below.
 *
 * Multi-policy plugins: A single heuristic plugin shared library may expose one or more
 * selection policies. Each policy is identified by a stable int64 policy ID (typically
 * derived from a canonical policy name via hipdnn_data_sdk::utilities::policyNameToId).
 * The host enumerates policies via hipdnnHeuristicPluginGetAllPolicyIds and resolves
 * names via hipdnnHeuristicPluginGetPolicyName. A single plugin handle is shared across
 * all policies of the same library; per-policy state lives in the policy descriptor.
 *
 * Status codes: Use hipdnnPluginStatus_t for all return values.
 * Serialized data: Device properties and graphs cross the ABI as hipdnnPluginConstData_t*.
 */

// Export macro for heuristic plugins - allows separate static/dynamic defines
#ifdef _WIN32
#ifdef HIPDNN_HEURISTIC_PLUGIN_STATIC_DEFINE
#define HIPDNN_HEURISTIC_PLUGIN_EXPORT
#else
#define HIPDNN_HEURISTIC_PLUGIN_EXPORT __declspec(dllexport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define HIPDNN_HEURISTIC_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#error "Unsupported platform or compiler"
#endif

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup HeuristicPluginDataTypes Heuristic Plugin Data Types
 * @brief Data types used in the Heuristics Plugin API.
 * @{
 */

/**
 * @brief Opaque handle for a heuristic plugin session.
 *
 * This handle represents a long-lived session object per loaded heuristic module
 * per hipdnnHandle. It stores plugin state (caches, tuning data, etc.) and receives
 * device properties via hipdnnHeuristicHandleSetDeviceProperties.
 *
 * Threading: This handle is NOT thread-safe (single-thread only). Concurrent use
 * requires separate hipdnnHandle instances.
 */
typedef struct hipdnnHeuristicHandle_opaque* hipdnnHeuristicHandle_t;

/**
 * @brief Opaque handle for a heuristic policy descriptor.
 *
 * This handle represents per-EngineHeuristicDescriptor slot state: candidate engine IDs,
 * serialized graph bytes, and finalize result. Owned by the EngineHeuristicDescriptor,
 * created when the policy list is established, destroyed with the descriptor.
 */
typedef struct hipdnnHeuristicPolicyDescriptor_opaque* hipdnnHeuristicPolicyDescriptor_t;

/** @} */ // End of HeuristicPluginDataTypes group

/**
 * @defgroup HeuristicPluginPolicyEnumeration Heuristic Plugin Policy Enumeration
 * @brief Functions for discovering the set of policies a plugin exposes.
 * @{
 */

/**
 * @brief Retrieves the IDs of all selection policies the plugin exposes.
 *
 * A heuristic plugin may expose one or more policies; each is identified by a
 * stable int64 policy ID (typically the FNV-1a hash of a canonical policy
 * name; see hipdnn_data_sdk::utilities::policyNameToId).
 *
 * This function follows the same two-pass query/retrieve pattern as
 * hipdnnEnginePluginGetAllEngineIds:
 *   1. Pass max_policies = 0 (and policy_ids may be NULL) to discover the
 *      total count, written to *num_policies.
 *   2. Allocate an array of that size and pass max_policies = N to fill it;
 *      *num_policies is set to the number of IDs actually written.
 *
 * @param[out] policy_ids Array to receive policy IDs, or NULL when querying count.
 * @param[in] max_policies Capacity of the policy_ids array; 0 to query count only.
 * @param[out] num_policies On count query: total available policies.
 *                          On retrieve: number of IDs written.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 *
 * @note Policy IDs must be unique within a plugin and stable for the lifetime
 *       of the loaded library.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPluginGetAllPolicyIds(int64_t* policy_ids,
                                         uint32_t max_policies,
                                         uint32_t* num_policies);

/**
 * @brief Retrieves the canonical name of a specific policy.
 *
 * The host validates that policyNameToId(name) == policy_id; mismatches cause
 * the plugin to be rejected at load time.
 *
 * @param[in] policy_id The policy ID (must come from hipdnnHeuristicPluginGetAllPolicyIds).
 * @param[out] name Pointer to receive a NUL-terminated string owned by the plugin.
 *                  Must remain valid for the lifetime of the loaded library.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_BAD_PARAM if policy_id is not exposed by this plugin.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPluginGetPolicyName(int64_t policy_id, const char** name);

/** @} */ // End of HeuristicPluginPolicyEnumeration group

/**
 * @defgroup HeuristicPluginHandleLifecycle Heuristic Plugin Handle Lifecycle
 * @brief Functions for managing the plugin session handle.
 * @{
 */

/**
 * @brief Creates a new heuristic plugin handle.
 *
 * The host calls this once per loaded heuristic module per hipdnnHandle.
 * The handle stores session state (caches, tuning data, etc.).
 *
 * @param[out] out_handle Pointer to receive the created handle.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 *
 * @note The caller must destroy the handle via hipdnnHeuristicHandleDestroy to avoid leaks.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle);

/**
 * @brief Destroys a heuristic plugin handle and releases associated resources.
 *
 * @param[in] handle The handle to destroy.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 *
 * @note The handle becomes invalid after this call.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle);

/**
 * @brief Sets device properties on the plugin handle.
 *
 * Provides serialized device properties (FlatBuffer) to the plugin. The host builds this
 * buffer from resolved DeviceProperties (via queryDeviceProperties() or descriptor override)
 * and passes it ONCE PER DISTINCT hipdnnHeuristicHandle_t BEFORE calling Finalize on any
 * policy descriptor created with that handle.
 *
 * The buffer contains a FlatBuffer-serialized device properties table (schema from data-SDK).
 * Plugins MUST verify the buffer with flatbuffers::Verifier and reject malformed/incompatible
 * data. Plugins MUST NOT call HIP APIs (hipGetDevice, hipGetDeviceProperties, etc.).
 *
 * Plugins query this handle state during Finalize (and as needed elsewhere on that session).
 * The policy descriptor does NOT carry a parallel device-properties buffer.
 *
 * @param[in] handle The plugin handle.
 * @param[in] device_props_serialized Pointer to hipdnnPluginConstData_t containing serialized
 *                                    device properties buffer. Must remain valid for the
 *                                    duration of this call.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_BAD_PARAM if buffer is malformed or incompatible.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleSetDeviceProperties(
        hipdnnHeuristicHandle_t handle, const hipdnnPluginConstData_t* device_props_serialized);

/** @} */ // End of HeuristicPluginHandleLifecycle group

/**
 * @defgroup HeuristicPolicyDescriptorLifecycle Heuristic Policy Descriptor Lifecycle
 * @brief Functions for managing policy descriptors (per-slot objects).
 * @{
 */

/**
 * @brief Creates a new policy descriptor for a specific policy of the plugin.
 *
 * The host calls this once per policy slot in EngineHeuristicDescriptor, binding the
 * descriptor to the given plugin handle and selecting which policy from the plugin
 * the descriptor will execute. The descriptor stores per-slot state: candidate
 * engine IDs, serialized graph, and finalize result.
 *
 * This BINDS the policy to the handle BEFORE Finalize, so selection code can treat the
 * handle as the source of device-properties session state (SetDeviceProperties).
 *
 * Lifecycle: Owned by EngineHeuristicDescriptor, created when the policy list is established,
 * destroyed with the descriptor.
 *
 * @param[in] plugin_handle The plugin handle this descriptor is bound to.
 * @param[in] policy_id The ID of the policy this descriptor will execute. Must be one
 *                      of the IDs returned by hipdnnHeuristicPluginGetAllPolicyIds.
 * @param[out] out_desc Pointer to receive the created policy descriptor.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_BAD_PARAM if policy_id is not exposed by this plugin.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorCreate(hipdnnHeuristicHandle_t plugin_handle,
                                          int64_t policy_id,
                                          hipdnnHeuristicPolicyDescriptor_t* out_desc);

/**
 * @brief Destroys a policy descriptor and releases associated resources.
 *
 * @param[in] desc The policy descriptor to destroy.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 *
 * @note The descriptor becomes invalid after this call.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc);

/** @} */ // End of HeuristicPolicyDescriptorLifecycle group

/**
 * @defgroup HeuristicPolicyInputs Heuristic Policy Inputs
 * @brief Functions for setting inputs on policy descriptors.
 * @{
 */

/**
 * @brief Sets the candidate engine IDs on the policy descriptor.
 *
 * Provides the list of candidate engine IDs from EnginePluginResourceManager::getApplicableEngineIds.
 * The plugin must produce a reordered subset or permutation of these IDs.
 *
 * @param[in] desc The policy descriptor.
 * @param[in] engine_ids Array of candidate engine IDs.
 * @param[in] engine_id_count Number of engine IDs in the array.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success, error code otherwise.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                      const int64_t* engine_ids,
                                      size_t engine_id_count);

/**
 * @brief Sets the serialized operation graph on the policy descriptor.
 *
 * Provides the FlatBuffer-serialized operation graph from GraphDescriptor::getSerializedGraph().
 * The buffer contains the canonical graph representation used by the backend.
 *
 * Plugins that need structured access should parse the buffer using data-SDK generated types,
 * subject to schema version rules.
 *
 * @param[in] desc The policy descriptor.
 * @param[in] serialized_graph Pointer to hipdnnPluginConstData_t containing serialized graph
 *                             buffer. Must remain valid for the duration of this call.
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_BAD_PARAM if buffer is malformed or incompatible.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                            const hipdnnPluginConstData_t* serialized_graph);

/** @} */ // End of HeuristicPolicyInputs group

/**
 * @defgroup HeuristicPolicySelection Heuristic Policy Selection
 * @brief Functions for executing policy selection and retrieving results.
 * @{
 */

/**
 * @brief Executes the policy selection logic.
 *
 * Performs applicability checking and engine ordering based on the inputs previously set
 * via SetEngineIds, SetSerializedGraph, and HandleSetDeviceProperties.
 *
 * IMPORTANT: This assumes current device-properties bytes were applied to this policy's
 * hipdnnHeuristicHandle_t via hipdnnHeuristicHandleSetDeviceProperties earlier in the
 * same EngineHeuristicDescriptor::finalize(). Plugins query that handle state as needed;
 * the host does NOT pass device properties again on this call.
 *
 * Two-phase design: This function performs the selection work; GetSortedEngineIds retrieves
 * the result. This allows future async implementations without changing function names.
 *
 * @param[in] desc The policy descriptor.
 * @param[out] out_applied Pointer to receive the result:
 *                         - Set to 1 if policy succeeded (host then calls GetSortedEngineIds)
 *                         - Set to 0 if not applicable or declined (host continues outer loop)
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success (check out_applied for applicability),
 *         error code on failure.
 *
 * @note Calls on descriptors bound to a given hipdnnHeuristicHandle_t must occur on a thread
 *       consistent with that handle's single-thread contract.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* out_applied);

/**
 * @brief Retrieves the sorted engine IDs after successful finalize.
 *
 * Valid only after Finalize returned with out_applied == 1.
 *
 * The output IDs must be a permutation or subset of the input IDs from SetEngineIds.
 * The host validates this constraint.
 *
 * Callers MUST use the two-call pattern:
 *   1. Query count: Pass engine_ids = NULL; on return *num_engines holds the
 *      total number of IDs the policy will produce.
 *   2. Retrieve IDs: Allocate an array of that exact size, set *num_engines to
 *      that capacity, and call again with engine_ids pointing at the array.
 *      On return *num_engines holds the number of IDs actually written.
 *
 * If the caller supplies a non-NULL engine_ids with a capacity smaller than the
 * policy's full result, the implementation silently truncates: it writes
 * min(*num_engines, total) IDs and sets *num_engines to that truncated count.
 * The return value is still HIPDNN_PLUGIN_STATUS_SUCCESS, so the caller cannot
 * distinguish "buffer was exactly right" from "buffer was too small" without
 * having queried the count first. Always query first.
 *
 * @param[in] desc The policy descriptor.
 * @param[out] engine_ids Array to receive the sorted engine IDs, or NULL to query count.
 * @param[in,out] num_engines Input: capacity of engine_ids array (ignored if engine_ids is NULL).
 *                            Output: number of IDs available (count query) or written
 *                            (retrieve, possibly truncated to the input capacity).
 *
 * @return HIPDNN_PLUGIN_STATUS_SUCCESS on success,
 *         HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED if descriptor not finalized,
 *         error code on other failures.
 */
HIPDNN_PLUGIN_NODISCARD HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                            int64_t* engine_ids,
                                            size_t* num_engines);

/** @} */ // End of HeuristicPolicySelection group

#ifdef __cplusplus
}
#endif
// NOLINTEND
