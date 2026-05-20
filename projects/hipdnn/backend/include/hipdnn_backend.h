// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file hipdnn_backend.h
 * @brief hipDNN Backend C API
 *
 * This file defines the public C API for hipDNN backend operations.
 * The backend provides handle management, descriptor operations, and
 * execution of computational graphs.
 *
 * @section backend_usage Basic Usage
 *
 * 1. Create a handle:
 *    @code{.c}
 *    hipdnnHandle_t handle;
 *    hipdnnCreate(&handle);
 *    @endcode
 *
 * 2. Create and configure descriptors:
 *    @code{.c}
 *    hipdnnBackendDescriptor_t graphDesc;
 *    hipdnnBackendCreateAndDeserializeGraph_ext(&graphDesc, serializedGraph, size);
 *    @endcode
 *
 * 3. Execute operations:
 *    @code{.c}
 *    hipdnnBackendExecute(handle, executionPlan, variantPack);
 *    @endcode
 *
 * 4. Clean up:
 *    @code{.c}
 *    hipdnnBackendDestroyDescriptor(graphDesc);
 *    hipdnnDestroy(handle);
 *    @endcode
 *
 * @see hipdnn_frontend.hpp for the C++ frontend API
 */

#pragma once

#include <hip/hip_runtime_api.h>

// Cmake Generated export header.
#include "hipdnn_backend_export.h"

#include "HipdnnBackendAttributeName.h"
#include "HipdnnBackendAttributeType.h"
#include "HipdnnBackendBehaviorNote.h"
#include "HipdnnBackendCallbackTypes.h"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnBackendHeuristicType.h"
#include "HipdnnBackendLimits.h"
#include "HipdnnBackendPluginLoadingMode.h"
#include "HipdnnBackendPluginUnloadingMode.h"
#include "HipdnnConvolutionMode.h"
#include "HipdnnDataType.h"
#include "HipdnnPaddingMode.h"
#include "HipdnnReduceTensorOp.h"
#include "HipdnnResampleMode.h"
#include "HipdnnStatus.h"

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/*!
* @brief Creates the hipdnnHandle_t type
*/
typedef struct hipdnnHandle* hipdnnHandle_t;

/*!
* @brief Creates the hipdnnBackendDescriptor_t type
*/
typedef struct HipdnnBackendDescriptor* hipdnnBackendDescriptor_t;

/*! @brief Creates a hipdnnHandle_t
 *
 * @param [in] handle        An instance of hipdnnHandle_t
 *
 * @retval HIPDNN_STATUS_SUCCESS            The creation was successful
 * @retval HIPDNN_STATUS_BAD_PARAM          The descriptor is not a valid (NULL) descriptor.
 * @retval HIPDNN_STATUS_ALLOC_FAILED       The memory allocation failed when creating handle object.
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle);

/*! @brief Destroyes hipdnnHandle_t
 *
 * @param [in] handle        An instance of hipdnnHandle_t
 *
 * @retval HIPDNN_STATUS_BAD_PARAM          The descriptor is not a valid (NULL) descriptor.
 * @retval HIPDNN_STATUS_SUCCESS            The destruction was successful
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle);

/*! @brief Sets the stream for the hipdnnHandle_t
 *
 * @param [in] handle        An instance of hipdnnHandle_t
 * @param [in] streamId      The stream to be set
 *
 * @retval HIPDNN_STATUS_BAD_PARAM                  invalid (NULL) handle.
 * @retval HIPDNN_STATUS_SUCCESS                    The creation was successful
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId);

/**
 * @brief Retrieves the HIP stream associated with the specified hipDNN handle.
 *
 * @param[in] handle The hipDNN handle whose associated stream is to be retrieved.
 * @param[out] streamId Pointer to a hipStream_t where the associated stream ID
 *                      will be stored upon successful execution.
 *
 * @retval HIPDNN_STATUS_BAD_PARAM                  invalid (NULL) handle.
 * @retval HIPDNN_STATUS_SUCCESS                    The creation was successful
 *
 * @note
 * - The handle must be valid and initialized before calling this function.
 * - The streamId pointer must not be null.
 * - This function uses a try-catch mechanism to handle exceptions internally.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId);

/*! @brief Creates a backend descriptor
 *
 * Allocates memory for a given descriptorType at the location pointed
 * by the descriptor
 *
 * @param [in]   descriptorType   One among the enumerated hipdnnBackendDescriptor_t
 * @param [out]  descriptor       Pointer to a descriptor
 *
 * @retval  HIPDNN_STATUS_SUCCESS            The creation was successful
 * @retval  HIPDNN_STATUS_NOT_SUPPORTED       Creating a descriptor of a given type is not supported.
 * @retval  HIPDNN_STATUS_ALLOC_FAILED        The memory allocation failed.
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptorType, hipdnnBackendDescriptor_t* descriptor);

/*! @brief Destroys an instance of hipdnnBackendDescriptor_t
 *
 * Destroys instances of hipdnnBackendDescriptor_t that were previously created using
 * hipdnnBackendCreateDescriptor(). The value pointed by the descriptor will be undefined after the
 * memory is free and done.
 *
 * **Undefined Behavior** if the descriptor was altered between the 'Create' and 'Destroy
 * Descriptor'
 *
 * @param  [in]  descriptor  Instance of hipdnnBackendDescriptor_t previously created by
 *                           hipdnnBackendCreateDescriptor()
 *
 * @retval  HIPDNN_STATUS_SUCCESS       The memory was destroyed successfully
 * @retval  HIPDNN_STATUS_ALLOC_FAILED   The destruction of memory failed.
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor);

/*! @brief Executes a graph
 *
 * Executes the given engine_configuration_plan on the variantPack and the finalized executionPlan
 * on the data. The data and the working space are encapsulated in the variantPack.
 *
 * @param  [in]  handle              An instance of hipdnnHandle_t
 * @param  [in]  executionPlan       Descriptor of the finalized executionPlan
 * @param  [in]  variantPack         Descriptor of the finalized variantPack consisting of:
 *                                   * Data pointer for each non-virtual pointer of the operation set in
 *                                     the execution plan.
 *                                   * Pointer to user-allocated workspace in global memory at least as
 *                                     large as the size queried
 *
 * @retval  HIPDNN_STATUS_SUCCESS               The executionPlan was executed successfully
 * @retval  HIPDNN_STATUS_BAD_PARAM             An incorrect or inconsistent value is encountered. For example, a required data pointer is invalid.
 * @retval  HIPDNN_STATUS_INTERNAL_ERROR        Some internal errors were encountered.
 * @retval  HIPDNN_STATUS_EXECUTION_FAILED      An error was encountered executing the plan with the variantPack.
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t handle,
                                                          hipdnnBackendDescriptor_t executionPlan,
                                                          hipdnnBackendDescriptor_t variantPack);

/*! @brief Finalizes a backend descriptor
*
* Finalizes the memory pointed to by the descriptor. The type of finalization is done depending on
* the descriptor_type argument with which the descriptor was created using
* hipdnnBackendCreateDescriptor() or initialized using hipdnnBackendInitialize().
*
* @param  [in]  descriptor  Instance of hipdnnBackendDescriptor_t to finalize
*
* @retval  HIPDNN_STATUS_SUCCESS           The descriptor was finalized successfully
* @retval  HIPDNN_STATUS_BAD_PARAM         Invalid descriptor attribute values or combination thereof is encountered.
* @retval  HIPDNN_STATUS_NOT_SUPPORTED     Descriptor attribute values or combinations therefore not supported by the current version of hipDNN are encountered.
* @retval  HIPDNN_STATUS_INTERNAL_ERROR    Some internal errors are encountered.

*/
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor);

/*! @brief Retrieves backend descriptor's attribute
 *
 * This function retrieves the values of an attribute of a descriptor. attributeName is the name of
 * the attribute whose value is requested. attributeType is the type of attribute.
 * requestedElementCount is the number of elements to be potentially retrieved. The number of
 * elements for the requested attribute is stored in elementCount. The retrieved values are stored
 * in arrayOfElements. When the attribute is expected to have a single value, arrayOfElements can be
 * pointer to the output value. This function will return HIPDNN_STATUS_NOT_INTIALIZED if the
 * descriptor has not been successfully finalized using hipdnnBackendFinalize()
 *
 * @param  [in]   descriptor               Instance of hipdnnBackendDescriptor_t whose attribute to
 *                                         retrieve
 * @param  [in]   attributeName            The name of the attribute being get from the descriptor
 * @param  [in]   attributeType            The type of attribute
 * @param  [in]   requestedElementCount    Number of elements to output to arrayOfElements
 * @param  [out]  elementCount             Output pointer for the number of elements the descriptor
 *                                         attribute has. Note that hipdnnBackendGetAttribute() will
 *                                         only write the least of this and requestedElementCount
 *                                         elements to arrayOfElements
 * @param  [out]  arrayOfElements          Array of elements of the datatype of the attributeType. The
 *                                         data type of the attributeType is listed in the mapping
 *                                         table of hipdnnBackendAttributeType_t
 *
 * @retval  HIPDNN_STATUS_SUCCESS           The attributeName was retrieved from the descriptor successfully
 * @retval  HIPDNN_STATUS_BAD_PARAM         One or more invalid or inconsistent argument values were encountered. Some examples include:
 *                                              attributeName is not a valid attribute for the descriptor.
 *                                              attributeType is not one of the valid types for the attribute.
 * @retval  HIPDNN_STATUS_NOT_INITIALIZED   The descriptor has not been successfully finalized using hipdnnBackendFinalize().
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attributeName,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements);

/*! @brief Sets an attribute of a descriptor
 *
 * This function sets an attribute of a descriptor to values provided as a pointer. descriptor is the descriptor to be set.
 * attributeName is the name of the attribute to be set. attributeType is the type of attribute.
 * The value to which the attribute is set, is pointed by the arrayOfElements. The number of elements is given by elementCount.
 * This function will return HIPDNN_STATUS_NOT_INITIALIZED if the descriptor is already successfully finalized using hipdnnBackendFinalize().
 *
 * @param  [in]  descriptor         Instance of hipdnnBackendDescriptor_t whose attribute is being set
 * @param  [in]  attributeName      The name of the attribute being set on the descriptor
 * @param  [in]  attributeType      The type of attribute
 * @param  [in]  elementCount       Number of elements being set
 * @param  [in]  arrayOfElements    The starting location for an array from where to read the values
 *                                  from. The elements of the array are expected to be of the datatype
 *                                  of the attributeType. The datatype of the attributeType is listed
 *                                  in the mapping table of hipdnnBackendAttributeType_t.
 *
 * @retval  HIPDNN_STATUS_SUCCESS           The attributeName was set to the descriptor
 * @retval  HIPDNN_STATUS_NOT_INITIALIZED   The backend descriptor pointed to by the descriptor is already in the finalized state.
 * @retval  HIPDNN_STATUS_BAD_PARAM         The function is called with arguments that correspond to invalid values. Some examples include:
 *                                              attributeName is not a settable attribute of descriptor.
 *                                              attributeType is incorrect for this attributeName.
 *                                              elementCount value is unexpected.
 *                                              arrayOfElements contains values invalid for the attributeType.
 * @retval  HIPDNN_STATUS_NOT_SUPPORTED     The values to which the attributes are being set are not supported by the current version of hipDNN.
 *
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                              hipdnnBackendAttributeName_t attributeName,
                              hipdnnBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              const void* arrayOfElements);

/**
 * @brief Converts the hipDNN status code to a NULL terminated static string.
 *
 * This function takes a hipdnnStatus_t status code and returns a pointer to a
 * static, NULL-terminated string describing the error or status.
 *
 * @param status The hipDNN status code to convert.
 * @return A pointer to a static, NULL-terminated string describing the status.
 */
HIPDNN_BACKEND_EXPORT const char* hipdnnGetErrorString(hipdnnStatus_t status);

/**
 * @brief Retrieves the last error message for the calling thread.
 *
 * This function copies the last error message associated with the calling thread into the provided
 * message buffer, up to max_size bytes (including the null terminator).
 * Note the max size for an error message is HIPDNN_ERROR_STRING_MAX_LENGTH characters.
 *
 * After retrieving the error message, this function clears the internal error state.
 * Use hipdnnPeekLastErrorString_ext() if you need to retrieve the error without clearing it.
 *
 * @param[out] message   Pointer to a character buffer where the error message will be copied.
 * @param[in]  maxSize   Maximum number of bytes to copy, including the null terminator.
 */
HIPDNN_BACKEND_EXPORT void hipdnnGetLastErrorString(char* message, size_t maxSize);

/*
 **************************************************************************************************
 *
 *  Extension API Below
 *
 **************************************************************************************************
 */

/**
 * @brief Retrieves the last error message for the calling thread without clearing it.
 *
 * This function copies the last error message associated with the calling thread into the provided
 * message buffer, up to max_size bytes (including the null terminator).
 * Note the max size for an error message is HIPDNN_ERROR_STRING_MAX_LENGTH characters.
 *
 * Unlike hipdnnGetLastErrorString(), this function does NOT clear the internal error state after
 * retrieval. This preserves the pre-existing behavior for cases where the error needs to be
 * queried multiple times.
 *
 * @param[out] message   Pointer to a character buffer where the error message will be copied.
 * @param[in]  maxSize   Maximum number of bytes to copy, including the null terminator.
 */
HIPDNN_BACKEND_EXPORT void hipdnnPeekLastErrorString_ext(char* message, size_t maxSize);

/*!
 * @brief Creates and deserializes a graph into a backend descriptor.
 *
 * This function creates a backend descriptor and deserializes a graph from a serialized byte array
 * into the descriptor. The serialized graph is provided as an input byte array, and the size of
 * the graph in bytes is specified. The created descriptor will encapsulate the deserialized graph.
 *
 * IMPORTANT: Hipdnn expects that the serialized graph is sorted in topological order, has no cycles,
 * and is fully connected (no orphan nodes). Additionally, all tensors in the graph must have unique uids.
 *
 * @param [out] descriptor        Pointer to a backend descriptor where the deserialized graph will
 *                                be stored.
 * @param [in]  serializedGraph   Pointer to the serialized graph data in a byte array.
 * @param [in]  graphByteSize     Size of the serialized graph in bytes.
 *
 * @retval HIPDNN_STATUS_SUCCESS                The graph was successfully deserialized and stored in the descriptor.
 * @retval HIPDNN_STATUS_BAD_PARAM_NULL_POINTER  descriptor or serializedGraph is null.
 * @retval HIPDNN_STATUS_BAD_PARAM               graphByteSize is zero.
 * @retval HIPDNN_STATUS_ALLOC_FAILED            Memory allocation for the descriptor or graph failed.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR          An internal error occurred during deserialization.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serializedGraph, size_t graphByteSize);

/*!
 * @brief Retrieves the binary-serialized graph from an operation graph descriptor.
 *
 * Uses the standard two-call pattern: call first with @p serializedGraph set to @c nullptr to query
 * the required buffer size, then call again with a caller-allocated buffer to receive the data.
 * The descriptor must be of type HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR. Finalization is not
 * required — if operations are set but the graph is not finalized, serialization builds from
 * operations directly. An empty operations list produces a valid (but empty) serialized graph.
 *
 * @param [in]  descriptor        An operation graph descriptor.
 * @param [in]  requestedByteSize Size of the caller-allocated buffer in bytes.
 *                                Ignored when @p serializedGraph is @c nullptr.
 * @param [out] graphByteSize     Pointer to receive the size of the serialized graph in bytes.
 *                                Always written on success.
 * @param [out] serializedGraph   Caller-allocated buffer to receive the serialized graph data,
 *                                or @c nullptr to query the required size only.
 *
 * @retval HIPDNN_STATUS_SUCCESS              The serialized graph was successfully retrieved,
 *                                            or the size query completed successfully.
 * @retval HIPDNN_STATUS_BAD_PARAM_NULL_POINTER  descriptor or graphByteSize is null.
 * @retval HIPDNN_STATUS_BAD_PARAM            The descriptor is not an operation graph descriptor.
 * @retval HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT  The requestedByteSize is smaller than the
 *                                                     serialized graph size.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR       An internal error occurred during serialization.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetSerializedBinaryGraph_ext(hipdnnBackendDescriptor_t descriptor,
                                              size_t requestedByteSize,
                                              size_t* graphByteSize,
                                              uint8_t* serializedGraph);

/*!
 * @brief Retrieves the JSON-serialized graph from an operation graph descriptor.
 *
 * Uses the standard two-call pattern: call first with @p serializedJsonGraph set to @c nullptr to
 * query the required buffer size, then call again with a caller-allocated buffer to receive the
 * data. The descriptor must be of type HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR. An empty
 * operations list produces a valid (but empty) serialized JSON graph.
 *
 * @param [in]  descriptor          An operation graph descriptor.
 * @param [in]  requestedByteSize   Size of the caller-allocated buffer in bytes.
 *                                  Ignored when @p serializedJsonGraph is @c nullptr.
 * @param [out] graphByteSize       Pointer to receive the size of the JSON graph in bytes.
 *                                  The reported size includes the null terminator.
 *                                  Always written on success.
 * @param [out] serializedJsonGraph Caller-allocated buffer to receive the JSON graph data,
 *                                  or @c nullptr to query the required size only.
 *
 * @retval HIPDNN_STATUS_SUCCESS                   The JSON graph was successfully retrieved,
 *                                                 or the size query completed successfully.
 * @retval HIPDNN_STATUS_BAD_PARAM_NULL_POINTER    descriptor or graphByteSize is null.
 * @retval HIPDNN_STATUS_BAD_PARAM                 The descriptor is not an operation graph
 *                                                 descriptor.
 * @retval HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT  The requestedByteSize is smaller than the
 *                                                     JSON graph size.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR            An internal error occurred during serialization.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetSerializedJsonGraph_ext(hipdnnBackendDescriptor_t descriptor,
                                            size_t requestedByteSize,
                                            size_t* graphByteSize,
                                            char* serializedJsonGraph);

/*!
 * @brief Retrieves the serialized backend execution plan from a finalized execution plan.
 *
 * Uses the standard two-call pattern: call first with @p serializedPlan set to @c nullptr to query
 * the required buffer size, then call again with a caller-allocated buffer to receive the data.
 * The descriptor must be a finalized HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR. The returned bytes
 * contain a hipDNN FlatBuffer envelope with routing metadata, plan tensor UIDs, and opaque
 * plugin-specific execution context bytes.
 *
 * @param [in]  descriptor        A finalized execution plan descriptor.
 * @param [in]  requestedByteSize Size of the caller-allocated buffer in bytes.
 *                                Ignored when @p serializedPlan is @c nullptr.
 * @param [out] planByteSize      Pointer to receive the size of the serialized plan in bytes.
 * @param [out] serializedPlan    Caller-allocated buffer to receive the serialized plan data,
 *                                or @c nullptr to query the required size only.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendGetSerializedExecutionPlan_ext(hipdnnBackendDescriptor_t descriptor,
                                                size_t requestedByteSize,
                                                size_t* planByteSize,
                                                uint8_t* serializedPlan);

/*!
 * @brief Creates and deserializes a backend execution plan descriptor.
 *
 * @param [in]  handle         A hipDNN handle used to access loaded engine plugins.
 * @param [out] descriptor     Pointer to the created execution plan descriptor.
 * @param [in]  serializedPlan Pointer to bytes produced by hipdnnBackendGetSerializedExecutionPlan_ext().
 * @param [in]  planByteSize   Size of @p serializedPlan in bytes.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnBackendCreateAndDeserializeExecutionPlan_ext(hipdnnHandle_t handle,
                                                       hipdnnBackendDescriptor_t* descriptor,
                                                       const uint8_t* serializedPlan,
                                                       size_t planByteSize);

/*!
 * @brief Creates and deserializes a graph from a JSON string into a backend descriptor.
 *
 * This function creates a backend descriptor and deserializes a graph from a JSON string
 * into the descriptor. The JSON is internally converted to binary and processed using the
 * standard deserialization path.
 *
 * @param [out] descriptor    Pointer to a backend descriptor where the deserialized graph will
 *                            be stored.
 * @param [in]  jsonGraph     Pointer to the JSON graph data as a character array. Does not need
 *                            to be null-terminated; the jsonByteSize parameter controls the
 *                            parsing length.
 * @param [in]  jsonByteSize  Size of the JSON graph in bytes.
 *                            May include or exclude the null terminator; the parser
 *                            handles both cases.
 *
 * @retval HIPDNN_STATUS_SUCCESS                The graph was successfully deserialized and stored in
 *                                              the descriptor.
 * @retval HIPDNN_STATUS_BAD_PARAM_NULL_POINTER  descriptor or jsonGraph is null.
 * @retval HIPDNN_STATUS_BAD_PARAM               jsonByteSize is zero.
 * @retval HIPDNN_STATUS_ALLOC_FAILED            Memory allocation for the descriptor or graph failed.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR          An internal error occurred during deserialization.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeJsonGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const char* jsonGraph, size_t jsonByteSize);

/*!
 * @brief Callback function for logging messages.
 *
 * This function is called by the hipDNN library to log messages. The severity level of the message
 * is provided along with the message itself.
 *
 * @param [in] severity  The severity level of the message.
 * @param [in] msg        The message to be logged.
 */
HIPDNN_BACKEND_EXPORT void hipdnnLoggingCallback_ext(hipdnnSeverity_t severity, const char* msg);

/**
 * @brief Sets the search paths for hipDNN engine plugins.
 *
 * This function configures the search paths for engine plugins and must be called before
 * creating a hipDNN handle, as plugins are loaded during handle creation.
 *
 * Paths can be either directories or specific plugin files. Relative paths are resolved
 * from the location of the libhipdnn_backend.so file. The backend can resolve
 * platform-agnostic names, allowing users to omit prefixes like `lib` and extensions
 * like `.so` or `.dll`.
 *
 * @param[in] numPaths       The number of paths in the `pluginPaths` array.
 * @param[in] pluginPaths    An array of relative or absolute path strings.
 * @param[in] loadingMode    Specifies whether to add paths to or replace the default search paths.
 *
 * @retval HIPDNN_STATUS_SUCCESS           The operation was successful.
 * @retval HIPDNN_STATUS_BAD_PARAM_NULL_POINTER         `pluginPaths` is nullptr when `numPaths` is greater than 0.
 * @retval HIPDNN_STATUS_NOT_SUPPORTED         Called with active handle.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR    An internal error occurred.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
    size_t numPaths, const char* const* pluginPaths, hipdnnPluginLoadingMode_ext_t loadingMode);

/**
 * @brief Sets the plugin unloading mode for hipDNN.
 *
 * This function controls when plugins are unloaded from memory. By default, lazy unloading (HIPDNN_PLUGIN_UNLOAD_LAZY)
 * is used, which keeps plugins loaded until the application exits or until
 * hipdnnSetEnginePluginPaths_ext is called. This avoids expensive plugin reloading when
 * handles are frequently created and destroyed.
 *
 * This function can be called at any time. When switching from lazy to eager mode (HIPDNN_PLUGIN_UNLOAD_EAGER) while
 * no handles exist, plugins will be unloaded immediately.
 *
 * @param[in] unloadingMode  Specifies when plugins should be unloaded from memory.
 *
 * @retval HIPDNN_STATUS_SUCCESS    The operation was successful.
 * @retval HIPDNN_STATUS_BAD_PARAM  An invalid unloading mode was specified.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnSetPluginUnloadMode_ext(hipdnnPluginUnloadingMode_ext_t unloadingMode);

/**
 * @brief Gets file paths of loaded engine plugins for a given handle.
 *
 * This function must be called twice:
 * 1. First call: Pass `pluginPaths` as `nullptr` to query the number of plugins and required buffer size.
 *    - Sets `numPluginPaths` to the total number of loaded plugins
 *    - Sets `maxStringLen` to the maximum string length needed (including null terminator)
 *
 * 2. Second call: Pass allocated buffers to retrieve the actual plugin paths.
 *    - Allocate an array of `numPluginPaths` char pointers
 *    - Each char pointer should point to a buffer of at least `maxStringLen` characters
 *    - The function will populate these buffers with the plugin paths
 *
 * @param[in]     handle           A valid hipDNN handle
 * @param[in,out] numPluginPaths   Pointer to number of plugins; updated with actual count.
 * @param[out]    pluginPaths      Array of character pointers for plugin paths, or `nullptr` to query sizes.
 * @param[in,out] maxStringLen     Pointer to max string length; updated with required length.
 *
 * @retval HIPDNN_STATUS_SUCCESS           Success.
 * @retval HIPDNN_STATUS_BAD_PARAM         Invalid handle, null pointers, or insufficient buffer sizes.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR    Internal error.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetLoadedEnginePluginPaths_ext(hipdnnHandle_t handle,
                                                                          size_t* numPluginPaths,
                                                                          char** pluginPaths,
                                                                          size_t* maxStringLen);

/**
 * @brief Callback mode (sync vs async).
 */
typedef enum
{
    HIPDNN_LOG_CALLBACK_SYNC = 0, ///< Callback invoked on logging thread (synchronous)
    HIPDNN_LOG_CALLBACK_ASYNC = 1 ///< Callback invoked on worker thread (asynchronous)
} hipdnnLogCallbackMode_t;

/**
 * @brief Set or update a user log callback.
 *
 * This API allows registering multiple user callbacks with individual log levels and sync/async modes.
 * Each callback is uniquely identified by the composite key (callback, userHandle).
 *
 * @note When a synchronous callback is registered, the synchronous callbacks will delay hipDNN
 *       until the callback returns, regardless of any async log callbacks also being registered.
 *       Synchronous callbacks are recommended only for debugging or testing purposes due to
 *       their blocking nature. Use async callbacks for production workloads.
 *
 * Behavior:
 * - If (callback, userHandle) already registered: UPDATES settings (level and/or sync/async mode)
 * - If (callback, userHandle) new: ADDS new registration
 * - If minLevel == SEV_OFF: REMOVES registration
 * - userHandle must be non-null
 *
 * Callback Removal (minLevel == SEV_OFF):
 * - No further logs will be received on the callback.
 * - Any pending async logs for this callback will be abandoned
 * - After this function returns, user can safely destroy data referenced by userHandle
 *
 * @param[in] callback   The callback function to invoke
 * @param[in] minLevel   Minimum severity level (SEV_OFF removes the callback). Note that
 *                        the logs produced on this callback will be limited by the global log
 *                        level set either by the HIPDNN_LOG_LEVEL environment variable or
 *                        the setGlobalLogLevel() API function.
 * @param[in] mode       Sync or async invocation mode
 * @param[in] userHandle Non-null user data (also serves as unique callback ID)
 *
 * @retval HIPDNN_STATUS_SUCCESS           The callback was set/updated/removed successfully
 * @retval HIPDNN_STATUS_BAD_PARAM         callback is NULL, userHandle is NULL, invalid mode,
 *                                         or attempting to remove non-existent callback
 * @retval HIPDNN_STATUS_NOT_INITIALIZED   Logging system not initialized
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t
    hipdnnSetUserLogCallback_ext(hipdnnUserLogCallback_t callback,
                                 hipdnnSeverity_t minLevel,
                                 hipdnnLogCallbackMode_t mode,
                                 hipdnnUserLogCallbackHandle_t userHandle);

/**
 * @brief Set the global log level for the backend.
 *
 * This controls which log messages are output to console/file AND to the global backend log output callback.
 * Valid levels: HIPDNN_SEV_INFO, HIPDNN_SEV_WARN, HIPDNN_SEV_ERROR, HIPDNN_SEV_FATAL, HIPDNN_SEV_OFF.
 *
 * @param[in] level   The severity level to set
 *
 * @retval HIPDNN_STATUS_SUCCESS      The log level was set successfully
 * @retval HIPDNN_STATUS_BAD_PARAM    Invalid log level
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendSetGlobalLogLevel_ext(hipdnnSeverity_t level);

/**
 * @brief Get the global log level for the backend.
 *
 * @param[out] level   Pointer to store the current log level
 *
 * @retval HIPDNN_STATUS_SUCCESS      The log level was retrieved successfully
 * @retval HIPDNN_STATUS_BAD_PARAM    level pointer is NULL
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendGetGlobalLogLevel_ext(hipdnnSeverity_t* level);

/**
 * @brief Gets the number of loaded engines for a given handle.
 *
 * @param[in]  handle       A valid hipDNN handle.
 * @param[out] numEngines   Pointer where the engine count will be stored.
 *
 * @retval HIPDNN_STATUS_SUCCESS           Success.
 * @retval HIPDNN_STATUS_BAD_PARAM         Invalid handle or null pointer.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR    Internal error.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetEngineCount_ext(hipdnnHandle_t handle,
                                                              size_t* numEngines);

/**
 * @brief Gets information about a loaded engine by index.
 *
 * Retrieves the id, name, version, type, and plugin name of the engine at the given index.
 * Valid indices are 0 to numEngines-1 as returned by hipdnnGetEngineCount_ext.
 * Engines are sorted alphabetically by name.
 *
 * This function uses a two-call pattern for string fields:
 * 1. First call: Pass all string buffers as `nullptr` to query required sizes.
 *    - Sets `nameLen`, `versionLen`, `typeLen`, and `pluginNameLen` to the required buffer sizes
 *      (including null terminator). Note: if any string buffer is null, all sizes are updated.
 *
 * 2. Second call: Pass allocated buffers with sizes set from the first call.
 *
 * @param[in]     handle           A valid hipDNN handle.
 * @param[in]     engineIndex      Zero-based index of the engine to query.
 * @param[out]    engineId         Pointer where the engine ID will be stored, or `nullptr` to skip.
 * @param[out]    engineName       Buffer for the engine name, or `nullptr` to query size.
 * @param[in,out] engineNameLen    Pointer to buffer size; updated with required size.
 * @param[out]    pluginName       Buffer for the plugin name, or `nullptr` to query size.
 * @param[in,out] pluginNameLen    Pointer to buffer size; updated with required size.
 * @param[out]    version          Buffer for the engine version, or `nullptr` to query size.
 * @param[in,out] versionLen       Pointer to buffer size; updated with required size.
 * @param[out]    type             Buffer for the engine type string, or `nullptr` to query size.
 * @param[in,out] typeLen          Pointer to buffer size; updated with required size.
 *
 * @retval HIPDNN_STATUS_SUCCESS           Success.
 * @retval HIPDNN_STATUS_BAD_PARAM         Invalid handle, null pointers, or out-of-range index.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR    Internal error.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnGetEngineInfo_ext(hipdnnHandle_t handle,
                                                             size_t engineIndex,
                                                             int64_t* engineId,
                                                             char* engineName,
                                                             size_t* engineNameLen,
                                                             char* pluginName,
                                                             size_t* pluginNameLen,
                                                             char* version,
                                                             size_t* versionLen,
                                                             char* type,
                                                             size_t* typeLen);

/**
 * @brief Returns hipdnn backend version string. Returns an error if nullptr is passed
 *
 * @deprecated Use hipdnnVersionString_ext instead
 *
 * @param[out] version pointer to where version string will be written
 *
 * @retval HIPDNN_STATUS_SUCCESS                  Success
 * @retval HIPDNN_STATUS_BAD_PARAM_NULL_POINTER   If version parameter is nullptr
 */
HIPDNN_BACKEND_DEPRECATED_EXPORT hipdnnStatus_t hipdnnGetVersion_ext(const char** version);

/**
 * @brief Returns hipdnn backend version string
 *
 * @return A string in the format "MAJOR.MINOR.PATCH.TWEAK". The returned value has a static lifetime
 */
HIPDNN_BACKEND_EXPORT const char* hipdnnVersionString_ext();

#ifdef __cplusplus
}
#endif
// NOLINTEND
