// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>

// Cmake Generated export header.
#include "hipdnn_backend_export.h"

#include "HipdnnBackendAttributeName.h"
#include "HipdnnBackendAttributeType.h"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnBackendHeuristicType.h"
#include "HipdnnBackendLimits.h"
#include "HipdnnBackendPluginLoadingMode.h"
#include "HipdnnStatus.h"
#include <hipdnn_sdk/logging/CallbackTypes.h>

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
 * @retval HIPDNN_STATUS_SUCCESS           The graph was successfully deserialized and stored in the descriptor.
 * @retval HIPDNN_STATUS_BAD_PARAM         Invalid or inconsistent parameter values were encountered, such as:
 *                                         - descriptor is null.
 *                                         - serializedGraph is null.
 *                                         - graphByteSize is zero.
 * @retval HIPDNN_STATUS_ALLOC_FAILED      Memory allocation for the descriptor or graph failed.
 * @retval HIPDNN_STATUS_INTERNAL_ERROR    An internal error occurred during deserialization.
 */
HIPDNN_BACKEND_EXPORT hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serializedGraph, size_t graphByteSize);

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

#ifdef __cplusplus
}
#endif
// NOLINTEND
