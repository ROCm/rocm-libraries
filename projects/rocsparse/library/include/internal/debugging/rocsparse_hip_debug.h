/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_HIP_DEBUG_H
#define ROCSPARSE_HIP_DEBUG_H

//
// API conditional to the existence of ROCSPARSE_DEBUGGING
//
#ifdef ROCSPARSE_DEBUGGING

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**@{*/
/*! \file
   *  \brief Notes on the HIP API debug
   *
   *  The rocSPARSE library offers debugging information on the HIP functions called during the execution of a rocSPARSE routine.
   *
   * The HIP functions used in rocSPARSE are summarized in \ref rocsparse_hip_debug_api.
   *
   *
   *
   * This set of HIP functions is divided in 2 groups.
   *
   * The first group contains: \ref rocsparse_hip_debug_api_hipMemsetAsync, \ref rocsparse_hip_debug_api_hipMemcpyAsync, \ref rocsparse_hip_debug_api_hipMallocAsync, \ref rocsparse_hip_debug_api_hipFreeAsync, \ref rocsparse_hip_debug_api_hipLaunchKernelGGL, \ref rocsparse_hip_debug_api_hipMemcpy2DAsync, and \ref rocsparse_hip_debug_api_hipStreamSynchronize.
   * The second group contains: \ref rocsparse_hip_debug_api_hipMemset, \ref rocsparse_hip_debug_api_hipMemcpy, \ref rocsparse_hip_debug_api_hipMalloc, \ref rocsparse_hip_debug_api_hipFree, and \ref rocsparse_hip_debug_api_hipDeviceSynchronize.
   * In the long term, the goal of this tool is to detect and prohibit the use of the second group.
   *
   * The first group, with the exception of \ref rocsparse_hip_debug_api_hipStreamSynchronize, is considered non-blocking with respect to the host; although some of these functions can be blocking depending on specific contexts, compiler versions and/or architectures. The user must see this classification as a tag of the HIP api call history.
   *
   * The second group is considered a group of functions that are blocking with respect to the host, again, this is a characterization of the API history, not an execution property.
   *
   * It results in the definition of \ref rocsparse_hip_debug_api_history, where none means no HIP function has been called, sync means the last  HIP operation synchronized the stream, psync means the stream is synchronized before the last HIP operation, async the stream is not synchronized.
   *
   *
   *
   */

/*! \ingroup aux_module
   * \details  Enumerate HIP debug api functions used in rocSPARSE.
   */
typedef enum rocsparse_hip_debug_api_
{
    rocsparse_hip_debug_api_unknown
    = 0, /**< Use to tag uninitialized enumeration, this value is considered as an error. */
    rocsparse_hip_debug_api_hipMalloc, /**< Refers to hipMalloc. */
    rocsparse_hip_debug_api_hipFree, /**< Refers to hipFree. */
    rocsparse_hip_debug_api_hipMallocAsync, /**< Refers to hipMallocAsync. */
    rocsparse_hip_debug_api_hipFreeAsync, /**< Refers to hipFreeAsync. */
    rocsparse_hip_debug_api_hipMemcpy, /**< Refers to hipMemcpy. */
    rocsparse_hip_debug_api_hipMemcpyAsync, /**< Refers to hipMemcpyAsync. */
    rocsparse_hip_debug_api_hipMemcpy2DAsync, /**< Refers to hipMemcpy2DAsync. */
    rocsparse_hip_debug_api_hipMemset, /**< Refers to hipMemset. */
    rocsparse_hip_debug_api_hipMemsetAsync, /**< Refers to hipMemsetAsync. */
    rocsparse_hip_debug_api_hipStreamSynchronize, /**< Refers to hipStreamSynchronize. */
    rocsparse_hip_debug_api_hipDeviceSynchronize, /**< Refers to hipDeviceSynchronize. */
    rocsparse_hip_debug_api_hipLaunchKernelGGL /**< Refers to hipLaunchKernelGGL. */
} rocsparse_hip_debug_api;

/*! \ingroup aux_module
   * \details  Categorize the hip function call history.
   * This enumeration compresses the history of the HIP api call history from the execution of a rocSPARSE routine.
   */
typedef enum rocsparse_hip_debug_api_history_
{
    rocsparse_hip_debug_api_history_none = 1, /**< The history is empty. */
    rocsparse_hip_debug_api_history_sync
    = 2, /**< The history ends with a HIP function call that synchronizes the stream. */
    rocsparse_hip_debug_api_history_psync
    = 4, /**< The history has a HIP function call that synchronizes the stream, but ends with a HIP function call that does not synchronize the stream. */
    rocsparse_hip_debug_api_history_async
    = 8 /**< The history only contains HIP function calls that do not synchronize the stream. */
} rocsparse_hip_debug_api_history;

/*! \ingroup aux_module
   * \details  Enumerate HIP debug API information about the functions used in rocSPARSE.
   */
typedef enum rocsparse_hip_debug_api_info_
{
    rocsparse_hip_debug_api_info_count, /**< Number of calls for a given \ref rocsparse_hip_debug_api */
} rocsparse_hip_debug_api_info;

/*!
    \ingroup aux_module
    \details Enumerate HIP debug information
  */
typedef enum rocsparse_hip_debug_info_
{
    rocsparse_hip_debug_info_api, /**< fetch the \ref rocsparse_hip_debug_api_history value of the HIP history. */
    rocsparse_hip_debug_info_transfer_in_gib, /**< fetch the total amount of memory hipMemcpyAsync, (and hipMemcpy if used). */
    //rocsparse_hip_debug_info_num_streams, /**< fetch the number of streams used in the HIP history. */
} rocsparse_hip_debug_info;

/*! \ingroup aux_module
   * \details Enable HIP debug.
   */
ROCSPARSE_EXPORT rocsparse_status rocsparse_hip_debug_enable();

/*! \ingroup aux_module
   * \details Disable HIP debug.
   */
ROCSPARSE_EXPORT rocsparse_status rocsparse_hip_debug_disable();

/*! \ingroup aux_module
   * \details State of hip_debug, 1: enable, 0 otherwise.
   */
ROCSPARSE_EXPORT int32_t rocsparse_hip_debug_state();

/*! \ingroup aux_module
   * \details Start HIP debug.
   *  @param[in]
   *  handle      the pointer to the handle to the rocSPARSE library context, this pointer is allowed to be null.
   *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *  \retval rocsparse_status_success if the operation completed successfully, otherwise see \ref rocsparse_status.
 */
ROCSPARSE_EXPORT rocsparse_status rocsparse_hip_debug_start(rocsparse_handle handle,
                                                            rocsparse_error* p_error);

/*! \ingroup aux_module
   *  \details Print information registered in the HIP debug data structure.
   *  @param[in]
   *  handle the pointer to the handle to the rocSPARSE library context, this pointer is allowed to be null.
   *  @param[out]
   *  p_error error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *  \retval rocsparse_status_success if the operation completed successfully, otherwise see \ref rocsparse_status.
   */
ROCSPARSE_EXPORT rocsparse_status rocsparse_hip_debug_print(rocsparse_handle handle,
                                                            rocsparse_error* p_error);

/*! \ingroup aux_module
   *  \brief Get the information about a HIP API call.
   *  @param[in]
   *  handle      the pointer to the handle to the rocSPARSE library context, note that a null pointer is valid.
   *  @param[in]
   *  debug_api   the referred HIP API from \ref rocsparse_hip_debug_api.
   *  @param[in]
   *  debug_api_info  the information of interest from \ref rocsparse_hip_debug_api_info.
   *  @param[out]
   *  data        pointer to the fetched data
 *  @param[in]
 *  data_size_in_bytes size in bytes of the fetched data.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p debug_api or \p debug_api_info is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size_in_bytes is invalid.
 *
*  \par Example
*  \snippet rocsparse_hip_debug_api_info_get.cpp doc example
 */
ROCSPARSE_EXPORT rocsparse_status
    rocsparse_hip_debug_api_info_get(rocsparse_handle             handle,
                                     rocsparse_hip_debug_api      debug_api,
                                     rocsparse_hip_debug_api_info debug_api_info,
                                     void*                        data,
                                     size_t                       data_size_in_bytes,
                                     rocsparse_error*             p_error);

/*! \ingroup aux_module
   *  \brief Get the information from the HIP debug data structure.
   *  @param[in]
   *  handle      the pointer to the handle to the rocSPARSE library context, note that a null pointer is valid.
   *  @param[in]
   *  debug_info  the information of interest from \ref rocsparse_hip_debug_info.
   *  @param[out]
   *  data        pointer to the fetched data
 *  @param[in]
 *  data_size_in_bytes size in bytes of the fetched data.
 *  @param[out]
 *  p_error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user does not require an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p debug_info is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size_in_bytes is invalid.
 *
*  \par Example
*  \snippet rocsparse_hip_debug_info_get.cpp doc example
 */
ROCSPARSE_EXPORT rocsparse_status rocsparse_hip_debug_info_get(rocsparse_handle         handle,
                                                               rocsparse_hip_debug_info debug_info,
                                                               void*                    data,
                                                               size_t           data_size_in_bytes,
                                                               rocsparse_error* p_error);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif // ROCSPARSE_DEBUGGING

#endif // ROCSPARSE_HIP_DEBUG_H
