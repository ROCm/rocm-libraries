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

/*! \file
 *  \brief rocsparse_mat_info.h provides auxiliary functions in rocsparse
 */

#ifndef ROCSPARSE_MAT_INFO_H
#define ROCSPARSE_MAT_INFO_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a matrix info structure
 *
 *  \details
 *  \p rocsparse_mat_info_create creates a structure that holds the matrix info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using \ref rocsparse_mat_info_destroy().
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[inout]
 *  info    the pointer to the info structure.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_mat_info_create(rocsparse_handle    handle,
                                           rocsparse_mat_info* info,
                                           rocsparse_error*    p_error);

/*! \ingroup aux_module
 *  \brief Copy a matrix info structure
 *  \details
 *  \p rocsparse_mat_info_copy copies a matrix info structure. Both, source and destination
 *  matrix info structure must be initialized prior to calling \p rocsparse_mat_info_copy.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[out]
 *  dest    the pointer to the destination matrix info structure.
 *  @param[in]
 *  src     the pointer to the source matrix info structure.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_mat_info_copy(rocsparse_handle         handle,
                                         rocsparse_mat_info       dest,
                                         const rocsparse_mat_info src,
                                         rocsparse_error*         p_error);

/*! \ingroup aux_module
 *  \brief Destroy a matrix info structure
 *
 *  \details
 *  \p rocsparse_mat_info_destroy destroys a matrix info structure.
 *
 *  @param[in]
 *  handle      the pointer to the handle to the rocSPARSE library context.
 *  @param[in]
 *  info    the info structure.
   *  @param[out]
   *  p_error  error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_mat_info_destroy(rocsparse_handle   handle,
                                            rocsparse_mat_info info,
                                            rocsparse_error*   p_error);

#ifdef __cplusplus
}
#endif

#endif
