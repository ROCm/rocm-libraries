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
 *  \brief rocsparse_mat_info_backward.h provides auxilary functions in rocsparse
 *  but without using a rocsparse_handle.
 * Causing a disruption in the stream to use, as the default is the only available.
 */

#ifndef ROCSPARSE_MAT_INFO_BACKWARD_H
#define ROCSPARSE_MAT_INFO_BACKWARD_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a matrix info structure
 *
 *  \details
 *  \p rocsparse_create_mat_info creates a structure that holds the matrix info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using rocsparse_destroy_mat_info().
 *
 *  @param[inout]
 *  info    the pointer to the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);

/*! \ingroup aux_module
 *  \brief Copy a matrix info structure
 *  \details
 *  \p rocsparse_copy_mat_info copies a matrix info structure. Both, source and destination
 *  matrix info structure must be initialized prior to calling \p rocsparse_copy_mat_info.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix info structure.
 *  @param[in]
 *  src     the pointer to the source matrix info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_mat_info(rocsparse_mat_info dest, const rocsparse_mat_info src);

/*! \ingroup aux_module
 *  \brief Destroy a matrix info structure
 *
 *  \details
 *  \p rocsparse_destroy_mat_info destroys a matrix info structure.
 *
 *  @param[in]
 *  info    the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);

#ifdef __cplusplus
}
#endif

#endif
