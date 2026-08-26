/*! \file */
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

#ifndef ROCSPARSE_ELLSV_H
#define ROCSPARSE_ELLSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \details
*  \p rocsparse_ellsv_zero_pivot returns \ref rocsparse_status_zero_pivot if a
*  numerical zero has been found during \ref rocsparse_ellsv_solve() computation.
*  The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using the same index base as the ELL matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_ellsv_zero_pivot is a blocking function. It might negatively influence
*  performance.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[in]
*  descr       descriptor of the sparse ELL matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
*              invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_zero_pivot zero pivot has been found.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ellsv_zero_pivot(rocsparse_handle          handle,
                                            const rocsparse_mat_descr descr,
                                            rocsparse_mat_info        info,
                                            rocsparse_int*            position);

/*! \ingroup level2_module
*  \details
*  \p rocsparse_ellsv_clear deallocates all memory that was allocated by
*  \ref rocsparse_ellsv_analysis(). This is especially useful if memory is an issue and the
*  analysis data is not required for further computation, for example, when switching to another
*  sparse matrix format.
*
*  Calling \p rocsparse_ellsv_clear is optional. All allocated resources will be cleared when the
*  opaque \ref rocsparse_mat_info struct is destroyed using \ref rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocSPARSE library context queue.
*  @param[in]
*  descr       descriptor of the sparse ELL matrix.
*  @param[inout]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer holding the meta data could not
*              be deallocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ellsv_clear(rocsparse_handle          handle,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_mat_info        info);

#ifdef __cplusplus
}
#endif

#endif // ROCSPARSE_ELLSV_H
