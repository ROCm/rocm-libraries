/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_COOSORT_H
#define ROCSPARSE_COOSORT_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \details
*  \p rocsparse_coosort_buffer_size returns the size of the temporary storage buffer that is
*  required by \ref rocsparse_coosort_by_row() and \ref rocsparse_coosort_by_column(). The
*  temporary storage buffer has to be allocated by the user.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse COO matrix.
*  @param[in]
*  n               number of columns of the sparse COO matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse COO matrix.
*  @param[in]
*  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
*                  COO matrix.
*  @param[in]
*  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
*                  COO matrix.
*  @param[out]
*  buffer_size     number of bytes of the temporary storage buffer required by
*                  \ref rocsparse_coosort_by_row() and \ref rocsparse_coosort_by_column().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
*              \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const rocsparse_int* coo_row_ind,
                                               const rocsparse_int* coo_col_ind,
                                               size_t*              buffer_size);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by row
*
*  \details
*  \p rocsparse_coosort_by_row sorts a matrix in COO format by row. The sorted
*  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
*  case, \p perm must be initialized as the identity permutation, see
*  \ref rocsparse_create_identity_permutation().
*
*  \p rocsparse_coosort_by_row requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  \ref rocsparse_coosort_buffer_size().
*
*  \note
*  \p perm can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse COO matrix.
*  @param[in]
*  n               number of columns of the sparse COO matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse COO matrix.
*  @param[inout]
*  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
*                  COO matrix.
*  @param[inout]
*  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
*                  COO matrix.
*  @param[inout]
*  perm            array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned by
*                  \ref rocsparse_coosort_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
*              \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  The following example sorts a \f$3 \times 3\f$ COO matrix by row indices.
*  \snippet example_rocsparse_coosort_by_row.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_by_row(rocsparse_handle handle,
                                          rocsparse_int    m,
                                          rocsparse_int    n,
                                          rocsparse_int    nnz,
                                          rocsparse_int*   coo_row_ind,
                                          rocsparse_int*   coo_col_ind,
                                          rocsparse_int*   perm,
                                          void*            temp_buffer);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by column
*
*  \details
*  \p rocsparse_coosort_by_column sorts a matrix in COO format by column. The sorted
*  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
*  case, \p perm must be initialized as the identity permutation, see
*  \ref rocsparse_create_identity_permutation().
*
*  \p rocsparse_coosort_by_column requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  \ref rocsparse_coosort_buffer_size().
*
*  \note
*  \p perm can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse COO matrix.
*  @param[in]
*  n               number of columns of the sparse COO matrix.
*  @param[in]
*  nnz             number of non-zero entries of the sparse COO matrix.
*  @param[inout]
*  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
*                  COO matrix.
*  @param[inout]
*  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
*                  COO matrix.
*  @param[inout]
*  perm            array of \p nnz integers containing the unsorted map indices, can be
*                  \p NULL.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned by
*                  \ref rocsparse_coosort_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
*              \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  The following example sorts a \f$3 \times 3\f$ COO matrix by column indices.
*  \snippet example_rocsparse_coosort_by_col.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_by_column(rocsparse_handle handle,
                                             rocsparse_int    m,
                                             rocsparse_int    n,
                                             rocsparse_int    nnz,
                                             rocsparse_int*   coo_row_ind,
                                             rocsparse_int*   coo_col_ind,
                                             rocsparse_int*   perm,
                                             void*            temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_COOSORT_H */
