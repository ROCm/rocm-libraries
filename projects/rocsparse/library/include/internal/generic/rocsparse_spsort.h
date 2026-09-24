/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_SPSORT_H
#define ROCSPARSE_SPSORT_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_spsort_buffer_size returns the size of the required buffer to execute the given stage of the SpSort operation.
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  descr        SpSort descriptor.
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor, the matrix to sort.
*  @param[in]
*  mat_B        sparse matrix \f$B\f$ descriptor, the sorted output matrix. \p mat_B can be the same
*               descriptor as \p mat_A to sort in place.
*  @param[in]
*  stage        SpSort stage for the SpSort computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p descr, \p mat_A, \p mat_B, or \p buffer_size pointer is invalid.
*  \retval rocsparse_status_invalid_size \p mat_A and \p mat_B do not have the same dimensions or number of non-zeros, or the batch strides make the batches overlap.
*  \retval rocsparse_status_invalid_value \p mat_A and \p mat_B do not have the same format, index types, data type, index base or batch count, or the direction is not supported for the format of \p mat_A.
*  \retval rocsparse_status_not_implemented \p mat_A is not in COO, CSR or CSC format.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spsort_buffer_size(rocsparse_handle            handle,
                                              rocsparse_spsort_descr      descr,
                                              rocsparse_const_spmat_descr mat_A,
                                              rocsparse_spmat_descr       mat_B,
                                              rocsparse_spsort_stage      stage,
                                              size_t*                     buffer_size,
                                              rocsparse_error*            error);

/*! \ingroup generic_module
*  \brief Sparse matrix sorting.
*
*  \details
*  \p rocsparse_spsort sorts the sparse matrix \f$A\f$ and writes the result to the sparse matrix
*  \f$B\f$. \f$A\f$ is left unchanged. \f$B\f$ must be created with the same format, dimensions,
*  number of non-zeros, index types, data type and index base as \f$A\f$. If the same descriptor
*  is passed for \p mat_A and \p mat_B, the matrix is sorted in place. Otherwise, the arrays of
*  \f$B\f$ must not overlap the arrays of \f$A\f$.
*
*  The supported formats are \ref rocsparse_format_coo, \ref rocsparse_format_csr and
*  \ref rocsparse_format_csc. The direction set with \ref rocsparse_spsort_input_direction
*  selects the sort order:
*  - COO: \ref rocsparse_direction_row sorts the entries by row and then by column, and
*    \ref rocsparse_direction_column sorts them by column and then by row.
*  - CSR: only the column indices within each row are sorted, so the direction must be
*    \ref rocsparse_direction_row. The row pointer of \f$B\f$ is a copy of the row pointer of \f$A\f$.
*  - CSC: only the row indices within each column are sorted, so the direction must be
*    \ref rocsparse_direction_column. The column pointer of \f$B\f$ is a copy of the column pointer of \f$A\f$.
*
*  Batched matrices are supported. They are set up with \ref rocsparse_coo_set_strided_batch,
*  \ref rocsparse_csr_set_strided_batch or \ref rocsparse_csc_set_strided_batch. Each batch is
*  sorted independently. \f$A\f$ and \f$B\f$ must have the same batch count, and the strides of
*  \f$A\f$ and \f$B\f$ may differ. Batches must not overlap:
*  - The COO batch stride, and the CSR/CSC columns/values batch stride, must be at least the
*    number of non-zeros.
*  - The CSR/CSC offsets batch stride must be zero, meaning all batches share one pointer
*    array, or at least the size of the pointer array. The offsets batch stride of \f$B\f$
*    can only be zero if the offsets batch stride of \f$A\f$ is zero.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  descr        SpSort descriptor.
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor, the matrix to sort.
*  @param[inout]
*  mat_B        sparse matrix \f$B\f$ descriptor, the sorted output matrix. \p mat_B can be the same
*               descriptor as \p mat_A to sort in place.
*  @param[in]
*  stage        SpSort stage for the SpSort computation.
*  @param[in]
*  buffer_size  number of bytes of the temporary storage buffer. \p buffer_size is
*               determined by calling \ref rocsparse_spsort_buffer_size.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p descr, \p mat_A, \p mat_B, or \p temp_buffer pointer is invalid.
*  \retval rocsparse_status_invalid_size \p mat_A and \p mat_B do not have the same dimensions or number of non-zeros, or the batch strides make the batches overlap.
*  \retval rocsparse_status_invalid_value \p mat_A and \p mat_B do not have the same format, index types, data type, index base or batch count, or the direction is not supported for the format of \p mat_A.
*  \retval rocsparse_status_not_implemented \p mat_A is not in COO, CSR or CSC format.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spsort(rocsparse_handle            handle,
                                  rocsparse_spsort_descr      descr,
                                  rocsparse_const_spmat_descr mat_A,
                                  rocsparse_spmat_descr       mat_B,
                                  rocsparse_spsort_stage      stage,
                                  size_t                      buffer_size,
                                  void*                       temp_buffer,
                                  rocsparse_error*            error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPSORT_H */
