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
*  source       source sparse matrix descriptor, the matrix to sort.
*  @param[in]
*  target       target sparse matrix descriptor, the sorted output matrix. \p target can be the same
*               descriptor as \p source to sort in place.
*  @param[in]
*  stage        SpSort stage for the SpSort computation.
*  @param[out]
*  buffer_size_in_bytes  size in bytes of the temporary storage buffer.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p descr, \p source, \p target, or \p buffer_size_in_bytes pointer is invalid.
*  \retval rocsparse_status_invalid_size \p source and \p target do not have the same dimensions or number of non-zeros, or the batch strides make the batches overlap.
*  \retval rocsparse_status_invalid_value \p source and \p target do not have the same format, index types, data type, index base or batch count, or the direction is not supported for the format of \p source.
*  \retval rocsparse_status_not_implemented \p source is not in COO, CSR or CSC format.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spsort_buffer_size(rocsparse_handle            handle,
                                              rocsparse_spsort_descr      descr,
                                              rocsparse_const_spmat_descr source,
                                              rocsparse_spmat_descr       target,
                                              rocsparse_spsort_stage      stage,
                                              size_t*                     buffer_size_in_bytes,
                                              rocsparse_error*            error);

/*! \ingroup generic_module
*  \brief Sparse matrix sorting.
*
*  \details
*  \p rocsparse_spsort sorts the source sparse matrix and writes the result to the target sparse
*  matrix. The source matrix is left unchanged. The target matrix must be created with the same
*  format, dimensions, number of non-zeros, index types, data type and index base as the source
*  matrix. If the same descriptor is passed for \p source and \p target, the matrix is sorted in
*  place. Otherwise, the arrays of the target matrix must not overlap the arrays of the source
*  matrix.
*
*  The supported formats are \ref rocsparse_format_coo, \ref rocsparse_format_csr and
*  \ref rocsparse_format_csc. The \ref rocsparse_direction set with
*  \ref rocsparse_spsort_input_direction selects the sort order:
*  - COO: \ref rocsparse_direction_row sorts the entries by row and then by column, and
*    \ref rocsparse_direction_column sorts them by column and then by row.
*  - CSR: only the column indices within each row are sorted, so the direction must be
*    \ref rocsparse_direction_row. The row pointer of the target matrix is a copy of the row
*    pointer of the source matrix.
*  - CSC: only the row indices within each column are sorted, so the direction must be
*    \ref rocsparse_direction_column. The column pointer of the target matrix is a copy of the
*    column pointer of the source matrix.
*
*  Batched matrices are supported. They are set up with \ref rocsparse_coo_set_strided_batch,
*  \ref rocsparse_csr_set_strided_batch or \ref rocsparse_csc_set_strided_batch. Each batch is
*  sorted independently. The source and target matrices must have the same batch count, and their
*  strides may differ. Batches must not overlap:
*  - The COO batch stride, and the CSR/CSC columns/values batch stride, must be at least the
*    number of non-zeros.
*  - The CSR/CSC offsets batch stride must be zero, meaning all batches share one pointer
*    array, or at least the size of the pointer array. The offsets batch stride of the target
*    matrix can only be zero if the offsets batch stride of the source matrix is zero.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  descr        SpSort descriptor.
*  @param[in]
*  source       source sparse matrix descriptor, the matrix to sort.
*  @param[inout]
*  target       target sparse matrix descriptor, the sorted output matrix. \p target can be the same
*               descriptor as \p source to sort in place.
*  @param[in]
*  stage        SpSort stage for the SpSort computation.
*  @param[in]
*  buffer_size_in_bytes  size in bytes of the temporary storage buffer. \p buffer_size_in_bytes is
*               determined by calling \ref rocsparse_spsort_buffer_size.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p descr, \p source, \p target, or \p temp_buffer pointer is invalid.
*  \retval rocsparse_status_invalid_size \p source and \p target do not have the same dimensions or number of non-zeros, or the batch strides make the batches overlap.
*  \retval rocsparse_status_invalid_value \p source and \p target do not have the same format, index types, data type, index base or batch count, or the direction is not supported for the format of \p source.
*  \retval rocsparse_status_not_implemented \p source is not in COO, CSR or CSC format.
*
*  \par Example
*  \snippet example_rocsparse_spsort.cpp doc example
*  \par Example
*  \snippet example_rocsparse_spsort_coo.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spsort(rocsparse_handle            handle,
                                  rocsparse_spsort_descr      descr,
                                  rocsparse_const_spmat_descr source,
                                  rocsparse_spmat_descr       target,
                                  rocsparse_spsort_stage      stage,
                                  size_t                      buffer_size_in_bytes,
                                  void*                       temp_buffer,
                                  rocsparse_error*            error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPSORT_H */
