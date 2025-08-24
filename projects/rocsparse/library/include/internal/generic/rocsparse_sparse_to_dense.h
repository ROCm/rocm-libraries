/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_SPARSE_TO_DENSE_H
#define ROCSPARSE_SPARSE_TO_DENSE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix to dense matrix conversion
*
*  \details
*  \p rocsparse_sparse_to_dense performs the conversion of a sparse matrix in CSR, CSC, or COO format to a dense matrix
*
*  \p rocsparse_sparse_to_dense requires multiple steps to complete. First, the user calls \p rocsparse_sparse_to_dense
*  with \p nullptr passed into \p temp_buffer:
*  \code{.c}
*   // Call sparse_to_dense to get required buffer size
*   size_t buffer_size = 0;
*   rocsparse_sparse_to_dense(handle,
*                             matA,
*                             matB,
*                             rocsparse_sparse_to_dense_alg_default,
*                             &buffer_size,
*                             nullptr);
*  \endcode
*  After this is called, the \p buffer_size will be filled with the size of the required buffer that must be then allocated by the
*  user. Finally, the conversion is completed by calling \p rocsparse_sparse_to_dense with both the \p buffer_size and \p temp_buffer:
*  \code{.c}
*   // Call dense_to_sparse to complete conversion
*   rocsparse_sparse_to_dense(handle,
*                             matA,
*                             matB,
*                             rocsparse_sparse_to_dense_alg_default,
*                             &buffer_size,
*                             temp_buffer);
*  \endcode
*  Currently, \p rocsparse_sparse_to_dense only supports the algorithm \ref rocsparse_sparse_to_dense_alg_default.
*  See full example below.
*
*  \p rocsparse_sparse_to_dense supports \ref rocsparse_datatype_f16_r, \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
*  \ref rocsparse_datatype_f32_c, and \ref rocsparse_datatype_f64_c for values arrays in the sparse matrix
*  (stored in CSR, CSC, or COO format) and the dense matrix. For the row/column offset and row/column index arrays of the
*  sparse matrix, \p rocsparse_sparse_to_dense supports the precisions \ref rocsparse_indextype_i32 and
*  \ref rocsparse_indextype_i64.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="sparse_to_dense_uniform">Uniform Precisions</caption>
*  <tr><th>A / B
*  <tr><td>rocsparse_datatype_f16_r
*  <tr><td>rocsparse_datatype_bf16_r
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the sparse to dense operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  mat_A        sparse matrix descriptor.
*  @param[in]
*  mat_B        dense matrix descriptor.
*  @param[in]
*  alg          algorithm for the sparse to dense computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the sparse to dense operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p mat_A, \p mat_B, or \p buffer_size
*               pointer is invalid.
*
*  \par Example
*  \snippet example_rocsparse_sparse_to_dense.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle              handle,
                                           rocsparse_const_spmat_descr   mat_A,
                                           rocsparse_dnmat_descr         mat_B,
                                           rocsparse_sparse_to_dense_alg alg,
                                           size_t*                       buffer_size,
                                           void*                         temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPARSE_TO_DENSE_H */
