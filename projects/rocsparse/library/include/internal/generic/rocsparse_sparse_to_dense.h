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
*  \code{.c}
*   //     1 4 0 0 0 0
*   // A = 0 2 3 0 0 0
*   //     5 0 0 7 8 0
*   //     0 0 9 0 6 0
*   int m   = 4;
*   int n   = 6;
*
*   std::vector<int> hcsr_row_ptr = {0, 2, 4, 7, 9};
*   std::vector<int> hcsr_col_ind = {0, 1, 1, 2, 0, 3, 4, 2, 4};
*   std::vector<float> hcsr_val   = {1, 4, 2, 3, 5, 7, 8, 9, 6};
*   std::vector<float> hdense(m * n, 0.0f);
*
*   int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];
*
*   // Offload data to device
*   int* dcsr_row_ptr;
*   int* dcsr_col_ind;
*   float* dcsr_val;
*   float* ddense;
*   hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
*   hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz);
*   hipMalloc((void**)&dcsr_val, sizeof(float) * nnz);
*   hipMalloc((void**)&ddense, sizeof(float) * m * n);
*
*   hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(ddense, hdense.data(), sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_spmat_descr matA;
*   rocsparse_dnmat_descr matB;
*
*   rocsparse_indextype row_idx_type = rocsparse_indextype_i32;
*   rocsparse_indextype col_idx_type = rocsparse_indextype_i32;
*   rocsparse_datatype  data_type = rocsparse_datatype_f32_r;
*   rocsparse_index_base idx_base = rocsparse_index_base_zero;
*
*   rocsparse_create_handle(&handle);
*
*   // Create sparse matrix A
*   rocsparse_create_csr_descr(&matA,
*                              m,
*                              n,
*                              nnz,
*                              dcsr_row_ptr,
*                              dcsr_col_ind,
*                              dcsr_val,
*                              row_idx_type,
*                              col_idx_type,
*                              idx_base,
*                              data_type);
*
*   // Create dense matrix B
*   rocsparse_create_dnmat_descr(&matB, m, n, m, ddense, data_type, rocsparse_order_column);
*
*   // Call sparse_to_dense
*   size_t buffer_size = 0;
*   rocsparse_sparse_to_dense(handle,
*                             matA,
*                             matB,
*                             rocsparse_sparse_to_dense_alg_default,
*                             &buffer_size,
*                             nullptr);
*
*   void* temp_buffer;
*   hipMalloc((void**)&temp_buffer, buffer_size);
*
*   rocsparse_sparse_to_dense(handle,
*                             matA,
*                             matB,
*                             rocsparse_sparse_to_dense_alg_default,
*                             &buffer_size,
*                             temp_buffer);
*
*   // Copy result back to host
*   hipMemcpy(hdense.data(), ddense, sizeof(float) * m * n, hipMemcpyDeviceToHost);
*
*   // Clear rocSPARSE
*   rocsparse_destroy_spmat_descr(matA);
*   rocsparse_destroy_dnmat_descr(matB);
*   rocsparse_destroy_handle(handle);
*
*   // Clear device memory
*   hipFree(dcsr_row_ptr);
*   hipFree(dcsr_col_ind);
*   hipFree(dcsr_val);
*   hipFree(ddense);
*  \endcode
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
