/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_DENSE_TO_SPARSE_H
#define ROCSPARSE_DENSE_TO_SPARSE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Dense matrix to sparse matrix conversion
*
*  \details
*  \p rocsparse_dense_to_sparse performs the conversion of a dense matrix to a sparse matrix in CSR, CSC, or COO format.
*
*  \p rocsparse_dense_to_sparse requires multiple steps to complete. First, the user calls \p rocsparse_dense_to_sparse 
*  with \p nullptr passed into \p temp_buffer:
*  \code{.c}
*   // Call dense_to_sparse to get required buffer size
*   size_t buffer_size = 0;
*   rocsparse_dense_to_sparse(handle,
*                             matA,
*                             matB,
*                             rocsparse_dense_to_sparse_alg_default,
*                             &buffer_size,
*                             nullptr);
*  \endcode
*  After this is called, the \p buffer_size will be filled with the size of the required buffer that must be then allocated by the 
*  user. Next the user calls \p rocsparse_dense_to_sparse with the newly allocated \p temp_buffer and \p nullptr passed into 
*  \p buffer_size:
*  \code{.c}
*   // Call dense_to_sparse to perform analysis
*   rocsparse_dense_to_sparse(handle,
*                             matA,
*                             matB,
*                             rocsparse_dense_to_sparse_alg_default,
*                             nullptr,
*                             temp_buffer);
*  \endcode
*  This will determine the number of non-zeros that will exist in the sparse matrix which can be queried using 
*  \ref rocsparse_spmat_get_size routine. With this, the user can allocate the sparse matrix device arrays and 
*  set them on the sparse matrix descriptor using \ref rocsparse_csr_set_pointers (CSR format), 
*  \ref rocsparse_csc_set_pointers (for CSC format), or rocsparse_coo_set_pointers (for COO format). Finally, the 
*  conversion is completed by calling \p rocsparse_dense_to_sparse with both the \p buffer_size and \p temp_buffer:
*  \code{.c}
*   // Call dense_to_sparse to complete conversion
*   rocsparse_dense_to_sparse(handle,
*                             matA,
*                             matB,
*                             rocsparse_dense_to_sparse_alg_default,
*                             &buffer_size,
*                             temp_buffer);
*  \endcode
*  Currently, \p rocsparse_dense_to_sparse only supports the algorithm \ref rocsparse_dense_to_sparse_alg_default. 
*  See full example below.
*
*  \p rocsparse_dense_to_sparse supports \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r, 
*  \ref rocsparse_datatype_f32_c, and \ref rocsparse_datatype_f64_c for values arrays in the sparse matrix (stored in 
*  CSR, CSC, or COO format) and the dense matrix. For the row/column offset and row/column index arrays of the sparse matrix, 
*  \p rocsparse_dense_to_sparse supports the precisions \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64.
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the dense to sparse operation, when a nullptr is passed for
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
*  mat_A        dense matrix descriptor.
*  @param[in]
*  mat_B        sparse matrix descriptor.
*  @param[in]
*  alg          algorithm for the dense to sparse computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the dense to sparse operation.
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
*   std::vector<float> hdense = {1, 0, 5, 0, 4, 2, 0, 0, 0, 3, 0, 9, 0, 0, 7, 0, 0, 0, 8, 6, 0, 0, 0, 0};
*
*   // Offload data to device
*   int* dcsr_row_ptr;
*   float* ddense;
*   hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
*   hipMalloc((void**)&ddense, sizeof(float) * m * n);
*
*   hipMemcpy(ddense, hdense.data(), sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_dnmat_descr matA;
*   rocsparse_spmat_descr matB;
*
*   rocsparse_indextype row_idx_type = rocsparse_indextype_i32;
*   rocsparse_indextype col_idx_type = rocsparse_indextype_i32;
*   rocsparse_datatype  data_type = rocsparse_datatype_f32_r;
*   rocsparse_index_base idx_base = rocsparse_index_base_zero;
*
*   rocsparse_create_handle(&handle);
*
*   // Create sparse matrix A
*   rocsparse_create_dnmat_descr(&matA, m, n, m, ddense, data_type, rocsparse_order_column);
*
*   // Create dense matrix B
*   rocsparse_create_csr_descr(&matB,
*                              m,
*                              n,
*                              0,
*                              dcsr_row_ptr,
*                              nullptr,
*                              nullptr,
*                              row_idx_type,
*                              col_idx_type,
*                              idx_base,
*                              data_type);
*
*   // Call dense_to_sparse to get required buffer size
*   size_t buffer_size = 0;
*   rocsparse_dense_to_sparse(handle,
*                             matA,
*                             matB,
*                             rocsparse_dense_to_sparse_alg_default,
*                             &buffer_size,
*                             nullptr);
*
*   void* temp_buffer;
*   hipMalloc((void**)&temp_buffer, buffer_size);
*
*   // Call dense_to_sparse to perform analysis
*   rocsparse_dense_to_sparse(handle,
*                             matA,
*                             matB,
*                             rocsparse_dense_to_sparse_alg_default,
*                             nullptr,
*                             temp_buffer);
*
*   int64_t num_rows_tmp, num_cols_tmp, nnz;
*   rocsparse_spmat_get_size(matB, &num_rows_tmp, &num_cols_tmp, &nnz);
*
*   int* dcsr_col_ind;
*   float* dcsr_val;
*   hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz);
*   hipMalloc((void**)&dcsr_val, sizeof(float) * nnz);
*
*   rocsparse_csr_set_pointers(matB, dcsr_row_ptr, dcsr_col_ind, dcsr_val);
*
*   // Call dense_to_sparse to complete conversion
*   rocsparse_dense_to_sparse(handle,
*                             matA,
*                             matB,
*                             rocsparse_dense_to_sparse_alg_default,
*                             &buffer_size,
*                             temp_buffer);
*
*   std::vector<int> hcsr_row_ptr(m + 1, 0);
*   std::vector<int> hcsr_col_ind(nnz, 0);
*   std::vector<float> hcsr_val(nnz, 0);
*
*   // Copy result back to host
*   hipMemcpy(hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost);
*   hipMemcpy(hcsr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost);
*   hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*
*   // Clear rocSPARSE
*   rocsparse_destroy_dnmat_descr(matA);
*   rocsparse_destroy_spmat_descr(matB);
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
rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle              handle,
                                           rocsparse_const_dnmat_descr   mat_A,
                                           rocsparse_spmat_descr         mat_B,
                                           rocsparse_dense_to_sparse_alg alg,
                                           size_t*                       buffer_size,
                                           void*                         temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_DENSE_TO_SPARSE_H */
