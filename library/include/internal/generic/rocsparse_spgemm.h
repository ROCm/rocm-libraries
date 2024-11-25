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

#ifndef ROCSPARSE_SPGEMM_H
#define ROCSPARSE_SPGEMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix sparse matrix multiplication
*
*  \details
*  \p rocsparse_spgemm multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$ and
*  adds the result to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot D,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spgemm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / D / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  This function does not produce deterministic results.
*
*  \note SpGEMM requires three stages to complete. The first stage
*  \ref rocsparse_spgemm_stage_buffer_size will return the size of the temporary storage buffer
*  that is required for subsequent calls to \ref rocsparse_spgemm. The second stage
*  \ref rocsparse_spgemm_stage_nnz will determine the number of non-zero elements of the
*  resulting \f$C\f$ matrix. If the sparsity pattern of \f$C\f$ is already known, this
*  stage can be skipped. In the final stage \ref rocsparse_spgemm_stage_compute, the actual
*  computation is performed.
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot op(A) \cdot op(B)\f$ will be
*  computed.
*  \note Currently only CSR and BSR formats are supported.
*  \note If \ref rocsparse_spgemm_stage_symbolic is selected then the symbolic computation is performed only.
*  \note If \ref rocsparse_spgemm_stage_numeric is selected then the numeric computation is performed only.
*  \note For the \ref rocsparse_spgemm_stage_symbolic and \ref rocsparse_spgemm_stage_numeric stages, only
*  CSR matrix format is currently supported.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note It is allowed to pass the same sparse matrix for \f$C\f$ and \f$D\f$, if both
*  matrices have the same sparsity pattern.
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for rare matrix products with more than 4096 non-zero entries
*  per row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans_A      sparse matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B      sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  B            sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[in]
*  D            sparse matrix \f$D\f$ descriptor.
*  @param[out]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SpGEMM computation.
*  @param[in]
*  alg          SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  stage        SpGEMM stage for the SpGEMM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpGEMM operation.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p A, \p B, \p D, \p C or \p buffer_size pointer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none or
*          \p trans_B != \ref rocsparse_operation_none.
*
*  \par Example
*  \code{.c}
*   // A - m x k
*   // B - k x n
*   // C - m x n
*   int m = 400;
*   int n = 400;
*   int k = 300;
*
*   std::vector<int> hcsr_row_ptr_A = {...}; // host A m x k matrix
*   std::vector<int> hcsr_col_ind_A = {...}; // host A m x k matrix
*   std::vector<float> hcsr_val_A = {...};   // host A m x k matrix
*
*   std::vector<int> hcsr_row_ptr_B = {...}; // host B k x n matrix
*   std::vector<int> hcsr_col_ind_B = {...}; // host B k x n matrix
*   std::vector<float> hcsr_val_B = {...};   // host B k x n matrix
*
*   int nnz_A = hcsr_val_A.size();
*   int nnz_B = hcsr_val_B.size();
*
*   float alpha            = 1.0f;
*   float beta             = 0.0f;
*
*   int* dcsr_row_ptr_A = nullptr;
*   int* dcsr_col_ind_A = nullptr;
*   float* dcsr_val_A = nullptr;
*
*   int* dcsr_row_ptr_B = nullptr;
*   int* dcsr_col_ind_B = nullptr;
*   float* dcsr_val_B = nullptr;
*
*   int* dcsr_row_ptr_C = nullptr;
*
*   hipMalloc((void**)&dcsr_row_ptr_A, (m + 1) * sizeof(int));
*   hipMalloc((void**)&dcsr_col_ind_A, nnz_A * sizeof(int));
*   hipMalloc((void**)&dcsr_val_A, nnz_A * sizeof(float));
*
*   hipMalloc((void**)&dcsr_row_ptr_B, (k + 1) * sizeof(int));
*   hipMalloc((void**)&dcsr_col_ind_B, nnz_B * sizeof(int));
*   hipMalloc((void**)&dcsr_val_B, nnz_B * sizeof(float));
*
*   hipMalloc((void**)&dcsr_row_ptr_C, (m + 1) * sizeof(int));
*
*   hipMemcpy(dcsr_row_ptr_A, hcsr_row_ptr_A.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind_A, hcsr_col_ind_A.data(), nnz_A * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val_A, hcsr_val_A.data(), nnz_A * sizeof(float), hipMemcpyHostToDevice);
*
*   hipMemcpy(dcsr_row_ptr_B, hcsr_row_ptr_B.data(), (k + 1) * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind_B, hcsr_col_ind_B.data(), nnz_B * sizeof(int), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val_B, hcsr_val_B.data(), nnz_B * sizeof(float), hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_spmat_descr matA, matB, matC, matD;
*   void*                temp_buffer    = NULL;
*   size_t               buffer_size = 0;
*
*   rocsparse_operation trans_A = rocsparse_operation_none;
*   rocsparse_operation trans_B = rocsparse_operation_none;
*   rocsparse_index_base index_base = rocsparse_index_base_zero;
*   rocsparse_indextype itype = rocsparse_indextype_i32;
*   rocsparse_indextype jtype = rocsparse_indextype_i32;
*   rocsparse_datatype  ttype = rocsparse_datatype_f32_r;
*
*   rocsparse_create_handle(&handle);
*
*   // Create sparse matrix A in CSR format
*   rocsparse_create_csr_descr(&matA, m, k, nnz_A,
*                       dcsr_row_ptr_A, dcsr_col_ind_A, dcsr_val_A,
*                       itype, jtype,
*                       index_base, ttype);
*
*   // Create sparse matrix B in CSR format
*   rocsparse_create_csr_descr(&matB, k, n, nnz_B,
*                       dcsr_row_ptr_B, dcsr_col_ind_B, dcsr_val_B,
*                       itype, jtype,
*                       index_base, ttype);
*
*   // Create sparse matrix C in CSR format
*   rocsparse_create_csr_descr(&matC, m, n, 0,
*                       dcsr_row_ptr_C, nullptr, nullptr,
*                       itype, jtype,
*                       index_base, ttype);
*
*   // Create sparse matrix D in CSR format
*   rocsparse_create_csr_descr(&matD, 0, 0, 0,
*                       nullptr, nullptr, nullptr,
*                       itype, jtype,
*                       index_base, ttype);
*
*   // Determine buffer size
*   rocsparse_spgemm(handle,
*                    trans_A,
*                    trans_B,
*                    &alpha,
*                    matA,
*                    matB,
*                    &beta,
*                    matD,
*                    matC,
*                    ttype,
*                    rocsparse_spgemm_alg_default,
*                    rocsparse_spgemm_stage_buffer_size,
*                    &buffer_size,
*                    nullptr);
*
*   hipMalloc(&temp_buffer, buffer_size);
*
*   // Determine number of non-zeros in C matrix
*   rocsparse_spgemm(handle,
*                    trans_A,
*                    trans_B,
*                    &alpha,
*                    matA,
*                    matB,
*                    &beta,
*                    matD,
*                    matC,
*                    ttype,
*                    rocsparse_spgemm_alg_default,
*                    rocsparse_spgemm_stage_nnz,
*                    &buffer_size,
*                    temp_buffer);
*
*   int64_t rows_C;
*   int64_t cols_C;
*   int64_t nnz_C;
*
*   // Extract number of non-zeros in C matrix so we can allocate the column indices and values arrays
*   rocsparse_spmat_get_size(matC, &rows_C, &cols_C, &nnz_C);
*
*   int* dcsr_col_ind_C;
*   float* dcsr_val_C;
*   hipMalloc((void**)&dcsr_col_ind_C, sizeof(int) * nnz_C);
*   hipMalloc((void**)&dcsr_val_C, sizeof(float) * nnz_C);
*
*   // Set C matrix pointers
*   rocsparse_csr_set_pointers(matC, dcsr_row_ptr_C, dcsr_col_ind_C, dcsr_val_C);
*
*   // SpGEMM computation
*   rocsparse_spgemm(handle,
*                    trans_A,
*                    trans_B,
*                    &alpha,
*                    matA,
*                    matB,
*                    &beta,
*                    matD,
*                    matC,
*                    ttype,
*                    rocsparse_spgemm_alg_default,
*                    rocsparse_spgemm_stage_compute,
*                    &buffer_size,
*                    temp_buffer);
*
*   // Copy C matrix result back to host
*   std::vector<int> hcsr_row_ptr_C(m + 1);
*   std::vector<int> hcsr_col_ind_C(nnz_C);
*   std::vector<float>  hcsr_val_C(nnz_C);
*
*   hipMemcpy(hcsr_row_ptr_C.data(), dcsr_row_ptr_C, sizeof(int) * (m + 1), hipMemcpyDeviceToHost);
*   hipMemcpy(hcsr_col_ind_C.data(), dcsr_col_ind_C, sizeof(int) * nnz_C, hipMemcpyDeviceToHost);
*   hipMemcpy(hcsr_val_C.data(), dcsr_val_C, sizeof(float) * nnz_C, hipMemcpyDeviceToHost);
*
*   // Destroy matrix descriptors
*   rocsparse_destroy_spmat_descr(matA);
*   rocsparse_destroy_spmat_descr(matB);
*   rocsparse_destroy_spmat_descr(matC);
*   rocsparse_destroy_spmat_descr(matD);
*   rocsparse_destroy_handle(handle);
*
*   // Free device arrays
*   hipFree(temp_buffer);
*   hipFree(dcsr_row_ptr_A);
*   hipFree(dcsr_col_ind_A);
*   hipFree(dcsr_val_A);
*
*   hipFree(dcsr_row_ptr_B);
*   hipFree(dcsr_col_ind_B);
*   hipFree(dcsr_val_B);
*
*   hipFree(dcsr_row_ptr_C);
*   hipFree(dcsr_col_ind_C);
*   hipFree(dcsr_val_C);
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spgemm(rocsparse_handle            handle,
                                  rocsparse_operation         trans_A,
                                  rocsparse_operation         trans_B,
                                  const void*                 alpha,
                                  rocsparse_const_spmat_descr A,
                                  rocsparse_const_spmat_descr B,
                                  const void*                 beta,
                                  rocsparse_const_spmat_descr D,
                                  rocsparse_spmat_descr       C,
                                  rocsparse_datatype          compute_type,
                                  rocsparse_spgemm_alg        alg,
                                  rocsparse_spgemm_stage      stage,
                                  size_t*                     buffer_size,
                                  void*                       temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPGEMM_H */
