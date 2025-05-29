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

#ifndef ROCSPARSE_SPMV_H
#define ROCSPARSE_SPMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix vector multiplication
*
*  \details
*  \p rocsparse_spmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$ matrix \f$op(A)\f$, defined in CSR,
*  CSC, COO, COO (AoS), BSR, or ELL format, with the dense vector \f$x\f$ and adds the result to the dense vector \f$y\f$
*  that is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  Performing the above operation involves multiple steps. First the user calls \p rocsparse_spmv with the stage parameter set to
*  \ref rocsparse_spmv_stage_buffer_size to determine the size of the required temporary storage buffer. The user then allocates this
*  buffer and calls \p rocsparse_spmv with the stage parameter set to \ref rocsparse_spmv_stage_preprocess. Depending on the algorithm
*  and sparse matrix format, this will perform analysis on the sparsity pattern of \f$op(A)\f$. Finally the user completes the operation
*  by calling \p rocsparse_spmv with the stage parmeter set to \ref rocsparse_spmv_stage_compute. The buffer size, buffer allocation, and
*  preprecess stages only need to be called once for a given sparse matrix \f$op(A)\f$ while the computation stage can be repeatedly used
*  with different \f$x\f$ and \f$y\f$ vectors. Once all calls to \p rocsparse_spmv are complete, the temporary buffer can be deallocated.
*
*  \p rocsparse_spmv supports multiple different algorithms. These algorithms have different trade offs depending on the sparsity
*  pattern of the matrix, whether or not the results need to be deterministic, and how many times the sparse-vector product will
*  be performed.
*
*  <table>
*  <caption id="spmv_csr_algorithms">CSR/CSC Algorithms</caption>
*  <tr><th>Algorithm                            <th>Deterministic  <th>Preprocessing  <th>Notes
*  <tr><td>rocsparse_spmv_alg_csr_rowsplit</td> <td>Yes</td>       <td>No</td>        <td>Is best suited for matrices with all rows having a similar number of non-zeros. Can out perform adaptive and LRB algorithms in certain sparsity patterns. Will perform very poorly if some rows have few non-zeros and some rows have many non-zeros.</td>
*  <tr><td>rocsparse_spmv_alg_csr_stream</td>   <td>Yes</td>       <td>No</td>        <td>[Deprecated] old name for rocsparse_spmv_alg_csr_rowsplit.</td>
*  <tr><td>rocsparse_spmv_alg_csr_adaptive</td> <td>No</td>        <td>Yes</td>       <td>Generally the fastest algorithm across all matrix sparsity patterns. This includes matrices that have some rows with many non-zeros and some rows with few non-zeros. Requires a lengthy preprocessing that needs to be amortized over many subsequent sparse vector products.</td>
*  <tr><td>rocsparse_spmv_alg_csr_lrb</td>      <td>No</td>        <td>Yes</td>       <td>Like adaptive algorithm, generally performs well accross all matrix sparsity patterns. Generally not as fast as adaptive algorithm, however uses a much faster pre-processing step. Good for when only a few number of sparse vector products will be performed.</td>
*  </table>
*
*  <table>
*  <caption id="spmv_coo_algorithms">COO Algorithms</caption>
*  <tr><th>COO Algorithms                     <th>Deterministic   <th>Preprocessing <th>Notes
*  <tr><td>rocsparse_spmv_alg_coo</td>        <td>Yes</td>        <td>Yes</td>      <td>Generally not as fast as atomic algorithm but is deterministic</td>
*  <tr><td>rocsparse_spmv_alg_coo_atomic</td> <td>No</td>         <td>No</td>       <td>Generally the fastest COO algorithm</td>
*  </table>
*
*  <table>
*  <caption id="spmv_ell_algorithms">ELL Algorithms</caption>
*  <tr><th>ELL Algorithms                <th>Deterministic   <th>Preprocessing <th>Notes
*  <tr><td>rocsparse_spmv_alg_ell</td>   <td>Yes</td>        <td>No</td>       <td></td>
*  </table>
*
*  <table>
*  <caption id="spmv_bsr_algorithms">BSR Algorithms</caption>
*  <tr><th>BSR Algorithm                 <th>Deterministic   <th>Preprocessing <th>Notes
*  <tr><td>rocsparse_spmv_alg_bsr</td>   <td>Yes</td>        <td>No</td>       <td></td>
*  </table>
*
*  \p rocsparse_spmv supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix \f$op(A)\f$ and the dense vectors \f$x\f$ and
*  \f$y\f$ and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmv_mixed">Mixed Precisions</caption>
*  <tr><th>A / X                    <th>Y                        <th>compute_type
*  <tr><td>rocsparse_datatype_i8_r  <td>rocsparse_datatype_i32_r <td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_i8_r  <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f16_r <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  </table>
*
*  \par Mixed-regular real precisions
*  <table>
*  <caption id="spmv_mixed_regular_real">Mixed-regular real precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c <td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed-regular Complex precisions
*  <table>
*  <caption id="spmv_mixed_regular_complex">Mixed-regular Complex precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f64_c
*  </table>
*
*  \p rocsparse_spmv supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index precisions
*  for storing the row pointer and column indices arrays of the sparse matrices.
*
*  \note
*  None of the algorithms above are deterministic when \f$A\f$ is transposed.
*
*  \note
*  The sparse matrix formats currently supported are: \ref rocsparse_format_bsr, \ref rocsparse_format_coo,
*  \ref rocsparse_format_coo_aos, \ref rocsparse_format_csr, \ref rocsparse_format_csc and \ref rocsparse_format_ell.
*
*  \note
*  Only the \ref rocsparse_spmv_stage_buffer_size stage and the \ref rocsparse_spmv_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spmv_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Only the \ref rocsparse_spmv_stage_buffer_size stage and the \ref rocsparse_spmv_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spmv_stage_preprocess stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans        matrix operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat          matrix descriptor.
*  @param[in]
*  x            vector descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  y            vector descriptor.
*  @param[in]
*  compute_type floating point precision for the SpMV computation.
*  @param[in]
*  alg          SpMV algorithm for the SpMV computation.
*  @param[in]
*  stage        SpMV stage for the SpMV computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When the
*               \ref rocsparse_spmv_stage_buffer_size stage is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpMV operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context \p handle was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat, \p x, \p beta, \p y or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_invalid_value the value of \p trans, \p compute_type, \p alg, or \p stage is incorrect.
*  \retval      rocsparse_status_not_implemented \p compute_type or \p alg is
*               currently not supported.
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
*   std::vector<float> hx(n, 1.0f);
*   std::vector<float> hy(m, 0.0f);
*
*   // Scalar alpha
*   float alpha = 3.7f;
*
*   // Scalar beta
*   float beta = 0.0f;
*
*   int nnz = hcsr_row_ptr[m] - hcsr_row_ptr[0];
*
*   // Offload data to device
*   int* dcsr_row_ptr;
*   int* dcsr_col_ind;
*   float* dcsr_val;
*   float* dx;
*   float* dy;
*   hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
*   hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz);
*   hipMalloc((void**)&dcsr_val, sizeof(float) * nnz);
*   hipMalloc((void**)&dx, sizeof(float) * n);
*   hipMalloc((void**)&dy, sizeof(float) * m);
*
*   hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(float) * nnz, hipMemcpyHostToDevice);
*   hipMemcpy(dx, hx.data(), sizeof(float) * n, hipMemcpyHostToDevice);
*
*   rocsparse_handle     handle;
*   rocsparse_spmat_descr matA;
*   rocsparse_dnvec_descr vecX;
*   rocsparse_dnvec_descr vecY;
*
*   rocsparse_indextype row_idx_type = rocsparse_indextype_i32;
*   rocsparse_indextype col_idx_type = rocsparse_indextype_i32;
*   rocsparse_datatype  data_type = rocsparse_datatype_f32_r;
*   rocsparse_datatype  compute_type = rocsparse_datatype_f32_r;
*   rocsparse_index_base idx_base = rocsparse_index_base_zero;
*   rocsparse_operation trans = rocsparse_operation_none;
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
*   // Create dense vector X
*   rocsparse_create_dnvec_descr(&vecX,
*                                n,
*                                dx,
*                                data_type);
*
*   // Create dense vector Y
*   rocsparse_create_dnvec_descr(&vecY,
*                                m,
*                                dy,
*                                data_type);
*
*   // Call spmv to get buffer size
*   size_t buffer_size;
*   rocsparse_spmv(handle,
*                  trans,
*                  &alpha,
*                  matA,
*                  vecX,
*                  &beta,
*                  vecY,
*                  compute_type,
*                  rocsparse_spmv_alg_csr_adaptive,
*                  rocsparse_spmv_stage_buffer_size,
*                  &buffer_size,
*                  nullptr);
*
*   void* temp_buffer;
*   hipMalloc((void**)&temp_buffer, buffer_size);
*
*   // Call spmv to perform analysis
*   rocsparse_spmv(handle,
*                  trans,
*                  &alpha,
*                  matA,
*                  vecX,
*                  &beta,
*                  vecY,
*                  compute_type,
*                  rocsparse_spmv_alg_csr_adaptive,
*                  rocsparse_spmv_stage_preprocess,
*                  &buffer_size,
*                  temp_buffer);
*
*   // Call spmv to perform computation
*   rocsparse_spmv(handle,
*                  trans,
*                  &alpha,
*                  matA,
*                  vecX,
*                  &beta,
*                  vecY,
*                  compute_type,
*                  rocsparse_spmv_alg_csr_adaptive,
*                  rocsparse_spmv_stage_compute,
*                  &buffer_size,
*                  temp_buffer);
*
*   // Copy result back to host
*   hipMemcpy(hy.data(), dy, sizeof(float) * m, hipMemcpyDeviceToHost);
*
*   // Clear rocSPARSE
*   rocsparse_destroy_spmat_descr(matA);
*   rocsparse_destroy_dnvec_descr(vecX);
*   rocsparse_destroy_dnvec_descr(vecY);
*   rocsparse_destroy_handle(handle);
*
*   // Clear device memory
*   hipFree(dcsr_row_ptr);
*   hipFree(dcsr_col_ind);
*   hipFree(dcsr_val);
*   hipFree(dx);
*   hipFree(dy);
*   hipFree(temp_buffer);
*  \endcode
*/
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_v2_spmv instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_spmv(rocsparse_handle            handle,
                   rocsparse_operation         trans,
                   const void*                 alpha,
                   rocsparse_const_spmat_descr mat,
                   rocsparse_const_dnvec_descr x,
                   const void*                 beta,
                   const rocsparse_dnvec_descr y,
                   rocsparse_datatype          compute_type,
                   rocsparse_spmv_alg          alg,
                   rocsparse_spmv_stage        stage,
                   size_t*                     buffer_size,
                   void*                       temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPMV_H */
