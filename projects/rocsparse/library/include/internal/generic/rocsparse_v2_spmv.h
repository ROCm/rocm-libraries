/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_V2_SPMV_H
#define ROCSPARSE_V2_SPMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_v2_spmv_buffer_size returns the size of the required buffer to execute the given stage of the Version 2 SpMV operation.
*  This routine is used in conjunction with \ref rocsparse_v2_spmv(). See \ref rocsparse_v2_spmv for full description and example.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        SpMV descriptor
*  @param[in]
*  mat          sparse matrix descriptor.
*  @param[in]
*  x            dense vector descriptor.
*  @param[in]
*  y            dense vector descriptor.
*  @param[in]
*  stage        Version 2 SpMV stage for the SpMV computation.
*  @param[out]
*  buffer_size_in_bytes  number of bytes of the buffer.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value the \p stage value is invalid.
*  \retval rocsparse_status_invalid_pointer \p mat, \p x, \p y, \p descr or \p buffer_size_in_bytes pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_v2_spmv_buffer_size(rocsparse_handle            handle,
                                               rocsparse_spmv_descr        descr,
                                               rocsparse_const_spmat_descr mat,
                                               rocsparse_const_dnvec_descr x,
                                               rocsparse_const_dnvec_descr y,
                                               rocsparse_v2_spmv_stage     stage,
                                               size_t*                     buffer_size_in_bytes,
                                               rocsparse_error*            error);

/*! \ingroup generic_module
*  \brief Sparse matrix vector multiplication
*
*  \details
*  \p rocsparse_v2_spmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$ matrix \f$op(A)\f$ with the dense vector \f$x\f$ and adds the result to the dense vector \f$y\f$
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
*  \note The sparse matrix format \ref rocsparse_format_bell is not supported.
*
*  Performing the above operation involves two stages. The first stage is \ref rocsparse_v2_spmv_stage_analysis. This will perform an analysis of the symbolic information of \f$op(A)\f$. The second stage is \ref rocsparse_v2_spmv_stage_compute which corresponds to the actual calculation. The size of the buffer required for each stage is determined with calling the routine \ref rocsparse_v2_spmv_buffer_size. The stage \ref rocsparse_v2_spmv_stage_analysis only needs to be called once for a given sparse matrix \f$op(A)\f$ while the computation stage can be repeatedly used
*  with different \f$x\f$ and \f$y\f$ vectors.
*
*  \note The stage \ref rocsparse_v2_spmv_stage_analysis is mandatory, an error will be returned if that stage was not executed before the stage \ref rocsparse_v2_spmv_stage_compute.
*
*  \p rocsparse_v2_spmv supports multiple algorithms. These algorithms have different trade offs depending on the sparsity pattern of the matrix, whether or not the results need to be deterministic, and how many times the sparse-vector product will be performed.
*
*  <table>
*  <caption id="v2_spmv_csr_algorithms">CSR/CSC Algorithms</caption>
*  <tr><th>Algorithm                            <th>Deterministic  <th>Notes
*  <tr><td>rocsparse_spmv_alg_csr_rowsplit</td> <td>Yes</td>       <td>Is best suited for matrices with all rows having a similar number of non-zeros. Can out perform adaptive and LRB algorithms in certain sparsity patterns. Will perform very poorly if some rows have few non-zeros and some rows have many non-zeros.</td>
*  <tr><td>rocsparse_spmv_alg_csr_stream</td>   <td>Yes</td>       <td>[Deprecated] old name for rocsparse_spmv_alg_csr_rowsplit.</td>
*  <tr><td>rocsparse_spmv_alg_csr_adaptive</td> <td>No</td>        <td>Generally the fastest algorithm across all matrix sparsity patterns. This includes matrices that have some rows with many non-zeros and some rows with few non-zeros. Requires a lengthy preprocessing that needs to be amortized over many subsequent sparse vector products.</td>
*  <tr><td>rocsparse_spmv_alg_csr_lrb</td>      <td>No</td>        <td>Like adaptive algorithm, generally performs well across all matrix sparsity patterns. Generally not as fast as adaptive algorithm, however uses a much faster pre-processing step. Good for when only a few number of sparse vector products will be performed.</td>
*  <tr><td>rocsparse_spmv_alg_csr_nnzsplit</td> <td>No</td>        <td>Like adaptive algorithm, generally performs well across all matrix sparsity patterns. Generally not as fast as adaptive algorithm but faster than LRB algorithm. It uses a much faster pre-processing step than LRB. Good for when the number of sparse vector products that will be performed is less than one hundred. If more products need to be computed, the adaptive algorithm is probably faster.</td>
*  </table>
*
*  <table>
*  <caption id="v2_spmv_coo_algorithms">COO Algorithms</caption>
*  <tr><th>COO Algorithms                     <th>Deterministic   <th>Notes
*  <tr><td>rocsparse_spmv_alg_coo</td>        <td>Yes</td>        <td>Generally not as fast as atomic algorithm but is deterministic</td>
*  <tr><td>rocsparse_spmv_alg_coo_atomic</td> <td>No</td>         <td>Generally the fastest COO algorithm</td>
*  </table>
*
*  <table>
*  <caption id="v2_spmv_ell_algorithms">ELL Algorithms</caption>
*  <tr><th>ELL Algorithms                <th>Deterministic   <th>Notes
*  <tr><td>rocsparse_spmv_alg_ell</td>   <td>Yes</td>        <td></td>
*  </table>
*
*  <table>
*  <caption id="v2_spmv_bsr_algorithms">BSR Algorithms</caption>
*  <tr><th>BSR Algorithm                 <th>Deterministic   <th>Notes
*  <tr><td>rocsparse_spmv_alg_bsr</td>   <td>Yes</td>        <td></td>
*  </table>
*
*  \p rocsparse_v2_spmv supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix \f$op(A)\f$ and the dense vectors \f$x\f$ and
*  \f$y\f$ and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="v2_spmv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="v2_spmv_mixed">Mixed Precisions</caption>
*  <tr><th>A / X                    <th>Y                        <th>compute_type
*  <tr><td>rocsparse_datatype_i8_r  <td>rocsparse_datatype_i32_r <td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_i8_r  <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f16_r <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  </table>
*
*  \par Mixed-regular real precisions
*  <table>
*  <caption id="v2_spmv_mixed_regular_real">Mixed-regular real precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c <td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed-regular Complex precisions
*  <table>
*  <caption id="v2_spmv_mixed_regular_complex">Mixed-regular Complex precisions</caption>
*  <tr><th>A                        <th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f64_c
*  </table>
*
*  \p rocsparse_v2_spmv supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index precisions
*  for storing the row pointer and column indices arrays of the sparse matrices.
*
*  \note
*  None of the algorithms above are deterministic when \f$A\f$ is transposed.
*
*  \note
*  All the sparse matrix formats are supported except \ref rocsparse_format_bell.
*
*  \note
*  The \ref rocsparse_v2_spmv_stage_compute stage is non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The stage \ref rocsparse_v2_spmv_stage_analysis is blocking with respect to the host.
*
*  \note
*  Only the stage \ref rocsparse_v2_spmv_stage_compute
*  supports execution in a hipGraph context. The \ref rocsparse_v2_spmv_stage_analysis stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        spmv descriptor.
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
*  stage        SpMV stage of the SpMV algorithm.
*  @param[in]
*  buffer_size_in_bytes  size in bytes of the buffer, must be greater or equal to the buffer size obtained from \ref rocsparse_v2_spmv_buffer_size.
*  @param[in]
*  buffer       temporary buffer allocated by the user.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context \p handle was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat, \p x, \p beta, \p y or
*               \p buffer pointer is invalid.
*  \retval      rocsparse_status_invalid_value the value of \p stage is invalid.
*  \retval      rocsparse_status_not_implemented if \p alg is not supported or if the mixed precision configuration is not supported.
*
*  \par Example
*  \snippet example_rocsparse_v2_spmv.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_v2_spmv(rocsparse_handle            handle,
                                   rocsparse_spmv_descr        descr,
                                   const void*                 alpha,
                                   rocsparse_const_spmat_descr mat,
                                   rocsparse_const_dnvec_descr x,
                                   const void*                 beta,
                                   rocsparse_dnvec_descr       y,
                                   rocsparse_v2_spmv_stage     stage,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer,
                                   rocsparse_error*            error);
#ifdef __cplusplus
}
#endif

#endif
