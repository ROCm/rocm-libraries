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

#ifndef ROCSPARSE_SPMM_H
#define ROCSPARSE_SPMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix dense matrix multiplication.
*
*  \details
*  \p rocsparse_spmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$ matrix \f$op(A)\f$,
*  defined in CSR, COO, BSR or Blocked ELL storage format, and the dense \f$k \times n\f$ matrix \f$op(B)\f$
*  and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that is multiplied by the scalar
*  \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
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
*  Both \f$B\f$ and \f$C\f$ can be in row or column order.
*
*  \p rocsparse_spmm requires three stages to complete. First, the user passes the \ref rocsparse_spmm_stage_buffer_size
*  stage to determine the size of the required temporary storage buffer. Next, the user allocates this buffer and calls
*  \p rocsparse_spmm again with the \ref rocsparse_spmm_stage_preprocess stage which will perform analysis on the sparse
*  matrix \f$op(A)\f$. Finally, the user calls \p rocsparse_spmm with the \ref rocsparse_spmm_stage_compute stage to perform
*  the actual computation. The buffer size, buffer allocation, and preprecess stages only need to be called once for a given
*  sparse matrix \f$op(A)\f$ while the computation stage can be repeatedly used with different \f$B\f$ and \f$C\f$ matrices.
*  Once all calls to \p rocsparse_spmm are complete, the temporary buffer can be deallocated.
*
*  As noted above, both \f$B\f$ and \f$C\f$ can be in row or column order (this includes mixing the order so that \f$B\f$ is
*  row order and \f$C\f$ is column order and vice versa). For best performance, use row order for both \f$B\f$ and \f$C\f$ as
*  this provides the best memory access.
*
*  \p rocsparse_spmm supports multiple different algorithms. These algorithms have different trade offs depending on the sparsity
*  pattern of the matrix, whether or not the results need to be deterministic, and how many times the sparse-matrix product will
*  be performed.
*
*  <table>
*  <caption id="spmm_csr_algorithms">CSR Algorithms</caption>
*  <tr><th>CSR Algorithms                         <th>Deterministic  <th>Preprocessing  <th>Notes
*  <tr><td>rocsparse_spmm_alg_csr</td>            <td>Yes</td>       <td>No</td>        <td>Default algorithm.</td>
*  <tr><td>rocsparse_spmm_alg_csr_row_split</td>  <td>Yes</td>       <td>No</td>        <td>Assigns a fixed number of threads per row regardless of the number of non-zeros in each row. This can perform well when each row in the matrix has roughly the same number of non-zeros</td>
*  <tr><td>rocsparse_spmm_alg_csr_nnz_split</td>  <td>No</td>        <td>Yes</td>       <td>Distributes work by having each thread block work on a fixed number of non-zeros regardless of the number of rows that might be involved. This can perform well when the matrix has some rows with few non-zeros and some rows with many non-zeros</td>
*  <tr><td>rocsparse_spmm_alg_csr_merge_path</td> <td>No</td>        <td>Yes</td>       <td>Attempts to combine the approaches of row-split and non-zero split by having each block work on a fixed amount of work which can be either non-zeros or rows</td>
*  </table>
*
*  <table>
*  <caption id="spmm_coo_algorithms">COO Algorithms</caption>
*  <tr><th>COO Algorithms                               <th>Deterministic   <th>Preprocessing <th>Notes
*  <tr><td>rocsparse_spmm_alg_coo_segmented</td>        <td>Yes</td>        <td>No</td>       <td>Generally not as fast as atomic algorithm but is deterministic</td>
*  <tr><td>rocsparse_spmm_alg_coo_atomic</td>           <td>No</td>         <td>No</td>       <td>Generally the fastest COO algorithm. Is the default algorithm</td>
*  <tr><td>rocsparse_spmm_alg_coo_segmented_atomic</td> <td>No</td>         <td>No</td>       <td> </td>
*  </table>
*
*  <table>
*  <caption id="spmm_bell_algorithms">Blocked-ELL Algorithms</caption>
*  <tr><th>ELL Algorithms                <th>Deterministic   <th>Preprocessing <th>Notes
*  <tr><td>rocsparse_spmm_alg_bell</td>  <td>Yes</td>        <td>No</td>       <td></td>
*  </table>
*
*  <table>
*  <caption id="spmm_bsr_algorithms">BSR Algorithms</caption>
*  <tr><th>BSR Algorithms                <th>Deterministic   <th>Preprocessing <th>Notes
*  <tr><td>rocsparse_spmm_alg_bsr</td>   <td>Yes</td>        <td>No</td>       <td></td>
*  </table>
*
*  One can also pass \ref rocsparse_spmm_alg_default which will automatically select from the algorithms listed above
*  based on the sparse matrix format. In the case of CSR matrices this will set the algorithm to be \ref rocsparse_spmm_alg_csr, in
*  the case of Blocked ELL matrices this will set the algorithm to be \ref rocsparse_spmm_alg_bell, in the case of BSR matrix this
*  will set the algorithm to be \ref rocsparse_spmm_alg_bsr and for COO matrices it will set the algorithm to be
*  \ref rocsparse_spmm_alg_coo_atomic.
*
*  When A is transposed, \p rocsparse_spmm will revert to using \ref rocsparse_spmm_alg_csr
*  for CSR format and \ref rocsparse_spmm_alg_coo_atomic for COO format regardless of algorithm selected.
*
*  \p rocsparse_spmm supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrix \f$op(A)\f$ and the dense matrices \f$op(B)\f$ and
*  \f$C\f$ and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spmm_mixed">Mixed Precisions</caption>
*  <tr><th>A / B                     <th>C                        <th>compute_type
*  <tr><td>rocsparse_datatype_i8_r   <td>rocsparse_datatype_i32_r <td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_i8_r   <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f16_r  <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_r
*  </table>
*
*  \p rocsparse_spmm supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index precisions
*  for storing the row pointer and column indices arrays of the sparse matrices.
*
*  \p rocsparse_spmm also supports batched computation for CSR and COO matrices. There are three supported batch modes:
*  \f[
*      C_i = A \times B_i \\
*      C_i = A_i \times B \\
*      C_i = A_i \times B_i
*  \f]
*
*  The batch mode is determined by the batch count and stride passed for each matrix. For example
*  to use the first batch mode (\f$C_i = A \times B_i\f$) with 100 batches for non-transposed \f$A\f$,
*  \f$B\f$, and \f$C\f$, one passes:
*  \f[
*      batch\_count\_A=1 \\
*      batch\_count\_B=100 \\
*      batch\_count\_C=100 \\
*      offsets\_batch\_stride\_A=0 \\
*      columns\_values\_batch\_stride\_A=0 \\
*      batch\_stride\_B=k*n \\
*      batch\_stride\_C=m*n
*  \f]
*  To use the second batch mode (\f$C_i = A_i \times B\f$) one could use:
*  \f[
*      batch\_count\_A=100 \\
*      batch\_count\_B=1 \\
*      batch\_count\_C=100 \\
*      offsets\_batch\_stride\_A=m+1 \\
*      columns\_values\_batch\_stride\_A=nnz \\
*      batch\_stride\_B=0 \\
*      batch\_stride\_C=m*n
*  \f]
*  And to use the third batch mode (\f$C_i = A_i \times B_i\f$) one could use:
*  \f[
*      batch\_count\_A=100 \\
*      batch\_count\_B=100 \\
*      batch\_count\_C=100 \\
*      offsets\_batch\_stride\_A=m+1 \\
*      columns\_values\_batch\_stride_A=nnz \\
*      batch\_stride_B=k*n \\
*      batch\_stride_C=m*n
*  \f]
*  See examples below.
*
*  \note
*  None of the algorithms above are deterministic when \f$A\f$ is transposed or conjugate transposed.
*
*  \note
*  All algorithms perform best when using row ordering for the dense \f$B\f$ and \f$C\f$ matrices
*
*  \note
*  The sparse matrix formats currently supported are: \ref rocsparse_format_coo, \ref rocsparse_format_csr,
*  \ref rocsparse_format_csc, \ref rocsparse_format_bsr, and \ref rocsparse_format_bell.
*
*  \note
*  Mixed precisions only supported for BSR, CSR, CSC, and COO matrix formats.
*
*  \note
*  Only the \ref rocsparse_spmm_stage_buffer_size stage and the \ref rocsparse_spmm_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spmm_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Currently, only \p trans_A == \ref rocsparse_operation_none is supported for COO and Blocked ELL formats.
*
*  \note
*  Only the \ref rocsparse_spmm_stage_buffer_size stage and the \ref rocsparse_spmm_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spmm_stage_preprocess stage does not support hipGraph.
*
*  \note
*  Currently, only CSR, COO, BSR and Blocked ELL sparse formats are supported.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans_A      matrix operation type.
*  @param[in]
*  trans_B      matrix operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        matrix descriptor.
*  @param[in]
*  mat_B        matrix descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[in]
*  mat_C        matrix descriptor.
*  @param[in]
*  compute_type floating point precision for the SpMM computation.
*  @param[in]
*  alg          SpMM algorithm for the SpMM computation.
*  @param[in]
*  stage        SpMM stage for the SpMM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When the
*               \ref rocsparse_spmm_stage_buffer_size stage is passed, the required
*               allocation size (in bytes) is written to \p buffer_size and function
*               returns without performing the SpMM operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat_A, \p mat_B, \p mat_C, \p beta, or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p trans_A, \p trans_B, \p compute_type or \p alg is
*               currently not supported.
*  \par Example
*  This example performs sparse matrix-dense matrix multiplication, \f$C := \alpha \cdot A \cdot B + \beta \cdot C\f$
*  \snippet example_rocsparse_spmm.cpp doc example
*
*  \par Example
*  An example of the first batch mode (\f$C_i = A \times B_i\f$) is provided below.
*  \snippet example_rocsparse_spmm_batched.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                rocsparse_operation         trans_A,
                                rocsparse_operation         trans_B,
                                const void*                 alpha,
                                rocsparse_const_spmat_descr mat_A,
                                rocsparse_const_dnmat_descr mat_B,
                                const void*                 beta,
                                const rocsparse_dnmat_descr mat_C,
                                rocsparse_datatype          compute_type,
                                rocsparse_spmm_alg          alg,
                                rocsparse_spmm_stage        stage,
                                size_t*                     buffer_size,
                                void*                       temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPMM_H */
