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
*  \f$m \times k\f$ matrix \f$op(A)\f$ and the sparse \f$k \times n\f$ matrix \f$op(B)\f$ and
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
*        A,   & \text{if trans_A == rocsparse_operation_none}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none}
*    \end{array}
*    \right.
*  \f]
*
*  \p rocsparse_spgemm requires three stages to complete. First, the user passes the \ref rocsparse_spgemm_stage_buffer_size
*  stage to determine the size of the required temporary storage buffer. Next, the user allocates this buffer and calls
*  \p rocsparse_spgemm again with the \ref rocsparse_spgemm_stage_nnz stage which will determine the number of non-zeros
*  in \f$C\f$. This stage will also fill in the row pointer array of \f$C\f$. Now that the number of non-zeros in \f$C\f$
*  is known, the user allocates space for the column indices and values arrays of \f$C\f$. Finally, the user calls
*  \p rocsparse_spgemm with the \ref rocsparse_spgemm_stage_compute stage to perform the actual computation which fills in
*  the column indices and values arrays of \f$C\f$. Once all calls to \p rocsparse_spgemm are complete, the temporary buffer
*  can be deallocated.
*
*  Alternatively, the user may also want to perform sparse matrix products multiple times with matrices having the same sparsity
*  pattern, but whose values differ. In this scenario, the process begins like before. First, the user calls \p rocsparse_spgemm
*  with stage \ref rocsparse_spgemm_stage_buffer_size to determine the required buffer size. The user again allocates this buffer
*  and calls \p rocsparse_spgemm with the stage \ref rocsparse_spgemm_stage_nnz to determine the number of non-zeros in \f$C\f$.
*  The user allocates the \f$C\f$ column indices and values arrays. Now, however, the user calls \p rocsparse_spgemm with the
*  \ref rocsparse_spgemm_stage_symbolic stage which will fill in the column indices array of \f$C\f$ but not the values array.
*  The user is then free to repeatedly change the values of \f$A\f$, \f$B\f$, and \f$D\f$ and call \p rocsparse_spgemm with
*  the \ref rocsparse_spgemm_stage_numeric stage which fill the values array of \f$C\f$. The use of the extra
*  \ref rocsparse_spgemm_stage_symbolic and \ref rocsparse_spgemm_stage_numeric stages allows the user to compute sparsity pattern
*  of \f$C\f$ once, but compute the values multiple times.
*
*  \p rocsparse_spgemm supports multiple combinations of data types and compute types. The tables below indicate the currently
*  supported different data types that can be used for for the sparse matrices \f$op(A)\f$, \f$op(B)\f$, \f$C\f$, and \f$D\f$
*  and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
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
*  \p rocsparse_spgemm supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index precisions for storing the row
*  pointer and column indices arrays of the sparse matrices.
*
*  In general, when multiplying two sparse matrices together, it is entirely possible that the resulting matrix will require a
*  larger index representation to store correctly. For example, when multiplying \f$A \times B\f$ using
*  \ref rocsparse_indextype_i32 index types for the row pointer and column indices arrays, it may be the case that the row pointer
*  of the resulting \f$C\f$ matrix would require index precision \ref rocsparse_indextype_i64. This is currently not supported.
*  In this scenario, the user would need to store the \f$A\f$ and \f$B\f$ matrices using the higher index precision.
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
*  \snippet example_rocsparse_spgemm.cpp doc example
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
