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

#ifndef ROCSPARSE_BSRSM_H
#define ROCSPARSE_BSRSM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level3_module
*  \details
*  \p rocsparse_bsrsm_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during
*  \ref rocsparse_sbsrsm_solve "rocsparse_Xbsrsm_solve()" computation. The first zero
*  pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same index base as
*  the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_bsrsm_zero_pivot is a blocking function. It might influence
*  performance negatively.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[inout]
*  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
*              invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_zero_pivot zero pivot has been found.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsrsm_zero_pivot(rocsparse_handle   handle,
                                            rocsparse_mat_info info,
                                            rocsparse_int*     position);

/*! \ingroup level3_module
*  \details
*  \p rocsparse_bsrsm_buffer_size returns the size of the temporary storage buffer that
*  is required by \ref rocsparse_sbsrsm_analysis "rocsparse_Xbsrsm_analysis()" and
*  \ref rocsparse_sbsrsm_solve "rocsparse_Xbsrsm_solve()". The temporary storage buffer
*  must be allocated by the user.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_X     matrix X operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix A.
*  @param[in]
*  nrhs        number of columns of the column-oriented dense matrix op(X).
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix A.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix A.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim   block dimension of the sparse BSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              \ref rocsparse_sbsrsm_analysis "rocsparse_Xbsrsm_analysis()" and
*              \ref rocsparse_sbsrsm_solve "rocsparse_Xbsrsm_solve()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nrhs, \p nnzb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p info or \p buffer_size pointer
*              is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_X == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_X,
                                              rocsparse_int             mb,
                                              rocsparse_int             nrhs,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const float*              bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_X,
                                              rocsparse_int             mb,
                                              rocsparse_int             nrhs,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const double*             bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsm_buffer_size(rocsparse_handle               handle,
                                              rocsparse_direction            dir,
                                              rocsparse_operation            trans_A,
                                              rocsparse_operation            trans_X,
                                              rocsparse_int                  mb,
                                              rocsparse_int                  nrhs,
                                              rocsparse_int                  nnzb,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* bsr_val,
                                              const rocsparse_int*           bsr_row_ptr,
                                              const rocsparse_int*           bsr_col_ind,
                                              rocsparse_int                  block_dim,
                                              rocsparse_mat_info             info,
                                              size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsm_buffer_size(rocsparse_handle                handle,
                                              rocsparse_direction             dir,
                                              rocsparse_operation             trans_A,
                                              rocsparse_operation             trans_X,
                                              rocsparse_int                   mb,
                                              rocsparse_int                   nrhs,
                                              rocsparse_int                   nnzb,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* bsr_val,
                                              const rocsparse_int*            bsr_row_ptr,
                                              const rocsparse_int*            bsr_col_ind,
                                              rocsparse_int                   block_dim,
                                              rocsparse_mat_info              info,
                                              size_t*                         buffer_size);
/**@}*/

/*! \ingroup level3_module
*  \details
*  \p rocsparse_bsrsm_analysis performs the analysis step for
*  \ref rocsparse_sbsrsm_solve "rocsparse_Xbsrsm_solve()". It is expected that this function
*  will be executed only once for a given matrix and particular operation type. The analysis
*  meta data can be cleared by \ref rocsparse_bsrsm_clear().
*
*  \p rocsparse_bsrsm_analysis can share its meta data with
*  \ref rocsparse_sbsrilu0_analysis "rocsparse_Xbsrilu0_analysis()",
*  \ref rocsparse_sbsric0_analysis "rocsparse_Xbsric0_analysis()",
*  \ref rocsparse_sbsrsv_analysis "rocsparse_Xbsrsv_analysis()". Selecting
*  \ref rocsparse_analysis_policy_reuse policy can greatly improve computation
*  performance of meta data. However, the user needs to make sure that the sparsity
*  pattern remains unchanged. If this cannot be assured,
*  \ref rocsparse_analysis_policy_force has to be used.
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_X     matrix X operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix A.
*  @param[in]
*  nrhs        number of columns of the column-oriented dense matrix op(X).
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix A.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix A.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix A.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix A.
*  @param[in]
*  bsr_col_ind array of \p nnzb containing the block column indices of the sparse
*              BSR matrix A.
*  @param[in]
*  block_dim   block dimension of the sparse BSR matrix A.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  analysis    \ref rocsparse_analysis_policy_reuse or
*              \ref rocsparse_analysis_policy_force.
*  @param[in]
*  solve       \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nrhs, \p nnzb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
*              \p bsr_col_ind, \p info or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_X == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsm_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_X,
                                           rocsparse_int             mb,
                                           rocsparse_int             nrhs,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const float*              bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsm_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_X,
                                           rocsparse_int             mb,
                                           rocsparse_int             nrhs,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const double*             bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsm_analysis(rocsparse_handle               handle,
                                           rocsparse_direction            dir,
                                           rocsparse_operation            trans_A,
                                           rocsparse_operation            trans_X,
                                           rocsparse_int                  mb,
                                           rocsparse_int                  nrhs,
                                           rocsparse_int                  nnzb,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* bsr_val,
                                           const rocsparse_int*           bsr_row_ptr,
                                           const rocsparse_int*           bsr_col_ind,
                                           rocsparse_int                  block_dim,
                                           rocsparse_mat_info             info,
                                           rocsparse_analysis_policy      analysis,
                                           rocsparse_solve_policy         solve,
                                           void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsm_analysis(rocsparse_handle                handle,
                                           rocsparse_direction             dir,
                                           rocsparse_operation             trans_A,
                                           rocsparse_operation             trans_X,
                                           rocsparse_int                   mb,
                                           rocsparse_int                   nrhs,
                                           rocsparse_int                   nnzb,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* bsr_val,
                                           const rocsparse_int*            bsr_row_ptr,
                                           const rocsparse_int*            bsr_col_ind,
                                           rocsparse_int                   block_dim,
                                           rocsparse_mat_info              info,
                                           rocsparse_analysis_policy       analysis,
                                           rocsparse_solve_policy          solve,
                                           void*                           temp_buffer);
/**@}*/

/*! \ingroup level3_module
*  \details
*  \p rocsparse_bsrsm_clear deallocates all memory that was allocated by
*  \ref rocsparse_sbsrsm_analysis "rocsparse_Xbsrsm_analysis()". This is especially useful,
*  if memory is an issue and the analysis data is not required for further computation, e.g.
*  when switching to another sparse matrix format. Calling \p rocsparse_bsrsm_clear is optional.
*  All allocated resources will be cleared when the opaque \ref rocsparse_mat_info struct
*  is destroyed using \ref rocsparse_destroy_mat_info().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[inout]
*  info        structure that holds the information collected during the analysis step.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
*  \retval     rocsparse_status_memory_error the buffer holding the meta data could not
*              be deallocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsrsm_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsm_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in BSR storage format, a column-oriented dense solution matrix
*  \f$X\f$ and the column-oriented dense right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$,
*  such that
*  \f[
*    op(A) \cdot op(X) = \alpha \cdot op(B),
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
*  ,
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_X == rocsparse_operation_none} \\
*        B^T, & \text{if trans_X == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if trans_X == rocsparse_operation_none} \\
*        X^T, & \text{if trans_X == rocsparse_operation_transpose} \\
*        X^H, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and where \f$m = block\_dim \times mb\f$.
*
*  Note that as indicated above, the operation type of both \f$op(B)\f$ and \f$op(X)\f$ is specified by the
*  \p trans_X parameter and that the operation type of B and X must match. For example, if \f$op(B)=B\f$ then
*  \f$op(X)=X\f$. Likewise, if \f$op(B)=B^T\f$ then \f$op(X)=X^T\f$.
*
*  Given that the sparse matrix A is a square matrix, its size is \f$m \times m\f$ regardless of
*  whether A is transposed or not. The size of the column-oriented dense matrices B and X have
*  size that depends on the value of \p trans_X:
*
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        ldb \times nrhs, \text{  } ldb \ge m, & \text{if trans_X == rocsparse_operation_none} \\
*        ldb \times m, \text{  } ldb \ge nrhs,  & \text{if trans_X == rocsparse_operation_transpose} \\
*        ldb \times m, \text{  } ldb \ge nrhs, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        ldb \times nrhs, \text{  } ldb \ge m, & \text{if trans_X == rocsparse_operation_none} \\
*        ldb \times m, \text{  } ldb \ge nrhs,  & \text{if trans_X == rocsparse_operation_transpose} \\
*        ldb \times m, \text{  } ldb \ge nrhs, & \text{if trans_X == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \p rocsparse_bsrsm_solve requires a user allocated temporary buffer. Its size is returned by
*  \ref rocsparse_sbsrsm_buffer_size "rocsparse_Xbsrsm_buffer_size()". The size of the required buffer is larger
*  when  \p trans_A equals \ref rocsparse_operation_transpose or \ref rocsparse_operation_conjugate_transpose and
*  when \p trans_X is \ref rocsparse_operation_none. The subsequent solve will also be faster when \f$A\f$ is
*  non-transposed and \f$B\f$ is transposed (or conjugate transposed). For example, instead of solving:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       a_{00} & a_{01} \\
*       a_{10} & a_{11}
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       a_{20} & a_{21} \\
*       a_{30} & a_{31}
*      \end{array} &
*      \begin{array}{c c}
*       a_{22} & a_{23} \\
*       a_{32} & a_{33}
*      \end{array} \\
*    \end{array}
*   \right]
*    \cdot
*    \begin{bmatrix}
*    x_{00} & x_{01} \\
*    x_{10} & x_{11} \\
*    x_{20} & x_{21} \\
*    x_{30} & x_{31} \\
*    \end{bmatrix}
*    =
*    \begin{bmatrix}
*    b_{00} & b_{01} \\
*    b_{10} & b_{11} \\
*    b_{20} & b_{21} \\
*    b_{30} & b_{31} \\
*    \end{bmatrix}
*  \f]
*
*  Consider solving:
*
*  \f[
*   \left[
*    \begin{array}{c | c}
*      \begin{array}{c c}
*       a_{00} & a_{01} \\
*       a_{10} & a_{11}
*      \end{array} &
*      \begin{array}{c c}
*       0 & 0 \\
*       0 & 0
*      \end{array} \\
*    \hline
*      \begin{array}{c c}
*       a_{20} & a_{21} \\
*       a_{30} & a_{31}
*      \end{array} &
*      \begin{array}{c c}
*       a_{22} & a_{23} \\
*       a_{32} & a_{33}
*      \end{array} \\
*    \end{array}
*   \right]
*    \cdot
*    \begin{bmatrix}
*    x_{00} & x_{10} & x_{20} & x_{30} \\
*    x_{01} & x_{11} & x_{21} & x_{31}
*    \end{bmatrix}^{T}
*    =
*    \begin{bmatrix}
*    b_{00} & b_{10} & b_{20} & b_{30} \\
*    b_{01} & b_{11} & b_{21} & b_{31}
*    \end{bmatrix}^{T}
*  \f]
*
*  Once the temporary storage buffer has been allocated, analysis meta data is required. It can be obtained
*  by rocsparse_sbsrsm_analysis "rocsparse_Xbsrsm_analysis()".
*
*  Solving a triangular system involves inverting the diagonal blocks. This means that if the sparse matrix is
*  missing the diagonal block (referred to as a structural zero) or the diagonal block is not invertible (referred
*  to as a numerical zero) then a solution is not possible. \p rocsparse_bsrsm_solve tracks the location of the first
*  zero pivot (either numerical or structural zero). The zero pivot status can be checked calling \ref rocsparse_bsrsm_zero_pivot().
*  If \ref rocsparse_bsrsm_zero_pivot() returns \ref rocsparse_status_success, then no zero pivot was found and therefore
*  the matrix does not have a structural or numerical zero.
*
*  The user can specify that the sparse matrix should be interpreted as having identity blocks on the diagonal by setting the diagonal
*  type on the descriptor \p descr to \ref rocsparse_diag_type_unit using \ref rocsparse_set_mat_diag_type. If
*  \ref rocsparse_diag_type == \ref rocsparse_diag_type_unit, no zero pivot will be reported, even if the diagonal block \f$A_{j,j}\f$
*  for some \f$j\f$ is not invertible.
*
*  The sparse CSR matrix passed to \p rocsparse_bsrsm_solve does not actually have to be a triangular matrix. Instead the
*  triangular upper or lower part of the sparse matrix is solved based on \ref rocsparse_fill_mode set on the descriptor
*  \p descr. If the fill mode is set to \ref rocsparse_fill_mode_lower, then the lower triangular matrix is solved. If the
*  fill mode is set to \ref rocsparse_fill_mode_upper then the upper triangular matrix is solved.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  Operation type of B and X must match, for example if \f$op(B)=B\f$ then \f$op(X)=X\f$.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans_A != \ref rocsparse_operation_conjugate_transpose and
*  \p trans_X != \ref rocsparse_operation_conjugate_transpose is supported.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_X     matrix X operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix A.
*  @param[in]
*  nrhs        number of columns of the column-oriented dense matrix op(X).
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix A.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix A.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  block_dim   block dimension of the sparse BSR matrix.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  B           column-oriented dense matrix B with leading dimension \p ldb.
*  @param[in]
*  ldb         leading dimension of rhs matrix B.
*  @param[out]
*  X           column-oriented dense solution matrix X with leading dimension \p ldx.
*  @param[in]
*  ldx         leading dimension of solution matrix X.
*  @param[in]
*  policy      \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nrhs, \p nnzb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p alpha, \p descr, \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p B, \p X \p info or \p temp_buffer pointer
*              is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_X == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  Consider the lower triangular \f$m \times m\f$ matrix \f$L\f$, stored in BSR
*  storage format with non-unit diagonal. The following example solves \f$L \cdot X = B\f$.
*  \snippet example_rocsparse_bsrsm.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsm_solve(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_X,
                                        rocsparse_int             mb,
                                        rocsparse_int             nrhs,
                                        rocsparse_int             nnzb,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             block_dim,
                                        rocsparse_mat_info        info,
                                        const float*              B,
                                        rocsparse_int             ldb,
                                        float*                    X,
                                        rocsparse_int             ldx,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsm_solve(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_X,
                                        rocsparse_int             mb,
                                        rocsparse_int             nrhs,
                                        rocsparse_int             nnzb,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             block_dim,
                                        rocsparse_mat_info        info,
                                        const double*             B,
                                        rocsparse_int             ldb,
                                        double*                   X,
                                        rocsparse_int             ldx,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsm_solve(rocsparse_handle               handle,
                                        rocsparse_direction            dir,
                                        rocsparse_operation            trans_A,
                                        rocsparse_operation            trans_X,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nrhs,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* bsr_val,
                                        const rocsparse_int*           bsr_row_ptr,
                                        const rocsparse_int*           bsr_col_ind,
                                        rocsparse_int                  block_dim,
                                        rocsparse_mat_info             info,
                                        const rocsparse_float_complex* B,
                                        rocsparse_int                  ldb,
                                        rocsparse_float_complex*       X,
                                        rocsparse_int                  ldx,
                                        rocsparse_solve_policy         policy,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsm_solve(rocsparse_handle                handle,
                                        rocsparse_direction             dir,
                                        rocsparse_operation             trans_A,
                                        rocsparse_operation             trans_X,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nrhs,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* bsr_val,
                                        const rocsparse_int*            bsr_row_ptr,
                                        const rocsparse_int*            bsr_col_ind,
                                        rocsparse_int                   block_dim,
                                        rocsparse_mat_info              info,
                                        const rocsparse_double_complex* B,
                                        rocsparse_int                   ldb,
                                        rocsparse_double_complex*       X,
                                        rocsparse_int                   ldx,
                                        rocsparse_solve_policy          policy,
                                        void*                           temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_BSRSM_H */
