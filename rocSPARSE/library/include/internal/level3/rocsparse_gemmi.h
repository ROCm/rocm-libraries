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

#ifndef ROCSPARSE_GEMMI_H
#define ROCSPARSE_GEMMI_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level3_module
*  \brief Dense matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p rocsparse_gemmi multiplies the scalar \f$\alpha\f$ with a column-oriented dense \f$m \times k\f$
*  matrix \f$op(A)\f$ and the sparse \f$k \times n\f$ matrix \f$op(B)\f$, defined in CSR
*  storage format and adds the result to the column-oriented dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
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
*  \note
*  Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*
*  \note
*  Currently, only \p trans_B == \ref rocsparse_operation_transpose is supported.
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
*  trans_A     matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B     matrix \f$B\f$ operation type.
*  @param[in]
*  m           number of rows of the column-oriented dense matrix \f$A\f$.
*  @param[in]
*  n           number of columns of the sparse CSR matrix \f$op(B)\f$ and \f$C\f$.
*  @param[in]
*  k           number of columns of the column-oriented dense matrix \f$A\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  A           array of dimension \f$lda \times k\f$ (\f$op(A) == A\f$) or
*              \f$lda \times m\f$ (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*  @param[in]
*  lda         leading dimension of \f$A\f$, must be at least \f$m\f$
*              (\f$op(A) == A\f$) or \f$k\f$ (\f$op(A) == A^T\f$ or
*              \f$op(A) == A^H\f$).
*  @param[in]
*  descr       descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse CSR
*              matrix \f$B\f$.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  C           column-oriented dense matrix of dimension \f$ldc \times n\f$ that holds the values of \f$C\f$.
*  @param[in]
*  ldc         leading dimension of \f$C\f$, must be at least \f$m\f$.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz, \p lda or \p ldc
*              is invalid.
*  \retval     rocsparse_status_invalid_pointer \p alpha, \p A, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p beta or \p C pointer is invalid.
*
*  \par Example
*  This example multiplies a column-oriented dense matrix with a CSC matrix.
*  \code{.c}
*      rocsparse_int m   = 2;
*      rocsparse_int n   = 5;
*      rocsparse_int k   = 3;
*      rocsparse_int nnz = 8;
*      rocsparse_int lda = m;
*      rocsparse_int ldc = m;
*
*      // Matrix A (m x k)
*      // (  9.0  10.0  11.0 )
*      // ( 12.0  13.0  14.0 )
*
*      // Matrix B (k x n)
*      // ( 1.0  2.0  0.0  3.0  0.0 )
*      // ( 0.0  4.0  5.0  0.0  0.0 )
*      // ( 6.0  0.0  0.0  7.0  8.0 )
*
*      // Matrix C (m x n)
*      // ( 15.0  16.0  17.0  18.0  19.0 )
*      // ( 20.0  21.0  22.0  23.0  24.0 )
*
*      A[lda * k]           = {9.0, 12.0, 10.0, 13.0, 11.0, 14.0};      // device memory
*      csc_col_ptr_B[n + 1] = {0, 2, 4, 5, 7, 8};                       // device memory
*      csc_row_ind_B[nnz]   = {0, 0, 1, 1, 2, 3, 3, 4};                 // device memory
*      csc_val_B[nnz]       = {1.0, 6.0, 2.0, 4.0, 5.0, 3.0, 7.0, 8.0}; // device memory
*      C[ldc * n]           = {15.0, 20.0, 16.0, 21.0, 17.0, 22.0,      // device memory
*                              18.0, 23.0, 19.0, 24.0};
*
*      // alpha and beta
*      float alpha = 1.0f;
*      float beta  = 0.0f;
*
*      // Perform the matrix multiplication
*      rocsparse_sgemmi(handle,
*                       rocsparse_operation_none,
*                       rocsparse_operation_transpose,
*                       m,
*                       n,
*                       k,
*                       nnz,
*                       &alpha,
*                       A,
*                       lda,
*                       descr_B,
*                       csc_val_B,
*                       csc_col_ptr_B,
*                       csc_row_ind_B,
*                       &beta,
*                       C,
*                       ldc);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgemmi(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const float*              A,
                                  rocsparse_int             lda,
                                  const rocsparse_mat_descr descr,
                                  const float*              csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const float*              beta,
                                  float*                    C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgemmi(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const double*             A,
                                  rocsparse_int             lda,
                                  const rocsparse_mat_descr descr,
                                  const double*             csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const double*             beta,
                                  double*                   C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgemmi(rocsparse_handle               handle,
                                  rocsparse_operation            trans_A,
                                  rocsparse_operation            trans_B,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  k,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_float_complex* A,
                                  rocsparse_int                  lda,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int*           csr_row_ptr,
                                  const rocsparse_int*           csr_col_ind,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       C,
                                  rocsparse_int                  ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgemmi(rocsparse_handle                handle,
                                  rocsparse_operation             trans_A,
                                  rocsparse_operation             trans_B,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   k,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_double_complex* A,
                                  rocsparse_int                   lda,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int*            csr_row_ptr,
                                  const rocsparse_int*            csr_col_ind,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       C,
                                  rocsparse_int                   ldc);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GEMMI_H */
