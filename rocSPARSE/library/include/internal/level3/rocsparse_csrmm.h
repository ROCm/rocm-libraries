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

#ifndef ROCSPARSE_CSRMM_H
#define ROCSPARSE_CSRMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level3_module
*  \brief Sparse matrix dense matrix multiplication using CSR storage format
*
*  \details
*  \p rocsparse_csrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
*  matrix \f$A\f$, defined in CSR storage format, and the column-oriented dense \f$k \times n\f$
*  matrix \f$B\f$ and adds the result to the column-oriented dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
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
*
*  \code{.c}
*      for(i = 0; i < ldc; ++i)
*      {
*          for(j = 0; j < n; ++j)
*          {
*              C[i][j] = beta * C[i][j];
*
*              for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
*              {
*                  C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
*              }
*          }
*      }
*  \endcode
*
*  \note
*  This function does not produce deterministic results when A is transposed.
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
*  m           number of rows of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  n           number of columns of the column-oriented dense matrix \f$op(B)\f$ and \f$C\f$.
*  @param[in]
*  k           number of columns of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix \f$A\f$. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix \f$A\f$.
*  @param[in]
*  B           column-oriented dense matrix of dimension \f$ldb \times n\f$ (\f$op(B) == B\f$),
*              \f$ldb \times k\f$ otherwise.
*  @param[in]
*  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$
*              (\f$op(B) == B\f$), \f$\max{(1, n)}\f$ otherwise.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  C           column-oriented dense matrix of dimension \f$ldc \times n\f$.
*  @param[in]
*  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$
*              (\f$op(A) == A\f$), \f$\max{(1, k)}\f$ otherwise.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz, \p ldb or \p ldc
*              is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p B, \p beta or \p C pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example multiplies a CSR matrix with a column-oriented dense matrix.
*  \code{.c}
*      //     1 2 0 3 0
*      // A = 0 4 5 0 0
*      //     6 0 0 7 8
*
*      rocsparse_int m   = 3;
*      rocsparse_int k   = 5;
*      rocsparse_int nnz = 8;
*
*      csr_row_ptr[m+1] = {0, 3, 5, 8};             // device memory
*      csr_col_ind[nnz] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
*      csr_val[nnz]     = {1, 2, 3, 4, 5, 6, 7, 8}; // device memory
*
*      // Set dimension n of B
*      rocsparse_int n = 64;
*
*      // Allocate and generate column-oriented dense matrix B
*      std::vector<float> hB(k * n);
*      for(rocsparse_int i = 0; i < k * n; ++i)
*      {
*          hB[i] = static_cast<float>(rand()) / RAND_MAX;
*      }
*
*      // Copy B to the device
*      float* B;
*      hipMalloc((void**)&B, sizeof(float) * k * n);
*      hipMemcpy(B, hB.data(), sizeof(float) * k * n, hipMemcpyHostToDevice);
*
*      // alpha and beta
*      float alpha = 1.0f;
*      float beta  = 0.0f;
*
*      // Allocate memory for the resulting matrix C
*      float* C;
*      hipMalloc((void**)&C, sizeof(float) * m * n);
*
*      // Perform the matrix multiplication
*      rocsparse_scsrmm(handle,
*                       rocsparse_operation_none,
*                       rocsparse_operation_none,
*                       m,
*                       n,
*                       k,
*                       nnz,
*                       &alpha,
*                       descr,
*                       csr_val,
*                       csr_row_ptr,
*                       csr_col_ind,
*                       B,
*                       k,
*                       &beta,
*                       C,
*                       m);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmm(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const float*              B,
                                  rocsparse_int             ldb,
                                  const float*              beta,
                                  float*                    C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmm(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const double*             B,
                                  rocsparse_int             ldb,
                                  const double*             beta,
                                  double*                   C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmm(rocsparse_handle               handle,
                                  rocsparse_operation            trans_A,
                                  rocsparse_operation            trans_B,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  k,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int*           csr_row_ptr,
                                  const rocsparse_int*           csr_col_ind,
                                  const rocsparse_float_complex* B,
                                  rocsparse_int                  ldb,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       C,
                                  rocsparse_int                  ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmm(rocsparse_handle                handle,
                                  rocsparse_operation             trans_A,
                                  rocsparse_operation             trans_B,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   k,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int*            csr_row_ptr,
                                  const rocsparse_int*            csr_col_ind,
                                  const rocsparse_double_complex* B,
                                  rocsparse_int                   ldb,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       C,
                                  rocsparse_int                   ldc);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRMM_H */
