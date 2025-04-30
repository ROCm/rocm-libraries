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

#ifndef ROCSPARSE_ELLMV_H
#define ROCSPARSE_ELLMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using ELL storage format
*
*  \details
*  \p rocsparse_ellmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in ELL storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
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
*  \code{.c}
*      for(i = 0; i < m; ++i)
*      {
*          y[i] = beta * y[i];
*
*          for(p = 0; p < ell_width; ++p)
*          {
*              idx = p * m + i;
*
*              if((ell_col_ind[idx] >= 0) && (ell_col_ind[idx] < n))
*              {
*                  y[i] = y[i] + alpha * ell_val[idx] * x[ell_col_ind[idx]];
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
*  trans       matrix operation type.
*  @param[in]
*  m           number of rows of the sparse ELL matrix.
*  @param[in]
*  n           number of columns of the sparse ELL matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse ELL matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  ell_val     array that contains the elements of the sparse ELL matrix. Padded
*              elements should be zero.
*  @param[in]
*  ell_col_ind array that contains the column indices of the sparse ELL matrix.
*              Padded column indices should be -1.
*  @param[in]
*  ell_width   number of non-zero elements per row of the sparse ELL matrix.
*  @param[in]
*  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
*              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
*              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p ell_val,
*              \p ell_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example performs a sparse matrix vector multiplication in ELL format. It also shows how to convert
*  from CSR to ELL format.
*  \code{.c}
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // A sparse matrix
*      // 1 0 3 4
*      // 0 0 5 1
*      // 0 2 0 0
*      // 4 0 0 8
*      rocsparse_int hAptr[5] = {0, 3, 5, 6, 8};
*      rocsparse_int hAcol[8] = {0, 2, 3, 2, 3, 1, 0, 3};
*      double        hAval[8] = {1.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0};
*
*      rocsparse_int m = 4;
*      rocsparse_int n = 4;
*      rocsparse_int nnz = 8;
*
*      double halpha = 1.0;
*      double hbeta  = 0.0;
*
*      double  hx[4] = {1.0, 2.0, 3.0, 4.0};
*      double  hy[4] = {4.0, 5.0, 6.0, 7.0};
*
*      // Matrix descriptors
*      rocsparse_mat_descr descrA;
*      rocsparse_create_mat_descr(&descrA);
*
*      rocsparse_mat_descr descrB;
*      rocsparse_create_mat_descr(&descrB);
*
*      // Offload data to device
*      rocsparse_int* dAptr = NULL;
*      rocsparse_int* dAcol = NULL;
*      double*        dAval = NULL;
*      double*        dx    = NULL;
*      double*        dy    = NULL;
*
*      hipMalloc((void**)&dAptr, sizeof(rocsparse_int) * (m + 1));
*      hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz);
*      hipMalloc((void**)&dAval, sizeof(double) * nnz);
*      hipMalloc((void**)&dx, sizeof(double) * n);
*      hipMalloc((void**)&dy, sizeof(double) * m);
*
*      hipMemcpy(dAptr, hAptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dAcol, hAcol, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dx, hx, sizeof(double) * n, hipMemcpyHostToDevice);
*
*      // Convert CSR matrix to ELL format
*      rocsparse_int* dBcol = NULL;
*      double*        dBval = NULL;
*
*      // Determine ELL width
*      rocsparse_int ell_width;
*      rocsparse_csr2ell_width(handle, m, descrA, dAptr, descrB, &ell_width);
*
*      // Allocate memory for ELL storage format
*      hipMalloc((void**)&dBcol, sizeof(rocsparse_int) * ell_width * m);
*      hipMalloc((void**)&dBval, sizeof(double) * ell_width * m);
*
*      // Convert matrix from CSR to ELL
*      rocsparse_dcsr2ell(handle, m, descrA, dAval, dAptr, dAcol, descrB, ell_width, dBval, dBcol);
*
*      // Clean up CSR structures
*      hipFree(dAptr);
*      hipFree(dAcol);
*      hipFree(dAval);
*
*      // Call rocsparse ellmv
*      rocsparse_dellmv(handle,
*                       rocsparse_operation_none,
*                       m,
*                       n,
*                       &halpha,
*                       descrB,
*                       dBval,
*                       dBcol,
*                       ell_width,
*                       dx,
*                       &hbeta,
*                       dy);
*
*      // Copy result back to host
*      hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost);
*
*      // Clear up on device
*      rocsparse_destroy_mat_descr(descrA);
*      rocsparse_destroy_mat_descr(descrB);
*      rocsparse_destroy_handle(handle);
*
*      hipFree(dBcol);
*      hipFree(dBval);
*      hipFree(dx);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sellmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              ell_val,
                                  const rocsparse_int*      ell_col_ind,
                                  rocsparse_int             ell_width,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dellmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             ell_val,
                                  const rocsparse_int*      ell_col_ind,
                                  rocsparse_int             ell_width,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cellmv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* ell_val,
                                  const rocsparse_int*           ell_col_ind,
                                  rocsparse_int                  ell_width,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zellmv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* ell_val,
                                  const rocsparse_int*            ell_col_ind,
                                  rocsparse_int                   ell_width,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_ELLMV_H */
