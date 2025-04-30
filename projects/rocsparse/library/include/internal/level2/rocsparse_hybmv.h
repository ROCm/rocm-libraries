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

#ifndef ROCSPARSE_HYBMV_H
#define ROCSPARSE_HYBMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using HYB storage format
*
*  \details
*  \p rocsparse_hybmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in HYB storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none}
*    \end{array}
*    \right.
*  \f]
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
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse HYB matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  hyb         matrix in HYB storage format.
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
*  \retval     rocsparse_status_invalid_size \p hyb structure was not initialized with
*              valid matrix sizes.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p hyb, \p x,
*              \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_invalid_value \p hyb structure was not initialized
*              with a valid partitioning type.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_memory_error the buffer could not be allocated.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example performs a sparse matrix vector multiplication in HYB format. Also
*  demonstrate conversion from CSR to HYB.
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
*      // Matrix descriptor
*      rocsparse_mat_descr descrA;
*      rocsparse_create_mat_descr(&descrA);
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
*      // Convert CSR matrix to HYB format
*      rocsparse_hyb_mat hybA;
*      rocsparse_create_hyb_mat(&hybA);
*
*      rocsparse_dcsr2hyb(handle, m, n, descrA, dAval, dAptr, dAcol, hybA, 0, rocsparse_hyb_partition_auto);
*
*      // Clean up CSR structures
*      hipFree(dAptr);
*      hipFree(dAcol);
*      hipFree(dAval);
*
*      // Call rocsparse hybmv
*      rocsparse_dhybmv(handle, rocsparse_operation_none, &halpha, descrA, hybA, dx, &hbeta, dy);
*
*      // Copy result back to host
*      hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost);
*
*      // Clear up on device
*      rocsparse_destroy_hyb_mat(hybA);
*      rocsparse_destroy_mat_descr(descrA);
*      rocsparse_destroy_handle(handle);
*
*      hipFree(dx);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_shybmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat   hyb,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dhybmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat   hyb,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_chybmv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_hyb_mat        hyb,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zhybmv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_hyb_mat         hyb,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_HYBMV_H */
