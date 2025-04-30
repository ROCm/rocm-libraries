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

#ifndef ROCSPARSE_GEMVI_H
#define ROCSPARSE_GEMVI_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
 *  \details
 *  \p rocsparse_gemvi_buffer_size returns the size of the temporary storage buffer
 *  required by \ref rocsparse_sgemvi "rocsparse_Xgemvi()". The temporary storage 
 *  buffer must be allocated by the user.
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
 *  m           number of rows of the dense matrix.
 *  @param[in]
 *  n           number of columns of the dense matrix.
 *  @param[in]
 *  nnz         number of non-zero entries in the sparse vector.
 *  @param[out]
 *  buffer_size temporary storage buffer size.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n, or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgemvi_buffer_size(rocsparse_handle    handle,
                                              rocsparse_operation trans,
                                              rocsparse_int       m,
                                              rocsparse_int       n,
                                              rocsparse_int       nnz,
                                              size_t*             buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgemvi_buffer_size(rocsparse_handle    handle,
                                              rocsparse_operation trans,
                                              rocsparse_int       m,
                                              rocsparse_int       n,
                                              rocsparse_int       nnz,
                                              size_t*             buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgemvi_buffer_size(rocsparse_handle    handle,
                                              rocsparse_operation trans,
                                              rocsparse_int       m,
                                              rocsparse_int       n,
                                              rocsparse_int       nnz,
                                              size_t*             buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgemvi_buffer_size(rocsparse_handle    handle,
                                              rocsparse_operation trans,
                                              rocsparse_int       m,
                                              rocsparse_int       n,
                                              rocsparse_int       nnz,
                                              size_t*             buffer_size);
/**@}*/

/*! \ingroup level2_module
 *  \brief Dense matrix sparse vector multiplication
 *
 *  \details
 *  \p rocsparse_gemvi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times n\f$
 *  matrix \f$A\f$ and the sparse vector \f$x\f$ and adds the result to the dense vector
 *  \f$y\f$ that is multiplied by the scalar \f$\beta\f$, such that
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
 *  Performing the above operation involves two steps. First, the user calls 
 *  \ref rocsparse_sgemvi_buffer_size "rocsparse_Xgemvi_buffer_size()" in order to determine the size of 
 *  the temporary storage buffer. Next, the user allocates this temporary buffer and passes it to 
 *  \p rocsparse_gemvi to complete the computation. Once all calls to \p rocsparse_gemvi are complete the
 *  temporary storage buffer can be freed.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the dense matrix.
 *  @param[in]
 *  n           number of columns of the dense matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  A           pointer to the dense matrix.
 *  @param[in]
 *  lda         leading dimension of the dense matrix
 *  @param[in]
 *  nnz         number of non-zero entries in the sparse vector
 *  @param[in]
 *  x_val       array of \p nnz elements containing the values of the sparse vector
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the sparse vector
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  idx_base    rocsparse_index_base_zero or rocsparse_index_base_one.
 *  @param[in]
 *  temp_buffer temporary storage buffer
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n, \p lda or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p alpha, \p A, \p x_val, \p x_ind,
 *              \p beta, \p y or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  \code{.c}
 *      // rocSPARSE handle
 *      rocsparse_handle handle;
 *      rocsparse_create_handle(&handle);
 *
 *      rocsparse_mat_descr descr;
 *      rocsparse_create_mat_descr(&descr);
 *
 *      // Number of rows and columns
 *      rocsparse_int m = 3;
 *      rocsparse_int n = 5;
 *
 *      // Dense matrix A in column-major
 *      rocsparse_int lda    = m;
 *      double        hA[15] = {9, 14, 19, 10, 15, 20, 11, 16, 21, 12, 17, 22, 13, 18, 23};
 *
 *      // Number of non-zero entries
 *      rocsparse_int nnz = 3;
 *
 *      // Vector x indices
 *      rocsparse_int hx_ind[3] = {0, 1, 3};
 *
 *      // Vector x values
 *      double hx_val[3] = {1, 2, 3};
 *
 *      // Vector y values
 *      double hy[3] = {4, 5, 6};
 *
 *      // Scalar alpha and beta
 *      double alpha = 3.7;
 *      double beta  = 1.3;
 *
 *      // Matrix operation
 *      rocsparse_operation trans = rocsparse_operation_none;
 *
 *      // Index base
 *      rocsparse_index_base base = rocsparse_index_base_zero;
 *
 *      // Offload data to device
 *      double*        dA;
 *      rocsparse_int* dx_ind;
 *      double*        dx_val;
 *      double*        dy;
 *      hipMalloc((void**)&dA, sizeof(double) * m * n);
 *      hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz);
 *      hipMalloc((void**)&dx_val, sizeof(double) * nnz);
 *      hipMalloc((void**)&dy, sizeof(double) * m);
 *
 *      hipMemcpy(dA, hA, sizeof(double) * m * n, hipMemcpyHostToDevice);
 *      hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
 *      hipMemcpy(dx_val, hx_val, sizeof(double) * nnz, hipMemcpyHostToDevice);
 *      hipMemcpy(dy, hy, sizeof(double) * m, hipMemcpyHostToDevice);
 *
 *      // Obtain buffer size
 *      size_t buffer_size;
 *      rocsparse_dgemvi_buffer_size(handle, trans, m, n, nnz, &buffer_size);
 *
 *      void* buffer;
 *      hipMalloc(&buffer, buffer_size);
 *
 *      // Call dgemvi
 *      rocsparse_dgemvi(handle, trans, m, n, &alpha, dA, lda, nnz, dx_val, dx_ind, &beta, dy, base, buffer);
 *
 *      // Copy result back to host
 *      hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost);
 *
 *      // Clear rocSPARSE
 *      rocsparse_destroy_handle(handle);
 *
 *      // Clear device memory
 *      hipFree(dA);
 *      hipFree(dx_ind);
 *      hipFree(dx_val);
 *      hipFree(dy);
 *      hipFree(buffer);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgemvi(rocsparse_handle     handle,
                                  rocsparse_operation  trans,
                                  rocsparse_int        m,
                                  rocsparse_int        n,
                                  const float*         alpha,
                                  const float*         A,
                                  rocsparse_int        lda,
                                  rocsparse_int        nnz,
                                  const float*         x_val,
                                  const rocsparse_int* x_ind,
                                  const float*         beta,
                                  float*               y,
                                  rocsparse_index_base idx_base,
                                  void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgemvi(rocsparse_handle     handle,
                                  rocsparse_operation  trans,
                                  rocsparse_int        m,
                                  rocsparse_int        n,
                                  const double*        alpha,
                                  const double*        A,
                                  rocsparse_int        lda,
                                  rocsparse_int        nnz,
                                  const double*        x_val,
                                  const rocsparse_int* x_ind,
                                  const double*        beta,
                                  double*              y,
                                  rocsparse_index_base idx_base,
                                  void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgemvi(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_float_complex* A,
                                  rocsparse_int                  lda,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* x_val,
                                  const rocsparse_int*           x_ind,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y,
                                  rocsparse_index_base           idx_base,
                                  void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgemvi(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_double_complex* A,
                                  rocsparse_int                   lda,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* x_val,
                                  const rocsparse_int*            x_ind,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y,
                                  rocsparse_index_base            idx_base,
                                  void*                           temp_buffer);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GEMVI_H */
