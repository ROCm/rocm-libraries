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

#ifndef ROCSPARSE_GEBSRMV_H
#define ROCSPARSE_GEBSRMV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using GEBSR storage format
*
*  \details
*  \p rocsparse_gebsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$m \times n\f$ matrix, defined in GEBSR storage format, and the dense vector 
*  \f$x\f$ and adds the result to the dense vector \f$y\f$ that is multiplied by 
*  the scalar \f$\beta\f$, such that
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
*  and where \f$m = mb \times row\_block\_dim\f$ and \f$n = nb \times col\_block\_dim\f$.
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
*  dir         matrix storage of GEBSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse GEBSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse GEBSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse GEBSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse GEBSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse GEBSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse GEBSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              GEBSR matrix.
*  @param[in]
*  row_block_dim row block dimension of the sparse GEBSR matrix.
*  @param[in]
*  col_block_dim column block dimension of the sparse GEBSR matrix.
*  @param[in]
*  x           array of \p nb*col_block_dim elements (\f$op(A) = A\f$) or \p mb*row_block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*row_block_dim elements (\f$op(A) = A\f$) or \p nb*col_block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb, \p nnzb, \p row_block_dim
*              or \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example performs a sparse matrix vector multiplication in GEBSR format.
*  \code{.c}
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // alpha * ( 1.0  0.0  2.0 ) * ( 1.0 ) + beta * ( 4.0 ) = (  31.1 )
*      //         ( 3.0  0.0  4.0 ) * ( 2.0 )          ( 5.0 ) = (  62.0 )
*      //         ( 5.0  6.0  0.0 ) * ( 3.0 )          ( 6.0 ) = (  70.7 )
*      //         ( 7.0  0.0  8.0 ) *                  ( 7.0 ) = ( 123.8 )
*
*      // GEBSR block dimensions
*      rocsparse_int row_block_dim = 2;
*      rocsparse_int col_block_dim = 3;
*
*      // Number of block rows and columns
*      rocsparse_int mb = 2;
*      rocsparse_int nb = 1;
*
*      // Number of non-zero blocks
*      rocsparse_int nnzb = 2;
*
*      // BSR row pointers
*      rocsparse_int hbsr_row_ptr[3] = {0, 1, 2};
*
*      // BSR column indices
*      rocsparse_int hbsr_col_ind[2] = {0, 0};
*
*      // BSR values
*      double hbsr_val[16] = {1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0};
*
*      // Block storage in column major
*      rocsparse_direction dir = rocsparse_direction_column;
*
*      // Transposition of the matrix
*      rocsparse_operation trans = rocsparse_operation_none;
*
*      // Scalar alpha and beta
*      double alpha = 3.7;
*      double beta  = 1.3;
*
*      // x and y
*      double hx[4] = {1.0, 2.0, 3.0, 0.0};
*      double hy[4] = {4.0, 5.0, 6.0, 7.0};
*
*      // Matrix descriptor
*      rocsparse_mat_descr descr;
*      rocsparse_create_mat_descr(&descr);
*
*      // Offload data to device
*      rocsparse_int* dbsr_row_ptr;
*      rocsparse_int* dbsr_col_ind;
*      double*        dbsr_val;
*      double*        dx;
*      double*        dy;
*
*      hipMalloc((void**)&dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1));
*      hipMalloc((void**)&dbsr_col_ind, sizeof(rocsparse_int) * nnzb);
*      hipMalloc((void**)&dbsr_val, sizeof(double) * nnzb * row_block_dim * col_block_dim);
*      hipMalloc((void**)&dx, sizeof(double) * nb * col_block_dim);
*      hipMalloc((void**)&dy, sizeof(double) * mb * row_block_dim);
*
*      hipMemcpy(dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice);
*      hipMemcpy(dbsr_val, hbsr_val, sizeof(double) * nnzb * row_block_dim * col_block_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dx, hx, sizeof(double) * nb * col_block_dim, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(double) * mb * row_block_dim, hipMemcpyHostToDevice);
*
*      // Call dbsrmv to perform y = alpha * A x + beta * y
*      rocsparse_dgebsrmv(handle,
*                         dir,
*                         trans,
*                         mb,
*                         nb,
*                         nnzb,
*                         &alpha,
*                         descr,
*                         dbsr_val,
*                         dbsr_row_ptr,
*                         dbsr_col_ind,
*                         row_block_dim,
*                         col_block_dim,
*                         dx,
*                         &beta,
*                         dy);
*
*      // Copy result back to host
*      hipMemcpy(hy, dy, sizeof(double) * mb * row_block_dim, hipMemcpyDeviceToHost);
*
*      // Clear rocSPARSE
*      rocsparse_destroy_mat_descr(descr);
*      rocsparse_destroy_handle(handle);
*
*      // Clear device memory
*      hipFree(dbsr_row_ptr);
*      hipFree(dbsr_col_ind);
*      hipFree(dbsr_val);
*      hipFree(dx);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsrmv(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             nnzb,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr,
                                    const float*              bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             row_block_dim,
                                    rocsparse_int             col_block_dim,
                                    const float*              x,
                                    const float*              beta,
                                    float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsrmv(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             nnzb,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr,
                                    const double*             bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             row_block_dim,
                                    rocsparse_int             col_block_dim,
                                    const double*             x,
                                    const double*             beta,
                                    double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsrmv(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_operation            trans,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  nb,
                                    rocsparse_int                  nnzb,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr,
                                    const rocsparse_float_complex* bsr_val,
                                    const rocsparse_int*           bsr_row_ptr,
                                    const rocsparse_int*           bsr_col_ind,
                                    rocsparse_int                  row_block_dim,
                                    rocsparse_int                  col_block_dim,
                                    const rocsparse_float_complex* x,
                                    const rocsparse_float_complex* beta,
                                    rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsrmv(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_operation             trans,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   nb,
                                    rocsparse_int                   nnzb,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr,
                                    const rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*            bsr_row_ptr,
                                    const rocsparse_int*            bsr_col_ind,
                                    rocsparse_int                   row_block_dim,
                                    rocsparse_int                   col_block_dim,
                                    const rocsparse_double_complex* x,
                                    const rocsparse_double_complex* beta,
                                    rocsparse_double_complex*       y);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GEBSRMV_H */
