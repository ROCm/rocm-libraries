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

#ifndef ROCSPARSE_GTSV_H
#define ROCSPARSE_GTSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \details
*  \p rocsparse_gtsv_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_sgtsv "rocsparse_Xgtsv()". The temporary
*  storage buffer must be allocated by the user.
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
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n           number of columns in the dense matrix B.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[in]
*  B           Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb         Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sgtsv(), rocsparse_dgtsv(), rocsparse_cgtsv()
*              and rocsparse_zgtsv().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p ldb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d, \p du,
*              \p B or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv_buffer_size(rocsparse_handle handle,
                                             rocsparse_int    m,
                                             rocsparse_int    n,
                                             const float*     dl,
                                             const float*     d,
                                             const float*     du,
                                             const float*     B,
                                             rocsparse_int    ldb,
                                             size_t*          buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv_buffer_size(rocsparse_handle handle,
                                             rocsparse_int    m,
                                             rocsparse_int    n,
                                             const double*    dl,
                                             const double*    d,
                                             const double*    du,
                                             const double*    B,
                                             rocsparse_int    ldb,
                                             size_t*          buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgtsv_buffer_size(rocsparse_handle               handle,
                                             rocsparse_int                  m,
                                             rocsparse_int                  n,
                                             const rocsparse_float_complex* dl,
                                             const rocsparse_float_complex* d,
                                             const rocsparse_float_complex* du,
                                             const rocsparse_float_complex* B,
                                             rocsparse_int                  ldb,
                                             size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgtsv_buffer_size(rocsparse_handle                handle,
                                             rocsparse_int                   m,
                                             rocsparse_int                   n,
                                             const rocsparse_double_complex* dl,
                                             const rocsparse_double_complex* d,
                                             const rocsparse_double_complex* du,
                                             const rocsparse_double_complex* B,
                                             rocsparse_int                   ldb,
                                             size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver with pivoting
*
*  \details
*  \p rocsparse_gtsv solves a tridiagonal system for multiple right hand sides using pivoting
*  \f[
*    T*B = B
*  \f]
*  where \f$T\f$ is a sparse tridiagonal matrix and \f$B\f$ is a dense \f$ldb \times n\f$ matrix storing the
*  right-hand side vectors in column order. The tridiagonal matrix \f$T\f$ is defined by three vectors: \p dl
*  for the lower diagonal, \p d for the main diagonal and \p du for the upper diagonal.
*
*  Solving the tridiagonal system involves two steps. First, the user calls
*  \ref rocsparse_sgtsv_buffer_size "rocsparse_Xgtsv_buffer_size()" in order to determine the size of the required
*  temporary storage buffer. Once determined, the user allocates this buffer and passes it to
*  \ref rocsparse_sgtsv "rocsparse_Xgtsv()" to perform the actual solve. The \f$B\f$ dense matrix, which initially
*  stores the \p n right-hand side vectors, is overwritten with the \p n solution vectors after the call to
*  \ref rocsparse_sgtsv "rocsparse_Xgtsv()".
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
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n           number of columns in the dense matrix B.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[inout]
*  B           Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb         Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p ldb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d,
*              \p du, \p B or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  \snippet example_rocsparse_gtsv.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv(rocsparse_handle handle,
                                 rocsparse_int    m,
                                 rocsparse_int    n,
                                 const float*     dl,
                                 const float*     d,
                                 const float*     du,
                                 float*           B,
                                 rocsparse_int    ldb,
                                 void*            temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv(rocsparse_handle handle,
                                 rocsparse_int    m,
                                 rocsparse_int    n,
                                 const double*    dl,
                                 const double*    d,
                                 const double*    du,
                                 double*          B,
                                 rocsparse_int    ldb,
                                 void*            temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgtsv(rocsparse_handle               handle,
                                 rocsparse_int                  m,
                                 rocsparse_int                  n,
                                 const rocsparse_float_complex* dl,
                                 const rocsparse_float_complex* d,
                                 const rocsparse_float_complex* du,
                                 rocsparse_float_complex*       B,
                                 rocsparse_int                  ldb,
                                 void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgtsv(rocsparse_handle                handle,
                                 rocsparse_int                   m,
                                 rocsparse_int                   n,
                                 const rocsparse_double_complex* dl,
                                 const rocsparse_double_complex* d,
                                 const rocsparse_double_complex* du,
                                 rocsparse_double_complex*       B,
                                 rocsparse_int                   ldb,
                                 void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_gtsv_no_pivot_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_sgtsv_no_pivot "rocsparse_Xgtsv_no_pivot()". The temporary
*  storage buffer must be allocated by the user.
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
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n           number of columns in the dense matrix B.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[in]
*  B           Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb         Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sgtsv_no_pivot(), rocsparse_dgtsv_no_pivot(), rocsparse_cgtsv_no_pivot()
*              and rocsparse_zgtsv_no_pivot().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p ldb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d, \p du,
*              \p B or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv_no_pivot_buffer_size(rocsparse_handle handle,
                                                      rocsparse_int    m,
                                                      rocsparse_int    n,
                                                      const float*     dl,
                                                      const float*     d,
                                                      const float*     du,
                                                      const float*     B,
                                                      rocsparse_int    ldb,
                                                      size_t*          buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv_no_pivot_buffer_size(rocsparse_handle handle,
                                                      rocsparse_int    m,
                                                      rocsparse_int    n,
                                                      const double*    dl,
                                                      const double*    d,
                                                      const double*    du,
                                                      const double*    B,
                                                      rocsparse_int    ldb,
                                                      size_t*          buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgtsv_no_pivot_buffer_size(rocsparse_handle               handle,
                                                      rocsparse_int                  m,
                                                      rocsparse_int                  n,
                                                      const rocsparse_float_complex* dl,
                                                      const rocsparse_float_complex* d,
                                                      const rocsparse_float_complex* du,
                                                      const rocsparse_float_complex* B,
                                                      rocsparse_int                  ldb,
                                                      size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgtsv_no_pivot_buffer_size(rocsparse_handle                handle,
                                                      rocsparse_int                   m,
                                                      rocsparse_int                   n,
                                                      const rocsparse_double_complex* dl,
                                                      const rocsparse_double_complex* d,
                                                      const rocsparse_double_complex* du,
                                                      const rocsparse_double_complex* B,
                                                      rocsparse_int                   ldb,
                                                      size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver (no pivoting)
*
*  \details
*  \p rocsparse_gtsv_no_pivot solves a tridiagonal linear system for multiple right-hand sides without pivoting
*  \f[
*    T*B = B
*  \f]
*  where \f$T\f$ is a sparse tridiagonal matrix and \f$B\f$ is a dense \f$ldb \times n\f$ matrix storing the
*  right-hand side vectors in column order. The tridiagonal matrix \f$T\f$ is defined by three vectors: \p dl
*  for the lower diagonal, \p d for the main diagonal and \p du for the upper diagonal.
*
*  Solving the tridiagonal system with multiple right-hand sides without pivoting involves two steps. First,
*  the user calls \ref rocsparse_sgtsv_no_pivot_buffer_size "rocsparse_Xgtsv_no_pivot_buffer_size()" in order
*  to determine the size of the required temporary storage buffer. Once determined, the user allocates this
*  buffer and passes it to \ref rocsparse_sgtsv_no_pivot "rocsparse_Xgtsv_no_pivot()" to perform the actual
*  solve. The \f$B\f$ dense matrix, which initially stores the \p n right-hand side vectors, is overwritten
*  with the \p n solution vectors after the call to \ref rocsparse_sgtsv_no_pivot "rocsparse_Xgtsv_no_pivot()".
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
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  n           number of columns in the dense matrix B.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[inout]
*  B           Dense matrix of size ( \p ldb, \p n ).
*  @param[in]
*  ldb         Leading dimension of B. Must satisfy \p ldb >= max(1, m).
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n or \p ldb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d,
*              \p du, \p B or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  \snippet example_rocsparse_gtsv_no_pivot.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv_no_pivot(rocsparse_handle handle,
                                          rocsparse_int    m,
                                          rocsparse_int    n,
                                          const float*     dl,
                                          const float*     d,
                                          const float*     du,
                                          float*           B,
                                          rocsparse_int    ldb,
                                          void*            temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv_no_pivot(rocsparse_handle handle,
                                          rocsparse_int    m,
                                          rocsparse_int    n,
                                          const double*    dl,
                                          const double*    d,
                                          const double*    du,
                                          double*          B,
                                          rocsparse_int    ldb,
                                          void*            temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgtsv_no_pivot(rocsparse_handle               handle,
                                          rocsparse_int                  m,
                                          rocsparse_int                  n,
                                          const rocsparse_float_complex* dl,
                                          const rocsparse_float_complex* d,
                                          const rocsparse_float_complex* du,
                                          rocsparse_float_complex*       B,
                                          rocsparse_int                  ldb,
                                          void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgtsv_no_pivot(rocsparse_handle                handle,
                                          rocsparse_int                   m,
                                          rocsparse_int                   n,
                                          const rocsparse_double_complex* dl,
                                          const rocsparse_double_complex* d,
                                          const rocsparse_double_complex* du,
                                          rocsparse_double_complex*       B,
                                          rocsparse_int                   ldb,
                                          void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_gtsv_no_pivot_strided_batch_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_sgtsv_no_pivot_strided_batch "rocsparse_Xgtsv_no_pivot_strided_batch()".
*  The temporary storage buffer must be allocated by the user.
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
*  m           size of the tri-diagonal linear system.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system where the ith system lower diagonal starts at \p dl+batch_stride*i.
*  @param[in]
*  d           main diagonal of tri-diagonal system where the ith system diagonal starts at \p d+batch_stride*i.
*  @param[in]
*  du          upper diagonal of tri-diagonal system where the ith system upper diagonal starts at \p du+batch_stride*i.
*  @param[inout]
*  x           Dense array of righthand-sides where the ith righthand-side starts at \p x+batch_stride*i.
*  @param[in]
*  batch_count The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate each system. Must satisfy \p batch_stride >= \p m.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              \ref rocsparse_sgtsv_no_pivot_strided_batch "rocsparse_Xgtsv_no_pivot_strided_batch()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p batch_count or \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d, \p du,
*              \p x or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle,
                                                                    rocsparse_int    m,
                                                                    const float*     dl,
                                                                    const float*     d,
                                                                    const float*     du,
                                                                    const float*     x,
                                                                    rocsparse_int    batch_count,
                                                                    rocsparse_int    batch_stride,
                                                                    size_t*          buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle,
                                                                    rocsparse_int    m,
                                                                    const double*    dl,
                                                                    const double*    d,
                                                                    const double*    du,
                                                                    const double*    x,
                                                                    rocsparse_int    batch_count,
                                                                    rocsparse_int    batch_stride,
                                                                    size_t*          buffer_size);

ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_cgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle               handle,
                                                       rocsparse_int                  m,
                                                       const rocsparse_float_complex* dl,
                                                       const rocsparse_float_complex* d,
                                                       const rocsparse_float_complex* du,
                                                       const rocsparse_float_complex* x,
                                                       rocsparse_int                  batch_count,
                                                       rocsparse_int                  batch_stride,
                                                       size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_zgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle                handle,
                                                       rocsparse_int                   m,
                                                       const rocsparse_double_complex* dl,
                                                       const rocsparse_double_complex* d,
                                                       const rocsparse_double_complex* du,
                                                       const rocsparse_double_complex* x,
                                                       rocsparse_int                   batch_count,
                                                       rocsparse_int                   batch_stride,
                                                       size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p rocsparse_gtsv_no_pivot_strided_batch solves a batched tridiagonal linear system
*  \f[
*    T^{i}*x^{i} = x^{i}
*  \f]
*  where for each batch \f$i=0\ldots\f$ \p batch_count, \f$T^{i}\f$ is a sparse tridiagonal matrix and
*  \f$x^{i}\f$ is a dense right-hand side vector. All of the tridiagonal matrices, \f$T^{i}\f$, are
*  packed one after the other into three vectors: \p dl for the lower diagonals, \p d for the main
*  diagonals and \p du for the upper diagonals. See below for a description of what this strided
*  memory pattern looks like.
*
*  Solving the batched tridiagonal system involves two steps. First, the user calls
*  \ref rocsparse_sgtsv_no_pivot_strided_batch_buffer_size "rocsparse_Xgtsv_no_pivot_strided_batch_buffer_size()"
*  in order to determine the size of the required temporary storage buffer. Once determined, the user allocates
*  this buffer and passes it to \ref rocsparse_sgtsv_no_pivot_strided_batch "rocsparse_Xgtsv_no_pivot_strided_batch()"
*  to perform the actual solve. The \f$x^{i}\f$ vectors, which initially stores the right-hand side values, are
*  overwritten with the solution after the call to
*  \ref rocsparse_sgtsv_no_pivot_strided_batch "rocsparse_Xgtsv_no_pivot_strided_batch()".
*
*  The strided batch routines write each batch matrix one after the other in memory. For example, consider
*  the following batch matrices:
*
*  \f[
*    \begin{bmatrix}
*    t^{0}_{00} & t^{0}_{01} & 0 \\
*    t^{0}_{10} & t^{0}_{11} & t^{0}_{12} \\
*    0 & t^{0}_{21} & t^{0}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{1}_{00} & t^{1}_{01} & 0 \\
*    t^{1}_{10} & t^{1}_{11} & t^{1}_{12} \\
*    0 & t^{1}_{21} & t^{1}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{2}_{00} & t^{2}_{01} & 0 \\
*    t^{2}_{10} & t^{2}_{11} & t^{2}_{12} \\
*    0 & t^{2}_{21} & t^{2}_{22}
*    \end{bmatrix}
*  \f]
*
*  In strided format, the upper, lower, and diagonal arrays would look like:
*  \f[
*    \begin{align}
*    \text{lower} &= \begin{bmatrix} 0 & t^{0}_{10} & t^{0}_{21} & 0 & t^{1}_{10} & t^{1}_{21} & 0 & t^{2}_{10} & t^{2}_{21} \end{bmatrix} \\
*    \text{diagonal} &= \begin{bmatrix} t^{0}_{00} & t^{0}_{11} & t^{0}_{22} & t^{1}_{00} & t^{1}_{11} & t^{1}_{22} & t^{2}_{00} & t^{2}_{11} & t^{2}_{22} \end{bmatrix} \\
*    \text{upper} &= \begin{bmatrix} t^{0}_{01} & t^{0}_{12} & 0 & t^{1}_{01} & t^{1}_{12} & 0 & t^{2}_{01} & t^{2}_{12} & 0 \end{bmatrix} \\
*    \end{align}
*  \f]
*  For the lower array, for each batch \p i, the \p i*batch_stride entries are zero and for the upper array the
*  \p i*batch_stride+batch_stride-1 entries are zero.
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
*  m           size of the tri-diagonal linear system (must be >= 2).
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. First entry must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. Last entry must be zero.
*  @param[inout]
*  x           Dense array of righthand-sides where the ith righthand-side starts at \p x+batch_stride*i.
*  @param[in]
*  batch_count The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate each system. Must satisfy \p batch_stride >= \p m.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p batch_count or \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d,
*              \p du, \p x or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  \snippet example_rocsparse_gtsv_no_pivot_strided_batch.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv_no_pivot_strided_batch(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        const float*     dl,
                                                        const float*     d,
                                                        const float*     du,
                                                        float*           x,
                                                        rocsparse_int    batch_count,
                                                        rocsparse_int    batch_stride,
                                                        void*            temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv_no_pivot_strided_batch(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        const double*    dl,
                                                        const double*    d,
                                                        const double*    du,
                                                        double*          x,
                                                        rocsparse_int    batch_count,
                                                        rocsparse_int    batch_stride,
                                                        void*            temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgtsv_no_pivot_strided_batch(rocsparse_handle               handle,
                                                        rocsparse_int                  m,
                                                        const rocsparse_float_complex* dl,
                                                        const rocsparse_float_complex* d,
                                                        const rocsparse_float_complex* du,
                                                        rocsparse_float_complex*       x,
                                                        rocsparse_int                  batch_count,
                                                        rocsparse_int                  batch_stride,
                                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgtsv_no_pivot_strided_batch(rocsparse_handle                handle,
                                                        rocsparse_int                   m,
                                                        const rocsparse_double_complex* dl,
                                                        const rocsparse_double_complex* d,
                                                        const rocsparse_double_complex* du,
                                                        rocsparse_double_complex*       x,
                                                        rocsparse_int                   batch_count,
                                                        rocsparse_int batch_stride,
                                                        void*         temp_buffer);
/**@}*/

/*! \ingroup precond_module
*  \details
*  \p rocsparse_gtsv_interleaved_batch_buffer_size returns the size of the temporary storage buffer
*  that is required by \ref rocsparse_sgtsv_interleaved_batch "rocsparse_Xgtsv_interleaved_batch()".
*  The temporary storage buffer must be allocated by the user.
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
*  alg         Algorithm to use when solving tridiagonal systems. Options are thomas ( \ref rocsparse_gtsv_interleaved_alg_thomas ),
*              LU ( \ref rocsparse_gtsv_interleaved_alg_lu ), or QR ( \ref rocsparse_gtsv_interleaved_alg_qr ). Passing
*              \ref rocsparse_gtsv_interleaved_alg_default defaults the algorithm to use QR. Thomas algorithm is the fastest but is not
*              stable while LU and QR are slower but are stable.
*  @param[in]
*  m           size of the tri-diagonal linear system.
*  @param[in]
*  dl          lower diagonal of tri-diagonal system. The first element of the lower diagonal must be zero.
*  @param[in]
*  d           main diagonal of tri-diagonal system.
*  @param[in]
*  du          upper diagonal of tri-diagonal system. The last element of the upper diagonal must be zero.
*  @param[inout]
*  x           Dense array of righthand-sides with dimension \p batch_stride by \p m.
*  @param[in]
*  batch_count The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate consecutive elements in a system. Must satisfy \p batch_stride >= \p batch_count.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              \ref rocsparse_sgtsv_interleaved_batch "rocsparse_Xgtsv_interleaved_batch()".
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p batch_count, \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d, \p du,
*              \p x or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gtsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const float*                   dl,
                                                               const float*                   d,
                                                               const float*                   du,
                                                               const float*                   x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gtsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const double*                  dl,
                                                               const double*                  d,
                                                               const double*                  du,
                                                               const double*                  x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgtsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gtsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const rocsparse_float_complex* dl,
                                                               const rocsparse_float_complex* d,
                                                               const rocsparse_float_complex* du,
                                                               const rocsparse_float_complex* x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgtsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gtsv_interleaved_alg  alg,
                                                               rocsparse_int                   m,
                                                               const rocsparse_double_complex* dl,
                                                               const rocsparse_double_complex* d,
                                                               const rocsparse_double_complex* du,
                                                               const rocsparse_double_complex* x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch tridiagonal solver
*
*  \details
*  \p rocsparse_gtsv_interleaved_batch solves a batched tridiagonal linear system
*  \f[
*    T^{i}*x^{i} = x^{i}
*  \f]
*  where for each batch \f$i=0\ldots\f$ \p batch_count, \f$T^{i}\f$ is a sparse tridiagonal matrix and
*  \f$x^{i}\f$ is a dense right-hand side vector. All of the tridiagonal matrices, \f$T^{i}\f$, are
*  packed in an interleaved fashion into three vectors: \p dl for the lower diagonals, \p d for the main
*  diagonals and \p du for the upper diagonals. See below for a description of what this interleaved
*  memory pattern looks like.
*
*  Solving the batched tridiagonal system involves two steps. First, the user calls
*  \ref rocsparse_sgtsv_interleaved_batch_buffer_size "rocsparse_Xgtsv_interleaved_batch_buffer_size()"
*  in order to determine the size of the required temporary storage buffer. Once determined, the user allocates
*  this buffer and passes it to \ref rocsparse_sgtsv_interleaved_batch "rocsparse_Xgtsv_interleaved_batch()"
*  to perform the actual solve. The \f$x^{i}\f$ vectors, which initially stores the right-hand side values, are
*  overwritten with the solution after the call to
*  \ref rocsparse_sgtsv_interleaved_batch "rocsparse_Xgtsv_interleaved_batch()".
*
*  The user can specify different algorithms for \p rocsparse_gtsv_interleaved_batch
*  to use. Options are thomas ( \ref rocsparse_gtsv_interleaved_alg_thomas ),
*  LU ( \ref rocsparse_gtsv_interleaved_alg_lu ), or QR ( \ref rocsparse_gtsv_interleaved_alg_qr ).
*  Passing \ref rocsparse_gtsv_interleaved_alg_default defaults the algorithm to use QR.
*
*  Unlike the strided batch routines which write each batch matrix one after the other in memory, the interleaved
*  routines write the batch matrices such that each element from each matrix is written consecutively one after
*  the other. For example, consider the following batch matrices:
*
*  \f[
*    \begin{bmatrix}
*    t^{0}_{00} & t^{0}_{01} & 0 \\
*    t^{0}_{10} & t^{0}_{11} & t^{0}_{12} \\
*    0 & t^{0}_{21} & t^{0}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{1}_{00} & t^{1}_{01} & 0 \\
*    t^{1}_{10} & t^{1}_{11} & t^{1}_{12} \\
*    0 & t^{1}_{21} & t^{1}_{22}
*    \end{bmatrix}
*    \begin{bmatrix}
*    t^{2}_{00} & t^{2}_{01} & 0 \\
*    t^{2}_{10} & t^{2}_{11} & t^{2}_{12} \\
*    0 & t^{2}_{21} & t^{2}_{22}
*    \end{bmatrix}
*  \f]
*
*  In interleaved format, the upper, lower, and diagonal arrays would look like:
*  \f[
*    \begin{align}
*    \text{lower} &= \begin{bmatrix} 0 & 0 & 0 & t^{0}_{10} & t^{1}_{10} & t^{1}_{10} & t^{0}_{21} & t^{1}_{21} & t^{2}_{21} \end{bmatrix} \\
*    \text{diagonal} &= \begin{bmatrix} t^{0}_{00} & t^{1}_{00} & t^{2}_{00} & t^{0}_{11} & t^{1}_{11} & t^{2}_{11} & t^{0}_{22} & t^{1}_{22} & t^{2}_{22} \end{bmatrix} \\
*    \text{upper} &= \begin{bmatrix} t^{0}_{01} & t^{1}_{01} & t^{2}_{01} & t^{0}_{12} & t^{1}_{12} & t^{2}_{12} & 0 & 0 & 0 \end{bmatrix} \\
*    \end{align}
*  \f]
*  For the lower array, the first \p batch_count entries are zero and for the upper array the last \p batch_count
*  entries are zero.
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
*  alg         Algorithm to use when solving tridiagonal systems. Options are thomas ( \ref rocsparse_gtsv_interleaved_alg_thomas ),
*              LU ( \ref rocsparse_gtsv_interleaved_alg_lu ), or QR ( \ref rocsparse_gtsv_interleaved_alg_qr ). Passing
*              \ref rocsparse_gtsv_interleaved_alg_default defaults the algorithm to use QR. Thomas algorithm is the fastest but is not
*              stable while LU and QR are slower but are stable.
*  @param[in]
*  m           size of the tri-diagonal linear system.
*  @param[inout]
*  dl          lower diagonal of tri-diagonal system. The first element of the lower diagonal must be zero.
*  @param[inout]
*  d           main diagonal of tri-diagonal system.
*  @param[inout]
*  du          upper diagonal of tri-diagonal system. The last element of the upper diagonal must be zero.
*  @param[inout]
*  x           Dense array of righthand-sides with dimension \p batch_stride by \p m.
*  @param[in]
*  batch_count The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate consecutive elements in a system. Must satisfy \p batch_stride >= \p batch_count.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p batch_count or \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p dl, \p d,
*              \p du, \p x or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  \snippet example_rocsparse_gtsv_interleaved_batch.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgtsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gtsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   float*                         dl,
                                                   float*                         d,
                                                   float*                         du,
                                                   float*                         x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgtsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gtsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   double*                        dl,
                                                   double*                        d,
                                                   double*                        du,
                                                   double*                        x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgtsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gtsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   rocsparse_float_complex*       dl,
                                                   rocsparse_float_complex*       d,
                                                   rocsparse_float_complex*       du,
                                                   rocsparse_float_complex*       x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgtsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gtsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   rocsparse_double_complex*      dl,
                                                   rocsparse_double_complex*      d,
                                                   rocsparse_double_complex*      du,
                                                   rocsparse_double_complex*      x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GTSV_H */
