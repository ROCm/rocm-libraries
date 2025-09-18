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

#ifndef ROCSPARSE_SPITSV_H
#define ROCSPARSE_SPITSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup generic_module
*  \brief Sparse iterative triangular solve
*
*  \details
*  \p rocsparse_spitsv solves, using the Jacobi iterative method, a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) y = \alpha x,
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
*  The Jacobi method applied to the sparse triangular linear system above gives
*  \f[
*     y_{k+1} = y_{k} + D^{-1} ( \alpha x - (D + T) y_{k} )
*  \f]
*  with \f$A = D + T\f$, \f$D\f$ the diagonal of \f$A\f$ and \f$T\f$ the strict triangular part of \f$A\f$.
*
*  The above equation can be also written as
*  \f[
*     y_{k+1} = y_{k} + D^{-1} r_k
*  \f]
*  where
*  \f[
*     r_k = \alpha x - (D + T) y_k.
*  \f]
*  Starting with \f$y_0 = \f$ \p y, the method iterates while \f$ k \lt \f$ \p host_nmaxiter and until
*  \f[
*     \Vert r_k \Vert_{\infty} \le \epsilon,
*  \f]
*  with \f$\epsilon\f$ = \p host_tol.
*
*  \p rocsparse_spitsv requires three stages to complete. First, the user passes the \ref rocsparse_spitsv_stage_buffer_size
*  stage to determine the size of the required temporary storage buffer. Next, the user allocates this buffer and calls
*  \p rocsparse_spitsv again with the \ref rocsparse_spitsv_stage_preprocess stage which will preprocess data and store it
*  in the temporary buffer. Finally, the user calls \p rocsparse_spitsv with the \ref rocsparse_spitsv_stage_compute stage to
*  perform the actual computation. Once all calls to \p rocsparse_spitsv are complete, the temporary buffer
*  can be deallocated.
*
*  \p rocsparse_spitsv supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index precisions for storing the
*  row pointer and column indices arrays of the sparse matrix. \p rocsparse_spitsv supports the following data types for
*  \f$op(A)\f$, \f$x\f$, \f$y\f$ and compute types for \f$\alpha\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spitsv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle        handle to the rocsparse library context queue.
*  @param[inout]
*  host_nmaxiter maximum number of iteration on input and number of iteration on output. If the output number of iterations is strictly less than the input maximum number of iterations, then the algorithm converged.
*  @param[in]
*  host_tol      if the pointer is null then loop will execute \p nmaxiter[0] iterations. The precision is float for f32 based calculation (including the complex case) and double for f64 based calculation (including the complex case).
*  @param[out]
*  host_history  Optional array to record the norm of the residual before each iteration. The precision is float for f32 based calculation (including the complex case) and double for f64 based calculation (including the complex case).
*  @param[in]
*  trans         matrix operation type.
*  @param[in]
*  alpha         scalar \f$\alpha\f$.
*  @param[in]
*  mat           matrix descriptor.
*  @param[in]
*  x             vector descriptor.
*  @param[inout]
*  y             vector descriptor.
*  @param[in]
*  compute_type  floating point precision for the SpITSV computation.
*  @param[in]
*  alg           SpITSV algorithm for the SpITSV computation.
*  @param[in]
*  stage         SpITSV stage for the SpITSV computation.
*  @param[out]
*  buffer_size   number of bytes of the temporary storage buffer.
*  @param[in]
*  temp_buffer   temporary storage buffer allocated by the user. When a nullptr is passed,
*                the required allocation size (in bytes) is written to \p buffer_size and
*                function returns without performing the SpITSV operation.
*
*  \retval       rocsparse_status_success the operation completed successfully.
*  \retval       rocsparse_status_invalid_handle the library context was not initialized.
*  \retval       rocsparse_status_invalid_pointer \p alpha, \p mat, \p x, \p y, \p descr or
*                \p buffer_size pointer is invalid.
*  \retval       rocsparse_status_not_implemented \p trans, \p compute_type, \p stage or \p alg is
*                currently not supported.
*
*  \par Example
*  \snippet example_rocsparse_spitsv.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spitsv(rocsparse_handle            handle,
                                  rocsparse_int*              host_nmaxiter,
                                  const void*                 host_tol,
                                  void*                       host_history,
                                  rocsparse_operation         trans,
                                  const void*                 alpha,
                                  const rocsparse_spmat_descr mat,
                                  const rocsparse_dnvec_descr x,
                                  const rocsparse_dnvec_descr y,
                                  rocsparse_datatype          compute_type,
                                  rocsparse_spitsv_alg        alg,
                                  rocsparse_spitsv_stage      stage,
                                  size_t*                     buffer_size,
                                  void*                       temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPITSV_H */
