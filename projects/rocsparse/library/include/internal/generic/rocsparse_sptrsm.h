/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_SPTRSM_H
#define ROCSPARSE_SPTRSM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_sptrsm_buffer_size returns the size of the required buffer to execute the given stage of the SpTrSM operation.
*  This routine is used in conjunction with \ref rocsparse_sptrsm(). See \ref rocsparse_sptrsm for full description and example.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  sptrsm_descr SpTrSM descriptor
*  @param[in]
*  A  sparse matrix descriptor.
*  @param[in]
*  X            dense matrix descriptor.
*  @param[in]
*  Y            dense matrix descriptor.
*  @param[in]
*  sptrsm_stage stage for the SpTrSM computation.
*  @param[out]
*  buffer_size_in_bytes  number of bytes of the buffer.
*  @param[out]
*  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value the \p sptrsm_stage value is invalid.
*  \retval rocsparse_status_invalid_pointer \p A, \p X, \p Y, \p sptrsm_descr or \p buffer_size_in_bytes pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsm_buffer_size(rocsparse_handle            handle,
                                              rocsparse_sptrsm_descr      sptrsm_descr,
                                              rocsparse_const_spmat_descr A,
                                              rocsparse_const_dnmat_descr X,
                                              rocsparse_const_dnmat_descr Y,
                                              rocsparse_sptrsm_stage      sptrsm_stage,
                                              size_t*                     buffer_size_in_bytes,
                                              rocsparse_error*            p_error);

/*! \ingroup generic_module
*  \brief Sparse triangular system solve with multiple right-hand sides
*
*  \details
*  \p rocsparse_sptrsm solves a triangular linear system of equations defined by a sparse \f$m \times m\f$ square matrix \f$op(A)\f$,
*  given in CSR or COO storage format, such that
*  \f[
*    op(A) \cdot Y = \alpha \cdot op(X),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if trans_B == rocsparse_operation_none} \\
*        X^T, & \text{if trans_B == rocsparse_operation_transpose}
*    \end{array}
*    \right.
*  \f]
*  and where \f$Y\f$ is the dense solution matrix and \f$X\f$ is the dense right-hand side matrix. Both \f$X\f$
*  and \f$Y\f$ can be in row or column order.
*
*  Performing the above operation requires two stages, the stage \ref rocsparse_sptrsm_stage_analysis and the stage \ref rocsparse_sptrsm_stage_compute.
*  The stage \ref rocsparse_sptrsm_stage_analysis is required to perform the stage \ref rocsparse_sptrsm_stage_compute and only need to be called once for a given sparse matrix \f$op(A)\f$ while the stage \ref rocsparse_sptrsm_stage_compute can be repeatedly used with different \f$X\f$ and \f$Y\f$ matrices.
*
*  As noted above, both \f$X\f$ and \f$Y\f$ can be in row or column order (this includes mixing the order so that \f$X\f$ is
*  row order and \f$Y\f$ is column order and vice versa). Internally however, rocSPARSE kernels solve the system assuming the
*  matrices \f$X\f$ and \f$Y\f$ are in row order as this provides the best memory access. This means that if the matrix
*  \f$Y\f$ is not in row order and/or the matrix \f$X\f$ is not row order (or \f$X^{T}\f$ is not column order as this is
*  equivalent to being in row order), then internally memory copies and/or transposing of data may be performed to get them
*  into the correct order (possbily using extra buffer size). Once computation is completed, additional memory copies and/or
*  transposing of data may be performed to get them back into the user arrays. For best performance and smallest required
*  temporary storage buffers, use row order for the matrix \f$Y\f$ and row order for the matrix \f$X\f$ (or column order if
*  \f$X\f$ is being transposed).
*
*  \p rocsparse_sptrsm supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index precisions for storing the
*  row pointer and column indices arrays of the sparse matrices. \p rocsparse_sptrsm supports the following data types for
*  \f$op(A)\f$, \f$op(X)\f$, \f$Y\f$ and compute types for \f$\alpha\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="sptrsm_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  The sparse matrix formats currently supported are: \ref rocsparse_format_coo and \ref rocsparse_format_csr.
*
*  \note
*  Only the \ref rocsparse_sptrsm_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_sptrsm_stage_analysis stage is blocking with respect to the host.
*
*  \note
*  Currently, only \p trans_A == \ref rocsparse_operation_none and \p trans_A == \ref rocsparse_operation_transpose is supported.
*  Currently, only \p trans_X == \ref rocsparse_operation_none and \p trans_X == \ref rocsparse_operation_transpose is supported.
*
*  \note
*  Only the stage \ref rocsparse_sptrsm_stage_compute
*  support execution in a hipGraph context. The \ref rocsparse_sptrsm_stage_analysis stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  sptrsm_descr           sptrsm routine descriptor.
*  @param[in]
*  A           sparse matrix descriptor.
*  @param[in]
*  X           dense matrix descriptor.
*  @param[inout]
*  Y           dense matrix descriptor.
*  @param[in]
*  sptrsm_stage Sptrsm stage for the Sptrsm computation.
*  @param[out]
*  buffer_size_in_bytes  number of bytes of the temporary storage buffer.
*  @param[in]
*  buffer  temporary storage buffer allocated by the user.
*  @param[out]
*  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p A, X, \p Y, \p sptrsm_descr or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented the configuration of the descriptor \p sptrsm_descr is currently not supported.
*  \par Example
*  \snippet example_rocsparse_sptrsm.cpp doc example
*
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsm(rocsparse_handle            handle,
                                  rocsparse_sptrsm_descr      sptrsm_descr,
                                  rocsparse_const_spmat_descr A,
                                  rocsparse_const_dnmat_descr X,
                                  rocsparse_dnmat_descr       Y,
                                  rocsparse_sptrsm_stage      sptrsm_stage,
                                  size_t                      buffer_size_in_bytes,
                                  void*                       buffer,
                                  rocsparse_error*            p_error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPTRSM_H */
