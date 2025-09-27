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

#ifndef ROCSPARSE_SPTRSV_H
#define ROCSPARSE_SPTRSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_sptrsv_buffer_size returns the size of the required buffer to execute the given stage of the SpTrSV operation.
*  This routine is used in conjunction with \ref rocsparse_sptrsv(). See \ref rocsparse_sptrsv for full description and example.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  sptrsv_descr SpTrSV descriptor
*  @param[in]
*  spmat_descr  sparse matrix descriptor.
*  @param[in]
*  x            dense vector descriptor.
*  @param[in]
*  y            dense vector descriptor.
*  @param[in]
*  sptrsv_stage stage for the SpTrSV computation.
*  @param[out]
*  buffer_size_in_bytes  number of bytes of the buffer.
*  @param[out]
*  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value the \p sptrsv_stage value is invalid.
*  \retval rocsparse_status_invalid_pointer \p sptrsv_descr, \p spmat_descr, \p x, \p y, or \p buffer_size_in_bytes pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv_buffer_size(rocsparse_handle            handle,
                                              rocsparse_sptrsv_descr      sptrsv_descr,
                                              rocsparse_const_spmat_descr spmat_descr,
                                              rocsparse_const_dnvec_descr x,
                                              rocsparse_const_dnvec_descr y,
                                              rocsparse_sptrsv_stage      sptrsv_stage,
                                              size_t*                     buffer_size_in_bytes,
                                              rocsparse_error*            p_error);

/*! \ingroup generic_module
*  \brief Sparse Triangular solve
*
*  \details
*  \p rocsparse_sptrsv solves a triangular linear system of equations defined by a sparse \f$m \times m\f$ square matrix \f$op(A)\f$,
*  such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if op == rocsparse_operation_none} \\
*        A^T, & \text{if op == rocsparse_operation_transpose} \\
*        A^H, & \text{if op == rocsparse_operation_conjugate_transpose} \\
*    \end{array}
*    \right.
*  \f]
*  and where \f$y\f$ is the dense solution vector and \f$x\f$ is the dense right-hand side vector.
*
*  Performing the above operation requires two stages, the stage \ref rocsparse_sptrsv_stage_analysis and the stage \ref rocsparse_sptrsv_stage_compute.
*  The stage \ref rocsparse_sptrsv_stage_analysis is required to perform the stage \ref rocsparse_sptrsv_stage_compute and only need to be called once for a given sparse matrix \f$op(A)\f$ while the stage \ref rocsparse_sptrsv_stage_compute can be repeatedly used with different \f$x\f$ and \f$y\f$ vectors.
*
*  \p rocsparse_sptrsv supports the following
*  data types for \f$op(A)\f$, \f$x\f$, \f$y\f$, and scalar \f$\alpha\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="sptrsv_uniform">Uniform Precisions</caption>
*  <tr><th>A / X / Y / scalar
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \note The descriptor \p rocsparse_sptrsv_descr needs to be configured with \ref rocsparse_sptrsv_set_input.
*  \note
*  The sparse matrix formats currently supported are: \ref rocsparse_format_coo and \ref rocsparse_format_csr.
*
*  \note
*  the \ref rocsparse_sptrsv_stage_compute stage is non blocking
*  and executed asynchronously with respect to the host. It may return before the actual computation has finished.
*  The \ref rocsparse_sptrsv_stage_analysis stage is blocking with respect to the host.
*
*  \note
*  Currently, only \p trans == \ref rocsparse_operation_none and \p trans == \ref rocsparse_operation_transpose is supported.
*  Only the \ref rocsparse_sptrsv_stage_compute stage
*  supports execution in a hipGraph context. The \ref rocsparse_sptrsv_stage_analysis stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  sptrsv_descr descriptor of the routine.
*  @param[in]
*  A            matrix descriptor.
*  @param[in]
*  x            vector descriptor.
*  @param[inout]
*  y            vector descriptor.
*  @param[in]
*  sptrsv_stage stage for the SpTRSV computation.
*  @param[in]
*  buffer_size_in_bytes  number of bytes of the buffer.
*  @param[in]
*  buffer       buffer allocated by the user.
*  @param[out]
*  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p sptrsv_descr, \p A, \p x, or \p y is invalid, or if \p buffer is null and \p buffer_size_in_bytes is non zero, or if \p buffer is non null and \p buffer_size_in_bytes is zero.
*  \retval      rocsparse_status_invalid_value \p sptrsv_stage is invalid.
*
*  \par Example
*  \snippet example_rocsparse_sptrsv.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv(rocsparse_handle            handle,
                                  rocsparse_sptrsv_descr      sptrsv_descr,
                                  rocsparse_const_spmat_descr A,
                                  rocsparse_const_dnvec_descr x,
                                  rocsparse_dnvec_descr       y,
                                  rocsparse_sptrsv_stage      sptrsv_stage,
                                  size_t                      buffer_size_in_bytes,
                                  void*                       buffer,
                                  rocsparse_error*            p_error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPTRSV_H */
