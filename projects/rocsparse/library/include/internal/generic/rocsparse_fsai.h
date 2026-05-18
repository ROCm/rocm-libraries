/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef ROCSPARSE_FSAI_H
#define ROCSPARSE_FSAI_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
 *  \brief Get buffer size for Factorized Sparse Approximate Inverse (FSAI) preconditioner.
 *
 *  \details
 *  \p rocsparse_fsai_buffer_size returns the size of the non-persistent buffer
 *  that is required by \ref rocsparse_fsai, and must be allocated by the user.
 *
 *  \note
 *  This function is non-blocking and executed asynchronously with respect to the host.
 *  It can return before the actual computation has finished.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  \note
 *  Supported format is \ref rocsparse_format_csr.
 *
 *  @param[in]
 *  handle       handle to the rocSPARSE library context queue.
 *  @param[in]
 *  descr        FSAI descriptor.
 *  @param[in]
 *  A            descriptor of the input matrix.
 *  @param[in]
 *  M            descriptor of the approximate inverse (output sparsity pattern).
 *  @param[in]
 *  stage        stage for the FSAI computation.
 *  @param[out]
 *  buffer_size_in_bytes  number of bytes of the buffer.
 *  @param[out]
 *  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success.
 *               A null pointer can be passed if the user is not interested in obtaining an error descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_not_implemented the sparse format is invalid.
 *  \retval rocsparse_status_invalid_value the \p stage value is invalid.
 *  \retval rocsparse_status_invalid_pointer \p descr, \p A, \p M, or \p buffer_size_in_bytes pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_fsai_buffer_size(rocsparse_handle            handle,
                                            rocsparse_fsai_descr        descr,
                                            rocsparse_const_spmat_descr A,
                                            rocsparse_const_spmat_descr M,
                                            rocsparse_fsai_stage        stage,
                                            size_t*                     buffer_size_in_bytes,
                                            rocsparse_error*            p_error);

/*! \ingroup generic_module
 *  \brief Factorized Sparse Approximate Inverse (FSAI) preconditioner.
 *
 *  \details
 *  \p rocsparse_fsai computes the Factorized Sparse Approximate Inverse preconditioner
 *  of a sparse \f$m \times m\f$ matrix \f$A\f$, such that
 *  \f[
 *    M \approx A^{-1}
 *  \f]
 *  where the sparse approximate inverse \f$M\f$ is computed by solving local least-squares
 *  problems for each row of \f$M\f$.
 *
 *  The FSAI preconditioner is useful for preconditioning iterative solvers such as
 *  Conjugate Gradient (CG) or GMRES. The sparsity pattern of \f$M\f$ is provided by the user.
 *
 *  Performing the above operation requires two stages:
 *  - \ref rocsparse_fsai_stage_analysis : Analyzes the sparsity pattern
 *  - \ref rocsparse_fsai_stage_compute : Computes the values of the approximate inverse
 *
 *  The analysis stage only needs to be called once for a given sparsity pattern of \f$A\f$ and \f$M\f$,
 *  while the compute stage can be repeatedly used with different values of \f$A\f$ that have
 *  the same sparsity pattern.
 *
 *  \p rocsparse_fsai supports the following uniform-precision data types for the sparse matrix \p A.
 *
 *  \par Uniform Precisions:
 *  <table>
 *  <caption id="fsai_uniform">Uniform Precisions</caption>
 *  <tr><th>A
 *  <tr><td>rocsparse_datatype_f32_r
 *  <tr><td>rocsparse_datatype_f64_r
 *  <tr><td>rocsparse_datatype_f32_c
 *  <tr><td>rocsparse_datatype_f64_c
 *  </table>
 *
 *  \note The descriptor \p descr needs to be configured with \ref rocsparse_fsai_descr_set_input.
 *  \note The sparse matrix format currently supported is \ref rocsparse_format_csr.
 *
 *  \note
 *  The \ref rocsparse_fsai_stage_compute stage is non-blocking
 *  and executed asynchronously with respect to the host. It can return before the actual
 *  computation has finished.
 *  The \ref rocsparse_fsai_stage_analysis stage is blocking with respect to the host.
 *
 *  \note
 *  Only the \ref rocsparse_fsai_stage_compute stage supports execution in a hipGraph context.
 *  The \ref rocsparse_fsai_stage_analysis stage does not support hipGraph.
 *
 *  @param[in]
 *  handle       handle to the rocSPARSE library context queue.
 *  @param[in]
 *  descr        FSAI descriptor.
 *  @param[in]
 *  A            descriptor of the input matrix.
 *  @param[out]
 *  M            descriptor of the approximate inverse. The sparsity pattern is input,
 *               the values are computed.
 *  @param[in]
 *  stage        stage for the FSAI computation.
 *  @param[in]
 *  buffer_size_in_bytes  number of bytes of the buffer.
 *  @param[in]
 *  buffer       buffer allocated by the user.
 *  @param[out]
 *  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success.
 *               A null pointer can be passed if an error descriptor is not required.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_not_implemented the sparse format is invalid.
 *  \retval rocsparse_status_invalid_value the \p stage value is invalid.
 *  \retval rocsparse_status_invalid_pointer \p descr, \p A, \p M, or \p buffer pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_fsai(rocsparse_handle            handle,
                                rocsparse_fsai_descr        descr,
                                rocsparse_const_spmat_descr A,
                                rocsparse_spmat_descr       M,
                                rocsparse_fsai_stage        stage,
                                size_t                      buffer_size_in_bytes,
                                void*                       buffer,
                                rocsparse_error*            p_error);

#ifdef __cplusplus
}
#endif

#endif // ROCSPARSE_FSAI_H
