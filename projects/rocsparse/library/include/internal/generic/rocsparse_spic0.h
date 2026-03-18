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
#ifndef ROCSPARSE_SPIC0_H
#define ROCSPARSE_SPIC0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting.
*  \details
*  \p rocsparse_spic0_buffer_size returns the size of the non-persistent buffer
*  that is required by \ref rocsparse_spic0, and must be allocated  by the user.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  \note
*  This routine only supports uniform batched computation, i.e. same sparsity pattern but batched values of the matrices.
*
*  \note
*  Supported formats are \ref rocsparse_format_csr, and \ref rocsparse_format_bsr.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  spic0_descr Spic0 descriptor
*  @param[in]
*  A            descriptor of the matrix to factorize.
*  @param[in]
*  P            descriptor of the factorization, in-place \p P = \p A is allowed.
*  @param[in]
*  spic0_stage stage for the Spic0 computation.
*  @param[out]
*  p_buffer_size_in_bytes  number of bytes of the buffer.
*  @param[out]
*  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_not_implemented the sparse format is invalid, or the preconditioner \p P is not identical to the matrix to factorize \p A.
*  \retval rocsparse_status_invalid_value the \p spic0_stage value is invalid.
*  \retval rocsparse_status_invalid_pointer \p spic0_descr, \p A, \p P, or \p p_buffer_size_in_bytes pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spic0_buffer_size(rocsparse_handle            handle,
                                             rocsparse_spic0_descr       spic0_descr,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_spmat_descr P,
                                             rocsparse_spic0_stage       spic0_stage,
                                             size_t*                     p_buffer_size_in_bytes,
                                             rocsparse_error*            p_error);

/*! \ingroup generic_module
   *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting.
   *
   *  \details
   *  \p rocsparse_spic0 computes the incomplete Cholesky factorization with 0 fill-ins
   *  and no pivoting of a sparse \f$m \times m\f$ matrix \f$A\f$, such that
   *  \f[
   *    A \approx LL^T
   *  \f]
   *  where the lower triangular matrix \f$L\f$ is computed using:
   *  \f[
   *    L_{ij} = \left\{
   *    \begin{array}{ll}
   *        \sqrt{A_{jj} - \sum_{k=0}^{j-1}(L_{jk})^{2}},   & \text{if i == j} \\
   *        \frac{1}{L_{jj}}(A_{ij} - \sum_{k=0}^{j-1}L_{ik} \times L_{jk}), & \text{if i > j}
   *    \end{array}
   *    \right.
   *  \f]
   *  for each entry found in the matrix \f$A\f$.
   *
   *  Performing the above operation requires two stages, the stage \ref rocsparse_spic0_stage_analysis and the stage \ref rocsparse_spic0_stage_compute.
   *  The stage \ref rocsparse_spic0_stage_analysis is required to perform the stage \ref rocsparse_spic0_stage_compute and only need to be called once for a given sparse matrix \f$A\f$ while the stage \ref rocsparse_spic0_stage_compute can be repeatedly used with different matrices \f$A\f$ but having the same sparsity pattern.
   *
   *  \p rocsparse_spic0 supports the following
   *  data types for \p A: \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r, \ref rocsparse_datatype_f32_c, and \ref rocsparse_datatype_f64_c.
   *
   *  \note The descriptor \p spic0_descr needs to be configured with \ref rocsparse_spic0_set_input.
   *  \note
   *  The sparse matrix formats currently supported are: \ref rocsparse_format_csr and \ref rocsparse_format_bsr.
   *
   *  \note
   *  the \ref rocsparse_spic0_stage_compute stage is non blocking
   *  and executed asynchronously with respect to the host. It may return before the actual computation has finished.
   *  The \ref rocsparse_spic0_stage_analysis stage is blocking with respect to the host.
   *
   *  \note
   *  Only the \ref rocsparse_spic0_stage_compute stage
   *  supports execution in a hipGraph context. The \ref rocsparse_spic0_stage_analysis stage does not support hipGraph.
   *
   *  \note
   *  This routine only supports uniform strided batched computation, i.e. same sparsity pattern but strided batched values of the matrices.
   *
   *  @param[in]
   *  handle       handle to the rocsparse library context queue.
   *  @param[in]
   *  spic0_descr Spic0 descriptor
   *  @param[in]
   *  A            descriptor of the matrix to factorize.
   *  @param[out]
   *  P            descriptor of the factorization, in-place \p P = \p A is allowed.
   *  @param[in]
   *  spic0_stage stage for the Spic0 computation.
   *  @param[in]
   *  buffer_size_in_bytes  number of bytes of the buffer.
   *  @param[in]
   *  buffer       buffer allocated by the user.
   *  @param[out]
   *  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
   *
   *  \retval rocsparse_status_success the operation completed successfully.
   *  \retval rocsparse_status_invalid_handle the library context was not initialized.
   *  \retval rocsparse_status_not_implemented the sparse format is invalid, or the preconditioner \p P is not identical to the matrix to factorize \p A.
   *  \retval rocsparse_status_invalid_value the \p spic0_stage value is invalid.
   *  \retval rocsparse_status_invalid_pointer \p spic0_descr, \p A, \p P, or \p buffer_size_in_bytes pointer is invalid.
   *
   *  \par Example
   *  \snippet example_rocsparse_spic0.cpp doc example
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spic0(rocsparse_handle            handle,
                                 rocsparse_spic0_descr       spic0_descr,
                                 rocsparse_const_spmat_descr A,
                                 rocsparse_spmat_descr       P,
                                 rocsparse_spic0_stage       spic0_stage,
                                 size_t                      buffer_size_in_bytes,
                                 void*                       buffer,
                                 rocsparse_error*            p_error);

#ifdef __cplusplus
}
#endif
#endif
