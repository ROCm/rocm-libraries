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

#ifndef ROCSPARSE_SPSM_H
#define ROCSPARSE_SPSM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse triangular system solve with multiple right-hand sides
*
*  \details
*  \p rocsparse_spsm solves a triangular linear system of equations defined by a sparse \f$m \times m\f$ square matrix \f$op(A)\f$,
*  given in CSR or COO storage format, such that
*  \f[
*    op(A) \cdot C = \alpha \cdot op(B),
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
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose}
*    \end{array}
*    \right.
*  \f]
*  and where \f$C\f$ is the dense solution matrix and \f$B\f$ is the dense right-hand side matrix. Both \f$B\f$
*  and \f$C\f$ can be in row or column order.
*
*  Performing the above operation requires three stages. First, \p rocsparse_spsm must be called with the stage
*  \ref rocsparse_spsm_stage_buffer_size which will determine the size of the required temporary storage buffer.
*  The user then allocates this buffer and calls \p rocsparse_spsm with the stage \ref rocsparse_spsm_stage_preprocess
*  which will perform analysis on the sparse matrix \f$op(A)\f$. Finally, the user completes the computation by calling
*  \p rocsparse_spsm with the stage \ref rocsparse_spsm_stage_compute. The buffer size, buffer allocation, and preprecess
*  stages only need to be called once for a given sparse triangular matrix \f$op(A)\f$ while the computation stage can be
*  repeatedly used with different \f$B\f$ and \f$C\f$ matrices.
*
*  As noted above, both \f$B\f$ and \f$C\f$ can be in row or column order (this includes mixing the order so that \f$B\f$ is
*  row order and \f$C\f$ is column order and vice versa). Internally however, rocSPARSE kernels solve the system assuming the
*  matrices \f$B\f$ and \f$C\f$ are in row order as this provides the best memory access. This means that if the matrix
*  \f$C\f$ is not in row order and/or the matrix \f$B\f$ is not row order (or \f$B^{T}\f$ is not column order as this is
*  equivalent to being in row order), then internally memory copies and/or transposing of data may be performed to get them
*  into the correct order (possbily using extra buffer size). Once computation is completed, additional memory copies and/or
*  transposing of data may be performed to get them back into the user arrays. For best performance and smallest required
*  temporary storage buffers, use row order for the matrix \f$C\f$ and row order for the matrix \f$B\f$ (or column order if
*  \f$B\f$ is being transposed).
*
*  \p rocsparse_spsm supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index precisions for storing the
*  row pointer and column indices arrays of the sparse matrices. \p rocsparse_spsm supports the following data types for
*  \f$op(A)\f$, \f$op(B)\f$, \f$C\f$ and compute types for \f$\alpha\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spsm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
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
*  Only the \ref rocsparse_spsm_stage_buffer_size stage and the \ref rocsparse_spsm_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spsm_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Currently, only \p trans_A == \ref rocsparse_operation_none and \p trans_A == \ref rocsparse_operation_transpose is supported.
*  Currently, only \p trans_B == \ref rocsparse_operation_none and \p trans_B == \ref rocsparse_operation_transpose is supported.
*
*  \note
*  Only the \ref rocsparse_spsm_stage_buffer_size stage and the \ref rocsparse_spsm_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spsm_stage_preprocess stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans_A      matrix operation type for the sparse matrix \f$op(A)\f$.
*  @param[in]
*  trans_B      matrix operation type for the dense matrix \f$op(B)\f$.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  matA         sparse matrix descriptor.
*  @param[in]
*  matB         dense matrix descriptor.
*  @param[inout]
*  matC         dense matrix descriptor.
*  @param[in]
*  compute_type floating point precision for the SpSM computation.
*  @param[in]
*  alg          SpSM algorithm for the SpSM computation.
*  @param[in]
*  stage        SpSM stage for the SpSM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When the
*               \ref rocsparse_spsm_stage_buffer_size stage is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpSM operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p matA, \p matB, \p matC, \p descr or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p trans_A, \p trans_B, \p compute_type, \p stage or \p alg is
*               currently not supported.
*
*  \par Example
*  \snippet example_rocsparse_spsm.cpp doc example
*/
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_sptrsm instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_spsm(rocsparse_handle            handle,
                   rocsparse_operation         trans_A,
                   rocsparse_operation         trans_B,
                   const void*                 alpha,
                   rocsparse_const_spmat_descr matA,
                   rocsparse_const_dnmat_descr matB,
                   const rocsparse_dnmat_descr matC,
                   rocsparse_datatype          compute_type,
                   rocsparse_spsm_alg          alg,
                   rocsparse_spsm_stage        stage,
                   size_t*                     buffer_size,
                   void*                       temp_buffer);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPSM_H */
