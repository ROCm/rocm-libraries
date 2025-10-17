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

#ifndef ROCSPARSE_SPSV_H
#define ROCSPARSE_SPSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse triangular system solve
*
*  \details
*  \p rocsparse_spsv solves a triangular linear system of equations defined by a sparse \f$m \times m\f$ square matrix \f$op(A)\f$,
*  given in CSR or COO storage format, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
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
*  and where \f$y\f$ is the dense solution vector and \f$x\f$ is the dense right-hand side vector.
*
*  Performing the above operation requires three stages. First, \p rocsparse_spsv must be called with the stage
*  \ref rocsparse_spsv_stage_buffer_size which will determine the size of the required temporary storage buffer.
*  The user then allocates this buffer and calls \p rocsparse_spsv with the stage \ref rocsparse_spsv_stage_preprocess
*  which will perform analysis on the sparse matrix \f$op(A)\f$. Finally, the user completes the computation by calling
*  \p rocsparse_spsv with the stage \ref rocsparse_spsv_stage_compute. The buffer size, buffer allocation, and preprecess
*  stages only need to be called once for a given sparse matrix \f$op(A)\f$ while the computation stage can be repeatedly
*  used with different \f$x\f$ and \f$y\f$ vectors.
*
*  \p rocsparse_spsv supports \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64 index types for
*  storing the row pointer and column indices arrays of the sparse matrices. \p rocsparse_spsv supports the following
*  data types for \f$op(A)\f$, \f$x\f$, \f$y\f$ and compute types for \f$\alpha\f$:
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spsv_uniform">Uniform Precisions</caption>
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
*  Only the \ref rocsparse_spsv_stage_buffer_size stage and the \ref rocsparse_spsv_stage_compute stage are non blocking
*  and executed asynchronously with respect to the host. They may return before the actual computation has finished.
*  The \ref rocsparse_spsv_stage_preprocess stage is blocking with respect to the host.
*
*  \note
*  Currently, only \p trans == \ref rocsparse_operation_none and \p trans == \ref rocsparse_operation_transpose is supported.
*
*  \note
*  Only the \ref rocsparse_spsv_stage_buffer_size stage and the \ref rocsparse_spsv_stage_compute stage
*  support execution in a hipGraph context. The \ref rocsparse_spsv_stage_preprocess stage does not support hipGraph.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans        matrix operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat          matrix descriptor.
*  @param[in]
*  x            vector descriptor.
*  @param[inout]
*  y            vector descriptor.
*  @param[in]
*  compute_type floating point precision for the SpSV computation.
*  @param[in]
*  alg          SpSV algorithm for the SpSV computation.
*  @param[in]
*  stage        SpSV stage for the SpSV computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When the
*               \ref rocsparse_spsv_stage_buffer_size stage is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpSV operation.
*               This buffer is non-persistent, no data are stored in it; therefore, this memory
*               can be freed or reuse for other tasks between the analysis phase and the compute phase.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat, \p x, \p y or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p trans, \p compute_type, \p stage or \p alg is
*               currently not supported.
*
*  \par Example
*  \snippet example_rocsparse_spsv.cpp doc example
*/
__attribute__((deprecated("This function is deprecated and will be removed in a future release. "
                          "Use rocsparse_sptrsv instead."))) ROCSPARSE_EXPORT rocsparse_status
    rocsparse_spsv(rocsparse_handle            handle,
                   rocsparse_operation         trans,
                   const void*                 alpha,
                   rocsparse_const_spmat_descr mat,
                   rocsparse_const_dnvec_descr x,
                   const rocsparse_dnvec_descr y,
                   rocsparse_datatype          compute_type,
                   rocsparse_spsv_alg          alg,
                   rocsparse_spsv_stage        stage,
                   size_t*                     buffer_size,
                   void*                       temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPSV_H */
