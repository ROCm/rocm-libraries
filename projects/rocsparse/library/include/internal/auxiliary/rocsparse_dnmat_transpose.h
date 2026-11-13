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

#ifndef ROCSPARSE_DNMAT_TRANSPOSE_H
#define ROCSPARSE_DNMAT_TRANSPOSE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup aux_module
 *  \brief Transpose a dense matrix into another dense matrix, optionally scaling it.
 *  \details
 *  \p rocsparse_dnmat_transpose writes the transpose of the \f$m \times n\f$ dense matrix \f$X\f$
 *  into the \f$n \times m\f$ dense matrix \f$Y\f$, optionally multiplying it by a scaling factor
 *  \f$\alpha\f$:
 *  \f[
 *    Y := \alpha \cdot X^T,
 *  \f]
 *  or \f$Y := X^T\f$ when \p alpha is null. The number of rows of \f$Y\f$ must be the number of
 *  columns of \f$X\f$, and the number of columns of \f$Y\f$ must be the number of rows of
 *  \f$X\f$. For complex data types the values are not conjugated.
 *
 *  \f$X\f$ and \f$Y\f$ can be in row or column order (\ref rocsparse_order), and can have
 *  different orders and leading dimensions.
 *
 *  \p alpha is not a plain scalar but a dense vector descriptor, so that a batched transpose can
 *  use a different scaling factor for each batch. Only the first value of each batch of \p alpha
 *  is read. To scale by a single value, create \p alpha with
 *  \ref rocsparse_dnvec_descr_create_scalar, which records whether the value is in host or device
 *  memory. To scale each batch by its own value, create \p alpha with
 *  \ref rocsparse_create_dnvec_descr over device memory and set its batch count and batch stride
 *  with \ref rocsparse_dnvec_set_strided_batch; the factor of batch \f$k\f$ is then the value at
 *  offset \f$k\f$ times the batch stride. The pointer mode of the handle does not apply to
 *  \p alpha.
 *
 *  \p rocsparse_dnmat_transpose supports strided batched computation. The number of batches
 *  \f$b\f$ is the batch count of \f$Y\f$, and for \f$k = 0, 1, \ldots, b - 1\f$ there are four
 *  supported batch modes:
 *  \f[
 *    Y_k = \alpha_k \cdot X_k^T \\
 *    Y_k = \alpha_k \cdot X^T \\
 *    Y_k = \alpha \cdot X_k^T \\
 *    Y_k = \alpha \cdot X^T
 *  \f]
 *  The batch mode is determined by the batch count and batch stride of \f$X\f$ (set with
 *  \ref rocsparse_dnmat_set_strided_batch) and of \p alpha (set with
 *  \ref rocsparse_dnvec_set_strided_batch). A batch count of 1 with a batch stride of 0 uses the
 *  same matrix or factor for every batch; otherwise the batch count must be \f$b\f$:
 *
 *  <table>
 *  <tr><th>Batch mode                    <th>Batch count of X   <th>Batch stride of X <th>Batch count of alpha <th>Batch stride of alpha
 *  <tr><td>Y_k = alpha_k * X_k^T</td>    <td>b</td>             <td>any</td>          <td>b</td>               <td>any</td>
 *  <tr><td>Y_k = alpha_k * X^T</td>      <td>1</td>             <td>0</td>            <td>b</td>               <td>any</td>
 *  <tr><td>Y_k = alpha * X_k^T</td>      <td>b</td>             <td>any</td>          <td>1</td>               <td>0</td>
 *  <tr><td>Y_k = alpha * X^T</td>        <td>1</td>             <td>0</td>            <td>1</td>               <td>0</td>
 *  </table>
 *
 *  A null \p alpha behaves like a factor of one in every batch. When \p alpha is in host memory
 *  (\ref rocsparse_pointer_mode_host), its first value is used for every batch, whatever its batch
 *  count, so only the last two modes are available; per-batch factors need \p alpha in device
 *  memory.
 *
 *  \f$X\f$, \f$Y\f$ and \p alpha must all have the same data type. Mixed precisions are not
 *  supported.
 *
 *  \par Uniform Precisions:
 *  <table>
 *  <tr><th>alpha / X / Y
 *  <tr><td>rocsparse_datatype_f16_r
 *  <tr><td>rocsparse_datatype_bf16_r
 *  <tr><td>rocsparse_datatype_f32_r
 *  <tr><td>rocsparse_datatype_f64_r
 *  <tr><td>rocsparse_datatype_f32_c
 *  <tr><td>rocsparse_datatype_f64_c
 *  <tr><td>rocsparse_datatype_i8_r
 *  <tr><td>rocsparse_datatype_u8_r
 *  <tr><td>rocsparse_datatype_i32_r
 *  <tr><td>rocsparse_datatype_u32_r
 *  </table>
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocSPARSE library context queue.
 *  @param[in]
 *  alpha       optional dense vector holding the scaling factor, or one factor per batch.
 *              Pass null to transpose without scaling.
 *  @param[in]
 *  X           source \f$m \times n\f$ dense matrix.
 *  @param[inout]
 *  Y           destination \f$n \times m\f$ dense matrix. Its values are overwritten.
 *  @param[out]
 *  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_invalid_pointer if \p X or \p Y is a null pointer.
 *  \retval rocsparse_status_invalid_value if the number of rows of \p Y is not the number of
 *          columns of \p X or the number of columns of \p Y is not the number of rows of \p X, or
 *          if the batch count and batch stride of \p X or of \p alpha match none of the batch
 *          modes above.
 *  \retval rocsparse_status_not_implemented if \p X, \p Y and \p alpha do not all have the same
 *          data type.
 *
 *  \par Example
 *  This example transposes a \f$3 \times 2\f$ matrix stored in column order into a
 *  \f$2 \times 3\f$ matrix, scaling it by \f$\alpha = 2\f$ held in host memory.
 *  \snippet example_rocsparse_dnmat_transpose.cpp doc example
 */
ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_transpose(rocsparse_handle            handle,
                                                            rocsparse_const_dnvec_descr alpha,
                                                            rocsparse_const_dnmat_descr X,
                                                            rocsparse_dnmat_descr       Y,
                                                            rocsparse_error*            p_error);
#ifdef __cplusplus
}
#endif

#endif
