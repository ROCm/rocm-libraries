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

#ifndef ROCSPARSE_ROT_H
#define ROCSPARSE_ROT_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup generic_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \p rocsparse_rot applies the Givens rotation matrix \f$G\f$ to the sparse vector
*  \f$x\f$ and the dense vector \f$y\f$, where
*  \f[
*    G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_tmp = x_val[i];
*          y_tmp = y[x_ind[i]];
*
*          x_val[i]    = c * x_tmp + s * y_tmp;
*          y[x_ind[i]] = c * y_tmp - s * x_tmp;
*      }
*  \endcode
*
*  \p rocsparse_rot supports the following uniform precision data types for the sparse and dense vectors x and
*  y and compute types for the scalars \f$c\f$ and \f$s\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="rot_uniform">Uniform Precisions</caption>
*  <tr><th>X / Y / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
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
*  c           pointer to the cosine element of \f$G\f$, can be on host or device.
*  @param[in]
*  s           pointer to the sine element of \f$G\f$, can be on host or device.
*  @param[inout]
*  x           sparse vector \f$x\f$.
*  @param[inout]
*  y           dense vector \f$y\f$.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p c, \p s, \p x or \p y pointer is
*              invalid.
*  \par Example
*  \snippet example_rocsparse_rot.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_rot(rocsparse_handle      handle,
                               const void*           c,
                               const void*           s,
                               rocsparse_spvec_descr x,
                               rocsparse_dnvec_descr y);
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_ROT_H */
