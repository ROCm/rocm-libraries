/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_ROTI_H
#define ROCSPARSE_ROTI_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level1_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \p rocsparse_roti applies the Givens rotation matrix \f$G\f$ to the sparse vector
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
*  nnz         number of non-zero entries of \f$x\f$.
*  @param[inout]
*  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
*  @param[in]
*  x_ind       array of \p nnz elements containing the indices of the non-zero
*              values of \f$x\f$.
*  @param[inout]
*  y           array of values in dense format.
*  @param[in]
*  c           pointer to the cosine element of \f$G\f$, can be on host or device.
*  @param[in]
*  s           pointer to the sine element of \f$G\f$, can be on host or device.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_value \p idx_base is invalid.
*  \retval     rocsparse_status_invalid_size \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p c, \p s, \p x_val, \p x_ind or \p y
*              pointer is invalid.
*
*  \par Example
*  \snippet example_rocsparse_roti.cpp doc example
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sroti(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 float*               x_val,
                                 const rocsparse_int* x_ind,
                                 float*               y,
                                 const float*         c,
                                 const float*         s,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_droti(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 double*              x_val,
                                 const rocsparse_int* x_ind,
                                 double*              y,
                                 const double*        c,
                                 const double*        s,
                                 rocsparse_index_base idx_base);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif // ROCSPARSE_ROTI_H
