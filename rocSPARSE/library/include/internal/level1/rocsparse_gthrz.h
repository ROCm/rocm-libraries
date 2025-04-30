/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_GTHRZ_H
#define ROCSPARSE_GTHRZ_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

extern "C" {

/*! \ingroup level1_module
*  \brief Gather and zero out elements from a dense vector and store them into a sparse
*  vector.
*
*  \details
*  \p rocsparse_gthrz gathers the elements that are listed in \p x_ind from the dense
*  vector \f$y\f$ and stores them in the sparse vector \f$x\f$. The gathered elements
*  in \f$y\f$ are replaced by zero.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_val[i]    = y[x_ind[i]];
*          y[x_ind[i]] = 0;
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
*  y           array of values in dense format.
*  @param[out]
*  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
*  @param[in]
*  x_ind       array of \p nnz elements containing the indices of the non-zero
*              values of \f$x\f$.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_value \p idx_base is invalid.
*  \retval     rocsparse_status_invalid_size \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p y, \p x_val or \p x_ind pointer is
*              invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgthrz(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  float*               y,
                                  float*               x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgthrz(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  double*              y,
                                  double*              x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgthrz(rocsparse_handle         handle,
                                  rocsparse_int            nnz,
                                  rocsparse_float_complex* y,
                                  rocsparse_float_complex* x_val,
                                  const rocsparse_int*     x_ind,
                                  rocsparse_index_base     idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgthrz(rocsparse_handle          handle,
                                  rocsparse_int             nnz,
                                  rocsparse_double_complex* y,
                                  rocsparse_double_complex* x_val,
                                  const rocsparse_int*      x_ind,
                                  rocsparse_index_base      idx_base);
/**@}*/
}

#endif // ROCSPARSE_GTHRZ_H
