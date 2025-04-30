/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_DOTCI_H
#define ROCSPARSE_DOTCI_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

extern "C" {

/*! \ingroup level1_module
*  \brief Compute the dot product of a complex conjugate sparse vector with a dense
*  vector.
*
*  \details
*  \p rocsparse_dotci computes the dot product of the complex conjugate sparse vector
*  \f$x\f$ with the dense vector \f$y\f$, such that
*  \f[
*    \text{result} := \bar{x}^H y
*  \f]
*
*  \code{.c}
*      result = 0
*      for(i = 0; i < nnz; ++i)
*      {
*          result += conj(x_val[i]) * y[x_ind[i]];
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
*  nnz         number of non-zero entries of vector \f$x\f$.
*  @param[in]
*  x_val       array of \p nnz values.
*  @param[in]
*  x_ind       array of \p nnz elements containing the indices of the non-zero
*              values of \f$x\f$.
*  @param[in]
*  y           array of values in dense format.
*  @param[out]
*  result      pointer to the result, can be host or device memory
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value \p idx_base is invalid.
*  \retval rocsparse_status_invalid_size \p nnz is invalid.
*  \retval rocsparse_status_invalid_pointer \p x_val, \p x_ind, \p y or \p result
*          pointer is invalid.
*  \retval rocsparse_status_memory_error the buffer for the dot product reduction
*          could not be allocated.
*  \retval rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdotci(rocsparse_handle               handle,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* x_val,
                                  const rocsparse_int*           x_ind,
                                  const rocsparse_float_complex* y,
                                  rocsparse_float_complex*       result,
                                  rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdotci(rocsparse_handle                handle,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* x_val,
                                  const rocsparse_int*            x_ind,
                                  const rocsparse_double_complex* y,
                                  rocsparse_double_complex*       result,
                                  rocsparse_index_base            idx_base);
/**@}*/
}

#endif // ROCSPARSE_DOTCI_H
