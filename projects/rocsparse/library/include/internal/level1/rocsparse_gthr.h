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

#ifndef ROCSPARSE_GTHR_H
#define ROCSPARSE_GTHR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level1_module
*  \brief Gather elements from a dense vector and store them into a sparse vector.
*
*  \details
*  \p rocsparse_gthr gathers the elements that are listed in \p x_ind from the dense
*  vector \f$y\f$ and stores them in the sparse vector \f$x\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_val[i] = y[x_ind[i]];
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
*  @param[in]
*  y           array of values in dense format.
*  @param[out]
*  x_val       array of \p nnz elements containing the values of \f$x\f$.
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
*
*  \par Example
*  \code{.c}
*      // Number of non-zeros of the sparse vector
*      rocsparse_int nnz = 3;
*
*      // Sparse index vector
*      rocsparse_int hx_ind[3] = {0, 3, 5};
*
*      // Sparse value vector
*      float hx_val[3];
*
*      // Dense vector
*      float hy[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
*
*      // Index base
*      rocsparse_index_base idx_base = rocsparse_index_base_zero;
*
*      // Offload data to device
*      rocsparse_int* dx_ind;
*      float*         dx_val;
*      float*         dy;
*
*      hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz);
*      hipMalloc((void**)&dx_val, sizeof(float) * nnz);
*      hipMalloc((void**)&dy, sizeof(float) * 9);
*
*      hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(float) * 9, hipMemcpyHostToDevice);
*
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // Call sgthr
*      rocsparse_sgthr(handle, nnz, dy, dx_val, dx_ind, idx_base);
*
*      // Copy result back to host
*      hipMemcpy(hx_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*
*      // Clear rocSPARSE
*      rocsparse_destroy_handle(handle);
*
*      // Clear device memory
*      hipFree(dx_ind);
*      hipFree(dx_val);
*      hipFree(dy);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgthr(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const float*         y,
                                 float*               x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgthr(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const double*        y,
                                 double*              x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgthr(rocsparse_handle               handle,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* y,
                                 rocsparse_float_complex*       x_val,
                                 const rocsparse_int*           x_ind,
                                 rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgthr(rocsparse_handle                handle,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* y,
                                 rocsparse_double_complex*       x_val,
                                 const rocsparse_int*            x_ind,
                                 rocsparse_index_base            idx_base);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif // ROCSPARSE_GTHR_H
