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

#ifndef ROCSPARSE_ROTI_H
#define ROCSPARSE_ROTI_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

extern "C" {

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
*  \code{.c}
*      // Number of non-zeros of the sparse vector
*      rocsparse_int nnz = 3;
*
*      // Sparse index vector
*      rocsparse_int hx_ind[3] = {0, 3, 5};
*
*      // Sparse value vector
*      float hx_val[3] = {1.0f, 2.0f, 3.0f};
*
*      // Dense vector
*      float hy[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
*
*      // c and s
*      float c = 3.7;
*      float s = 1.3;
*
*      // Index base
*      rocsparse_index_base idx_base = rocsparse_index_base_zero;
*
*      // Offload data to device
*      rocsparse_int* dx_ind;
*      float*        dx_val;
*      float*        dy;
*
*      hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz);
*      hipMalloc((void**)&dx_val, sizeof(float) * nnz);
*      hipMalloc((void**)&dy, sizeof(float) * 9);
*
*      hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dx_val, hx_val, sizeof(float) * nnz, hipMemcpyHostToDevice);
*      hipMemcpy(dy, hy, sizeof(float) * 9, hipMemcpyHostToDevice);
*
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // Call sroti
*      rocsparse_sroti(handle, nnz, dx_val, dx_ind, dy, &c, &s, idx_base);
*
*      // Copy result back to host
*      hipMemcpy(hx_val, dx_val, sizeof(float) * nnz, hipMemcpyDeviceToHost);
*      hipMemcpy(hy, dy, sizeof(float) * 9, hipMemcpyDeviceToHost);
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
}

#endif // ROCSPARSE_ROTI_H
