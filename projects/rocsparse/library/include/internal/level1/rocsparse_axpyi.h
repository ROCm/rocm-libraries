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

#ifndef ROCSPARSE_AXPYI_H
#define ROCSPARSE_AXPYI_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

extern "C" {

/*
* ===========================================================================
*    level 1 SPARSE
* ===========================================================================
*/

/*! \ingroup level1_module
*  \brief Scale a sparse vector and add it to a dense vector.
*
*  \details
*  \p rocsparse_axpyi multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
*  adds the result to the dense vector \f$y\f$, such that
*
*  \f[
*      y := y + \alpha \cdot x
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] = y[x_ind[i]] + alpha * x_val[i];
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
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  x_val       array of \p nnz elements containing the values of \f$x\f$.
*  @param[in]
*  x_ind       array of \p nnz elements containing the indices of the non-zero
*              values of \f$x\f$.
*  @param[inout]
*  y           array of values in dense format.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_value \p idx_base is invalid.
*  \retval rocsparse_status_invalid_size \p nnz is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p x_val, \p x_ind or \p y pointer
*          is invalid.
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
*      // Scalar alpha
*      float alpha = 3.7f;
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
*      // Call saxpyi to perform y = y + alpha * x
*      rocsparse_saxpyi(handle, nnz, &alpha, dx_val, dx_ind, dy, idx_base);
*
*      // Copy result back to host
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
rocsparse_status rocsparse_saxpyi(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  const float*         alpha,
                                  const float*         x_val,
                                  const rocsparse_int* x_ind,
                                  float*               y,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_daxpyi(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  const double*        alpha,
                                  const double*        x_val,
                                  const rocsparse_int* x_ind,
                                  double*              y,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_caxpyi(rocsparse_handle               handle,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_float_complex* x_val,
                                  const rocsparse_int*           x_ind,
                                  rocsparse_float_complex*       y,
                                  rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zaxpyi(rocsparse_handle                handle,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_double_complex* x_val,
                                  const rocsparse_int*            x_ind,
                                  rocsparse_double_complex*       y,
                                  rocsparse_index_base            idx_base);
/**@}*/
}

#endif // ROCSPARSE_AXPYI_H
