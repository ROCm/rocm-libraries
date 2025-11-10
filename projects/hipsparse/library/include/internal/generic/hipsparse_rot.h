/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef HIPSPARSE_ROT_H
#define HIPSPARSE_ROT_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \p hipsparseRot applies the Givens rotation matrix \f$G\f$ to the sparse vector
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
*  \p hipsparseRot supports the following uniform precision data types for the sparse and dense vectors \f$x\f$ and
*  \f$y\f$ and compute types for the scalars \f$c\f$ and \f$s\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="rot_uniform">Uniform Precisions</caption>
*  <tr><th>X / Y / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  c_coeff     pointer to the cosine element of \f$G\f$, can be on host or device.
*  @param[in]
*  s_coeff     pointer to the sine element of \f$G\f$, can be on host or device.
*  @param[inout]
*  vecX        sparse vector descriptor \f$x\f$.
*  @param[inout]
*  vecY        dense vector descriptor \f$y\f$.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p c_coeff, \p s_coeff, \p vecX or \p vecY pointer is
*              invalid.
*
*  \par Example
*  \snippet example_hipsparse_rot.cpp doc example
*/
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 11000 && CUDART_VERSION < 13000))
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseRot(hipsparseHandle_t     handle,
                               const void*           c_coeff,
                               const void*           s_coeff,
                               hipsparseSpVecDescr_t vecX,
                               hipsparseDnVecDescr_t vecY);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_ROT_H */
