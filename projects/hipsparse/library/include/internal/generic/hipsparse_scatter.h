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
#ifndef HIPSPARSE_SCATTER_H
#define HIPSPARSE_SCATTER_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Scatter elements from a sparse vector into a dense vector.
*
*  \details
*  \p hipsparseScatter scatters the elements from the sparse vector \f$x\f$ in the dense
*  vector \f$y\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] = x_val[i];
*      }
*  \endcode
*
*  \p hipsparseScatter supports the following uniform precision data types for the sparse and dense vectors \f$x\f$ and
*  \f$y\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="scatter_uniform">Uniform Precisions</caption>
*  <tr><th>X / Y
*  <tr><td>HIP_R_8I
*  <tr><td>HIP_R_16F
*  <tr><td>HIP_R_16BF
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  vecX         sparse vector descriptor \f$x\f$.
*  @param[out]
*  vecY         dense vector descriptor \f$y\f$.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX or \p vecY pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_scatter.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScatter(hipsparseHandle_t          handle,
                                   hipsparseConstSpVecDescr_t vecX,
                                   hipsparseDnVecDescr_t      vecY);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScatter(hipsparseHandle_t     handle,
                                   hipsparseSpVecDescr_t vecX,
                                   hipsparseDnVecDescr_t vecY);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SCATTER_H */
