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
#ifndef HIPSPARSE_GATHER_H
#define HIPSPARSE_GATHER_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Gather elements from a dense vector and store them into a sparse vector.
*
*  \details
*  \p hipsparseGather gathers the elements from the dense vector \f$y\f$ and stores
*  them in the sparse vector \f$x\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_val[i] = y[x_ind[i]];
*      }
*  \endcode
*
*  \p hipsparseGather supports the following uniform precision data types for the sparse and dense vectors \f$x\f$ and
*  \f$y\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="gather_uniform">Uniform Precisions</caption>
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
*  vecY         dense vector descriptor \f$y\f$.
*  @param[out]
*  vecX         sparse vector descriptor \f$x\f$.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX or \p vecY pointer is invalid.
*
*  \par Example
*  \snippet example_hipsparse_gather.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGather(hipsparseHandle_t          handle,
                                  hipsparseConstDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t      vecX);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGather(hipsparseHandle_t     handle,
                                  hipsparseDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t vecX);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_GATHER_H */
