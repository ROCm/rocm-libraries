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
#ifndef HIPSPARSE_AXPBY_H
#define HIPSPARSE_AXPBY_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Scale a sparse vector and add it to a scaled dense vector.
*
*  \details
*  \p hipsparseAxpby multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
*  adds the result to the dense vector \f$y\f$ that is multiplied with scalar
*  \f$\beta\f$, such that
*
*  \f[
*      y := \alpha \cdot x + \beta \cdot y
*  \f]
*
*  \code{.c}
*      for(i = 0; i < size; ++i)
*      {
*          y[i] = beta * y[i]
*      }
*      for(i = 0; i < nnz; ++i)
*      {
*          y[xInd[i]] += alpha * xVal[i]
*      }
*  \endcode
*
*  \p hipsparseAxpby supports the following precision data types for the sparse and dense vectors \f$x\f$ and
*  \f$y\f$ and compute types for the scalars \f$\alpha\f$ and \f$\beta\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="axpby_uniform">Uniform Precisions</caption>
*  <tr><th>X / Y / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="axpby_mixed">Mixed Precisions</caption>
*  <tr><th>X / Y      <th>compute_type
*  <tr><td>HIP_R_16F  <td>HIP_R_32F
*  <tr><td>HIP_R_16BF <td>HIP_R_32F
*  </table>
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  vecX        sparse matrix descriptor.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  vecY        dense matrix descriptor.
*
*  \retval HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p alpha, \p vecX, \p beta or \p vecY pointer is
*          invalid.
*
*  \par Example
*  \snippet example_hipsparse_axpby.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t          handle,
                                 const void*                alpha,
                                 hipsparseConstSpVecDescr_t vecX,
                                 const void*                beta,
                                 hipsparseDnVecDescr_t      vecY);
#elif(CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t     handle,
                                 const void*           alpha,
                                 hipsparseSpVecDescr_t vecX,
                                 const void*           beta,
                                 hipsparseDnVecDescr_t vecY);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_AXPBY_H */
