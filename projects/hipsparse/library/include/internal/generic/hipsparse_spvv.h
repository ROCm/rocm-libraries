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
#ifndef HIPSPARSE_SPVV_H
#define HIPSPARSE_SPVV_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p hipsparseSpVV_bufferSize computes the required user allocated buffer size needed when computing the
*  inner dot product of a sparse vector with a dense vector:
*  \f[
*    \text{result} := op(x) \cdot y,
*  \f]
*
*  \p hipsparseSpVV_bufferSize supports multiple combinations of data types and compute types. See \ref hipsparseSpVV for a complete
*  listing of all the data type and compute type combinations available.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  opX                 sparse vector operation type.
*  @param[in]
*  vecX                sparse vector descriptor.
*  @param[in]
*  vecY                dense vector descriptor.
*  @param[out]
*  result              pointer to the result, can be host or device memory
*  @param[in]
*  computeType         floating point precision for the SpVV computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX, \p vecY, \p result or \p pBufferSizeInBytes
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType is currently not
*               supported.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opX,
                                           hipsparseConstSpVecDescr_t vecX,
                                           hipsparseConstDnVecDescr_t vecY,
                                           void*                      result,
                                           hipDataType                computeType,
                                           size_t*                    pBufferSizeInBytes);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \details
*  \p hipsparseSpVV computes the inner dot product of the sparse vector \f$x\f$ with the
*  dense vector \f$y\f$, such that
*  \f[
*    \text{result} := op(x) \cdot y,
*  \f]
*  with
*  \f[
*    op(x) = \left\{
*    \begin{array}{ll}
*        x,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        \bar{x}, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      result = 0;
*      for(i = 0; i < nnz; ++i)
*      {
*          result += x_val[i] * y[x_ind[i]];
*      }
*  \endcode
*
*  Performing the above operation involves two steps. First, the user calls \p hipsparseSpVV_bufferSize which will return the
*  required temporary buffer size. The user then allocates this buffer. Finally, the user then completes the computation by
*  calling \p hipsparseSpVV with the newly allocated buffer. Once the computation is complete, the user is free to deallocate
*  the buffer.
*
*  \p hipsparseSpVV supports the following uniform and mixed precision data types for the sparse and dense vectors \f$x\f$ and
*  \f$y\f$ and compute types for the scalar \f$result\f$.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spvv_uniform">Uniform Precisions</caption>
*  <tr><th>X / Y / compute_type
*  <tr><td>HIP_R_32F
*  <tr><td>HIP_R_64F
*  <tr><td>HIP_C_32F
*  <tr><td>HIP_C_64F
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="spvv_mixed">Mixed Precisions</caption>
*  <tr><th>X / Y     <th>compute_type
*  <tr><td>HIP_R_8I  <td>HIP_R_32I
*  <tr><td>HIP_R_8I  <td>HIP_R_32F
*  <tr><td>HIP_R_16F <td>HIP_R_32F
*  <tr><td>HIP_R_16BF <td>HIP_R_32F
*  </table>
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  opX             sparse vector operation type.
*  @param[in]
*  vecX            sparse vector descriptor.
*  @param[in]
*  vecY            dense vector descriptor.
*  @param[out]
*  result          pointer to the result, can be host or device memory
*  @param[in]
*  computeType     floating point precision for the SpVV computation.
*  @param[out]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p vecX, \p vecY, \p result or \p externalBuffer
*               pointer is invalid.
*  \retval      HIPSPARSE_STATUS_NOT_SUPPORTED \p computeType is currently not
*               supported.
*
*  \par Example
*  \snippet example_hipsparse_spvv.cpp doc example
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t          handle,
                                hipsparseOperation_t       opX,
                                hipsparseConstSpVecDescr_t vecX,
                                hipsparseConstDnVecDescr_t vecY,
                                void*                      result,
                                hipDataType                computeType,
                                void*                      externalBuffer);
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPVV_H */
