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
#ifndef HIPSPARSE_SPARSE2DENSE_H
#define HIPSPARSE_SPARSE2DENSE_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix to dense matrix conversion
*
*  \details
*  \p hipsparseSparseToDense_bufferSize computes the required user allocated buffer size needed when converting 
*  a sparse matrix to a dense matrix.
*
*  @param[in]
*  handle              handle to the hipsparse library context queue.
*  @param[in]
*  matA                sparse matrix descriptor.
*  @param[in]
*  matB                dense matrix descriptor.
*  @param[in]
*  alg                 algorithm for the sparse to dense computation.
*  @param[out]
*  pBufferSizeInBytes  number of bytes of the temporary storage buffer.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p pBufferSizeInBytes
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseConstSpMatDescr_t  matA,
                                                    hipsparseDnMatDescr_t       matB,
                                                    hipsparseSparseToDenseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseSpMatDescr_t       matA,
                                                    hipsparseDnMatDescr_t       matB,
                                                    hipsparseSparseToDenseAlg_t alg,
                                                    size_t*                     pBufferSizeInBytes);
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix to dense matrix conversion
*
*  \details
*  \p hipsparseSparseToDense converts a sparse matrix to a dense matrix. This routine takes a user allocated buffer 
*  whose size must first be computed by calling \ref hipsparseSparseToDense_bufferSize
*
*  @param[in]
*  handle          handle to the hipsparse library context queue.
*  @param[in]
*  matA            sparse matrix descriptor.
*  @param[in]
*  matB            dense matrix descriptor.
*  @param[in]
*  alg             algorithm for the sparse to dense computation.
*  @param[in]
*  externalBuffer  temporary storage buffer allocated by the user.
*
*  \retval      HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval      HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p matA, \p matB, or \p externalBuffer
*               pointer is invalid.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseDnMatDescr_t       matB,
                                         hipsparseSparseToDenseAlg_t alg,
                                         void*                       externalBuffer);
#elif(CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                         hipsparseSpMatDescr_t       matA,
                                         hipsparseDnMatDescr_t       matB,
                                         hipsparseSparseToDenseAlg_t alg,
                                         void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_SPARSE2DENSE_H */
