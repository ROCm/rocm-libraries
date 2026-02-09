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
#ifndef HIPSPARSE_PRUNE_CSR2CSR_H
#define HIPSPARSE_PRUNE_CSR2CSR_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSize returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
 *  temporary storage buffer must be allocated by the user.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  m                  number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                  number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA               number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA             descriptor of the sparse CSR matrix A. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA            array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA         array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold          pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC            array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC         array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[out]
 *  pBufferSizeInBytes number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnz(),
 *                     hipsparseDpruneCsr2csrNnz(), hipsparseSpruneCsr2csr(), and hipsparseDpruneCsr2csr().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const float*              csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const float*              threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const float*              csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const double*             csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const double*             threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const double*             csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   pBufferSizeInBytes);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSizeExt returns the size of the temporary buffer that
 *  is required by \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()" and
 *  \ref hipsparseSpruneCsr2csr "hipsparseXpruneCsr2csr()". The temporary storage buffer
 *  must be allocated by the user.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  m                  number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                  number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA               number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA             descriptor of the sparse CSR matrix A. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA            array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA         array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold          pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC            array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC         array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[out]
 *  pBufferSizeInBytes number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnz(),
 *                     hipsparseDpruneCsr2csrNnz(), hipsparseSpruneCsr2csr(), and hipsparseDpruneCsr2csr().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const float*              csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const float*              threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const float*              csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t* pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const double*             csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const double*             threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const double*             csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrNnz computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
 *
 *  @param[in]
 *  handle             handle to the hipsparse library context queue.
 *  @param[in]
 *  m                  number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                  number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA               number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA             descriptor of the sparse CSR matrix A. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA            array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA         array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold          pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[out]
 *  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
 *  @param[out]
 *  buffer             buffer allocated by the user whose size is determined by calling \ref hipsparseSpruneCsr2csr_bufferSize
 *                     "hipsparseXpruneCsr2csr_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p threshold, \p descrA,
 *              \p descrC, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p csrRowPtrC, \p nnzTotalDevHostPtr
 *              or \p buffer pointer is invalid.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const float*              threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const double*             threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls \ref hipsparseSpruneCsr2csr_bufferSize "hipsparseXpruneCsr2csr_bufferSize()" to
 *  determine the size of the buffer used by \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()"
 *  and \p hipsparseXpruneCsr2csr() which the user then allocates. The user then allocates \p csrRowPtrC to
 *  have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills in the \p csrRowPtrC array
 *  stores then number of elements that are larger than the pruning \p threshold in \p nnzTotalDevHostPtr.
 *  The user then calls \p hipsparseXpruneCsr2csr() to complete the conversion. It is executed asynchronously
 *  with respect to the host and may return control to the application on the host before the entire result is ready.
 *
 *  @param[in]
 *  handle        handle to the hipsparse library context queue.
 *  @param[in]
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA          number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA        descriptor of the sparse CSR matrix A. Currently, only
 *                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA       array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA    array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA    array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  threshold     pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  descrC        descriptor of the sparse CSR matrix C. Currently, only
 *                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrValC       array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC    array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[out]
 *  csrColIndC    array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  buffer        buffer allocated by the user whose size is determined by calling \ref hipsparseSpruneCsr2csr_bufferSize
 *                "hipsparseXpruneCsr2csr_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p threshold, \p descrA, \p descrC, \p csrValA,
 *              \p csrRowPtrA, \p csrcolindA, \p csrvalC, \p csrrowptrC, \p csrcolIndC or \p buffer pointer is invalid.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const float*              threshold,
                                         const hipsparseMatDescr_t descrC,
                                         float*                    csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const double*             threshold,
                                         const hipsparseMatDescr_t descrC,
                                         double*                   csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_PRUNE_CSR2CSR_H */
