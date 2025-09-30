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
#ifndef HIPSPARSE_PRUNE_CSR2CSR_BY_PRECENTAGE_H
#define HIPSPARSE_PRUNE_CSR2CSR_BY_PRECENTAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrByPercentage_bufferSize returns the size of the temporary buffer that
 *  is required by \ref hipsparseSpruneCsr2csrNnzByPercentage "hipsparseXpruneCsr2csrNnzByPercentage()".
 *  The temporary storage buffer must be allocated by the user.
 *
 *  @param[in]
 *  handle              handle to the hipsparse library context queue.
 *  @param[in]
 *  m                   number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                   number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA                number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA              descriptor of the sparse CSR matrix A. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA             array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA          array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage          \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC              descriptor of the sparse CSR matrix C. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC             array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC          array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info                prune info structure.
 *  @param[out]
 *  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnzByPercentage(),
 *                      hipsparseDpruneCsr2csrNnzByPercentage(), hipsparseSpruneCsr2csrByPercentage(),
 *                      and hipsparseDpruneCsr2csrByPercentage().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const float*              csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                float      percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const float*              csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     pBufferSizeInBytes);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const double*             csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                double     percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const double*             csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrByPercentage_bufferSizeExt returns the size of the temporary buffer that
 *  is required by \ref hipsparseSpruneCsr2csrNnzByPercentage "hipsparseXpruneCsr2csrNnzByPercentage()".
 *  The temporary storage buffer must be allocated by the user.
 *
 *  @param[in]
 *  handle              handle to the hipsparse library context queue.
 *  @param[in]
 *  m                   number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n                   number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnzA                number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  descrA              descriptor of the sparse CSR matrix A. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValA             array of \p nnzA elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csrRowPtrA          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix A.
 *  @param[in]
 *  csrColIndA          array of \p nnzA elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage          \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC              descriptor of the sparse CSR matrix C. Currently, only
 *                      \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[in]
 *  csrValC             array of \p nnzC elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC          array of \p m+1 elements that point to the start of every row of the
 *                      sparse CSR matrix C.
 *  @param[in]
 *  csrColIndC          array of \p nnzC elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info                prune info structure.
 *  @param[out]
 *  pBufferSizeInBytes  number of bytes of the temporary storage buffer required by hipsparseSpruneCsr2csrNnzByPercentage(),
 *                      hipsparseDpruneCsr2csrNnzByPercentage(), hipsparseSpruneCsr2csrByPercentage(),
 *                      and hipsparseDpruneCsr2csrByPercentage().
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle or \p pBufferSizeInBytes pointer is invalid.
 *  \retval     HIPSPARSE_STATUS_INTERNAL_ERROR an internal error occurred.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const float* csrValA,
                                                                   const int*   csrRowPtrA,
                                                                   const int*   csrColIndA,
                                                                   float        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const float* csrValC,
                                                                   const int*   csrRowPtrC,
                                                                   const int*   csrColIndC,
                                                                   pruneInfo_t  info,
                                                                   size_t*      pBufferSizeInBytes);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const double* csrValA,
                                                                   const int*    csrRowPtrA,
                                                                   const int*    csrColIndA,
                                                                   double        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const double* csrValC,
                                                                   const int*    csrRowPtrC,
                                                                   const int*    csrColIndC,
                                                                   pruneInfo_t   info,
                                                                   size_t* pBufferSizeInBytes);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrNnzByPercentage computes the number of nonzero elements per row and the total
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
 *  percentage         \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC             descriptor of the sparse CSR matrix C. Currently, only
 *                     \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrRowPtrC         array of \p m+1 elements that point to the start of every row of the
 *                     sparse CSR matrix C.
 *  @param[out]
 *  nnzTotalDevHostPtr total number of nonzero elements in device or host memory.
 *  @param[in]
 *  info               prune info structure.
 *  @param[out]
 *  buffer             buffer allocated by the user whose size is determined by calling
 *                     \ref hipsparseSpruneCsr2csrByPercentage_bufferSize "hipsparseXpruneCsr2csrByPercentage_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p percentage, \p descrA, \p descrC,
 *              \p info, \p csrValA, \p csrRowPtrA, \p csrColIndA, \p csrRowPtrC, \p nnzTotalDevHostPtr or \p buffer
 *              pointer is invalid.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const float*              csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        float                     percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const double*             csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        double                    percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls \ref hipsparseSpruneCsr2csr_bufferSize "hipsparseXpruneCsr2csr_bufferSize()" to
 *  determine the size of the buffer used by \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()" and
 *  \p hipsparseXpruneCsr2csr() which the user then allocates. The user then allocates \p csrRowPtrC to have
 *  \p m+1 elements and then calls \ref hipsparseSpruneCsr2csrNnz "hipsparseXpruneCsr2csrNnz()" which fills
 *  in the \p csrRowPtrC array stores then number of elements that are larger than the pruning \p threshold
 *  in \p nnzTotalDevHostPtr. The user then calls \p hipsparseXpruneCsr2csr() to complete the conversion. It
 *  is executed asynchronously with respect to the host and may return control to the application on the host
 *  before the entire result is ready.
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
 *  percentage    \p percentage>=0 and \p percentage<=100.
 *  @param[in]
 *  descrC        descriptor of the sparse CSR matrix C. Currently, only
 *                \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
 *  @param[out]
 *  csrValC       array of \p nnz_C elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csrRowPtrC    array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[out]
 *  csrColIndC    array of \p nnz_C elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info          prune info structure.
 *  @param[in]
 *  buffer        buffer allocated by the user whose size is determined by calling
 *                \ref hipsparseSpruneCsr2csrByPercentage_bufferSize "hipsparseXpruneCsr2csrByPercentage_bufferSize()".
 *
 *  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
 *  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p nnzA, \p percentage, \p descrA, \p descrC, \p info,
 *              \p csrValA, \p csrRowPtrA, \p csrColIndA, \p csrValC, \p csrRowPtrC, \p csrColIndC or \p buffer pointer is
 *              invalid.
 */
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const float*              csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     float                     percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     float*                    csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer);

DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const double*             csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     double                    percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     double*                   csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_PRUNE_CSR2CSR_BY_PRECENTAGE_H */
