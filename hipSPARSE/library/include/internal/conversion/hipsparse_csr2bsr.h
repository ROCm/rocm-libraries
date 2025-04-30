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
#ifndef HIPSPARSE_CSR2BSR_H
#define HIPSPARSE_CSR2BSR_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  BSR matrix given a sparse CSR matrix as input.
*
*  \details
*  Consider the matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7 \\
*    1 & 2 & 3 & 4
*    \end{bmatrix}
*  \f]
*
*  stored as a sparse CSR matrix. This function computes both the BSR row pointer array as well as the total number 
*  of non-zero blocks that results when converting the CSR matrix to the BSR format. Assuming a block dimension of 2, 
*  the above matrix once converted to BSR format looks like:
*
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c} 
*       1 & 0 \\ 
*       3 & 4 
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 0 
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c} 
*       5 & 0 \\ 
*       1 & 2 
*      \end{array} & 
*      \begin{array}{c c} 
*       6 & 7 \\ 
*       3 & 4 
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  and the resulting BSR row pointer array and total non-zero blocks once \p hipsparseXcsr2bsrNnz has been called:
*
*  \f[
*    \begin{align}
*    \text{bsrRowPtrC} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \text{bsrNnzb} &= 4
*    \end{align}
*  \f]
*
*  In general, when converting a CSR matrix of size \p m x \p n to a BSR matrix, the resulting BSR matrix will have size 
*  \p mb x \p nb where \p mb and \p nb equal:
*
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / blockDim + 1} \\
*    \text{nb} &= \text{(n - 1) / blockDim + 1}
*    \end{align}
*  \f]
*
*  In particular, it may be the case that \p blockDim does not divide evenly into \p m and/or \p n. In these cases, the 
*  CSR matrix is expanded in size in order to fit full BSR blocks. For example, using the original CSR matrix and block 
*  dimension 3 instead of 2, the function \p hipsparseXcsr2bsrNnz computes the BSR row pointer array and total number of 
*  non-zero blocks for the BSR matrix:
*
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c c} 
*       1 & 0 & 0 \\ 
*       3 & 4 & 0 \\
*       5 & 0 & 6
*      \end{array} & 
*      \begin{array}{c c c} 
*       2 & 0 & 0 \\ 
*       0 & 0 & 0 \\
*       7 & 0 & 0
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c c} 
*       1 & 2 & 3 \\ 
*       0 & 0 & 0 \\
*       0 & 0 & 0
*      \end{array} & 
*      \begin{array}{c c c} 
*       4 & 0 & 0 \\ 
*       0 & 0 & 0 \\
*       0 & 0 & 0
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  See hipsparseScsr2bsr() for full code example.
*
*  \note
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dirA        direction that specified whether to count nonzero elements by \ref HIPSPARSE_DIRECTION_ROW or by
*              \ref HIPSPARSE_DIRECTION_COLUMN.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  descrA      descriptor of the sparse CSR matrix. Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrRowPtrA  integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*  @param[in]
*  csrColIndA  integer array of the column indices for each non-zero element in the CSR matrix
*  @param[in]
*  blockDim    the block dimension of the BSR matrix. Between \f$1\f$ and \f$\min(m, n)\f$
*  @param[in]
*  descrC      descriptor of the sparse BSR matrix. Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrRowPtrC  integer array containing \p mb+1 elements that point to the start of each block row of the BSR matrix
*  @param[out]
*  bsrNnzb     total number of nonzero elements in device or host memory.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p blockDim, \p csrRowPtrA, \p csrColIndA, 
*              \p bsrRowPtrC or \p bsrNnzb pointer is invalid.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2bsrNnz(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dirA,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       int                       blockDim,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      bsrRowPtrC,
                                       int*                      bsrNnzb);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse BSR matrix
*
*  \details
*  \p hipsparseXcsr2bsr completes the conversion of a CSR matrix into a BSR matrix. 
*  It is assumed, that \p bsrValC, \p bsrColIndC and \p bsrRowPtrC are allocated. The 
*  allocation size for \p bsrRowPtr is computed as \p mb+1 where \p mb is the number of 
*  block rows in the BSR matrix defined as:
*
*  \f[
*    \begin{align}
*    \text{mb} &= \text{(m - 1) / blockDim + 1}
*    \end{align}
*  \f]
*
*  The allocation size for \p bsrColIndC, i.e. \p bsrNnzb, is computed using
*  \ref hipsparseXcsr2bsrNnz() which also fills the \p bsrRowPtrC array. The allocation size 
*  for \p bsrValC is then equal to:
*
*  \f[
*    \text{bsrNnzb * blockDim * blockDim}
*  \f]
*
*  For example, given the CSR matrix:
*  \f[
*    \begin{bmatrix}
*    1 & 0 & 0 & 2 \\
*    3 & 4 & 0 & 0 \\
*    5 & 0 & 6 & 7 \\
*    1 & 2 & 3 & 4
*    \end{bmatrix}
*  \f]
*
*  The resulting BSR matrix using block dimension 2 would look like:
*  \f[
*   \left[ 
*    \begin{array}{c | c} 
*      \begin{array}{c c} 
*       1 & 0 \\ 
*       3 & 4 
*      \end{array} & 
*      \begin{array}{c c} 
*       0 & 2 \\ 
*       0 & 0 
*      \end{array} \\ 
*    \hline 
*      \begin{array}{c c} 
*       5 & 0 \\ 
*       1 & 2 
*      \end{array} & 
*      \begin{array}{c c} 
*       6 & 7 \\ 
*       3 & 4 
*      \end{array} \\ 
*   \end{array} 
*  \right] 
*  \f]
*
*  The call to \ref hipsparseXcsr2bsrNnz results in the BSR row pointer array:
*  \f[
*    \begin{align}
*    \text{bsrRowPtrC} &= \begin{bmatrix} 0 & 2 & 4 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  and the call to \p hipsparseXcsr2bsr completes the conversion resulting in the BSR column indices and values arrays:
*  \f[
*    \begin{align}
*    \text{bsrColIndC} &= \begin{bmatrix} 0 & 1 & 0 & 1 \end{bmatrix} \\
*    \text{bsrValC} &= \begin{bmatrix} 1 & 0 & 3 & 4 & 0 & 2 & 0 & 0 & 5 & 0 & 1 & 2 & 6 & 7 & 3 & 4 \end{bmatrix} \\
*    \end{align}
*  \f]
*
*  The \p dirA parameter determines the order of the BSR block values. The example above uses row order. Using column ordering
*  would result instead in the BSR values array:
*
*  \f[
*    \text{bsrValC} &= \begin{bmatrix} 1 & 3 & 0 & 4 & 0 & 0 & 2 & 0 & 5 & 1 & 0 & 2 & 6 & 3 & 7 & 4 \end{bmatrix} \\
*  \f]
*
*  \note
*  \p hipsparseXcsr2bsr requires extra temporary storage that is allocated internally if
*  \p blockDim>16
*
*  @param[in]
*  handle       handle to the hipsparse library context queue.
*  @param[in]
*  dirA         the storage format of the blocks, \ref HIPSPARSE_DIRECTION_ROW or \ref HIPSPARSE_DIRECTION_COLUMN
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  descrA       descriptor of the sparse CSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  csrValA      array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csrRowPtrA   array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csrColIndA   array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  blockDim     size of the blocks in the sparse BSR matrix.
*  @param[in]
*  descrC       descriptor of the sparse BSR matrix. Currently, only
*               \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[out]
*  bsrValC      array of \p nnzb*blockDim*blockDim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsrRowPtrC   array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsrColIndC   array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p m, \p n, \p blockDim, \p bsrValC, \p bsrRowPtrC, 
*              \p bsrColIndC, \p csrValA, \p csrRowPtrA or \p csrColIndA pointer is invalid.
*
*  \par Example
*  \code{.c}
*    // hipSPARSE handle
*    hipsparseHandle_t handle;
*    hipsparseCreate(&handle);
*
*    // Matrix descriptor
*    hipsparseMatDescr_t csr_descr;
*    hipsparseCreateMatDescr(&csr_descr);
*
*    hipsparseMatDescr_t bsr_descr;
*    hipsparseCreateMatDescr(&bsr_descr);
*
*    // Sparse matrix in CSR format
*    //     1 2 0 3 0
*    // A = 0 4 5 0 0
*    //     6 0 0 7 8
*    int hcsrRowPtr[4] = {0, 3, 5, 8};
*    int hcsrColInd[8] = {0, 1, 3, 1, 2, 0, 3, 4};
*    float hcsrVal[8]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}; 
*
*    int m         = 3;
*    int n         = 5;
*    int nnz       = 8;
*    int blockDim  = 3;
*    hipsparseDirection_t dir = HIPSPARSE_DIRECTION_ROW;
*
*    int mb = (m + blockDim - 1) / blockDim;
*    int nb = (n + blockDim - 1) / blockDim;
*
*    int* dcsrRowPtr = nullptr;
*    int* dcsrColInd = nullptr;
*    float* dcsrVal = nullptr;
*    hipMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1));
*    hipMalloc((void**)&dcsrColInd, sizeof(int) * nnz);
*    hipMalloc((void**)&dcsrVal, sizeof(float) * nnz);
*
*    hipMemcpy(dcsrRowPtr, hcsrRowPtr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
*    hipMemcpy(dcsrColInd, hcsrColInd, sizeof(int) * nnz, hipMemcpyHostToDevice);
*    hipMemcpy(dcsrVal, hcsrVal, sizeof(float) * nnz, hipMemcpyHostToDevice);
*
*    int* dbsrRowPtr = nullptr;
*    hipMalloc((void**)&dbsrRowPtr, sizeof(int) * (mb + 1));
*
*    int nnzb;
*    hipsparseXcsr2bsrNnz(handle, dir, m, n, csr_descr, dcsrRowPtr, dcsrColInd, blockDim, bsr_descr, dbsrRowPtr, &nnzb);
*
*    int* dbsrColInd = nullptr;
*    float* dbsrVal = nullptr;
*    hipMalloc((void**)&dbsrColInd, sizeof(int) * nnzb);
*    hipMalloc((void**)&dbsrVal, sizeof(float) * blockDim * blockDim * nnzb);
*
*    hipsparseScsr2bsr(handle, dir, m, n, csr_descr, dcsrVal, dcsrRowPtr, dcsrColInd, blockDim, bsr_descr, dbsrVal, dbsrRowPtr, dbsrColInd);
*
*    hipFree(dcsrRowPtr);
*    hipFree(dcsrColInd);
*    hipFree(dcsrVal);
*   
*    hipFree(dbsrRowPtr);
*    hipFree(dbsrColInd);
*    hipFree(dbsrVal);
*
*    hipsparseDestroyMatDescr(csr_descr);
*    hipsparseDestroyMatDescr(bsr_descr);
*    hipsparseDestroy(handle);
*  \endcode
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_CSR2BSR_H */
