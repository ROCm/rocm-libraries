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
#ifndef HIPSPARSE_BSRXMV_H
#define HIPSPARSE_BSRXMV_H

#ifdef __cplusplus
extern "C" {
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication with mask operation using BSR storage format
*
*  \details
*  \p hipsparseXbsrxmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \times \text{blockDim}) \times (nb \times \text{blockDim})\f$
*  modified matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \left( \alpha \cdot op(A) \cdot x + \beta \cdot y \right)\left( \text{mask} \right),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  The \f$\text{mask}\f$ is defined as an array of block row indices.
*  The input sparse matrix is defined with a modified BSR storage format where the beginning and the end of each row
*  is defined with two arrays, \p bsrRowPtr and \p bsr_end_ptr (both of size \p mb), rather the usual \p bsrRowPtr of size \p mb+1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  Currently, \p blockDim == 1 is not supported.
*
*  @param[in]
*  handle      handle to the hipsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  sizeOfMask  number of updated block rows of the array \p y.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Currently, only
*              \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  @param[in]
*  bsrVal      array of \p nnzb blocks of the sparse BSR matrix.
*
*  @param[in]
*  bsrMaskPtr  array of \p sizeOfMask elements that give the indices of the updated block rows.
*
*  @param[in]
*  bsrRowPtr   array of \p mb elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrEndPtr   array of \p mb elements that point to the end of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsrColInd   array of \p nnzb elements containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  blockDim    block dimension of the sparse BSR matrix.
*  @param[in]
*  x           array of \p nb*blockDim elements (\f$op(A) = A\f$) or \p mb*blockDim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*blockDim elements (\f$op(A) = A\f$) or \p nb*blockDim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     HIPSPARSE_STATUS_SUCCESS the operation completed successfully.
*  \retval     HIPSPARSE_STATUS_INVALID_VALUE \p handle, \p mb, \p nb, \p nnzb, \p blockDim, 
*              \p sizeOfMask, \p descr, \p alpha, \p bsrVal, \p bsrRowPtr, \p bsrEndPtr, 
*              \p bsrColInd, \p x, \p beta or \p y is invalid or if \p sizeOfMask is greater 
*              than \p mb.
*  \retval     HIPSPARSE_STATUS_ARCH_MISMATCH the device is not supported.
*  \retval     HIPSPARSE_STATUS_NOT_SUPPORTED
*              \p blockDim==1, \p trans != \ref HIPSPARSE_OPERATION_NON_TRANSPOSE or
*              \ref hipsparseMatrixType_t != \ref HIPSPARSE_MATRIX_TYPE_GENERAL.
*/
/**@{*/
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const float*              alpha,
                                   const hipsparseMatDescr_t descr,
                                   const float*              bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const float*              x,
                                   const float*              beta,
                                   float*                    y);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const double*             alpha,
                                   const hipsparseMatDescr_t descr,
                                   const double*             bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const double*             x,
                                   const double*             beta,
                                   double*                   y);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const hipComplex*         alpha,
                                   const hipsparseMatDescr_t descr,
                                   const hipComplex*         bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const hipComplex*         x,
                                   const hipComplex*         beta,
                                   hipComplex*               y);
DEPRECATED_CUDA_12000("The routine will be removed in CUDA 13")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const hipDoubleComplex*   alpha,
                                   const hipsparseMatDescr_t descr,
                                   const hipDoubleComplex*   bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const hipDoubleComplex*   x,
                                   const hipDoubleComplex*   beta,
                                   hipDoubleComplex*         y);
/**@}*/
#endif

#ifdef __cplusplus
}
#endif

#endif /* HIPSPARSE_BSRXMV_H */
