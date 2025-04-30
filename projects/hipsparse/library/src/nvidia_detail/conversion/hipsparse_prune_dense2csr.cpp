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
#include "hipsparse.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#include "../utility.h"

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const float*              A,
                                                      int                       lda,
                                                      const float*              threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const float*              csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const double*             A,
                                                      int                       lda,
                                                      const double*             threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const double*             csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const float*              A,
                                                         int                       lda,
                                                         const float*              threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const float*              csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const double*             A,
                                                         int                       lda,
                                                         const double*             threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const double*             csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const float*              A,
                                              int                       lda,
                                              const float*              threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrNnz((cusparseHandle_t)handle,
                                   m,
                                   n,
                                   A,
                                   lda,
                                   threshold,
                                   (const cusparseMatDescr_t)descr,
                                   csrRowPtr,
                                   nnzTotalDevHostPtr,
                                   buffer));
}

hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const double*             A,
                                              int                       lda,
                                              const double*             threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrNnz((cusparseHandle_t)handle,
                                   m,
                                   n,
                                   A,
                                   lda,
                                   threshold,
                                   (const cusparseMatDescr_t)descr,
                                   csrRowPtr,
                                   nnzTotalDevHostPtr,
                                   buffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const float*              A,
                                           int                       lda,
                                           const float*              threshold,
                                           const hipsparseMatDescr_t descr,
                                           float*                    csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csr((cusparseHandle_t)handle,
                                m,
                                n,
                                A,
                                lda,
                                threshold,
                                (const cusparseMatDescr_t)descr,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                buffer));
}

hipsparseStatus_t hipsparseDpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const double*             A,
                                           int                       lda,
                                           const double*             threshold,
                                           const hipsparseMatDescr_t descr,
                                           double*                   csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csr((cusparseHandle_t)handle,
                                m,
                                n,
                                A,
                                lda,
                                threshold,
                                (const cusparseMatDescr_t)descr,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                buffer));
}
#endif
