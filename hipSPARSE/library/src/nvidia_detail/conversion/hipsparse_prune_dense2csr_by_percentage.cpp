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
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const float*      A,
                                                                  int               lda,
                                                                  float             percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const float*              csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const double*     A,
                                                                  int               lda,
                                                                  double            percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const double*             csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t
    hipsparseSpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const float*              csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          pBufferSizeInBytes));
}

hipsparseStatus_t
    hipsparseDpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const double*             csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const float*              A,
                                                          int                       lda,
                                                          float                     percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrNnzByPercentage((cusparseHandle_t)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               percentage,
                                               (const cusparseMatDescr_t)descr,
                                               csrRowPtr,
                                               nnzTotalDevHostPtr,
                                               (pruneInfo_t)info,
                                               buffer));
}

hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const double*             A,
                                                          int                       lda,
                                                          double                    percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrNnzByPercentage((cusparseHandle_t)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               percentage,
                                               (const cusparseMatDescr_t)descr,
                                               csrRowPtr,
                                               nnzTotalDevHostPtr,
                                               (pruneInfo_t)info,
                                               buffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       float*                    csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrByPercentage((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            A,
                                            lda,
                                            percentage,
                                            (const cusparseMatDescr_t)descr,
                                            csrVal,
                                            csrRowPtr,
                                            csrColInd,
                                            (pruneInfo_t)info,
                                            buffer));
}

hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       double*                   csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrByPercentage((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            A,
                                            lda,
                                            percentage,
                                            (const cusparseMatDescr_t)descr,
                                            csrVal,
                                            csrRowPtr,
                                            csrColInd,
                                            (pruneInfo_t)info,
                                            buffer));
}
#endif
