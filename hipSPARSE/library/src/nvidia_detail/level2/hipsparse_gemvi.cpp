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

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
hipsparseStatus_t hipsparseSgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgemvi_bufferSize((cusparseHandle_t)handle,
                                  hipsparse::hipOperationToCudaOperation(transA),
                                  m,
                                  n,
                                  nnz,
                                  pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgemvi_bufferSize((cusparseHandle_t)handle,
                                  hipsparse::hipOperationToCudaOperation(transA),
                                  m,
                                  n,
                                  nnz,
                                  pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgemvi_bufferSize((cusparseHandle_t)handle,
                                  hipsparse::hipOperationToCudaOperation(transA),
                                  m,
                                  n,
                                  nnz,
                                  pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgemvi_bufferSize((cusparseHandle_t)handle,
                                  hipsparse::hipOperationToCudaOperation(transA),
                                  m,
                                  n,
                                  nnz,
                                  pBufferSizeInBytes));
}
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const float*         alpha,
                                  const float*         A,
                                  int                  lda,
                                  int                  nnz,
                                  const float*         x,
                                  const int*           xInd,
                                  const float*         beta,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgemvi((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       alpha,
                       A,
                       lda,
                       nnz,
                       x,
                       xInd,
                       beta,
                       y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                       pBuffer));
}

hipsparseStatus_t hipsparseDgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const double*        alpha,
                                  const double*        A,
                                  int                  lda,
                                  int                  nnz,
                                  const double*        x,
                                  const int*           xInd,
                                  const double*        beta,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgemvi((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       alpha,
                       A,
                       lda,
                       nnz,
                       x,
                       xInd,
                       beta,
                       y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                       pBuffer));
}

hipsparseStatus_t hipsparseCgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const hipComplex*    alpha,
                                  const hipComplex*    A,
                                  int                  lda,
                                  int                  nnz,
                                  const hipComplex*    x,
                                  const int*           xInd,
                                  const hipComplex*    beta,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgemvi((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       (const cuComplex*)alpha,
                       (const cuComplex*)A,
                       lda,
                       nnz,
                       (const cuComplex*)x,
                       xInd,
                       (const cuComplex*)beta,
                       (cuComplex*)y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                       pBuffer));
}

hipsparseStatus_t hipsparseZgemvi(hipsparseHandle_t       handle,
                                  hipsparseOperation_t    transA,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  int                     nnz,
                                  const hipDoubleComplex* x,
                                  const int*              xInd,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase,
                                  void*                   pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgemvi((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       (const cuDoubleComplex*)alpha,
                       (const cuDoubleComplex*)A,
                       lda,
                       nnz,
                       (const cuDoubleComplex*)x,
                       xInd,
                       (const cuDoubleComplex*)beta,
                       (cuDoubleComplex*)y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase),
                       pBuffer));
}
#endif
