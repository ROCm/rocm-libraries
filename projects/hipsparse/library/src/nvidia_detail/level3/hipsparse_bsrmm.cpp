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

hipsparseStatus_t hipsparseSbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrmm((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       hipsparse::hipOperationToCudaOperation(transB),
                       mb,
                       n,
                       kb,
                       nnzb,
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       bsrValA,
                       bsrRowPtrA,
                       bsrColIndA,
                       blockDim,
                       B,
                       ldb,
                       beta,
                       C,
                       ldc));
}

hipsparseStatus_t hipsparseDbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrmm((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       hipsparse::hipOperationToCudaOperation(transB),
                       mb,
                       n,
                       kb,
                       nnzb,
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       bsrValA,
                       bsrRowPtrA,
                       bsrColIndA,
                       blockDim,
                       B,
                       ldb,
                       beta,
                       C,
                       ldc));
}

hipsparseStatus_t hipsparseCbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrmm((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       hipsparse::hipOperationToCudaOperation(transB),
                       mb,
                       n,
                       kb,
                       nnzb,
                       (const cuComplex*)alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cuComplex*)bsrValA,
                       bsrRowPtrA,
                       bsrColIndA,
                       blockDim,
                       (const cuComplex*)B,
                       ldb,
                       (const cuComplex*)beta,
                       (cuComplex*)C,
                       ldc));
}

hipsparseStatus_t hipsparseZbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrmm((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       hipsparse::hipOperationToCudaOperation(transB),
                       mb,
                       n,
                       kb,
                       nnzb,
                       (const cuDoubleComplex*)alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cuDoubleComplex*)bsrValA,
                       bsrRowPtrA,
                       bsrColIndA,
                       blockDim,
                       (const cuDoubleComplex*)B,
                       ldb,
                       (const cuDoubleComplex*)beta,
                       (cuDoubleComplex*)C,
                       ldc));
}
