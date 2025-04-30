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

hipsparseStatus_t hipsparseSbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrmv((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       mb,
                       nb,
                       nnzb,
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       bsrSortedValA,
                       bsrSortedRowPtrA,
                       bsrSortedColIndA,
                       blockDim,
                       x,
                       beta,
                       y));
}

hipsparseStatus_t hipsparseDbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrmv((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       mb,
                       nb,
                       nnzb,
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       bsrSortedValA,
                       bsrSortedRowPtrA,
                       bsrSortedColIndA,
                       blockDim,
                       x,
                       beta,
                       y));
}

hipsparseStatus_t hipsparseCbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrmv((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       mb,
                       nb,
                       nnzb,
                       (const cuComplex*)alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cuComplex*)bsrSortedValA,
                       bsrSortedRowPtrA,
                       bsrSortedColIndA,
                       blockDim,
                       (const cuComplex*)x,
                       (const cuComplex*)beta,
                       (cuComplex*)y));
}

hipsparseStatus_t hipsparseZbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrmv((cusparseHandle_t)handle,
                       hipsparse::hipDirectionToCudaDirection(dirA),
                       hipsparse::hipOperationToCudaOperation(transA),
                       mb,
                       nb,
                       nnzb,
                       (const cuDoubleComplex*)alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cuDoubleComplex*)bsrSortedValA,
                       bsrSortedRowPtrA,
                       bsrSortedColIndA,
                       blockDim,
                       (const cuDoubleComplex*)x,
                       (const cuDoubleComplex*)beta,
                       (cuDoubleComplex*)y));
}
