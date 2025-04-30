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

#if CUDART_VERSION < 11000
hipsparseStatus_t hipsparseShybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseShybmv((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cusparseHybMat_t)hybA,
                       x,
                       beta,
                       y));
}

hipsparseStatus_t hipsparseDhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDhybmv((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cusparseHybMat_t)hybA,
                       x,
                       beta,
                       y));
}

hipsparseStatus_t hipsparseChybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseChybmv((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       (const cuComplex*)alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cusparseHybMat_t)hybA,
                       (const cuComplex*)x,
                       (const cuComplex*)beta,
                       (cuComplex*)y));
}

hipsparseStatus_t hipsparseZhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZhybmv((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       (const cuDoubleComplex*)alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cusparseHybMat_t)hybA,
                       (const cuDoubleComplex*)x,
                       (const cuDoubleComplex*)beta,
                       (cuDoubleComplex*)y));
}
#endif
