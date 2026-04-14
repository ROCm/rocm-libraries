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
                                   float*                    y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrxmv((cusparseHandle_t)handle,
                        hipsparse::hipDirectionToCudaDirection(dir),
                        hipsparse::hipOperationToCudaOperation(trans),
                        sizeOfMask,
                        mb,
                        nb,
                        nnzb,
                        alpha,
                        (const cusparseMatDescr_t)descr,
                        bsrVal,
                        bsrMaskPtr,
                        bsrRowPtr,
                        bsrEndPtr,
                        bsrColInd,
                        blockDim,
                        x,
                        beta,
                        y));
}

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
                                   double*                   y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrxmv((cusparseHandle_t)handle,
                        hipsparse::hipDirectionToCudaDirection(dir),
                        hipsparse::hipOperationToCudaOperation(trans),
                        sizeOfMask,
                        mb,
                        nb,
                        nnzb,
                        alpha,
                        (const cusparseMatDescr_t)descr,
                        bsrVal,
                        bsrMaskPtr,
                        bsrRowPtr,
                        bsrEndPtr,
                        bsrColInd,
                        blockDim,
                        x,
                        beta,
                        y));
}

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
                                   hipComplex*               y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrxmv((cusparseHandle_t)handle,
                        hipsparse::hipDirectionToCudaDirection(dir),
                        hipsparse::hipOperationToCudaOperation(trans),
                        sizeOfMask,
                        mb,
                        nb,
                        nnzb,
                        (const cuComplex*)alpha,
                        (const cusparseMatDescr_t)descr,
                        (const cuComplex*)bsrVal,
                        bsrMaskPtr,
                        bsrRowPtr,
                        bsrEndPtr,
                        bsrColInd,
                        blockDim,
                        (const cuComplex*)x,
                        (const cuComplex*)beta,
                        (cuComplex*)y));
}

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
                                   hipDoubleComplex*         y)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrxmv((cusparseHandle_t)handle,
                        hipsparse::hipDirectionToCudaDirection(dir),
                        hipsparse::hipOperationToCudaOperation(trans),
                        sizeOfMask,
                        mb,
                        nb,
                        nnzb,
                        (const cuDoubleComplex*)alpha,
                        (const cusparseMatDescr_t)descr,
                        (const cuDoubleComplex*)bsrVal,
                        bsrMaskPtr,
                        bsrRowPtr,
                        bsrEndPtr,
                        bsrColInd,
                        blockDim,
                        (const cuDoubleComplex*)x,
                        (const cuDoubleComplex*)beta,
                        (cuDoubleComplex*)y));
}
#endif
