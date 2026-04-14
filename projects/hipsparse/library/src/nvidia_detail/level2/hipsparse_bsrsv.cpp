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
hipsparseStatus_t
    hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXbsrsv2_zeroPivot((cusparseHandle_t)handle, (bsrsv2Info_t)info, position));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dir),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dir),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dir),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dir),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}
#endif

hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dir),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dir),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dir),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dir),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dir),
                              hipsparse::hipOperationToCudaOperation(transA),
                              mb,
                              nnzb,
                              alpha,
                              (cusparseMatDescr_t)descrA,
                              bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsv2Info_t)info,
                              f,
                              x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dir),
                              hipsparse::hipOperationToCudaOperation(transA),
                              mb,
                              nnzb,
                              alpha,
                              (cusparseMatDescr_t)descrA,
                              bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsv2Info_t)info,
                              f,
                              x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dir),
                              hipsparse::hipOperationToCudaOperation(transA),
                              mb,
                              nnzb,
                              (const cuComplex*)alpha,
                              (cusparseMatDescr_t)descrA,
                              (const cuComplex*)bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsv2Info_t)info,
                              (const cuComplex*)f,
                              (cuComplex*)x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dir),
                              hipsparse::hipOperationToCudaOperation(transA),
                              mb,
                              nnzb,
                              (const cuDoubleComplex*)alpha,
                              (cusparseMatDescr_t)descrA,
                              (const cuDoubleComplex*)bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsv2Info_t)info,
                              (const cuDoubleComplex*)f,
                              (cuDoubleComplex*)x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}
#endif
