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
    hipsparseXbsrsm2_zeroPivot(hipsparseHandle_t handle, bsrsm2Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXbsrsm2_zeroPivot((cusparseHandle_t)handle, (bsrsm2Info_t)info, position));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   hipsparse::hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   hipsparse::hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   hipsparse::hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   (cuComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   hipsparse::hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dirA),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseDbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dirA),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseCbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dirA),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseZbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipDirectionToCudaDirection(dirA),
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const float*              B,
                                         int                       ldb,
                                         float*                    X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsm2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dirA),
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transX),
                              mb,
                              nrhs,
                              nnzb,
                              alpha,
                              (const cusparseMatDescr_t)descrA,
                              bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsm2Info_t)info,
                              B,
                              ldb,
                              X,
                              ldx,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseDbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const double*             B,
                                         int                       ldb,
                                         double*                   X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsm2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dirA),
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transX),
                              mb,
                              nrhs,
                              nnzb,
                              alpha,
                              (const cusparseMatDescr_t)descrA,
                              bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsm2Info_t)info,
                              B,
                              ldb,
                              X,
                              ldx,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseCbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipComplex*         B,
                                         int                       ldb,
                                         hipComplex*               X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsm2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dirA),
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transX),
                              mb,
                              nrhs,
                              nnzb,
                              (const cuComplex*)alpha,
                              (const cusparseMatDescr_t)descrA,
                              (const cuComplex*)bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsm2Info_t)info,
                              (const cuComplex*)B,
                              ldb,
                              (cuComplex*)X,
                              ldx,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseZbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipDoubleComplex*   B,
                                         int                       ldb,
                                         hipDoubleComplex*         X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsm2_solve((cusparseHandle_t)handle,
                              hipsparse::hipDirectionToCudaDirection(dirA),
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transX),
                              mb,
                              nrhs,
                              nnzb,
                              (const cuDoubleComplex*)alpha,
                              (const cusparseMatDescr_t)descrA,
                              (const cuDoubleComplex*)bsrSortedValA,
                              bsrSortedRowPtrA,
                              bsrSortedColIndA,
                              blockDim,
                              (bsrsm2Info_t)info,
                              (const cuDoubleComplex*)B,
                              ldb,
                              (cuDoubleComplex*)X,
                              ldx,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}
#endif
