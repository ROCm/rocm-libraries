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
    hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXbsric02_zeroPivot((cusparseHandle_t)handle, (bsric02Info_t)info, position));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipsparse::hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipsparse::hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipsparse::hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    (cuComplex*)bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipsparse::hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    (cuDoubleComplex*)bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsric02_analysis((cusparseHandle_t)handle,
                                  hipsparse::hipDirectionToCudaDirection(dirA),
                                  mb,
                                  nnzb,
                                  (cusparseMatDescr_t)descrA,
                                  bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  (bsric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}

hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsric02_analysis((cusparseHandle_t)handle,
                                  hipsparse::hipDirectionToCudaDirection(dirA),
                                  mb,
                                  nnzb,
                                  (cusparseMatDescr_t)descrA,
                                  bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  (bsric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}

hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsric02_analysis((cusparseHandle_t)handle,
                                  hipsparse::hipDirectionToCudaDirection(dirA),
                                  mb,
                                  nnzb,
                                  (cusparseMatDescr_t)descrA,
                                  (const cuComplex*)bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  (bsric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}

hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsric02_analysis((cusparseHandle_t)handle,
                                  hipsparse::hipDirectionToCudaDirection(dirA),
                                  mb,
                                  nnzb,
                                  (cusparseMatDescr_t)descrA,
                                  (const cuDoubleComplex*)bsrValA,
                                  bsrRowPtrA,
                                  bsrColIndA,
                                  blockDim,
                                  (bsric02Info_t)info,
                                  hipsparse::hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsric02((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         mb,
                         nnzb,
                         (cusparseMatDescr_t)descrA,
                         bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         (bsric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}

hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsric02((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         mb,
                         nnzb,
                         (cusparseMatDescr_t)descrA,
                         bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         (bsric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}

hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsric02((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         mb,
                         nnzb,
                         (cusparseMatDescr_t)descrA,
                         (cuComplex*)bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         (bsric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}

hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsric02((cusparseHandle_t)handle,
                         hipsparse::hipDirectionToCudaDirection(dirA),
                         mb,
                         nnzb,
                         (cusparseMatDescr_t)descrA,
                         (cuDoubleComplex*)bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         (bsric02Info_t)info,
                         hipsparse::hipPolicyToCudaPolicy(policy),
                         pBuffer));
}
#endif
