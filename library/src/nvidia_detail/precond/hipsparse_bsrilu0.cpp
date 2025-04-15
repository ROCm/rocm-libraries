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
    hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle, bsrilu02Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXbsrilu02_zeroPivot((cusparseHandle_t)handle, (bsrilu02Info_t)info, position));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSbsrilu02_numericBoost(
        (cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseDbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDbsrilu02_numericBoost(
        (cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCbsrilu02_numericBoost(
        (cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, (cuComplex*)boost_val));
}

hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrilu02_numericBoost((cusparseHandle_t)handle,
                                       (bsrilu02Info_t)info,
                                       enable_boost,
                                       tol,
                                       (cuDoubleComplex*)boost_val));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipsparse::hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipsparse::hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipsparse::hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     (cuComplex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipsparse::hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     (cuDoubleComplex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

hipsparseStatus_t hipsparseDbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

hipsparseStatus_t hipsparseCbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

hipsparseStatus_t hipsparseZbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipsparse::hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipsparse::hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}
#endif

#if CUDART_VERSION < 13000
hipsparseStatus_t hipsparseSbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrilu02((cusparseHandle_t)handle,
                          hipsparse::hipDirectionToCudaDirection(dirA),
                          mb,
                          nnzb,
                          (cusparseMatDescr_t)descrA,
                          bsrSortedValA_valM,
                          bsrSortedRowPtrA,
                          bsrSortedColIndA,
                          blockDim,
                          (bsrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}

hipsparseStatus_t hipsparseDbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrilu02((cusparseHandle_t)handle,
                          hipsparse::hipDirectionToCudaDirection(dirA),
                          mb,
                          nnzb,
                          (cusparseMatDescr_t)descrA,
                          bsrSortedValA_valM,
                          bsrSortedRowPtrA,
                          bsrSortedColIndA,
                          blockDim,
                          (bsrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}

hipsparseStatus_t hipsparseCbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrilu02((cusparseHandle_t)handle,
                          hipsparse::hipDirectionToCudaDirection(dirA),
                          mb,
                          nnzb,
                          (cusparseMatDescr_t)descrA,
                          (cuComplex*)bsrSortedValA_valM,
                          bsrSortedRowPtrA,
                          bsrSortedColIndA,
                          blockDim,
                          (bsrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}

hipsparseStatus_t hipsparseZbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrilu02((cusparseHandle_t)handle,
                          hipsparse::hipDirectionToCudaDirection(dirA),
                          mb,
                          nnzb,
                          (cusparseMatDescr_t)descrA,
                          (cuDoubleComplex*)bsrSortedValA_valM,
                          bsrSortedRowPtrA,
                          bsrSortedColIndA,
                          blockDim,
                          (bsrilu02Info_t)info,
                          hipsparse::hipPolicyToCudaPolicy(policy),
                          pBuffer));
}
#endif
