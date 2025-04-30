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

#if CUDART_VERSION < 12000
hipsparseStatus_t
    hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXcsrsv2_zeroPivot((cusparseHandle_t)handle, (csrsv2Info_t)info, position));
}
#endif

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   (cuComplex*)csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipsparse::hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrsv2Info_t)info,
                                   pBufferSizeInBytes));
}
#endif

hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseScsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (csrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseDcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (csrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseCcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuComplex*)csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (csrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseZcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsv2_analysis((cusparseHandle_t)handle,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (csrsv2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}
#endif

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseScsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipOperationToCudaOperation(transA),
                              m,
                              nnz,
                              alpha,
                              (cusparseMatDescr_t)descrA,
                              csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              (csrsv2Info_t)info,
                              f,
                              x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseDcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipOperationToCudaOperation(transA),
                              m,
                              nnz,
                              alpha,
                              (cusparseMatDescr_t)descrA,
                              csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              (csrsv2Info_t)info,
                              f,
                              x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseCcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipOperationToCudaOperation(transA),
                              m,
                              nnz,
                              (const cuComplex*)alpha,
                              (cusparseMatDescr_t)descrA,
                              (const cuComplex*)csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              (csrsv2Info_t)info,
                              (const cuComplex*)f,
                              (cuComplex*)x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseZcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsv2_solve((cusparseHandle_t)handle,
                              hipsparse::hipOperationToCudaOperation(transA),
                              m,
                              nnz,
                              (const cuDoubleComplex*)alpha,
                              (cusparseMatDescr_t)descrA,
                              (const cuDoubleComplex*)csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              (csrsv2Info_t)info,
                              (const cuDoubleComplex*)f,
                              (cuDoubleComplex*)x,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}
#endif
