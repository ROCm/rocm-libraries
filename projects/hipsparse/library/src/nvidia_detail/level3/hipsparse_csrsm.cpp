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
    hipsparseXcsrsm2_zeroPivot(hipsparseHandle_t handle, csrsm2Info_t info, int* position)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseXcsrsm2_zeroPivot((cusparseHandle_t)handle, (csrsm2Info_t)info, position));
}
#endif

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const float*              alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const float*              csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const float*              B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipsparse::hipOperationToCudaOperation(transA),
                                      hipsparse::hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipsparse::hipPolicyToCudaPolicy(policy),
                                      pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const double*             alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const double*             csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const double*             B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipsparse::hipOperationToCudaOperation(transA),
                                      hipsparse::hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipsparse::hipPolicyToCudaPolicy(policy),
                                      pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipComplex*         alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipComplex*         B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipsparse::hipOperationToCudaOperation(transA),
                                      hipsparse::hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      (const cuComplex*)alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      (const cuComplex*)csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      (const cuComplex*)B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipsparse::hipPolicyToCudaPolicy(policy),
                                      pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipDoubleComplex*   alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipDoubleComplex*   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipDoubleComplex*   B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipsparse::hipOperationToCudaOperation(transA),
                                      hipsparse::hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      (const cuDoubleComplex*)alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      (const cuDoubleComplex*)csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      (const cuDoubleComplex*)B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipsparse::hipPolicyToCudaPolicy(policy),
                                      pBufferSizeInBytes));
}
#endif

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseScsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const float*              alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const float*              B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseDcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const double*             alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const double*             B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseCcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipComplex*         alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipComplex*         B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 (const cuComplex*)alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuComplex*)csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (const cuComplex*)B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseZcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipDoubleComplex*   alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipDoubleComplex*   B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipsparse::hipOperationToCudaOperation(transA),
                                 hipsparse::hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 (const cuDoubleComplex*)alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (const cuDoubleComplex*)B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipsparse::hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}
#endif

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseScsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         float*                    B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsm2_solve((cusparseHandle_t)handle,
                              algo,
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transB),
                              m,
                              nrhs,
                              nnz,
                              alpha,
                              (const cusparseMatDescr_t)descrA,
                              csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              B,
                              ldb,
                              (csrsm2Info_t)info,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseDcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         double*                   B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsm2_solve((cusparseHandle_t)handle,
                              algo,
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transB),
                              m,
                              nrhs,
                              nnz,
                              alpha,
                              (const cusparseMatDescr_t)descrA,
                              csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              B,
                              ldb,
                              (csrsm2Info_t)info,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseCcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipComplex*               B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsm2_solve((cusparseHandle_t)handle,
                              algo,
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transB),
                              m,
                              nrhs,
                              nnz,
                              (const cuComplex*)alpha,
                              (const cusparseMatDescr_t)descrA,
                              (const cuComplex*)csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              (cuComplex*)B,
                              ldb,
                              (csrsm2Info_t)info,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}

hipsparseStatus_t hipsparseZcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipDoubleComplex*         B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsm2_solve((cusparseHandle_t)handle,
                              algo,
                              hipsparse::hipOperationToCudaOperation(transA),
                              hipsparse::hipOperationToCudaOperation(transB),
                              m,
                              nrhs,
                              nnz,
                              (const cuDoubleComplex*)alpha,
                              (const cusparseMatDescr_t)descrA,
                              (const cuDoubleComplex*)csrSortedValA,
                              csrSortedRowPtrA,
                              csrSortedColIndA,
                              (cuDoubleComplex*)B,
                              ldb,
                              (csrsm2Info_t)info,
                              hipsparse::hipPolicyToCudaPolicy(policy),
                              pBuffer));
}
#endif
