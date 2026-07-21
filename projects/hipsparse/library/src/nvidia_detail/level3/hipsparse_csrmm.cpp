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
hipsparseStatus_t hipsparseScsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrmm((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       k,
                       nnz,
                       alpha,
                       (cusparseMatDescr_t)descrA,
                       csrSortedValA,
                       csrSortedRowPtrA,
                       csrSortedColIndA,
                       B,
                       ldb,
                       beta,
                       C,
                       ldc));
}

hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrmm((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       k,
                       nnz,
                       alpha,
                       (cusparseMatDescr_t)descrA,
                       csrSortedValA,
                       csrSortedRowPtrA,
                       csrSortedColIndA,
                       B,
                       ldb,
                       beta,
                       C,
                       ldc));
}

hipsparseStatus_t hipsparseCcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrmm((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       k,
                       nnz,
                       (const cuComplex*)alpha,
                       (cusparseMatDescr_t)descrA,
                       (const cuComplex*)csrSortedValA,
                       csrSortedRowPtrA,
                       csrSortedColIndA,
                       (const cuComplex*)B,
                       ldb,
                       (const cuComplex*)beta,
                       (cuComplex*)C,
                       ldc));
}

hipsparseStatus_t hipsparseZcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrmm((cusparseHandle_t)handle,
                       hipsparse::hipOperationToCudaOperation(transA),
                       m,
                       n,
                       k,
                       nnz,
                       (const cuDoubleComplex*)alpha,
                       (cusparseMatDescr_t)descrA,
                       (const cuDoubleComplex*)csrSortedValA,
                       csrSortedRowPtrA,
                       csrSortedColIndA,
                       (const cuDoubleComplex*)B,
                       ldb,
                       (const cuDoubleComplex*)beta,
                       (cuDoubleComplex*)C,
                       ldc));
}
#endif

#if CUDART_VERSION < 11000
hipsparseStatus_t hipsparseScsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const float*              alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const float*              csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const float*              B,
                                   int                       ldb,
                                   const float*              beta,
                                   float*                    C,
                                   int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseScsrmm2((cusparseHandle_t)handle,
                        hipsparse::hipOperationToCudaOperation(transA),
                        hipsparse::hipOperationToCudaOperation(transB),
                        m,
                        n,
                        k,
                        nnz,
                        alpha,
                        (cusparseMatDescr_t)descrA,
                        csrSortedValA,
                        csrSortedRowPtrA,
                        csrSortedColIndA,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc));
}

hipsparseStatus_t hipsparseDcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const double*             alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const double*             csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const double*             B,
                                   int                       ldb,
                                   const double*             beta,
                                   double*                   C,
                                   int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrmm2((cusparseHandle_t)handle,
                        hipsparse::hipOperationToCudaOperation(transA),
                        hipsparse::hipOperationToCudaOperation(transB),
                        m,
                        n,
                        k,
                        nnz,
                        alpha,
                        (cusparseMatDescr_t)descrA,
                        csrSortedValA,
                        csrSortedRowPtrA,
                        csrSortedColIndA,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc));
}

hipsparseStatus_t hipsparseCcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipComplex*         alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipComplex*         csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipComplex*         B,
                                   int                       ldb,
                                   const hipComplex*         beta,
                                   hipComplex*               C,
                                   int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrmm2((cusparseHandle_t)handle,
                        hipsparse::hipOperationToCudaOperation(transA),
                        hipsparse::hipOperationToCudaOperation(transB),
                        m,
                        n,
                        k,
                        nnz,
                        (const cuComplex*)alpha,
                        (cusparseMatDescr_t)descrA,
                        (const cuComplex*)csrSortedValA,
                        csrSortedRowPtrA,
                        csrSortedColIndA,
                        (const cuComplex*)B,
                        ldb,
                        (const cuComplex*)beta,
                        (cuComplex*)C,
                        ldc));
}

hipsparseStatus_t hipsparseZcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipDoubleComplex*   alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipDoubleComplex*   csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipDoubleComplex*   B,
                                   int                       ldb,
                                   const hipDoubleComplex*   beta,
                                   hipDoubleComplex*         C,
                                   int                       ldc)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrmm2((cusparseHandle_t)handle,
                        hipsparse::hipOperationToCudaOperation(transA),
                        hipsparse::hipOperationToCudaOperation(transB),
                        m,
                        n,
                        k,
                        nnz,
                        (const cuDoubleComplex*)alpha,
                        (cusparseMatDescr_t)descrA,
                        (const cuDoubleComplex*)csrSortedValA,
                        csrSortedRowPtrA,
                        csrSortedColIndA,
                        (const cuDoubleComplex*)B,
                        ldb,
                        (const cuDoubleComplex*)beta,
                        (cuDoubleComplex*)C,
                        ldc));
}
#endif
