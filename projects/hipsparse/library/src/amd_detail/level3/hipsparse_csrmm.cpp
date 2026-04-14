/*! \file */
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

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../utility.h"

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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         rocsparse_operation_none,
                         m,
                         n,
                         k,
                         nnz,
                         alpha,
                         (rocsparse_mat_descr)descrA,
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         rocsparse_operation_none,
                         m,
                         n,
                         k,
                         nnz,
                         alpha,
                         (rocsparse_mat_descr)descrA,
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         rocsparse_operation_none,
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_float_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_float_complex*)B,
                         ldb,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)C,
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         rocsparse_operation_none,
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_double_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_double_complex*)B,
                         ldb,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)C,
                         ldc));
}

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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         m,
                         n,
                         k,
                         nnz,
                         alpha,
                         (rocsparse_mat_descr)descrA,
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         m,
                         n,
                         k,
                         nnz,
                         alpha,
                         (rocsparse_mat_descr)descrA,
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_float_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_float_complex*)B,
                         ldb,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)C,
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrmm((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_double_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_double_complex*)B,
                         ldb,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)C,
                         ldc));
}
