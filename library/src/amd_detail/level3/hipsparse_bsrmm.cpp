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

hipsparseStatus_t hipsparseSbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrmm((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         mb,
                         n,
                         kb,
                         nnzb,
                         alpha,
                         (rocsparse_mat_descr)descrA,
                         bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         B,
                         ldb,
                         beta,
                         C,
                         ldc));
}

hipsparseStatus_t hipsparseDbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrmm((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         mb,
                         n,
                         kb,
                         nnzb,
                         alpha,
                         (rocsparse_mat_descr)descrA,
                         bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         B,
                         ldb,
                         beta,
                         C,
                         ldc));
}

hipsparseStatus_t hipsparseCbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrmm((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         mb,
                         n,
                         kb,
                         nnzb,
                         (const rocsparse_float_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         (const rocsparse_float_complex*)B,
                         ldb,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)C,
                         ldc));
}

hipsparseStatus_t hipsparseZbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrmm((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         hipsparse::hipOperationToHCCOperation(transB),
                         mb,
                         n,
                         kb,
                         nnzb,
                         (const rocsparse_double_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)bsrValA,
                         bsrRowPtrA,
                         bsrColIndA,
                         blockDim,
                         (const rocsparse_double_complex*)B,
                         ldb,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)C,
                         ldc));
}
