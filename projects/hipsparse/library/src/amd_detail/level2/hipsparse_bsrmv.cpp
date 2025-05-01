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

hipsparseStatus_t hipsparseSbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrmv((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         mb,
                         nb,
                         nnzb,
                         alpha,
                         (const rocsparse_mat_descr)descrA,
                         bsrSortedValA,
                         bsrSortedRowPtrA,
                         bsrSortedColIndA,
                         blockDim,
                         nullptr,
                         x,
                         beta,
                         y));
}

hipsparseStatus_t hipsparseDbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrmv((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         mb,
                         nb,
                         nnzb,
                         alpha,
                         (const rocsparse_mat_descr)descrA,
                         bsrSortedValA,
                         bsrSortedRowPtrA,
                         bsrSortedColIndA,
                         blockDim,
                         nullptr,
                         x,
                         beta,
                         y));
}

hipsparseStatus_t hipsparseCbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrmv((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         mb,
                         nb,
                         nnzb,
                         (const rocsparse_float_complex*)alpha,
                         (const rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)bsrSortedValA,
                         bsrSortedRowPtrA,
                         bsrSortedColIndA,
                         blockDim,
                         nullptr,
                         (const rocsparse_float_complex*)x,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)y));
}

hipsparseStatus_t hipsparseZbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrmv((rocsparse_handle)handle,
                         hipsparse::hipDirectionToHCCDirection(dirA),
                         hipsparse::hipOperationToHCCOperation(transA),
                         mb,
                         nb,
                         nnzb,
                         (const rocsparse_double_complex*)alpha,
                         (const rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)bsrSortedValA,
                         bsrSortedRowPtrA,
                         bsrSortedColIndA,
                         blockDim,
                         nullptr,
                         (const rocsparse_double_complex*)x,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)y));
}
