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

hipsparseStatus_t hipsparseSgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const float*      alpha,
                                  const float*      A,
                                  int               lda,
                                  const float*      cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const float*      beta,
                                  float*            C,
                                  int               ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    hipsparseStatus_t status
        = hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_sgemmi((rocsparse_handle)handle,
                                                                 rocsparse_operation_none,
                                                                 rocsparse_operation_transpose,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 nnz,
                                                                 alpha,
                                                                 A,
                                                                 lda,
                                                                 descr,
                                                                 cscValB,
                                                                 cscColPtrB,
                                                                 cscRowIndB,
                                                                 beta,
                                                                 C,
                                                                 ldc));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_descr(descr);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const double*     alpha,
                                  const double*     A,
                                  int               lda,
                                  const double*     cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const double*     beta,
                                  double*           C,
                                  int               ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    hipsparseStatus_t status
        = hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dgemmi((rocsparse_handle)handle,
                                                                 rocsparse_operation_none,
                                                                 rocsparse_operation_transpose,
                                                                 m,
                                                                 n,
                                                                 k,
                                                                 nnz,
                                                                 alpha,
                                                                 A,
                                                                 lda,
                                                                 descr,
                                                                 cscValB,
                                                                 cscColPtrB,
                                                                 cscRowIndB,
                                                                 beta,
                                                                 C,
                                                                 ldc));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_descr(descr);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const hipComplex* alpha,
                                  const hipComplex* A,
                                  int               lda,
                                  const hipComplex* cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const hipComplex* beta,
                                  hipComplex*       C,
                                  int               ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgemmi((rocsparse_handle)handle,
                         rocsparse_operation_none,
                         rocsparse_operation_transpose,
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_float_complex*)alpha,
                         (const rocsparse_float_complex*)A,
                         lda,
                         descr,
                         (const rocsparse_float_complex*)cscValB,
                         cscColPtrB,
                         cscRowIndB,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)C,
                         ldc));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_descr(descr);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgemmi(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  const hipDoubleComplex* cscValB,
                                  const int*              cscColPtrB,
                                  const int*              cscRowIndB,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       C,
                                  int                     ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgemmi((rocsparse_handle)handle,
                         rocsparse_operation_none,
                         rocsparse_operation_transpose,
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_double_complex*)alpha,
                         (const rocsparse_double_complex*)A,
                         lda,
                         descr,
                         (const rocsparse_double_complex*)cscValB,
                         cscColPtrB,
                         cscRowIndB,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)C,
                         ldc));

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_destroy_mat_descr(descr);

        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}
