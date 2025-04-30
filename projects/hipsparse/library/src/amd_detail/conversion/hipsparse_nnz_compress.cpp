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

hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         float                     tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_snnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      tol));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         double                    tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      tol));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipComplex                tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cnnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      (const rocsparse_float_complex*)csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      {hipCrealf(tol), hipCimagf(tol)}));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipDoubleComplex          tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_znnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      (const rocsparse_double_complex*)csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      {hipCreal(tol), hipCimag(tol)}));
    return HIPSPARSE_STATUS_SUCCESS;
}
