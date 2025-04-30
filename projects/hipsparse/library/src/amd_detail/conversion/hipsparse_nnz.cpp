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

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const float*              A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_snnz((rocsparse_handle)handle,
                                             hipsparse::hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const double*             A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnnz((rocsparse_handle)handle,
                                             hipsparse::hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipComplex*         A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cnnz((rocsparse_handle)handle,
                                             hipsparse::hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             (const rocsparse_float_complex*)A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipDoubleComplex*   A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_znnz((rocsparse_handle)handle,
                                             hipsparse::hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             (const rocsparse_double_complex*)A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}
