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

hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const float*              A,
                                                      int                       lda,
                                                      const float*              threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const float*              csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               pBufferSizeInBytes));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const double*             A,
                                                      int                       lda,
                                                      const double*             threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const double*             csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               pBufferSizeInBytes));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const float*              A,
                                                         int                       lda,
                                                         const float*              threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const float*              csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               pBufferSizeInBytes));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const double*             A,
                                                         int                       lda,
                                                         const double*             threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const double*             csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t* pBufferSizeInBytes)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               pBufferSizeInBytes));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const float*              A,
                                              int                       lda,
                                              const float*              threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sprune_dense2csr_nnz((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             threshold,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrRowPtr,
                                                             nnzTotalDevHostPtr,
                                                             buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const double*             A,
                                              int                       lda,
                                              const double*             threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dprune_dense2csr_nnz((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             threshold,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrRowPtr,
                                                             nnzTotalDevHostPtr,
                                                             buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const float*              A,
                                           int                       lda,
                                           const float*              threshold,
                                           const hipsparseMatDescr_t descr,
                                           float*                    csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sprune_dense2csr((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         A,
                                                         lda,
                                                         threshold,
                                                         (const rocsparse_mat_descr)descr,
                                                         csrVal,
                                                         csrRowPtr,
                                                         csrColInd,
                                                         buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const double*             A,
                                           int                       lda,
                                           const double*             threshold,
                                           const hipsparseMatDescr_t descr,
                                           double*                   csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dprune_dense2csr((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         A,
                                                         lda,
                                                         threshold,
                                                         (const rocsparse_mat_descr)descr,
                                                         csrVal,
                                                         csrRowPtr,
                                                         csrColInd,
                                                         buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}
