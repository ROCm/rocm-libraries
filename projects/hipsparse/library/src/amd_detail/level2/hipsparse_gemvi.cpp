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

hipsparseStatus_t hipsparseSgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sgemvi_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     n,
                                     nnz,
                                     &buffer_size));

    *pBufferSizeInBytes = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseDgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dgemvi_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     n,
                                     nnz,
                                     &buffer_size));

    *pBufferSizeInBytes = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseCgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgemvi_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     n,
                                     nnz,
                                     &buffer_size));

    *pBufferSizeInBytes = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseZgemvi_bufferSize(hipsparseHandle_t    handle,
                                             hipsparseOperation_t transA,
                                             int                  m,
                                             int                  n,
                                             int                  nnz,
                                             int*                 pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgemvi_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     n,
                                     nnz,
                                     &buffer_size));

    *pBufferSizeInBytes = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const float*         alpha,
                                  const float*         A,
                                  int                  lda,
                                  int                  nnz,
                                  const float*         x,
                                  const int*           xInd,
                                  const float*         beta,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sgemvi((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         m,
                         n,
                         alpha,
                         A,
                         lda,
                         nnz,
                         x,
                         xInd,
                         beta,
                         y,
                         hipsparse::hipBaseToHCCBase(idxBase),
                         pBuffer));
}

hipsparseStatus_t hipsparseDgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const double*        alpha,
                                  const double*        A,
                                  int                  lda,
                                  int                  nnz,
                                  const double*        x,
                                  const int*           xInd,
                                  const double*        beta,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dgemvi((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         m,
                         n,
                         alpha,
                         A,
                         lda,
                         nnz,
                         x,
                         xInd,
                         beta,
                         y,
                         hipsparse::hipBaseToHCCBase(idxBase),
                         pBuffer));
}

hipsparseStatus_t hipsparseCgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const hipComplex*    alpha,
                                  const hipComplex*    A,
                                  int                  lda,
                                  int                  nnz,
                                  const hipComplex*    x,
                                  const int*           xInd,
                                  const hipComplex*    beta,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgemvi((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         m,
                         n,
                         (const rocsparse_float_complex*)alpha,
                         (const rocsparse_float_complex*)A,
                         lda,
                         nnz,
                         (const rocsparse_float_complex*)x,
                         xInd,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)y,
                         hipsparse::hipBaseToHCCBase(idxBase),
                         pBuffer));
}

hipsparseStatus_t hipsparseZgemvi(hipsparseHandle_t       handle,
                                  hipsparseOperation_t    transA,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  int                     nnz,
                                  const hipDoubleComplex* x,
                                  const int*              xInd,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase,
                                  void*                   pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgemvi((rocsparse_handle)handle,
                         hipsparse::hipOperationToHCCOperation(transA),
                         m,
                         n,
                         (const rocsparse_double_complex*)alpha,
                         (const rocsparse_double_complex*)A,
                         lda,
                         nnz,
                         (const rocsparse_double_complex*)x,
                         xInd,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)y,
                         hipsparse::hipBaseToHCCBase(idxBase),
                         pBuffer));
}
