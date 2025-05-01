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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSgemmi((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  nnz,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  cscValB,
                                                                  cscColPtrB,
                                                                  cscRowIndB,
                                                                  beta,
                                                                  C,
                                                                  ldc));
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDgemmi((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  nnz,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  cscValB,
                                                                  cscColPtrB,
                                                                  cscRowIndB,
                                                                  beta,
                                                                  C,
                                                                  ldc));
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCgemmi((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  nnz,
                                                                  (const cuComplex*)alpha,
                                                                  (const cuComplex*)A,
                                                                  lda,
                                                                  (const cuComplex*)cscValB,
                                                                  cscColPtrB,
                                                                  cscRowIndB,
                                                                  (const cuComplex*)beta,
                                                                  (cuComplex*)C,
                                                                  ldc));
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
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseZgemmi((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  nnz,
                                                                  (const cuDoubleComplex*)alpha,
                                                                  (const cuDoubleComplex*)A,
                                                                  lda,
                                                                  (const cuDoubleComplex*)cscValB,
                                                                  cscColPtrB,
                                                                  cscRowIndB,
                                                                  (const cuDoubleComplex*)beta,
                                                                  (cuDoubleComplex*)C,
                                                                  ldc));
}
#endif
