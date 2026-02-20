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
hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      float*                    csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSdense2csr((cusparseHandle_t)handle,
                           m,
                           n,
                           (const cusparseMatDescr_t)descr,
                           A,
                           ld,
                           nnzPerRow,
                           csrVal,
                           csrRowPtr,
                           csrColInd));
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      double*                   csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDdense2csr((cusparseHandle_t)handle,
                           m,
                           n,
                           (const cusparseMatDescr_t)descr,
                           A,
                           ld,
                           nnzPerRow,
                           csrVal,
                           csrRowPtr,
                           csrColInd));
}

hipsparseStatus_t hipsparseCdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipComplex*               csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCdense2csr((cusparseHandle_t)handle,
                           m,
                           n,
                           (const cusparseMatDescr_t)descr,
                           (const cuComplex*)A,
                           ld,
                           nnzPerRow,
                           (cuComplex*)csrVal,
                           csrRowPtr,
                           csrColInd));
}

hipsparseStatus_t hipsparseZdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipDoubleComplex*         csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZdense2csr((cusparseHandle_t)handle,
                           m,
                           n,
                           (const cusparseMatDescr_t)descr,
                           (const cuDoubleComplex*)A,
                           ld,
                           nnzPerRow,
                           (cuDoubleComplex*)csrVal,
                           csrRowPtr,
                           csrColInd));
}
#endif
