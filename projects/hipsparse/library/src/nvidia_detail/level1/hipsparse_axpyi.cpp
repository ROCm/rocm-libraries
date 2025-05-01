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
hipsparseStatus_t hipsparseSaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const float*         alpha,
                                  const float*         xVal,
                                  const int*           xInd,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSaxpyi((cusparseHandle_t)handle,
                       nnz,
                       alpha,
                       xVal,
                       xInd,
                       y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const double*        alpha,
                                  const double*        xVal,
                                  const int*           xInd,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDaxpyi((cusparseHandle_t)handle,
                       nnz,
                       alpha,
                       xVal,
                       xInd,
                       y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseCaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    alpha,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCaxpyi((cusparseHandle_t)handle,
                       nnz,
                       (const cuComplex*)alpha,
                       (const cuComplex*)xVal,
                       xInd,
                       (cuComplex*)y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZaxpyi(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZaxpyi((cusparseHandle_t)handle,
                       nnz,
                       (const cuDoubleComplex*)alpha,
                       (const cuDoubleComplex*)xVal,
                       xInd,
                       (cuDoubleComplex*)y,
                       hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}
#endif
