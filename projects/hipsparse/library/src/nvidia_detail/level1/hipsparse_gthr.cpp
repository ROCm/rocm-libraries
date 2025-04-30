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
hipsparseStatus_t hipsparseSgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         y,
                                 float*               xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgthr((cusparseHandle_t)handle,
                      nnz,
                      y,
                      xVal,
                      xInd,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        y,
                                 double*              xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgthr((cusparseHandle_t)handle,
                      nnz,
                      y,
                      xVal,
                      xInd,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseCgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    y,
                                 hipComplex*          xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgthr((cusparseHandle_t)handle,
                      nnz,
                      (const cuComplex*)y,
                      (cuComplex*)xVal,
                      xInd,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZgthr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       xVal,
                                 const int*              xInd,
                                 hipsparseIndexBase_t    idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgthr((cusparseHandle_t)handle,
                      nnz,
                      (const cuDoubleComplex*)y,
                      (cuDoubleComplex*)xVal,
                      xInd,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}
#endif
