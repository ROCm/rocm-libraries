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

#if CUDART_VERSION < 11000
hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 const float*         y,
                                 float*               result,
                                 hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSdoti((cusparseHandle_t)handle,
                      nnz,
                      xVal,
                      xInd,
                      y,
                      result,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 const double*        y,
                                 double*              result,
                                 hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDdoti((cusparseHandle_t)handle,
                      nnz,
                      xVal,
                      xInd,
                      y,
                      result,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseCdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 const hipComplex*    y,
                                 hipComplex*          result,
                                 hipsparseIndexBase_t idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCdoti((cusparseHandle_t)handle,
                      nnz,
                      (const cuComplex*)xVal,
                      xInd,
                      (const cuComplex*)y,
                      (cuComplex*)result,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZdoti(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       result,
                                 hipsparseIndexBase_t    idxBase)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZdoti((cusparseHandle_t)handle,
                      nnz,
                      (const cuDoubleComplex*)xVal,
                      xInd,
                      (const cuDoubleComplex*)y,
                      (cuDoubleComplex*)result,
                      hipsparse::hipIndexBaseToCudaIndexBase(idxBase)));
}
#endif
