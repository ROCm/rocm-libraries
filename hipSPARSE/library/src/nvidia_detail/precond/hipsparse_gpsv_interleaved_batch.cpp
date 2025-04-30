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

hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      ds,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      dw,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSgpsvInterleavedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     ds,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     dw,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDgpsvInterleavedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* ds,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* dw,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCgpsvInterleavedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* ds,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* dw,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseZgpsvInterleavedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            ds,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            dw,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSgpsvInterleavedBatch(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
}

hipsparseStatus_t hipsparseDgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           ds,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           dw,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDgpsvInterleavedBatch(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
}

hipsparseStatus_t hipsparseCgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       ds,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       dw,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCgpsvInterleavedBatch(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
}

hipsparseStatus_t hipsparseZgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* ds,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* dw,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseZgpsvInterleavedBatch(
        (cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer));
}
