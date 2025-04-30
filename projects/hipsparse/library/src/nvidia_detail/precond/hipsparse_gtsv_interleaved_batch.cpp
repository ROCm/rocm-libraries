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

hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSgtsvInterleavedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDgtsvInterleavedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgtsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle,
                                                    algo,
                                                    m,
                                                    (const cuComplex*)dl,
                                                    (const cuComplex*)d,
                                                    (const cuComplex*)du,
                                                    (const cuComplex*)x,
                                                    batchCount,
                                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgtsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle,
                                                    algo,
                                                    m,
                                                    (const cuDoubleComplex*)dl,
                                                    (const cuDoubleComplex*)d,
                                                    (const cuDoubleComplex*)du,
                                                    (const cuDoubleComplex*)x,
                                                    batchCount,
                                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSgtsvInterleavedBatch(
        (cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBuffer));
}

hipsparseStatus_t hipsparseDgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer)

{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDgtsvInterleavedBatch(
        (cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBuffer));
}

hipsparseStatus_t hipsparseCgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer)

{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgtsvInterleavedBatch((cusparseHandle_t)handle,
                                      algo,
                                      m,
                                      (cuComplex*)dl,
                                      (cuComplex*)d,
                                      (cuComplex*)du,
                                      (cuComplex*)x,
                                      batchCount,
                                      pBuffer));
}

hipsparseStatus_t hipsparseZgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer)

{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgtsvInterleavedBatch((cusparseHandle_t)handle,
                                      algo,
                                      m,
                                      (cuDoubleComplex*)dl,
                                      (cuDoubleComplex*)d,
                                      (cuDoubleComplex*)du,
                                      (cuDoubleComplex*)x,
                                      batchCount,
                                      pBuffer));
}
