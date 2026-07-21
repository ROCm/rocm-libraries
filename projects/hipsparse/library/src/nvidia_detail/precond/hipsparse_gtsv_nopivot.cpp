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

hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const float*      dl,
                                                        const float*      d,
                                                        const float*      du,
                                                        const float*      B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSgtsv2_nopivot_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const double*     dl,
                                                        const double*     d,
                                                        const double*     du,
                                                        const double*     B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseDgtsv2_nopivot_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const hipComplex* dl,
                                                        const hipComplex* d,
                                                        const hipComplex* du,
                                                        const hipComplex* B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseCgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle,
                                             m,
                                             n,
                                             (const cuComplex*)dl,
                                             (const cuComplex*)d,
                                             (const cuComplex*)du,
                                             (const cuComplex*)B,
                                             ldb,
                                             pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t       handle,
                                                        int                     m,
                                                        int                     n,
                                                        const hipDoubleComplex* dl,
                                                        const hipDoubleComplex* d,
                                                        const hipDoubleComplex* du,
                                                        const hipDoubleComplex* B,
                                                        int                     ldb,
                                                        size_t*                 pBufferSizeInBytes)
{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle,
                                             m,
                                             n,
                                             (const cuDoubleComplex*)dl,
                                             (const cuDoubleComplex*)d,
                                             (const cuDoubleComplex*)du,
                                             (const cuDoubleComplex*)B,
                                             ldb,
                                             pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const float*      dl,
                                          const float*      d,
                                          const float*      du,
                                          float*            B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const double*     dl,
                                          const double*     d,
                                          const double*     du,
                                          double*           B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseDgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const hipComplex* dl,
                                          const hipComplex* d,
                                          const hipComplex* du,
                                          hipComplex*       B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseCgtsv2_nopivot((cusparseHandle_t)handle,
                                                                          m,
                                                                          n,
                                                                          (const cuComplex*)dl,
                                                                          (const cuComplex*)d,
                                                                          (const cuComplex*)du,
                                                                          (cuComplex*)B,
                                                                          ldb,
                                                                          pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2_nopivot(hipsparseHandle_t       handle,
                                          int                     m,
                                          int                     n,
                                          const hipDoubleComplex* dl,
                                          const hipDoubleComplex* d,
                                          const hipDoubleComplex* du,
                                          hipDoubleComplex*       B,
                                          int                     ldb,
                                          void*                   pBuffer)

{
    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseZgtsv2_nopivot((cusparseHandle_t)handle,
                               m,
                               n,
                               (const cuDoubleComplex*)dl,
                               (const cuDoubleComplex*)d,
                               (const cuDoubleComplex*)du,
                               (cuDoubleComplex*)B,
                               ldb,
                               pBuffer));
}
