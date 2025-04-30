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
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const float* dummy = static_cast<const float*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sgpsv_interleaved_batch_buffer_size((rocsparse_handle)handle,
                                                      (rocsparse_gpsv_interleaved_alg)algo,
                                                      m,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      batchCount,
                                                      batchCount,
                                                      pBufferSizeInBytes));
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
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const double* dummy = static_cast<const double*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dgpsv_interleaved_batch_buffer_size((rocsparse_handle)handle,
                                                      (rocsparse_gpsv_interleaved_alg)algo,
                                                      m,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      dummy,
                                                      batchCount,
                                                      batchCount,
                                                      pBufferSizeInBytes));
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
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const hipComplex* dummy = static_cast<const hipComplex*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgpsv_interleaved_batch_buffer_size((rocsparse_handle)handle,
                                                      (rocsparse_gpsv_interleaved_alg)algo,
                                                      m,
                                                      (const rocsparse_float_complex*)dummy,
                                                      (const rocsparse_float_complex*)dummy,
                                                      (const rocsparse_float_complex*)dummy,
                                                      (const rocsparse_float_complex*)dummy,
                                                      (const rocsparse_float_complex*)dummy,
                                                      (const rocsparse_float_complex*)dummy,
                                                      batchCount,
                                                      batchCount,
                                                      pBufferSizeInBytes));
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
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const hipDoubleComplex* dummy = static_cast<const hipDoubleComplex*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgpsv_interleaved_batch_buffer_size((rocsparse_handle)handle,
                                                      (rocsparse_gpsv_interleaved_alg)algo,
                                                      m,
                                                      (const rocsparse_double_complex*)dummy,
                                                      (const rocsparse_double_complex*)dummy,
                                                      (const rocsparse_double_complex*)dummy,
                                                      (const rocsparse_double_complex*)dummy,
                                                      (const rocsparse_double_complex*)dummy,
                                                      (const rocsparse_double_complex*)dummy,
                                                      batchCount,
                                                      batchCount,
                                                      pBufferSizeInBytes));
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sgpsv_interleaved_batch((rocsparse_handle)handle,
                                          (rocsparse_gpsv_interleaved_alg)algo,
                                          m,
                                          ds,
                                          dl,
                                          d,
                                          du,
                                          dw,
                                          x,
                                          batchCount,
                                          batchCount,
                                          pBuffer));
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dgpsv_interleaved_batch((rocsparse_handle)handle,
                                          (rocsparse_gpsv_interleaved_alg)algo,
                                          m,
                                          ds,
                                          dl,
                                          d,
                                          du,
                                          dw,
                                          x,
                                          batchCount,
                                          batchCount,
                                          pBuffer));
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgpsv_interleaved_batch((rocsparse_handle)handle,
                                          (rocsparse_gpsv_interleaved_alg)algo,
                                          m,
                                          (rocsparse_float_complex*)ds,
                                          (rocsparse_float_complex*)dl,
                                          (rocsparse_float_complex*)d,
                                          (rocsparse_float_complex*)du,
                                          (rocsparse_float_complex*)dw,
                                          (rocsparse_float_complex*)x,
                                          batchCount,
                                          batchCount,
                                          pBuffer));
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgpsv_interleaved_batch((rocsparse_handle)handle,
                                          (rocsparse_gpsv_interleaved_alg)algo,
                                          m,
                                          (rocsparse_double_complex*)ds,
                                          (rocsparse_double_complex*)dl,
                                          (rocsparse_double_complex*)d,
                                          (rocsparse_double_complex*)du,
                                          (rocsparse_double_complex*)dw,
                                          (rocsparse_double_complex*)x,
                                          batchCount,
                                          batchCount,
                                          pBuffer));
}
