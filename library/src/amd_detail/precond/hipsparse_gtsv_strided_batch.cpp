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

hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const float* dummy = static_cast<const float*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sgtsv_no_pivot_strided_batch_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           dummy,
                                                           dummy,
                                                           dummy,
                                                           dummy,
                                                           batchCount,
                                                           batchStride,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const double* dummy = static_cast<const double*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dgtsv_no_pivot_strided_batch_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           dummy,
                                                           dummy,
                                                           dummy,
                                                           dummy,
                                                           batchCount,
                                                           batchStride,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const hipComplex* dummy = static_cast<const hipComplex*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgtsv_no_pivot_strided_batch_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           (const rocsparse_float_complex*)dummy,
                                                           (const rocsparse_float_complex*)dummy,
                                                           (const rocsparse_float_complex*)dummy,
                                                           (const rocsparse_float_complex*)dummy,
                                                           batchCount,
                                                           batchStride,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* x,
                                                            int                     batchCount,
                                                            int                     batchStride,
                                                            size_t* pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const hipDoubleComplex* dummy = static_cast<const hipDoubleComplex*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv_no_pivot_strided_batch_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           (const rocsparse_double_complex*)dummy,
                                                           (const rocsparse_double_complex*)dummy,
                                                           (const rocsparse_double_complex*)dummy,
                                                           (const rocsparse_double_complex*)dummy,
                                                           batchCount,
                                                           batchStride,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)

{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_sgtsv_no_pivot_strided_batch(
        (rocsparse_handle)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)

{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dgtsv_no_pivot_strided_batch(
        (rocsparse_handle)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgtsv_no_pivot_strided_batch((rocsparse_handle)handle,
                                               m,
                                               (const rocsparse_float_complex*)dl,
                                               (const rocsparse_float_complex*)d,
                                               (const rocsparse_float_complex*)du,
                                               (rocsparse_float_complex*)x,
                                               batchCount,
                                               batchStride,
                                               pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t       handle,
                                              int                     m,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       x,
                                              int                     batchCount,
                                              int                     batchStride,
                                              void*                   pBuffer)

{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv_no_pivot_strided_batch((rocsparse_handle)handle,
                                               m,
                                               (const rocsparse_double_complex*)dl,
                                               (const rocsparse_double_complex*)d,
                                               (const rocsparse_double_complex*)du,
                                               (rocsparse_double_complex*)x,
                                               batchCount,
                                               batchStride,
                                               pBuffer));
}
