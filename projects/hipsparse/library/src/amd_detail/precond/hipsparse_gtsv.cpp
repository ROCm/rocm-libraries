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

hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const float*      dl,
                                                const float*      d,
                                                const float*      du,
                                                const float*      B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const float* dummy = static_cast<const float*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_sgtsv_buffer_size(
        (rocsparse_handle)handle, m, n, dummy, dummy, dummy, dummy, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const double*     dl,
                                                const double*     d,
                                                const double*     du,
                                                const double*     B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const double* dummy = static_cast<const double*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dgtsv_buffer_size(
        (rocsparse_handle)handle, m, n, dummy, dummy, dummy, dummy, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const hipComplex* dl,
                                                const hipComplex* d,
                                                const hipComplex* du,
                                                const hipComplex* B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const hipComplex* dummy = static_cast<const hipComplex*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cgtsv_buffer_size((rocsparse_handle)handle,
                                    m,
                                    n,
                                    (const rocsparse_float_complex*)dummy,
                                    (const rocsparse_float_complex*)dummy,
                                    (const rocsparse_float_complex*)dummy,
                                    (const rocsparse_float_complex*)dummy,
                                    ldb,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(hipsparseHandle_t       handle,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* dl,
                                                const hipDoubleComplex* d,
                                                const hipDoubleComplex* du,
                                                const hipDoubleComplex* B,
                                                int                     ldb,
                                                size_t*                 pBufferSizeInBytes)
{
    // cusparse allows passing nullptr's for dl, d, du, and B. On the other hand rocsparse checks
    // if they are nullptr and returns invalid pointer if they are. In both cases the pointers are
    // never actually de-referenced. In order to work in the same way regardless of the backend
    // that a user chooses, just pass in non-null dummy pointer.
    const hipDoubleComplex* dummy = static_cast<const hipDoubleComplex*>((void*)0x4);
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv_buffer_size((rocsparse_handle)handle,
                                    m,
                                    n,
                                    (const rocsparse_double_complex*)dummy,
                                    (const rocsparse_double_complex*)dummy,
                                    (const rocsparse_double_complex*)dummy,
                                    (const rocsparse_double_complex*)dummy,
                                    ldb,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const float*      dl,
                                  const float*      d,
                                  const float*      du,
                                  float*            B,
                                  int               ldb,
                                  void*             pBuffer)

{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sgtsv((rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const double*     dl,
                                  const double*     d,
                                  const double*     du,
                                  double*           B,
                                  int               ldb,
                                  void*             pBuffer)

{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dgtsv((rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const hipComplex* dl,
                                  const hipComplex* d,
                                  const hipComplex* du,
                                  hipComplex*       B,
                                  int               ldb,
                                  void*             pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_cgtsv((rocsparse_handle)handle,
                                                                 m,
                                                                 n,
                                                                 (const rocsparse_float_complex*)dl,
                                                                 (const rocsparse_float_complex*)d,
                                                                 (const rocsparse_float_complex*)du,
                                                                 (rocsparse_float_complex*)B,
                                                                 ldb,
                                                                 pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* dl,
                                  const hipDoubleComplex* d,
                                  const hipDoubleComplex* du,
                                  hipDoubleComplex*       B,
                                  int                     ldb,
                                  void*                   pBuffer)

{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv((rocsparse_handle)handle,
                        m,
                        n,
                        (const rocsparse_double_complex*)dl,
                        (const rocsparse_double_complex*)d,
                        (const rocsparse_double_complex*)du,
                        (rocsparse_double_complex*)B,
                        ldb,
                        pBuffer));
}
