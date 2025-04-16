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

hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const float*         csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    float*               cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                      m,
                                      n,
                                      nnz,
                                      csrSortedRowPtr,
                                      csrSortedColInd,
                                      hipsparse::hipActionToHCCAction(copyValues),
                                      &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t       stream;
    hipsparseStatus_t status = hipsparseGetStream(handle, &stream);

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        RETURN_IF_HIP_ERROR(hipFree(buffer));

        return status;
    }

    // Format conversion
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsr2csc((rocsparse_handle)handle,
                           m,
                           n,
                           nnz,
                           csrSortedVal,
                           csrSortedRowPtr,
                           csrSortedColInd,
                           cscSortedVal,
                           cscSortedRowInd,
                           cscSortedColPtr,
                           hipsparse::hipActionToHCCAction(copyValues),
                           hipsparse::hipBaseToHCCBase(idxBase),
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return status;
}

hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const double*        csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    double*              cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                      m,
                                      n,
                                      nnz,
                                      csrSortedRowPtr,
                                      csrSortedColInd,
                                      hipsparse::hipActionToHCCAction(copyValues),
                                      &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t       stream;
    hipsparseStatus_t status = hipsparseGetStream(handle, &stream);

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        RETURN_IF_HIP_ERROR(hipFree(buffer));

        return status;
    }

    // Format conversion
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsr2csc((rocsparse_handle)handle,
                           m,
                           n,
                           nnz,
                           csrSortedVal,
                           csrSortedRowPtr,
                           csrSortedColInd,
                           cscSortedVal,
                           cscSortedRowInd,
                           cscSortedColPtr,
                           hipsparse::hipActionToHCCAction(copyValues),
                           hipsparse::hipBaseToHCCBase(idxBase),
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return status;
}

hipsparseStatus_t hipsparseCcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const hipComplex*    csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    hipComplex*          cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                      m,
                                      n,
                                      nnz,
                                      csrSortedRowPtr,
                                      csrSortedColInd,
                                      hipsparse::hipActionToHCCAction(copyValues),
                                      &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t       stream;
    hipsparseStatus_t status = hipsparseGetStream(handle, &stream);

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        RETURN_IF_HIP_ERROR(hipFree(buffer));

        return status;
    }

    // Format conversion
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsr2csc((rocsparse_handle)handle,
                           m,
                           n,
                           nnz,
                           (const rocsparse_float_complex*)csrSortedVal,
                           csrSortedRowPtr,
                           csrSortedColInd,
                           (rocsparse_float_complex*)cscSortedVal,
                           cscSortedRowInd,
                           cscSortedColPtr,
                           hipsparse::hipActionToHCCAction(copyValues),
                           hipsparse::hipBaseToHCCBase(idxBase),
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return status;
}

hipsparseStatus_t hipsparseZcsr2csc(hipsparseHandle_t       handle,
                                    int                     m,
                                    int                     n,
                                    int                     nnz,
                                    const hipDoubleComplex* csrSortedVal,
                                    const int*              csrSortedRowPtr,
                                    const int*              csrSortedColInd,
                                    hipDoubleComplex*       cscSortedVal,
                                    int*                    cscSortedRowInd,
                                    int*                    cscSortedColPtr,
                                    hipsparseAction_t       copyValues,
                                    hipsparseIndexBase_t    idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                      m,
                                      n,
                                      nnz,
                                      csrSortedRowPtr,
                                      csrSortedColInd,
                                      hipsparse::hipActionToHCCAction(copyValues),
                                      &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t       stream;
    hipsparseStatus_t status = hipsparseGetStream(handle, &stream);

    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        RETURN_IF_HIP_ERROR(hipFree(buffer));

        return status;
    }

    // Format conversion
    status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsr2csc((rocsparse_handle)handle,
                           m,
                           n,
                           nnz,
                           (const rocsparse_double_complex*)csrSortedVal,
                           csrSortedRowPtr,
                           csrSortedColInd,
                           (rocsparse_double_complex*)cscSortedVal,
                           cscSortedRowInd,
                           cscSortedColPtr,
                           hipsparse::hipActionToHCCAction(copyValues),
                           hipsparse::hipBaseToHCCBase(idxBase),
                           buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return status;
}

hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(hipsparseHandle_t     handle,
                                                 int                   m,
                                                 int                   n,
                                                 int                   nnz,
                                                 const void*           csrVal,
                                                 const int*            csrRowPtr,
                                                 const int*            csrColInd,
                                                 void*                 cscVal,
                                                 int*                  cscColPtr,
                                                 int*                  cscRowInd,
                                                 hipDataType           valType,
                                                 hipsparseAction_t     copyValues,
                                                 hipsparseIndexBase_t  idxBase,
                                                 hipsparseCsr2CscAlg_t alg,
                                                 size_t*               pBufferSizeInBytes)
{
    switch(valType)
    {
    case HIP_R_32F:
    case HIP_R_64F:
    case HIP_C_32F:
    case HIP_C_64F:
        return hipsparse::rocSPARSEStatusToHIPStatus(
            rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                          m,
                                          n,
                                          nnz,
                                          csrRowPtr,
                                          csrColInd,
                                          hipsparse::hipActionToHCCAction(copyValues),
                                          pBufferSizeInBytes));
    case HIP_R_8I:
    {
        // Build Source
        rocsparse_const_spmat_descr source;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_create_const_csr_descr(&source,
                                             m,
                                             n,
                                             nnz,
                                             csrRowPtr,
                                             csrColInd,
                                             (const int8_t*)csrVal,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             hipsparse::hipBaseToHCCBase(idxBase),
                                             rocsparse_datatype_i8_r));

        // Build target
        rocsparse_spmat_descr target;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csc_descr(&target,
                                                             m,
                                                             n,
                                                             nnz,
                                                             cscColPtr,
                                                             cscRowInd,
                                                             (int8_t*)cscVal,
                                                             rocsparse_indextype_i32,
                                                             rocsparse_indextype_i32,
                                                             hipsparse::hipBaseToHCCBase(idxBase),
                                                             rocsparse_datatype_i8_r));

        // Create descriptor
        rocsparse_sparse_to_sparse_descr descr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_sparse_to_sparse_descr(
            &descr, source, target, rocsparse_sparse_to_sparse_alg_default));

        size_t buffer_size_analysis;
        size_t buffer_size_compute;

        // Analysis phase
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse_buffer_size((rocsparse_handle)handle,
                                                   descr,
                                                   source,
                                                   target,
                                                   rocsparse_sparse_to_sparse_stage_analysis,
                                                   &buffer_size_analysis));

        // Calculation phase
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse_buffer_size((rocsparse_handle)handle,
                                                   descr,
                                                   source,
                                                   target,
                                                   rocsparse_sparse_to_sparse_stage_compute,
                                                   &buffer_size_compute));

        *pBufferSizeInBytes = std::max(buffer_size_analysis, buffer_size_compute);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(source));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(target));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_sparse_to_sparse_descr(descr));

        return HIPSPARSE_STATUS_SUCCESS;
    }
    default:
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    }

    return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCsr2cscEx2(hipsparseHandle_t     handle,
                                      int                   m,
                                      int                   n,
                                      int                   nnz,
                                      const void*           csrVal,
                                      const int*            csrRowPtr,
                                      const int*            csrColInd,
                                      void*                 cscVal,
                                      int*                  cscColPtr,
                                      int*                  cscRowInd,
                                      hipDataType           valType,
                                      hipsparseAction_t     copyValues,
                                      hipsparseIndexBase_t  idxBase,
                                      hipsparseCsr2CscAlg_t alg,
                                      void*                 buffer)
{
    switch(valType)
    {
    case HIP_R_32F:
        return hipsparse::rocSPARSEStatusToHIPStatus(
            rocsparse_scsr2csc((rocsparse_handle)handle,
                               m,
                               n,
                               nnz,
                               (const float*)csrVal,
                               csrRowPtr,
                               csrColInd,
                               (float*)cscVal,
                               cscRowInd,
                               cscColPtr,
                               hipsparse::hipActionToHCCAction(copyValues),
                               hipsparse::hipBaseToHCCBase(idxBase),
                               buffer));
    case HIP_R_64F:
        return hipsparse::rocSPARSEStatusToHIPStatus(
            rocsparse_dcsr2csc((rocsparse_handle)handle,
                               m,
                               n,
                               nnz,
                               (const double*)csrVal,
                               csrRowPtr,
                               csrColInd,
                               (double*)cscVal,
                               cscRowInd,
                               cscColPtr,
                               hipsparse::hipActionToHCCAction(copyValues),
                               hipsparse::hipBaseToHCCBase(idxBase),
                               buffer));
    case HIP_C_32F:
        return hipsparse::rocSPARSEStatusToHIPStatus(
            rocsparse_ccsr2csc((rocsparse_handle)handle,
                               m,
                               n,
                               nnz,
                               (const rocsparse_float_complex*)csrVal,
                               csrRowPtr,
                               csrColInd,
                               (rocsparse_float_complex*)cscVal,
                               cscRowInd,
                               cscColPtr,
                               hipsparse::hipActionToHCCAction(copyValues),
                               hipsparse::hipBaseToHCCBase(idxBase),
                               buffer));
    case HIP_C_64F:
        return hipsparse::rocSPARSEStatusToHIPStatus(
            rocsparse_zcsr2csc((rocsparse_handle)handle,
                               m,
                               n,
                               nnz,
                               (const rocsparse_double_complex*)csrVal,
                               csrRowPtr,
                               csrColInd,
                               (rocsparse_double_complex*)cscVal,
                               cscRowInd,
                               cscColPtr,
                               hipsparse::hipActionToHCCAction(copyValues),
                               hipsparse::hipBaseToHCCBase(idxBase),
                               buffer));
    case HIP_R_8I:
    {
        // Build Source
        rocsparse_const_spmat_descr source;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_create_const_csr_descr(&source,
                                             m,
                                             n,
                                             nnz,
                                             csrRowPtr,
                                             csrColInd,
                                             (const int8_t*)csrVal,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             hipsparse::hipBaseToHCCBase(idxBase),
                                             rocsparse_datatype_i8_r));

        // Build target
        rocsparse_spmat_descr target;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csc_descr(&target,
                                                             m,
                                                             n,
                                                             nnz,
                                                             cscColPtr,
                                                             cscRowInd,
                                                             (int8_t*)cscVal,
                                                             rocsparse_indextype_i32,
                                                             rocsparse_indextype_i32,
                                                             hipsparse::hipBaseToHCCBase(idxBase),
                                                             rocsparse_datatype_i8_r));

        // Create descriptor
        rocsparse_sparse_to_sparse_descr descr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_sparse_to_sparse_descr(
            &descr, source, target, rocsparse_sparse_to_sparse_alg_default));

        size_t buffer_size;

        // Analysis phase
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse_buffer_size((rocsparse_handle)handle,
                                                   descr,
                                                   source,
                                                   target,
                                                   rocsparse_sparse_to_sparse_stage_analysis,
                                                   &buffer_size));
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse((rocsparse_handle)handle,
                                       descr,
                                       source,
                                       target,
                                       rocsparse_sparse_to_sparse_stage_analysis,
                                       buffer_size,
                                       buffer));

        // Calculation phase
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse_buffer_size((rocsparse_handle)handle,
                                                   descr,
                                                   source,
                                                   target,
                                                   rocsparse_sparse_to_sparse_stage_compute,
                                                   &buffer_size));
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse((rocsparse_handle)handle,
                                       descr,
                                       source,
                                       target,
                                       rocsparse_sparse_to_sparse_stage_compute,
                                       buffer_size,
                                       buffer));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(source));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(target));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_sparse_to_sparse_descr(descr));

        return HIPSPARSE_STATUS_SUCCESS;
    }
    default:
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    }

    return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
