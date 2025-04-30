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

hipsparseStatus_t
    hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_bsric0_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_sbsric0_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dbsric0_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_cbsric0_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           (rocsparse_float_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zbsric0_buffer_size((rocsparse_handle)handle,
                                           hipsparse::hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           (rocsparse_double_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sbsric0_analysis((rocsparse_handle)handle,
                                   hipsparse::hipDirectionToHCCDirection(dirA),
                                   mb,
                                   nnzb,
                                   (rocsparse_mat_descr)descrA,
                                   bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   blockDim,
                                   (rocsparse_mat_info)info,
                                   rocsparse_analysis_policy_force,
                                   rocsparse_solve_policy_auto,
                                   pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dbsric0_analysis((rocsparse_handle)handle,
                                   hipsparse::hipDirectionToHCCDirection(dirA),
                                   mb,
                                   nnzb,
                                   (rocsparse_mat_descr)descrA,
                                   bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   blockDim,
                                   (rocsparse_mat_info)info,
                                   rocsparse_analysis_policy_force,
                                   rocsparse_solve_policy_auto,
                                   pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_cbsric0_analysis((rocsparse_handle)handle,
                                   hipsparse::hipDirectionToHCCDirection(dirA),
                                   mb,
                                   nnzb,
                                   (rocsparse_mat_descr)descrA,
                                   (const rocsparse_float_complex*)bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   blockDim,
                                   (rocsparse_mat_info)info,
                                   rocsparse_analysis_policy_force,
                                   rocsparse_solve_policy_auto,
                                   pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zbsric0_analysis((rocsparse_handle)handle,
                                   hipsparse::hipDirectionToHCCDirection(dirA),
                                   mb,
                                   nnzb,
                                   (rocsparse_mat_descr)descrA,
                                   (const rocsparse_double_complex*)bsrValA,
                                   bsrRowPtrA,
                                   bsrColIndA,
                                   blockDim,
                                   (rocsparse_mat_info)info,
                                   rocsparse_analysis_policy_force,
                                   rocsparse_solve_policy_auto,
                                   pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsric0((rocsparse_handle)handle,
                          hipsparse::hipDirectionToHCCDirection(dirA),
                          mb,
                          nnzb,
                          (rocsparse_mat_descr)descrA,
                          bsrValA,
                          bsrRowPtrA,
                          bsrColIndA,
                          blockDim,
                          (rocsparse_mat_info)info,
                          rocsparse_solve_policy_auto,
                          pBuffer));
}

hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsric0((rocsparse_handle)handle,
                          hipsparse::hipDirectionToHCCDirection(dirA),
                          mb,
                          nnzb,
                          (rocsparse_mat_descr)descrA,
                          bsrValA,
                          bsrRowPtrA,
                          bsrColIndA,
                          blockDim,
                          (rocsparse_mat_info)info,
                          rocsparse_solve_policy_auto,
                          pBuffer));
}

hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsric0((rocsparse_handle)handle,
                          hipsparse::hipDirectionToHCCDirection(dirA),
                          mb,
                          nnzb,
                          (rocsparse_mat_descr)descrA,
                          (rocsparse_float_complex*)bsrValA,
                          bsrRowPtrA,
                          bsrColIndA,
                          blockDim,
                          (rocsparse_mat_info)info,
                          rocsparse_solve_policy_auto,
                          pBuffer));
}

hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsric0((rocsparse_handle)handle,
                          hipsparse::hipDirectionToHCCDirection(dirA),
                          mb,
                          nnzb,
                          (rocsparse_mat_descr)descrA,
                          (rocsparse_double_complex*)bsrValA,
                          bsrRowPtrA,
                          bsrColIndA,
                          blockDim,
                          (rocsparse_mat_info)info,
                          rocsparse_solve_policy_auto,
                          pBuffer));
}
