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
    hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv zero pivot
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsv_zero_pivot(
        (rocsparse_handle)handle, nullptr, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_scsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dcsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_ccsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_float_complex*)csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zcsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_double_complex*)csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_float_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_double_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseScsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_scsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  m,
                                  nnz,
                                  (rocsparse_mat_descr)descrA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dcsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  m,
                                  nnz,
                                  (rocsparse_mat_descr)descrA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_ccsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  m,
                                  nnz,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zcsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  m,
                                  nnz,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               m,
                               nnz,
                               alpha,
                               (rocsparse_mat_descr)descrA,
                               csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_mat_info)info,
                               f,
                               x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseDcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               m,
                               nnz,
                               alpha,
                               (rocsparse_mat_descr)descrA,
                               csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_mat_info)info,
                               f,
                               x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseCcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               m,
                               nnz,
                               (const rocsparse_float_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_mat_info)info,
                               (const rocsparse_float_complex*)f,
                               (rocsparse_float_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               m,
                               nnz,
                               (const rocsparse_double_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_mat_info)info,
                               (const rocsparse_double_complex*)f,
                               (rocsparse_double_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}
