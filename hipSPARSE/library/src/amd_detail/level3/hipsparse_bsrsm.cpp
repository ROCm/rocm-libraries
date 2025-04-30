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
    hipsparseXbsrsm2_zeroPivot(hipsparseHandle_t handle, bsrsm2Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse bsrsm2_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsrsm zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_bsrsm_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseDbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseCbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     (rocsparse_float_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseZbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dirA),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     (rocsparse_double_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseSbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dirA),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transX),
                                  mb,
                                  nrhs,
                                  nnzb,
                                  (const rocsparse_mat_descr)descrA,
                                  bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseDbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dirA),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transX),
                                  mb,
                                  nrhs,
                                  nnzb,
                                  (const rocsparse_mat_descr)descrA,
                                  bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseCbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dirA),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transX),
                                  mb,
                                  nrhs,
                                  nnzb,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseZbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dirA),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transX),
                                  mb,
                                  nrhs,
                                  nnzb,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseSbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const float*              B,
                                         int                       ldb,
                                         float*                    X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dirA),
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transX),
                               mb,
                               nrhs,
                               nnzb,
                               alpha,
                               (const rocsparse_mat_descr)descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               B,
                               ldb,
                               X,
                               ldx,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseDbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const double*             B,
                                         int                       ldb,
                                         double*                   X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dirA),
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transX),
                               mb,
                               nrhs,
                               nnzb,
                               alpha,
                               (const rocsparse_mat_descr)descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               B,
                               ldb,
                               X,
                               ldx,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseCbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipComplex*         B,
                                         int                       ldb,
                                         hipComplex*               X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dirA),
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transX),
                               mb,
                               nrhs,
                               nnzb,
                               (const rocsparse_float_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_float_complex*)B,
                               ldb,
                               (rocsparse_float_complex*)X,
                               ldx,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipDoubleComplex*   B,
                                         int                       ldb,
                                         hipDoubleComplex*         X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dirA),
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transX),
                               mb,
                               nrhs,
                               nnzb,
                               (const rocsparse_double_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_double_complex*)B,
                               ldb,
                               (rocsparse_double_complex*)X,
                               ldx,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}
