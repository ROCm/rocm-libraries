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
    hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_bsrsv_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));
}

hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_sbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipDirectionToHCCDirection(dir),
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipDirectionToHCCDirection(dir),
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_cbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipDirectionToHCCDirection(dir),
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_float_complex*)bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipsparse::hipDirectionToHCCDirection(dir),
                                          hipsparse::hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_double_complex*)bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return hipsparse::rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dir),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dir),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dir),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_float_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipDirectionToHCCDirection(dir),
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_double_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dir),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  mb,
                                  nnzb,
                                  (rocsparse_mat_descr)descrA,
                                  bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dir),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  mb,
                                  nnzb,
                                  (rocsparse_mat_descr)descrA,
                                  bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dir),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  mb,
                                  nnzb,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsv_analysis((rocsparse_handle)handle,
                                  hipsparse::hipDirectionToHCCDirection(dir),
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  mb,
                                  nnzb,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dir),
                               hipsparse::hipOperationToHCCOperation(transA),
                               mb,
                               nnzb,
                               alpha,
                               (rocsparse_mat_descr)descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               f,
                               x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dir),
                               hipsparse::hipOperationToHCCOperation(transA),
                               mb,
                               nnzb,
                               alpha,
                               (rocsparse_mat_descr)descrA,
                               bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               f,
                               x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dir),
                               hipsparse::hipOperationToHCCOperation(transA),
                               mb,
                               nnzb,
                               (const rocsparse_float_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_float_complex*)f,
                               (rocsparse_float_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsv_solve((rocsparse_handle)handle,
                               hipsparse::hipDirectionToHCCDirection(dir),
                               hipsparse::hipOperationToHCCOperation(transA),
                               mb,
                               nnzb,
                               (const rocsparse_double_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_double_complex*)f,
                               (rocsparse_double_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}
