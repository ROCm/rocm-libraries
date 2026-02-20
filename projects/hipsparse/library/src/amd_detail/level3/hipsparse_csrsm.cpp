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
    hipsparseXcsrsm2_zeroPivot(hipsparseHandle_t handle, csrsm2Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse csrsm2_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsm zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csrsm_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const float*              alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const float*              csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const float*              B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const double*             alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const double*             csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const double*             B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipComplex*         alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipComplex*         B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     (const rocsparse_float_complex*)alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     (const rocsparse_float_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (const rocsparse_float_complex*)B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipDoubleComplex*   alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipDoubleComplex*   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipDoubleComplex*   B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsm_buffer_size((rocsparse_handle)handle,
                                     hipsparse::hipOperationToHCCOperation(transA),
                                     hipsparse::hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     (const rocsparse_double_complex*)alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     (const rocsparse_double_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (const rocsparse_double_complex*)B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseScsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const float*              alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const float*              B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transB),
                                  m,
                                  nrhs,
                                  nnz,
                                  alpha,
                                  (const rocsparse_mat_descr)descrA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  B,
                                  ldb,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseDcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const double*             alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const double*             B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transB),
                                  m,
                                  nrhs,
                                  nnz,
                                  alpha,
                                  (const rocsparse_mat_descr)descrA,
                                  csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  B,
                                  ldb,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseCcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipComplex*         alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipComplex*         B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transB),
                                  m,
                                  nrhs,
                                  nnz,
                                  (const rocsparse_float_complex*)alpha,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (const rocsparse_float_complex*)B,
                                  ldb,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseZcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipDoubleComplex*   alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipDoubleComplex*   B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsm_analysis((rocsparse_handle)handle,
                                  hipsparse::hipOperationToHCCOperation(transA),
                                  hipsparse::hipOperationToHCCOperation(transB),
                                  m,
                                  nrhs,
                                  nnz,
                                  (const rocsparse_double_complex*)alpha,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (const rocsparse_double_complex*)B,
                                  ldb,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseScsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         float*                    B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transB),
                               m,
                               nrhs,
                               nnz,
                               alpha,
                               (const rocsparse_mat_descr)descrA,
                               csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               B,
                               ldb,
                               (rocsparse_mat_info)info,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseDcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         double*                   B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transB),
                               m,
                               nrhs,
                               nnz,
                               alpha,
                               (const rocsparse_mat_descr)descrA,
                               csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               B,
                               ldb,
                               (rocsparse_mat_info)info,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseCcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipComplex*               B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transB),
                               m,
                               nrhs,
                               nnz,
                               (const rocsparse_float_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_float_complex*)B,
                               ldb,
                               (rocsparse_mat_info)info,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipDoubleComplex*         B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsm_solve((rocsparse_handle)handle,
                               hipsparse::hipOperationToHCCOperation(transA),
                               hipsparse::hipOperationToHCCOperation(transB),
                               m,
                               nrhs,
                               nnz,
                               (const rocsparse_double_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_double_complex*)B,
                               ldb,
                               (rocsparse_mat_info)info,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}
