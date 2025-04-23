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
    hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csrilu0_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dscsrilu0_numeric_boost(
        (rocsparse_handle)handle, (rocsparse_mat_info)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseDcsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dcsrilu0_numeric_boost(
        (rocsparse_handle)handle, (rocsparse_mat_info)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dccsrilu0_numeric_boost((rocsparse_handle)handle,
                                          (rocsparse_mat_info)info,
                                          enable_boost,
                                          tol,
                                          (rocsparse_float_complex*)boost_val));
}

hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrilu0_numeric_boost((rocsparse_handle)handle,
                                         (rocsparse_mat_info)info,
                                         enable_boost,
                                         tol,
                                         (rocsparse_double_complex*)boost_val));
}

hipsparseStatus_t hipsparseScsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_scsrilu0_buffer_size((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseDcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dcsrilu0_buffer_size((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseCcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_ccsrilu0_buffer_size((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseZcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zcsrilu0_buffer_size((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_scsrilu0_buffer_size((rocsparse_handle)handle,
                                       m,
                                       nnz,
                                       (rocsparse_mat_descr)descrA,
                                       csrSortedValA,
                                       csrSortedRowPtrA,
                                       csrSortedColIndA,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrilu0_buffer_size((rocsparse_handle)handle,
                                       m,
                                       nnz,
                                       (rocsparse_mat_descr)descrA,
                                       csrSortedValA,
                                       csrSortedRowPtrA,
                                       csrSortedColIndA,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrilu0_buffer_size((rocsparse_handle)handle,
                                       m,
                                       nnz,
                                       (rocsparse_mat_descr)descrA,
                                       (rocsparse_float_complex*)csrSortedValA,
                                       csrSortedRowPtrA,
                                       csrSortedColIndA,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrilu0_buffer_size((rocsparse_handle)handle,
                                       m,
                                       nnz,
                                       (rocsparse_mat_descr)descrA,
                                       (rocsparse_double_complex*)csrSortedValA,
                                       csrSortedRowPtrA,
                                       csrSortedColIndA,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseScsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const float*              csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsrilu0_analysis((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseDcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const double*             csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsrilu0_analysis((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseCcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_ccsrilu0_analysis((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseZcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipDoubleComplex*   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zcsrilu0_analysis((rocsparse_handle)handle,
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

hipsparseStatus_t hipsparseScsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_scsrilu0((rocsparse_handle)handle,
                                                                    m,
                                                                    nnz,
                                                                    (rocsparse_mat_descr)descrA,
                                                                    csrSortedValA_valM,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (rocsparse_mat_info)info,
                                                                    rocsparse_solve_policy_auto,
                                                                    pBuffer));
}

hipsparseStatus_t hipsparseDcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_dcsrilu0((rocsparse_handle)handle,
                                                                    m,
                                                                    nnz,
                                                                    (rocsparse_mat_descr)descrA,
                                                                    csrSortedValA_valM,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (rocsparse_mat_info)info,
                                                                    rocsparse_solve_policy_auto,
                                                                    pBuffer));
}

hipsparseStatus_t hipsparseCcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrilu0((rocsparse_handle)handle,
                           m,
                           nnz,
                           (rocsparse_mat_descr)descrA,
                           (rocsparse_float_complex*)csrSortedValA_valM,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_mat_info)info,
                           rocsparse_solve_policy_auto,
                           pBuffer));
}

hipsparseStatus_t hipsparseZcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrilu0((rocsparse_handle)handle,
                           m,
                           nnz,
                           (rocsparse_mat_descr)descrA,
                           (rocsparse_double_complex*)csrSortedValA_valM,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_mat_info)info,
                           rocsparse_solve_policy_auto,
                           pBuffer));
}
