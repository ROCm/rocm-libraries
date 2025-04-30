/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

/*! \file
 *  \brief Implementation of the compatibility refactor APIs that require especial
 *  calls to hipSOLVER or cuSOLVER.
 */

#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"

#include <cusolverRf.h>

extern "C" {

/******************** HANDLE ********************/
hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfCreate((cusolverRfHandle_t*)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfDestroy((cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** NON-BATCHED ********************/
hipsolverStatus_t hipsolverRfSetupDevice(int                 n,
                                         int                 nnzA,
                                         int*                csrRowPtrA,
                                         int*                csrColIndA,
                                         double*             csrValA,
                                         int                 nnzL,
                                         int*                csrRowPtrL,
                                         int*                csrColIndL,
                                         double*             csrValL,
                                         int                 nnzU,
                                         int*                csrRowPtrU,
                                         int*                csrColIndU,
                                         double*             csrValU,
                                         int*                P,
                                         int*                Q,
                                         hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfSetupDevice(n,
                                                            nnzA,
                                                            csrRowPtrA,
                                                            csrColIndA,
                                                            csrValA,
                                                            nnzL,
                                                            csrRowPtrL,
                                                            csrColIndL,
                                                            csrValL,
                                                            nnzU,
                                                            csrRowPtrU,
                                                            csrColIndU,
                                                            csrValU,
                                                            P,
                                                            Q,
                                                            (cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetupHost(int                 n,
                                       int                 nnzA,
                                       int*                h_csrRowPtrA,
                                       int*                h_csrColIndA,
                                       double*             h_csrValA,
                                       int                 nnzL,
                                       int*                h_csrRowPtrL,
                                       int*                h_csrColIndL,
                                       double*             h_csrValL,
                                       int                 nnzU,
                                       int*                h_csrRowPtrU,
                                       int*                h_csrColIndU,
                                       double*             h_csrValU,
                                       int*                h_P,
                                       int*                h_Q,
                                       hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfSetupHost(n,
                                                          nnzA,
                                                          h_csrRowPtrA,
                                                          h_csrColIndA,
                                                          h_csrValA,
                                                          nnzL,
                                                          h_csrRowPtrL,
                                                          h_csrColIndL,
                                                          h_csrValL,
                                                          nnzU,
                                                          h_csrRowPtrU,
                                                          h_csrColIndU,
                                                          h_csrValU,
                                                          h_P,
                                                          h_Q,
                                                          (cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfAccessBundledFactorsDevice(
    hipsolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfAccessBundledFactorsDevice((cusolverRfHandle_t)handle, nnzM, Mp, Mi, Mx));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfAnalyze((cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfExtractBundledFactorsHost(
    hipsolverRfHandle_t handle, int* h_nnzM, int** h_Mp, int** h_Mi, double** h_Mx)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfExtractBundledFactorsHost((cusolverRfHandle_t)handle, h_nnzM, h_Mp, h_Mi, h_Mx));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfExtractSplitFactorsHost(hipsolverRfHandle_t handle,
                                                     int*                h_nnzL,
                                                     int**               h_Lp,
                                                     int**               h_Li,
                                                     double**            h_Lx,
                                                     int*                h_nnzU,
                                                     int**               h_Up,
                                                     int**               h_Ui,
                                                     double**            h_Ux)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfExtractSplitFactorsHost(
        (cusolverRfHandle_t)handle, h_nnzL, h_Lp, h_Li, h_Lx, h_nnzU, h_Up, h_Ui, h_Ux));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfGet_Algs(hipsolverRfHandle_t           handle,
                                      hipsolverRfFactorization_t*   fact_alg,
                                      hipsolverRfTriangularSolve_t* solve_alg)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!fact_alg)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!solve_alg)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    cusolverRfFactorization_t   fact_algC;
    cusolverRfTriangularSolve_t solve_algC;
    cusolverStatus_t            status
        = cusolverRfGetAlgs((cusolverRfHandle_t)handle, &fact_algC, &solve_algC);

    *fact_alg  = hipsolver::cuda2hip_factorization(fact_algC);
    *solve_alg = hipsolver::cuda2hip_trisolve(solve_algC);
    return hipsolver::cuda2hip_status(status);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetMatrixFormat(hipsolverRfHandle_t        handle,
                                             hipsolverRfMatrixFormat_t* format,
                                             hipsolverRfUnitDiagonal_t* diag)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!format)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!diag)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    cusolverRfMatrixFormat_t formatC;
    cusolverRfUnitDiagonal_t diagC;
    cusolverStatus_t         status
        = cusolverRfGetMatrixFormat((cusolverRfHandle_t)handle, &formatC, &diagC);

    *format = hipsolver::cuda2hip_matrixformat(formatC);
    *diag   = hipsolver::cuda2hip_unitdiag(diagC);
    return hipsolver::cuda2hip_status(status);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetNumericBoostReport(hipsolverRfHandle_t              handle,
                                                   hipsolverRfNumericBoostReport_t* report)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!report)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    cusolverRfNumericBoostReport_t reportC;
    cusolverStatus_t status = cusolverRfGetNumericBoostReport((cusolverRfHandle_t)handle, &reportC);

    *report = hipsolver::cuda2hip_boostrep(reportC);
    return hipsolver::cuda2hip_status(status);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t
    hipsolverRfGetNumericProperties(hipsolverRfHandle_t handle, double* zero, double* boost)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfGetNumericProperties((cusolverRfHandle_t)handle, zero, boost));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetResetValuesFastMode(hipsolverRfHandle_t               handle,
                                                    hipsolverRfResetValuesFastMode_t* fastMode)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!fastMode)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    cusolverRfResetValuesFastMode_t fastModeC;
    cusolverStatus_t                status
        = cusolverRfGetResetValuesFastMode((cusolverRfHandle_t)handle, &fastModeC);

    *fastMode = hipsolver::cuda2hip_resetvalsfm(fastModeC);
    return hipsolver::cuda2hip_status(status);
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfRefactor((cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfResetValues(int                 n,
                                         int                 nnzA,
                                         int*                csrRowPtrA,
                                         int*                csrColIndA,
                                         double*             csrValA,
                                         int*                P,
                                         int*                Q,
                                         hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfResetValues(
        n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, (cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetAlgs(hipsolverRfHandle_t          handle,
                                     hipsolverRfFactorization_t   fact_alg,
                                     hipsolverRfTriangularSolve_t solve_alg)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfSetAlgs((cusolverRfHandle_t)handle,
                                                        hipsolver::hip2cuda_factorization(fact_alg),
                                                        hipsolver::hip2cuda_trisolve(solve_alg)));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetMatrixFormat(hipsolverRfHandle_t       handle,
                                             hipsolverRfMatrixFormat_t format,
                                             hipsolverRfUnitDiagonal_t diag)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfSetMatrixFormat((cusolverRfHandle_t)handle,
                                  hipsolver::hip2cuda_matrixformat(format),
                                  hipsolver::hip2cuda_unitdiag(diag)));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetNumericProperties(hipsolverRfHandle_t handle,
                                                  double              effective_zero,
                                                  double              boost_val)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfSetNumericProperties((cusolverRfHandle_t)handle, effective_zero, boost_val));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetResetValuesFastMode(hipsolverRfHandle_t              handle,
                                                    hipsolverRfResetValuesFastMode_t fastMode)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfSetResetValuesFastMode(
        (cusolverRfHandle_t)handle, hipsolver::hip2cuda_resetvalsfm(fastMode)));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSolve(hipsolverRfHandle_t handle,
                                   int*                P,
                                   int*                Q,
                                   int                 nrhs,
                                   double*             Temp,
                                   int                 ldt,
                                   double*             XF,
                                   int                 ldxf)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfSolve((cusolverRfHandle_t)handle, P, Q, nrhs, Temp, ldt, XF, ldxf));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** BATCHED ********************/
hipsolverStatus_t hipsolverRfBatchSetupHost(int                 batchSize,
                                            int                 n,
                                            int                 nnzA,
                                            int*                h_csrRowPtrA,
                                            int*                h_csrColIndA,
                                            double*             h_csrValA_array[],
                                            int                 nnzL,
                                            int*                h_csrRowPtrL,
                                            int*                h_csrColIndL,
                                            double*             h_csrValL,
                                            int                 nnzU,
                                            int*                h_csrRowPtrU,
                                            int*                h_csrColIndU,
                                            double*             h_csrValU,
                                            int*                h_P,
                                            int*                h_Q,
                                            hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfBatchSetupHost(batchSize,
                                                               n,
                                                               nnzA,
                                                               h_csrRowPtrA,
                                                               h_csrColIndA,
                                                               h_csrValA_array,
                                                               nnzL,
                                                               h_csrRowPtrL,
                                                               h_csrColIndL,
                                                               h_csrValL,
                                                               nnzU,
                                                               h_csrRowPtrU,
                                                               h_csrColIndU,
                                                               h_csrValU,
                                                               h_P,
                                                               h_Q,
                                                               (cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchAnalyze(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfBatchAnalyze((cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchRefactor(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfBatchRefactor((cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchResetValues(int                 batchSize,
                                              int                 n,
                                              int                 nnzA,
                                              int*                csrRowPtrA,
                                              int*                csrColIndA,
                                              double*             csrValA_array[],
                                              int*                P,
                                              int*                Q,
                                              hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverRfBatchResetValues(batchSize,
                                                                 n,
                                                                 nnzA,
                                                                 csrRowPtrA,
                                                                 csrColIndA,
                                                                 csrValA_array,
                                                                 P,
                                                                 Q,
                                                                 (cusolverRfHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchSolve(hipsolverRfHandle_t handle,
                                        int*                P,
                                        int*                Q,
                                        int                 nrhs,
                                        double*             Temp,
                                        int                 ldt,
                                        double*             XF_array[],
                                        int                 ldxf)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfBatchSolve((cusolverRfHandle_t)handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle, int* position)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverRfBatchZeroPivot((cusolverRfHandle_t)handle, position));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

} //extern C
