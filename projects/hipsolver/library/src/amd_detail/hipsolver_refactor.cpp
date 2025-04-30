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
 *  calls to hipSOLVER or rocSOLVER.
 */

#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"
#include "lib_macros.hpp"

#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>

extern "C" {

/******************** HANDLE ********************/
struct hipsolverRfHandle
{
    hipsolverRfResetValuesFastMode_t fast_mode;
    hipsolverRfMatrixFormat_t        matrix_format;
    hipsolverRfUnitDiagonal_t        diag_format;
    hipsolverRfNumericBoostReport_t  numeric_boost;

    hipsolverRfFactorization_t   fact_alg;
    hipsolverRfTriangularSolve_t solve_alg;

    rocblas_handle   handle;
    rocsolver_rfinfo rfinfo;

    rocblas_int n, nnzA, nnzL, nnzU, nnzLU, batch_count;
    double      effective_zero;
    double      boost_val;

    rocblas_int* dPtrA;
    rocblas_int* dIndA;
    double*      dValA;

    rocblas_int *dPtrL, *hPtrL;
    rocblas_int *dIndL, *hIndL;
    double *     dValL, *hValL;

    rocblas_int *dPtrU, *hPtrU;
    rocblas_int *dIndU, *hIndU;
    double *     dValU, *hValU;

    rocblas_int *dPtrLU, *hPtrLU;
    rocblas_int *dIndLU, *hIndLU;
    double *     dValLU, *hValLU;

    rocblas_int* dP;
    rocblas_int* dQ;

    char *d_buffer, *h_buffer;

    // Constructor
    explicit hipsolverRfHandle()
        : fast_mode(HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF)
        , matrix_format(HIPSOLVERRF_MATRIX_FORMAT_CSR)
        , diag_format(HIPSOLVERRF_UNIT_DIAGONAL_STORED_L)
        , numeric_boost(HIPSOLVERRF_NUMERIC_BOOST_NOT_USED)
        , fact_alg(HIPSOLVERRF_FACTORIZATION_ALG0)
        , solve_alg(HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1)
        , n(0)
        , nnzA(0)
        , nnzL(0)
        , nnzU(0)
        , nnzLU(0)
        , batch_count(0)
        , effective_zero(0.0)
        , boost_val(0.0)
        , d_buffer(nullptr)
        , h_buffer(nullptr)
    {
    }

    // Allocate device memory
    hipsolverStatus_t malloc_device(int n, int nnzA, int nnzL, int nnzU)
    {
        if(n < 0 || nnzA < 0 || nnzL < 0 || nnzU < 0)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        if(this->n != n || this->nnzA != nnzA || this->nnzL != nnzL || this->nnzU != nnzU)
        {
            int nnzLU = nnzL - n + nnzU;

            if(this->h_buffer)
            {
                free(this->h_buffer);
                this->h_buffer = nullptr;
            }

            if(this->d_buffer)
            {
                if(hipFree(this->d_buffer) != hipSuccess)
                    return HIPSOLVER_STATUS_INTERNAL_ERROR;
                this->d_buffer = nullptr;
            }

            size_t size_dPtrA = sizeof(rocblas_int) * (n + 1);
            size_t size_dIndA = sizeof(rocblas_int) * nnzA;
            size_t size_dValA = sizeof(double) * nnzA;

            size_t size_dPtrL = sizeof(rocblas_int) * (n + 1);
            size_t size_dIndL = sizeof(rocblas_int) * nnzL;
            size_t size_dValL = sizeof(double) * nnzL;

            size_t size_dPtrU = sizeof(rocblas_int) * (n + 1);
            size_t size_dIndU = sizeof(rocblas_int) * nnzU;
            size_t size_dValU = sizeof(double) * nnzU;

            size_t size_dPtrLU = sizeof(rocblas_int) * (n + 1);
            size_t size_dIndLU = sizeof(rocblas_int) * nnzLU;
            size_t size_dValLU = sizeof(double) * nnzLU;

            size_t size_dP = sizeof(rocblas_int) * n;
            size_t size_dQ = sizeof(rocblas_int) * n;

            // 128 byte alignment
            size_dPtrL  = ((size_dPtrL - 1) / 128 + 1) * 128;
            size_dIndL  = ((size_dIndL - 1) / 128 + 1) * 128;
            size_dValL  = ((size_dValL - 1) / 128 + 1) * 128;
            size_dPtrU  = ((size_dPtrU - 1) / 128 + 1) * 128;
            size_dIndU  = ((size_dIndU - 1) / 128 + 1) * 128;
            size_dValU  = ((size_dValU - 1) / 128 + 1) * 128;
            size_dPtrLU = ((size_dPtrLU - 1) / 128 + 1) * 128;
            size_dIndLU = ((size_dIndLU - 1) / 128 + 1) * 128;
            size_dValLU = ((size_dValLU - 1) / 128 + 1) * 128;
            size_dP     = ((size_dP - 1) / 128 + 1) * 128;
            size_dQ     = ((size_dQ - 1) / 128 + 1) * 128;

            size_t size_buffer = size_dPtrA + size_dIndA + size_dValA + size_dPtrL + size_dIndL
                                 + size_dValL + size_dPtrU + size_dIndU + size_dValU + size_dPtrLU
                                 + size_dIndLU + size_dValLU + size_dP + size_dQ;

            if(hipMalloc(&this->d_buffer, size_buffer) != hipSuccess)
                return HIPSOLVER_STATUS_ALLOC_FAILED;

            char* temp_buf;
            this->dPtrA  = (rocblas_int*)(temp_buf = this->d_buffer);
            this->dPtrL  = (rocblas_int*)(temp_buf += size_dPtrA);
            this->dPtrU  = (rocblas_int*)(temp_buf += size_dPtrL);
            this->dPtrLU = (rocblas_int*)(temp_buf += size_dPtrU);

            this->dIndA  = (rocblas_int*)(temp_buf += size_dPtrLU);
            this->dIndL  = (rocblas_int*)(temp_buf += size_dIndA);
            this->dIndU  = (rocblas_int*)(temp_buf += size_dIndL);
            this->dIndLU = (rocblas_int*)(temp_buf += size_dIndU);

            this->dP = (rocblas_int*)(temp_buf += size_dIndLU);
            this->dQ = (rocblas_int*)(temp_buf += size_dP);

            this->dValA  = (double*)(temp_buf += size_dQ);
            this->dValL  = (double*)(temp_buf += size_dValA);
            this->dValU  = (double*)(temp_buf += size_dValL);
            this->dValLU = (double*)(temp_buf += size_dValU);

            this->n     = n;
            this->nnzA  = nnzA;
            this->nnzL  = nnzL;
            this->nnzU  = nnzU;
            this->nnzLU = nnzLU;
        }

        return HIPSOLVER_STATUS_SUCCESS;
    }

    // Allocate host memory
    hipsolverStatus_t malloc_host()
    {
        if(!this->h_buffer)
        {
            size_t size_hPtrL = sizeof(rocblas_int) * (n + 1);
            size_t size_hIndL = sizeof(rocblas_int) * nnzL;
            size_t size_hValL = sizeof(double) * nnzL;

            size_t size_hPtrU = sizeof(rocblas_int) * (n + 1);
            size_t size_hIndU = sizeof(rocblas_int) * nnzU;
            size_t size_hValU = sizeof(double) * nnzU;

            size_t size_hPtrLU = sizeof(rocblas_int) * (n + 1);
            size_t size_hIndLU = sizeof(rocblas_int) * nnzLU;
            size_t size_hValLU = sizeof(double) * nnzLU;

            // 128 byte alignment
            size_hPtrL  = ((size_hPtrL - 1) / 128 + 1) * 128;
            size_hIndL  = ((size_hIndL - 1) / 128 + 1) * 128;
            size_hValL  = ((size_hValL - 1) / 128 + 1) * 128;
            size_hPtrU  = ((size_hPtrU - 1) / 128 + 1) * 128;
            size_hIndU  = ((size_hIndU - 1) / 128 + 1) * 128;
            size_hValU  = ((size_hValU - 1) / 128 + 1) * 128;
            size_hPtrLU = ((size_hPtrLU - 1) / 128 + 1) * 128;
            size_hIndLU = ((size_hIndLU - 1) / 128 + 1) * 128;
            size_hValLU = ((size_hValLU - 1) / 128 + 1) * 128;

            size_t size_buffer = size_hPtrL + size_hIndL + size_hValL + size_hPtrU + size_hIndU
                                 + size_hValU + size_hPtrLU + size_hIndLU + size_hValLU;

            this->h_buffer = (char*)malloc(size_buffer);
            if(!this->h_buffer)
                return HIPSOLVER_STATUS_ALLOC_FAILED;

            char* temp_buf;
            this->hPtrL  = (rocblas_int*)(temp_buf = this->h_buffer);
            this->hPtrU  = (rocblas_int*)(temp_buf += size_hPtrL);
            this->hPtrLU = (rocblas_int*)(temp_buf += size_hPtrU);

            this->hIndL  = (rocblas_int*)(temp_buf += size_hPtrLU);
            this->hIndU  = (rocblas_int*)(temp_buf += size_hIndL);
            this->hIndLU = (rocblas_int*)(temp_buf += size_hIndU);

            this->hValL  = (double*)(temp_buf += size_hIndLU);
            this->hValU  = (double*)(temp_buf += size_hValL);
            this->hValLU = (double*)(temp_buf += size_hValU);
        }

        return HIPSOLVER_STATUS_SUCCESS;
    }

    // Free memory
    void free_all()
    {
        if(this->h_buffer)
        {
            free(this->h_buffer);
            this->h_buffer = nullptr;
        }

        if(this->d_buffer)
        {
            hipFree(this->d_buffer);
            this->d_buffer = nullptr;
        }
    }
};

hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = new hipsolverRfHandle;
    rocblas_status     status;

    if((status = rocblas_create_handle(&rf->handle)) != rocblas_status_success)
    {
        delete rf;
        return hipsolver::rocblas2hip_status(status);
    }

    if((status = rocsolver_create_rfinfo(&rf->rfinfo, rf->handle)) != rocblas_status_success)
    {
        rocblas_destroy_handle(rf->handle);
        delete rf;
        return hipsolver::rocblas2hip_status(status);
    }

    *handle = rf;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    rf->free_all();
    rocsolver_destroy_rfinfo(rf->rfinfo);
    rocblas_destroy_handle(rf->handle);
    delete rf;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!csrRowPtrA || !csrColIndA || !csrValA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrL || !csrColIndL || !csrValL)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrU || !csrColIndU || !csrValU)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!P || !Q)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    CHECK_HIPSOLVER_ERROR(rf->malloc_device(n, nnzA, nnzL, nnzU));

    CHECK_HIP_ERROR(
        hipMemcpy(rf->dPtrA, csrRowPtrA, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->dIndA, csrColIndA, sizeof(rocblas_int) * nnzA, hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(hipMemcpy(rf->dValA, csrValA, sizeof(double) * nnzA, hipMemcpyDeviceToDevice));

    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_sumlu(rf->handle,
                                               n,
                                               nnzL,
                                               csrRowPtrL,
                                               csrColIndL,
                                               csrValL,
                                               nnzU,
                                               csrRowPtrU,
                                               csrColIndU,
                                               csrValU,
                                               rf->dPtrLU,
                                               rf->dIndLU,
                                               rf->dValLU));

    CHECK_HIP_ERROR(hipMemcpy(rf->dP, P, sizeof(rocblas_int) * n, hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(hipMemcpy(rf->dQ, Q, sizeof(rocblas_int) * n, hipMemcpyDeviceToDevice));

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetupHost(int                 n,
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
    if(!csrRowPtrA || !csrColIndA || !csrValA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrL || !csrColIndL || !csrValL)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrU || !csrColIndU || !csrValU)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!P || !Q)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    CHECK_HIPSOLVER_ERROR(rf->malloc_device(n, nnzA, nnzL, nnzU));

    CHECK_HIP_ERROR(
        hipMemcpy(rf->dPtrA, csrRowPtrA, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->dIndA, csrColIndA, sizeof(rocblas_int) * nnzA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(rf->dValA, csrValA, sizeof(double) * nnzA, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(
        hipMemcpy(rf->dPtrL, csrRowPtrL, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->dIndL, csrColIndL, sizeof(rocblas_int) * nnzL, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(rf->dValL, csrValL, sizeof(double) * nnzL, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(
        hipMemcpy(rf->dPtrU, csrRowPtrU, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->dIndU, csrColIndU, sizeof(rocblas_int) * nnzU, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(rf->dValU, csrValU, sizeof(double) * nnzU, hipMemcpyHostToDevice));

    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_sumlu(rf->handle,
                                               n,
                                               nnzL,
                                               rf->dPtrL,
                                               rf->dIndL,
                                               rf->dValL,
                                               nnzU,
                                               rf->dPtrU,
                                               rf->dIndU,
                                               rf->dValU,
                                               rf->dPtrLU,
                                               rf->dIndLU,
                                               rf->dValLU));

    CHECK_HIP_ERROR(hipMemcpy(rf->dP, P, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(rf->dQ, Q, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!nnzM || !Mp || !Mi || !Mx)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    if(!rf->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *nnzM = rf->nnzLU;
    *Mp   = rf->dPtrLU;
    *Mi   = rf->dIndLU;
    *Mx   = rf->dValLU;

    return HIPSOLVER_STATUS_SUCCESS;
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

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    if(!rf->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return hipsolver::rocblas2hip_status(rocsolver_dcsrrf_analysis(rf->handle,
                                                                   rf->n,
                                                                   1,
                                                                   rf->nnzA,
                                                                   rf->dPtrA,
                                                                   rf->dIndA,
                                                                   rf->dValA,
                                                                   rf->nnzLU,
                                                                   rf->dPtrLU,
                                                                   rf->dIndLU,
                                                                   rf->dValLU,
                                                                   rf->dP,
                                                                   rf->dQ,
                                                                   // pass dummy values for B
                                                                   rf->dValA,
                                                                   rf->n,
                                                                   rf->rfinfo));
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
    if(!h_nnzM || !h_Mp || !h_Mi || !h_Mx)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    if(!rf->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    CHECK_HIPSOLVER_ERROR(rf->malloc_host());

    CHECK_HIP_ERROR(hipMemcpy(
        rf->hPtrLU, rf->dPtrLU, sizeof(rocblas_int) * (rf->n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->hIndLU, rf->dIndLU, sizeof(rocblas_int) * rf->nnzLU, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->hValLU, rf->dValLU, sizeof(double) * rf->nnzLU, hipMemcpyDeviceToHost));

    *h_nnzM = rf->nnzLU;
    *h_Mp   = rf->hPtrLU;
    *h_Mi   = rf->hIndLU;
    *h_Mx   = rf->hValLU;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!h_nnzL || !h_Lp || !h_Li || !h_Lx)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!h_nnzU || !h_Up || !h_Ui || !h_Ux)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    if(!rf->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    CHECK_HIPSOLVER_ERROR(rf->malloc_host());

    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_splitlu(rf->handle,
                                                 rf->n,
                                                 rf->nnzLU,
                                                 rf->dPtrLU,
                                                 rf->dIndLU,
                                                 rf->dValLU,
                                                 rf->dPtrL,
                                                 rf->dIndL,
                                                 rf->dValL,
                                                 rf->dPtrU,
                                                 rf->dIndU,
                                                 rf->dValU));

    CHECK_HIP_ERROR(
        hipMemcpy(rf->hPtrL, rf->dPtrL, sizeof(rocblas_int) * (rf->n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->hIndL, rf->dIndL, sizeof(rocblas_int) * rf->nnzL, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->hValL, rf->dValL, sizeof(double) * rf->nnzL, hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(
        hipMemcpy(rf->hPtrU, rf->dPtrU, sizeof(rocblas_int) * (rf->n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->hIndU, rf->dIndU, sizeof(rocblas_int) * rf->nnzU, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(rf->hValU, rf->dValU, sizeof(double) * rf->nnzU, hipMemcpyDeviceToHost));

    *h_nnzL = rf->nnzL;
    *h_Lp   = rf->hPtrL;
    *h_Li   = rf->hIndL;
    *h_Lx   = rf->hValL;

    *h_nnzU = rf->nnzU;
    *h_Up   = rf->hPtrU;
    *h_Ui   = rf->hIndU;
    *h_Ux   = rf->hValU;

    return HIPSOLVER_STATUS_SUCCESS;
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

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    *fact_alg             = rf->fact_alg;
    *solve_alg            = rf->solve_alg;

    return HIPSOLVER_STATUS_SUCCESS;
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

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    *format               = rf->matrix_format;
    *diag                 = rf->diag_format;

    return HIPSOLVER_STATUS_SUCCESS;
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

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    *report               = rf->numeric_boost;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!zero)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!boost)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    *zero                 = rf->effective_zero;
    *boost                = rf->boost_val;

    return HIPSOLVER_STATUS_SUCCESS;
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

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    *fastMode             = rf->fast_mode;

    return HIPSOLVER_STATUS_SUCCESS;
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

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    if(!rf->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return hipsolver::rocblas2hip_status(rocsolver_dcsrrf_refactlu(rf->handle,
                                                                   rf->n,
                                                                   rf->nnzA,
                                                                   rf->dPtrA,
                                                                   rf->dIndA,
                                                                   rf->dValA,
                                                                   rf->nnzLU,
                                                                   rf->dPtrLU,
                                                                   rf->dIndLU,
                                                                   rf->dValLU,
                                                                   rf->dP,
                                                                   rf->dQ,
                                                                   rf->rfinfo));
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
    if(!csrRowPtrA || !csrColIndA || !csrValA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!P || !Q)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    if(rf->n != n)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(rf->nnzA != nnzA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!rf->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    CHECK_HIP_ERROR(hipMemcpy(rf->dValA, csrValA, sizeof(double) * nnzA, hipMemcpyDeviceToDevice));

    return HIPSOLVER_STATUS_SUCCESS;
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
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetResetValuesFastMode(hipsolverRfHandle_t              handle,
                                                    hipsolverRfResetValuesFastMode_t fastMode)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
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

    hipsolverRfHandle* rf = (hipsolverRfHandle*)handle;
    if(!rf->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return hipsolver::rocblas2hip_status(rocsolver_dcsrrf_solve(rf->handle,
                                                                rf->n,
                                                                nrhs,
                                                                rf->nnzLU,
                                                                rf->dPtrLU,
                                                                rf->dIndLU,
                                                                rf->dValLU,
                                                                rf->dP,
                                                                rf->dQ,
                                                                XF,
                                                                ldxf,
                                                                rf->rfinfo));
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
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchAnalyze(hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchRefactor(hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle, int* position)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

} //extern C
