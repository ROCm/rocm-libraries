/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>

#include "clientcommon.hpp"

static bool test_left_eigenvectors
    = false; // Computing left eigenvectors is not supported in cuSOLVER.
static bool test_right_eigenvectors = true;

template <testAPI_t API,
          typename I,
          typename SIZE,
          typename Td,
          typename Wd,
          typename INTd,
          typename Th>
void geev_checkBadArgs(const hipsolverHandle_t   handle,
                       const hipsolverDnParams_t params,
                       const hipsolverEigMode_t  jobvl,
                       const hipsolverEigMode_t  jobvr,
                       const I                   n,
                       Td                        dA,
                       const I                   lda,
                       Wd                        dW,
                       Td                        dVL,
                       const I                   ldvl,
                       Td                        dVR,
                       const I                   ldvr,
                       Td                        dWork,
                       const SIZE                dlwork,
                       Th                        hWork,
                       const SIZE                hlwork,
                       INTd                      dinfo)
{
    // Check handle.
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         (hipsolverHandle_t) nullptr,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    //
    // Check pointers.
    //
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         (hipsolverDnParams_t) nullptr,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         (Td) nullptr,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         (Wd) nullptr,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         HIPSOLVER_EIG_MODE_VECTOR,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         (Td) nullptr,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         HIPSOLVER_EIG_MODE_VECTOR,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         (Td) nullptr,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    /* Note: `workOnDevice` is still allowed to be `nullptr` */
    /* EXPECT_ROCBLAS_STATUS(hipsolver_geev(API, */
    /*                                      handle, */
    /*                                      params, */
    /*                                      jobvl, */
    /*                                      jobvr, */
    /*                                      n, */
    /*                                      dA, */
    /*                                      lda, */
    /*                                      dW, */
    /*                                      dVL, */
    /*                                      ldvl, */
    /*                                      dVR, */
    /*                                      ldvr, */
    /*                                      (Td) nullptr, */
    /*                                      dlwork, */
    /*                                      hWork, */
    /*                                      hlwork, */
    /*                                      dinfo), */
    /*                       HIPSOLVER_STATUS_INVALID_VALUE); */
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         (Th) nullptr,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         (INTd) nullptr),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    //
    // Check values.
    //
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         hipsolverEigMode_t(-1),
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         hipsolverEigMode_t(-1),
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         -1,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         -1,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         -1,
                                         dVR,
                                         ldvr,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                         handle,
                                         params,
                                         jobvl,
                                         jobvr,
                                         n,
                                         dA,
                                         lda,
                                         dW,
                                         dVL,
                                         ldvl,
                                         dVR,
                                         -1,
                                         dWork,
                                         dlwork,
                                         hWork,
                                         hlwork,
                                         dinfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API,
          bool      BATCHED,
          bool      STRIDED,
          typename T,
          typename W,
          typename I,
          typename SIZE>
void testing_geev_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    hipsolverEigMode_t     jobvl = HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverEigMode_t     jobvr = HIPSOLVER_EIG_MODE_NOVECTOR;
    I                      n     = 1;
    I                      lda   = 1;
    I                      ldvl  = 1;
    I                      ldvr  = 1;

    if(BATCHED)
    {
        // unsupported
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<W>   dW(1, 1, 1, 1);
        device_strided_batch_vector<T>   dVL(1, 1, 1, 1);
        device_strided_batch_vector<T>   dVR(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dVL.memcheck());
        CHECK_HIP_ERROR(dVR.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        SIZE size_dW, size_hW;
        CHECK_ROCBLAS_ERROR(hipsolver_geev_bufferSize(API,
                                                      handle,
                                                      params,
                                                      jobvl,
                                                      jobvr,
                                                      n,
                                                      dA.data(),
                                                      lda,
                                                      dW.data(),
                                                      dVL.data(),
                                                      ldvl,
                                                      dVR.data(),
                                                      ldvr,
                                                      &size_dW,
                                                      &size_hW));
        host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
        device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        geev_checkBadArgs<API>(handle,
                               params,
                               jobvl,
                               jobvr,
                               n,
                               dA.data(),
                               lda,
                               dW.data(),
                               dVL.data(),
                               ldvl,
                               dVR.data(),
                               ldvr,
                               dWork.data(),
                               size_dW,
                               hWork.data(),
                               size_hW,
                               dInfo.data());
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void geev_initData(const hipsolverHandle_t   handle,
                   const hipsolverDnParams_t params,
                   const I                   n,
                   Td&                       dA,
                   const I                   lda,
                   Th&                       hA)
{
    if(CPU)
    {
        // TODO: Add option to create defective matrices.

        // Create a (non-symmetric) Sylvester-Kac/Clement matrix.  For a given
        // dimension `n`, the spectrum of hA[0] is composed of the following `n`
        // distinct integers:
        //
        //    -(n-1), -(n-1) + 2, -(n-1) + 4, ..., (n-1) - 2, (n-1).
        //
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                hA[0][i + j * lda] = T(0);
            }
        }

        for(int i = 1; i < n; ++i)
        {
            hA[0][i + (i - 1) * lda] = T(n - i);
            hA[0][(i - 1) + i * lda] = T(i);
        }

        /* // Temporary: Create a symmetric Clement matrix to simplify testing. */
        /* // */
        /* // The main reason to rely on the symmetric version of the Clement */
        /* // matrices is to guarantee that computed eigenvalues will be real */
        /* // (numerically) for all relevant test sizes; this will be removed in */
        /* // favor of the general (non-symmetric) Clement matrices when the test */
        /* // routine is complete. */
        /* for(int i = 0; i < n; ++i) */
        /* { */
        /*     for(int j = 0; j < n; ++j) */
        /*     { */
        /*         if(i == j + 1) */
        /*         { */
        /*             hA[0][i + j * lda] = std::sqrt(i * (n - i)); */
        /*             hA[0][j + i * lda] = std::sqrt(i * (n - i)); */
        /*         } */
        /*         else */
        /*         { */
        /*             hA[0][i + j * lda] = T(0); */
        /*         } */
        /*     } */
        /* } */
        /* print_array(hA[0], n, n, lda, "hA[0] = "); */
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <testAPI_t API,
          typename T,
          typename W,
          typename I,
          typename SIZE,
          typename Td,
          typename Wd,
          typename INTd,
          typename Th,
          typename Wh,
          typename CTh,
          typename INTh>
void geev_getError(const hipsolverHandle_t   handle,
                   const hipsolverDnParams_t params,
                   const hipsolverEigMode_t  jobvl,
                   const hipsolverEigMode_t  jobvr,
                   const I                   n,
                   Td&                       dA,
                   const I                   lda,
                   Wd&                       dW,
                   Td&                       dVL,
                   const I                   ldvl,
                   Td&                       dVR,
                   const I                   ldvr,
                   Td&                       dWork,
                   const SIZE                dlwork,
                   Th&                       hWork,
                   const SIZE                hlwork,
                   INTd&                     dInfo,
                   Th&                       hA,
                   CTh&                      hCA,
                   Th&                       hARes,
                   Wh&                       hW,
                   Wh&                       hWRes,
                   CTh&                      hVL,
                   Th&                       hVLRes,
                   CTh&                      hVR,
                   Th&                       hVRRes,
                   INTh&                     hInfo,
                   INTh&                     hInfoRes,
                   Th&                       hD,
                   CTh&                      hCD,
                   Th&                       hErr,
                   CTh&                      hCErr,
                   double*                   max_err)
{
    // Initialize extra type used to simplify testing when type `T` is real.
    using CT = std::conditional_t<
        std::is_same_v<T, float>,
        hipsolverComplex,
        std::conditional_t<std::is_same_v<T, double>, hipsolverDoubleComplex, T>>;

    // Initialize variables used in 2nd run.
    bool               vl_2nd_run = test_left_eigenvectors;
    bool               vr_2nd_run = test_right_eigenvectors;
    hipsolverEigMode_t jobvl_2nd_run
        = test_left_eigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverEigMode_t jobvr_2nd_run
        = test_right_eigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR;
    I ldvl_2nd_run = n;
    I ldvr_2nd_run = n;
    I ldvl_{ldvl};
    I ldvr_{ldvr};

    // input data initialization
    geev_initData<true, true, T>(handle, params, n, dA, lda, hA);

    //
    // Execute computations
    // GPU lapack
    //
    CHECK_ROCBLAS_ERROR(hipsolver_geev(API,
                                       handle,
                                       params,
                                       jobvl,
                                       jobvr,
                                       n,
                                       dA.data(),
                                       lda,
                                       dW.data(),
                                       dVL.data(),
                                       ldvl,
                                       dVR.data(),
                                       ldvr,
                                       dWork.data(),
                                       dlwork,
                                       hWork.data(),
                                       hlwork,
                                       dInfo.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    if(jobvl != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
        vl_2nd_run    = false;
        jobvl_2nd_run = HIPSOLVER_EIG_MODE_NOVECTOR;
    }
    if(jobvr != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));
        vr_2nd_run    = false;
        jobvr_2nd_run = HIPSOLVER_EIG_MODE_NOVECTOR;
    }
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    //
    // `geev` is supposed to converge for all input test matrices, if
    // `info != 0` test has failed.
    //
    EXPECT_EQ(hInfoRes[0][0], 0);
    if(hInfoRes[0][0] != 0)
    {
        *max_err += 1;
        return;
    }

    //
    // Execute `geev` a second time to get missing left and right eigenvectors
    // (if necessary).
    //
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    if(vl_2nd_run || vr_2nd_run)
    {
        CHECK_ROCBLAS_ERROR(hipsolver_geev(API,
                                           handle,
                                           params,
                                           jobvl_2nd_run,
                                           jobvr_2nd_run,
                                           n,
                                           dA.data(),
                                           lda,
                                           dW.data(),
                                           dVL.data(),
                                           ldvl_2nd_run,
                                           dVR.data(),
                                           ldvr_2nd_run,
                                           dWork.data(),
                                           dlwork,
                                           hWork.data(),
                                           hlwork,
                                           dInfo.data()));

        if(jobvl_2nd_run != HIPSOLVER_EIG_MODE_NOVECTOR)
        {
            CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
            ldvl_ = ldvl_2nd_run;
        }
        if(jobvr_2nd_run != HIPSOLVER_EIG_MODE_NOVECTOR)
        {
            CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));
            ldvr_ = ldvr_2nd_run;
        }
    }

    //
    // [Not implemented:] Execute `geev` a third time (if necessary) to check if eigenvalues change when
    // `geev`'s input parameters change.
    //

    //
    // Compute error(s).
    // *** Note: FOR NOW, the tests' code require the spectrum of A to be real. ***
    //
    double err{};
    double normA = snorm('F', n, n, hA[0], lda);
    CT     alpha, beta;

    CT* A   = hCA[0];
    CT* D   = hCD[0];
    CT* VR  = hVR[0];
    CT* VL  = hVL[0];
    CT* Err = hCErr[0];
    if constexpr(!std::is_same<W, CT>())
    {
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                if(i == j)
                    D[i + j * n] = {hWRes[0][j], hWRes[0][j + n]};
                else
                    D[i + j * n] = CT(0);
            }
        }
    }
    else // is_same<W, CT>
    {
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                if(i == j)
                    D[i + j * n] = hWRes[0][j];
                else
                    D[i + j * n] = CT(0);
            }
        }
    }

    if constexpr(!is_complex<T>)
    {
        for(int j = 0; j < n; ++j)
        {
            auto VRij = [ld = ldvr_, V = hVRRes[0]](I i, I j) -> T { return V[i + j * ld]; };
            auto VLij = [ld = ldvl_, V = hVLRes[0]](I i, I j) -> T { return V[i + j * ld]; };
            for(int i = 0; i < n; ++i)
            {
                A[i + j * lda] = {hA[0][i + j * lda], 0};

                if(D[j + j * n].imag() > 0)
                {
                    if(test_right_eigenvectors)
                        VR[i + j * ldvr_] = {VRij(i, j), VRij(i, j + 1)};
                    if(test_left_eigenvectors)
                        VL[i + j * ldvl_] = {VLij(i, j), VLij(i, j + 1)};
                }
                else if(D[j + j * n].imag() < 0)
                {
                    if(test_right_eigenvectors)
                        VR[i + j * ldvr_] = {VRij(i, j - 1), -VRij(i, j)};
                    if(test_left_eigenvectors)
                        VL[i + j * ldvl_] = {VLij(i, j - 1), -VLij(i, j)};
                }
                else
                {
                    if(test_right_eigenvectors)
                        VR[i + j * ldvr_] = {VRij(i, j), 0};
                    if(test_left_eigenvectors)
                        VL[i + j * ldvl_] = {VLij(i, j), 0};
                }
            }
        }
    }
    else // is_complex<T> == true
    {
        VR = hVRRes[0];
        VL = hVLRes[0];
        A  = hA[0];
    }

    // [Not implemented:] 1. Check if eigenvalues computed with different parameters match.

    // 2a. If computing left eigenvectors: check if ||VL_i|| == 1, for 0 <= i < n, and if entry
    // with largest absolute value of VL is real.
    if(test_left_eigenvectors && (jobvl != HIPSOLVER_EIG_MODE_NOVECTOR))
    {
        for(int j = 0; j < n; ++j)
        {
            auto   VLij    = [ld = ldvl_, V = VL](I i, I j) -> CT { return V[i + j * ld]; };
            double norm    = 0.;
            double abs_max = 0.;
            I      imax    = I(0);
            for(int i = 0; i < n; ++i)
            {
                auto vij = VLij(i, j);
                norm += std::norm<double>({vij.real(), vij.imag()});
                if(std::abs(VLij(i, j)) >= abs_max)
                {
                    imax    = i;
                    abs_max = std::abs(VLij(i, j));
                }
            }
            err      = (std::sqrt(norm) - 1.) / normA;
            *max_err = *max_err < err ? err : *max_err;

            if(std::abs(VLij(imax, j).imag()) > 0)
            {
                *max_err += 1.;
            }
        }
    }

    // 2b. If computing right eigenvectors: check if ||VR_i|| == 1, for 0 <= i < n, and if entry
    // with largest absolute value of VR is real.
    if(test_right_eigenvectors && (jobvr != HIPSOLVER_EIG_MODE_NOVECTOR))
    {
        for(int j = 0; j < n; ++j)
        {
            auto   VRij    = [ld = ldvr_, V = VR](I i, I j) -> CT { return V[i + j * ld]; };
            double norm    = 0.;
            double abs_max = 0.;
            I      imax    = I(0);
            for(int i = 0; i < n; ++i)
            {
                auto vij = VRij(i, j);
                norm += std::norm<double>({vij.real(), vij.imag()});
                if(std::abs(VRij(i, j)) >= abs_max)
                {
                    imax    = i;
                    abs_max = std::abs(VRij(i, j));
                }
            }
            err      = (std::sqrt(norm) - 1.) / normA;
            *max_err = *max_err < err ? err : *max_err;

            if(std::abs(VRij(imax, j).imag()) > 0)
            {
                *max_err += 1.;
            }
        }
    }

    // 3a. Check reconstruction with right eigenvectors (VR): ||A*VR - VR*W||/|A|| <= n * eps
    if(test_right_eigenvectors && (jobvr != HIPSOLVER_EIG_MODE_NOVECTOR))
    {
        alpha = T(1);
        beta  = T(0);
        cpu_gemm(HIPSOLVER_OP_N, HIPSOLVER_OP_N, n, n, n, alpha, A, lda, VR, ldvr_, beta, Err, n);

        alpha = T(-1);
        beta  = T(1);
        cpu_gemm(HIPSOLVER_OP_N, HIPSOLVER_OP_N, n, n, n, alpha, VR, ldvr_, D, n, beta, Err, n);

        err = snorm('F', n, n, Err, n) / normA;
        /* std::cout << "err (VR) = " << err << std::endl; */

        *max_err = *max_err < err ? err : *max_err;
    }

    // 3b. Check reconstruction with left eigenvectors (VL): ||A'*VL - VL*W||/||A|| <= n * eps
    if(test_left_eigenvectors && (jobvl != HIPSOLVER_EIG_MODE_NOVECTOR))
    {
        alpha = T(1);
        beta  = T(0);
        cpu_gemm(HIPSOLVER_OP_T, HIPSOLVER_OP_N, n, n, n, alpha, A, lda, VL, ldvl_, beta, Err, n);

        alpha = T(-1);
        beta  = T(1);
        cpu_gemm(HIPSOLVER_OP_N, HIPSOLVER_OP_N, n, n, n, alpha, VL, ldvl_, D, n, beta, Err, n);

        err = snorm('F', n, n, Err, n) / normA;
        /* std::cout << "err (VL) = " << err << std::endl; */

        *max_err = *max_err < err ? err : *max_err;
    }

    // TODO
    // Save different errors in different variables instead of accumulating everything in `max_err`.
}

template <testAPI_t API,
          typename T,
          typename W,
          typename I,
          typename SIZE,
          typename Td,
          typename Wd,
          typename INTd,
          typename Th>
void geev_getPerfData(const hipsolverHandle_t   handle,
                      const hipsolverDnParams_t params,
                      const hipsolverEigMode_t  jobvl,
                      const hipsolverEigMode_t  jobvr,
                      const I                   n,
                      Td&                       dA,
                      const I                   lda,
                      Wd&                       dW,
                      Td&                       dVL,
                      const I                   ldvl,
                      Td&                       dVR,
                      const I                   ldvr,
                      Td&                       dWork,
                      const SIZE                dlwork,
                      Th&                       hWork,
                      const SIZE                hlwork,
                      INTd&                     dInfo,
                      Th&                       hA,
                      double*                   gpu_time_used,
                      double*                   cpu_time_used,
                      const int                 hot_calls,
                      const bool                perf)
{
    if(!perf)
    {
        // TODO
    }

    geev_initData<true, false, T>(handle, params, n, dA, lda, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geev_initData<false, true, T>(handle, params, n, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(hipsolver_geev(API,
                                           handle,
                                           params,
                                           jobvl,
                                           jobvr,
                                           n,
                                           dA.data(),
                                           lda,
                                           dW.data(),
                                           dVL.data(),
                                           ldvl,
                                           dVR.data(),
                                           ldvr,
                                           dWork.data(),
                                           dlwork,
                                           hWork.data(),
                                           hlwork,
                                           dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        geev_initData<false, true, T>(handle, params, n, dA, lda, hA);

        start = get_time_us_sync(stream);
        hipsolver_geev(API,
                       handle,
                       params,
                       jobvl,
                       jobvr,
                       n,
                       dA.data(),
                       lda,
                       dW.data(),
                       dVL.data(),
                       ldvl,
                       dVR.data(),
                       ldvr,
                       dWork.data(),
                       dlwork,
                       hWork.data(),
                       hlwork,
                       dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API,
          bool      BATCHED,
          bool      STRIDED,
          typename T,
          typename W,
          typename I,
          typename SIZE>
void testing_geev(Arguments& argus)
{
    // Initialize extra type used to simplify testing when type `T` is real.
    using CT = std::conditional_t<
        std::is_same_v<T, float>,
        hipsolverComplex,
        std::conditional_t<std::is_same_v<T, double>, hipsolverDoubleComplex, T>>;

    // get arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    char                   jobvlC = argus.get<char>("jobvl", 'N');
    char                   jobvrC = argus.get<char>("jobvr");
    I                      n      = argus.get<int>("n");
    I                      lda    = argus.get<int>("lda", n);
    I                      ldvl   = argus.get<int>("ldvl", n);
    I                      ldvr   = argus.get<int>("ldvr", n);

    hipsolverEigMode_t jobvl     = char2hipsolver_evect(jobvlC);
    hipsolverEigMode_t jobvr     = char2hipsolver_evect(jobvrC);
    int                bc        = argus.batch_count;
    int                hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_W    = is_complex<W> ? size_t(n) : size_t(2 * n);
    size_t size_VL   = jobvlC == 'N' ? size_t(n) * n : size_t(ldvl) * n;
    size_t size_VR   = jobvrC == 'N' ? size_t(n) * n : size_t(ldvr) * n;
    size_t size_D    = size_t(n) * n;
    size_t size_Err  = size_t(n) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes  = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes  = (argus.unit_check || argus.norm_check) ? size_W : 0;
    size_t size_VLRes = (argus.unit_check || argus.norm_check) ? size_VL : 0;
    size_t size_VRRes = (argus.unit_check || argus.norm_check) ? size_VR : 0;

    // check invalid sizes
    bool invalid_size    = (n < 0 || lda < n || bc < 0);
    bool invalid_size_vl = ldvl < (jobvl == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : n);
    bool invalid_size_vr = ldvr < (jobvr == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : n);
    if(invalid_size || invalid_size_vl || invalid_size_vr)
    {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
        if(BATCHED)
        {
            // unsupported
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_geev(API,
                                                 handle,
                                                 params,
                                                 jobvl,
                                                 jobvr,
                                                 n,
                                                 (T*)nullptr,
                                                 lda,
                                                 (W*)nullptr,
                                                 (T*)nullptr,
                                                 ldvl,
                                                 (T*)nullptr,
                                                 ldvr,
                                                 (T*)nullptr,
                                                 0,
                                                 (T*)nullptr,
                                                 0,
                                                 (int*)nullptr),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }
#endif

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // Memory size query is necessary.
    //
    // Note: current test implementation requires the computation of left and right eigenvectors.
    SIZE size_dW, size_hW;
    CHECK_ROCBLAS_ERROR(
        hipsolver_geev_bufferSize(API,
                                  handle,
                                  params,
                                  test_left_eigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : jobvl,
                                  test_right_eigenvectors ? HIPSOLVER_EIG_MODE_VECTOR : jobvr,
                                  n,
                                  (T*)nullptr,
                                  lda,
                                  (W*)nullptr,
                                  (T*)nullptr,
                                  std::max(ldvl, n),
                                  (T*)nullptr,
                                  std::max(ldvr, n),
                                  &size_dW,
                                  &size_hW));

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_dW);
        return;
    }

    if(BATCHED)
    {
        // unsupported
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, size_A, bc);
        host_strided_batch_vector<CT>    hCA(size_A, 1, size_A, bc);
        host_strided_batch_vector<T>     hARes(size_ARes, 1, size_ARes, bc);
        host_strided_batch_vector<W>     hW(size_W, 1, size_W, bc);
        host_strided_batch_vector<W>     hWRes(size_WRes, 1, size_WRes, bc);
        host_strided_batch_vector<CT>    hVL(size_VL, 1, size_VL, bc);
        host_strided_batch_vector<T>     hVLRes(size_VLRes, 1, size_VLRes, bc);
        host_strided_batch_vector<CT>    hVR(size_VR, 1, size_VR, bc);
        host_strided_batch_vector<T>     hVRRes(size_VRRes, 1, size_VRRes, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        host_strided_batch_vector<T>     hWork(size_hW, 1, size_hW, 1); // size_hW accounts for bc
        host_strided_batch_vector<T>     hD(size_D, 1, size_D, bc);
        host_strided_batch_vector<CT>    hCD(size_D, 1, size_D, bc);
        host_strided_batch_vector<T>     hErr(size_Err, 1, size_Err, bc);
        host_strided_batch_vector<CT>    hCErr(size_Err, 1, size_Err, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, size_A, bc);
        device_strided_batch_vector<W>   dW(size_W, 1, size_W, bc);
        device_strided_batch_vector<T>   dVL(size_VL, 1, size_VL, bc);
        device_strided_batch_vector<T>   dVR(size_VR, 1, size_VR, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_dW, 1, size_dW, 1); // size_dW accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dW.memcheck());
        if(size_VL)
            CHECK_HIP_ERROR(dVL.memcheck());
        if(size_VR)
            CHECK_HIP_ERROR(dVR.memcheck());
        if(size_dW)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            geev_getError<API, T, W>(handle,
                                     params,
                                     jobvl,
                                     jobvr,
                                     n,
                                     dA,
                                     lda,
                                     dW,
                                     dVL,
                                     ldvl,
                                     dVR,
                                     ldvr,
                                     dWork,
                                     size_dW,
                                     hWork,
                                     size_hW,
                                     dInfo,
                                     hA,
                                     hCA,
                                     hARes,
                                     hW,
                                     hWRes,
                                     hVL,
                                     hVLRes,
                                     hVR,
                                     hVRRes,
                                     hInfo,
                                     hInfoRes,
                                     hD,
                                     hCD,
                                     hErr,
                                     hCErr,
                                     &max_error);

        // collect performance data
        if(argus.timing)
            geev_getPerfData<API, T, W>(handle,
                                        params,
                                        jobvl,
                                        jobvr,
                                        n,
                                        dA,
                                        lda,
                                        dW,
                                        dVL,
                                        ldvl,
                                        dVR,
                                        ldvr,
                                        dWork,
                                        size_dW,
                                        hWork,
                                        size_hW,
                                        dInfo,
                                        hA,
                                        &gpu_time_used,
                                        &cpu_time_used,
                                        hot_calls,
                                        argus.perf);
    }

    // validate results for hipsolver-test
    // using 5 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 5 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("jobvl", "jobvr", "n", "lda", "ldvl", "ldvr");
            rocsolver_bench_output(jobvlC, jobvrC, n, lda, ldvl, ldvr);

            std::cerr << "\n============================================\n";
            std::cerr << "Results:\n";
            std::cerr << "============================================\n";
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time", "gpu_time");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            std::cerr << std::endl;
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}
