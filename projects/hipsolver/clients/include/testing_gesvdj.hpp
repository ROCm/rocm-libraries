/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "clientcommon.hpp"

template <testAPI_t API, bool STRIDED, typename T, typename TT, typename W, typename U>
void gesvdj_checkBadArgs(const hipsolverHandle_t     handle,
                         hipsolverEigMode_t          jobz,
                         const int                   econ,
                         const int                   m,
                         const int                   n,
                         W                           dA,
                         const int                   lda,
                         const int                   stA,
                         TT                          dS,
                         const int                   stS,
                         T                           dU,
                         const int                   ldu,
                         const int                   stU,
                         T                           dV,
                         const int                   ldv,
                         const int                   stV,
                         T                           dWork,
                         const int                   lwork,
                         U                           dinfo,
                         const hipsolverGesvdjInfo_t params,
                         const int                   bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                           STRIDED,
                                           nullptr,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           dS,
                                           stS,
                                           dU,
                                           ldu,
                                           stU,
                                           dV,
                                           ldv,
                                           stV,
                                           dWork,
                                           lwork,
                                           dinfo,
                                           params,
                                           bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                           STRIDED,
                                           handle,
                                           hipsolverEigMode_t(-1),
                                           econ,
                                           m,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           dS,
                                           stS,
                                           dU,
                                           ldu,
                                           stU,
                                           dV,
                                           ldv,
                                           stV,
                                           dWork,
                                           lwork,
                                           dinfo,
                                           params,
                                           bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                           STRIDED,
                                           handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           (W) nullptr,
                                           lda,
                                           stA,
                                           dS,
                                           stS,
                                           dU,
                                           ldu,
                                           stU,
                                           dV,
                                           ldv,
                                           stV,
                                           dWork,
                                           lwork,
                                           dinfo,
                                           params,
                                           bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                           STRIDED,
                                           handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           (TT) nullptr,
                                           stS,
                                           dU,
                                           ldu,
                                           stU,
                                           dV,
                                           ldv,
                                           stV,
                                           dWork,
                                           lwork,
                                           dinfo,
                                           params,
                                           bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                           STRIDED,
                                           handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           dS,
                                           stS,
                                           (T) nullptr,
                                           ldu,
                                           stU,
                                           dV,
                                           ldv,
                                           stV,
                                           dWork,
                                           lwork,
                                           dinfo,
                                           params,
                                           bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                           STRIDED,
                                           handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           dS,
                                           stS,
                                           dU,
                                           ldu,
                                           stU,
                                           (T) nullptr,
                                           ldv,
                                           stV,
                                           dWork,
                                           lwork,
                                           dinfo,
                                           params,
                                           bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                           STRIDED,
                                           handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           dS,
                                           stS,
                                           dU,
                                           ldu,
                                           stU,
                                           dV,
                                           ldv,
                                           stV,
                                           dWork,
                                           lwork,
                                           (U) nullptr,
                                           params,
                                           bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_gesvdj_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle      handle;
    hipsolver_local_gesvdj_info params;
    hipsolverEigMode_t          jobz = HIPSOLVER_EIG_MODE_VECTOR;
    int                         econ = 0;
    int                         m    = 2;
    int                         n    = 2;
    int                         lda  = 2;
    int                         ldu  = 2;
    int                         ldv  = 2;
    int                         stA  = 2;
    int                         stS  = 2;
    int                         stU  = 2;
    int                         stV  = 2;
    int                         bc   = 1;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T> dA(1, 1, 1);
        // device_strided_batch_vector<S> dS(1, 1, 1, 1);
        // device_strided_batch_vector<T> dU(1, 1, 1, 1);
        // device_strided_batch_vector<T> dV(1, 1, 1, 1);
        // device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dS.memcheck());
        // CHECK_HIP_ERROR(dU.memcheck());
        // CHECK_HIP_ERROR(dV.memcheck());
        // CHECK_HIP_ERROR(dinfo.memcheck());

        // int size_W;
        // hipsolver_gesvdj_bufferSize(API,
        //                             STRIDED,
        //                             handle,
        //                             jobz,
        //                             econ,
        //                             m,
        //                             n,
        //                             dA.data(),
        //                             lda,
        //                             dS.data(),
        //                             dU.data(),
        //                             ldu,
        //                             dV.data(),
        //                             ldv,
        //                             &size_W,
        //                             params,
        //                             bc);
        // size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;
        // device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // gesvdj_checkBadArgs<API, STRIDED>(handle,
        //                                   jobz,
        //                                   econ,
        //                                   m,
        //                                   n,
        //                                   dA.data(),
        //                                   lda,
        //                                   stA,
        //                                   dS.data(),
        //                                   stS,
        //                                   dU.data(),
        //                                   ldu,
        //                                   stU,
        //                                   dV.data(),
        //                                   ldv,
        //                                   stV,
        //                                   dWork.data(),
        //                                   size_W,
        //                                   dinfo.data(),
        //                                   params,
        //                                   bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<S>   dS(1, 1, 1, 1);
        device_strided_batch_vector<T>   dU(1, 1, 1, 1);
        device_strided_batch_vector<T>   dV(1, 1, 1, 1);
        device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        int size_W;
        hipsolver_gesvdj_bufferSize(API,
                                    STRIDED,
                                    handle,
                                    jobz,
                                    econ,
                                    m,
                                    n,
                                    dA.data(),
                                    lda,
                                    dS.data(),
                                    dU.data(),
                                    ldu,
                                    dV.data(),
                                    ldv,
                                    &size_W,
                                    params,
                                    bc);
        size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr
                             ? size_W
                             : sizeof(T) * size_W;
        device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        gesvdj_checkBadArgs<API, STRIDED>(handle,
                                          jobz,
                                          econ,
                                          m,
                                          n,
                                          dA.data(),
                                          lda,
                                          stA,
                                          dS.data(),
                                          stS,
                                          dU.data(),
                                          ldu,
                                          stU,
                                          dV.data(),
                                          ldv,
                                          stV,
                                          dWork.data(),
                                          size_W,
                                          dinfo.data(),
                                          params,
                                          bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void gesvdj_initData(const hipsolverHandle_t handle,
                     hipsolverEigMode_t      jobz,
                     const int               m,
                     const int               n,
                     Td&                     dA,
                     const int               lda,
                     const int               bc,
                     Th&                     hA,
                     std::vector<T>&         A,
                     bool                    test = true)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // make copy of original data to test vectors if required
            if(test && jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
            {
                for(int i = 0; i < m; i++)
                {
                    for(int j = 0; j < n; j++)
                        A[b * lda * n + i + j * lda] = hA[b][i + j * lda];
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <testAPI_t API,
          bool      STRIDED,
          typename T,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Vh,
          typename Ih>
void gesvdj_getError(const hipsolverHandle_t handle,
                     hipsolverEigMode_t      jobz,
                     const int               econ,
                     const int               m,
                     const int               n,
                     Wd&                     dA,
                     const int               lda,
                     const int               stA,
                     Td&                     dS,
                     const int               stS,
                     Ud&                     dU,
                     const int               ldu,
                     const int               stU,
                     Ud&                     dV,
                     const int               ldv,
                     const int               stV,
                     Ud&                     dWork,
                     const int               lwork,
                     Id&                     dinfo,
                     hipsolverGesvdjInfo_t   params,
                     const double            abstol,
                     const int               max_sweeps,
                     const int               sort_eig,
                     const int               bc,
                     Wh&                     hA,
                     Th&                     hS,
                     Th&                     hSres,
                     Uh&                     Ures,
                     Uh&                     Vres,
                     Ih&                     hinfo,
                     Ih&                     hinfoRes,
                     Vh&                     hResidualRes,
                     Ih&                     hSweepsRes,
                     double*                 max_err,
                     double*                 max_errv)
{
    using S = decltype(std::real(T{}));

    int            size_W = 5 * max(m, n);
    std::vector<S> hE(size_W);
    std::vector<T> hWork(size_W);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    gesvdj_initData<true, true, T>(handle, jobz, m, n, dA, lda, bc, hA, A);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_gesvdj(API,
                                         STRIDED,
                                         handle,
                                         jobz,
                                         econ,
                                         m,
                                         n,
                                         dA.data(),
                                         lda,
                                         stA,
                                         dS.data(),
                                         stS,
                                         dU.data(),
                                         ldu,
                                         stU,
                                         dV.data(),
                                         ldv,
                                         stV,
                                         dWork.data(),
                                         lwork,
                                         dinfo.data(),
                                         params,
                                         bc));

    CHECK_HIP_ERROR(hSres.transfer_from(dS));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        CHECK_HIP_ERROR(Ures.transfer_from(dU));
        CHECK_HIP_ERROR(Vres.transfer_from(dV));
    }

    hipsolverXgesvdjGetResidual(handle, params, hResidualRes.data());
    hipsolverXgesvdjGetSweeps(handle, params, hSweepsRes.data());

    // CPU lapack
    // Only singular values needed
    for(int b = 0; b < bc; ++b)
        cpu_gesvd<T>('N',
                     'N',
                     m,
                     n,
                     hA[b],
                     lda,
                     hS[b],
                     nullptr,
                     ldu,
                     nullptr,
                     ldv,
                     hWork.data(),
                     size_W,
                     hE.data(),
                     hinfo[b]);

    // Check info for non-convergence
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;
    }

    if(!STRIDED)
    {
        // Also check validity of residual
        for(rocblas_int b = 0; b < bc; ++b)
        {
            EXPECT_GE(hResidualRes[b][0], 0) << "where b = " << b;
            if(hResidualRes[b][0] < 0)
                *max_err += 1;
        }

        // Also check validity of sweeps
        for(rocblas_int b = 0; b < bc; ++b)
        {
            EXPECT_GE(hResidualRes[b][0], 0) << "where b = " << b;
            EXPECT_LE(hSweepsRes[b][0], max_sweeps) << "where b = " << b;
            if(hSweepsRes[b][0] < 0 || hSweepsRes[b][0] > max_sweeps)
                *max_err += 1;
        }
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    double err = 0;
    *max_errv  = 0;

    for(int b = 0; b < bc; ++b)
    {
        // error is ||hS - hSres||
        if(sort_eig)
            err = norm_error('F', 1, min(m, n), 1, hS[b], hSres[b]);
        *max_err = err > *max_err ? err : *max_err;

        // Check the singular vectors if required
        if(hinfo[b][0] == 0 && jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
        {
            err = 0;
            // check singular vectors implicitly (A*v_k = s_k*u_k)
            for(int k = 0; k < min(m, n); ++k)
            {
                for(int i = 0; i < m; ++i)
                {
                    T tmp = 0;
                    for(int j = 0; j < n; ++j)
                        tmp += A[b * lda * n + i + j * lda] * Vres[b][j + k * ldv];
                    tmp -= hSres[b][k] * Ures[b][i + k * ldu];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
            err       = std::sqrt(err) / double(snorm('F', m, n, A.data() + b * lda * n, lda));
            *max_errv = err > *max_errv ? err : *max_errv;
        }
    }
}

template <testAPI_t API,
          bool      STRIDED,
          typename T,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Ih>
void gesvdj_getPerfData(const hipsolverHandle_t handle,
                        hipsolverEigMode_t      jobz,
                        const int               econ,
                        const int               m,
                        const int               n,
                        Wd&                     dA,
                        const int               lda,
                        const int               stA,
                        Td&                     dS,
                        const int               stS,
                        Ud&                     dU,
                        const int               ldu,
                        const int               stU,
                        Ud&                     dV,
                        const int               ldv,
                        const int               stV,
                        Ud&                     dWork,
                        const int               lwork,
                        Id&                     dinfo,
                        hipsolverGesvdjInfo_t   params,
                        const int               bc,
                        Wh&                     hA,
                        Th&                     hS,
                        Uh&                     hU,
                        Uh&                     hV,
                        Ih&                     hinfo,
                        double*                 gpu_time_used,
                        double*                 cpu_time_used,
                        const int               hot_calls,
                        const bool              perf)
{
    using S = decltype(std::real(T{}));

    int            size_W = 5 * max(m, n);
    std::vector<S> hE(size_W);
    std::vector<T> hWork(size_W);
    std::vector<T> A;

    char svect     = (jobz == HIPSOLVER_EIG_MODE_NOVECTOR ? 'N' : (econ == 0 ? 'A' : 'S'));
    int  ldv_trans = (jobz == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : (econ == 0 ? n : min(m, n)));

    if(!perf)
    {
        gesvdj_initData<true, false, T>(handle, jobz, m, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_gesvd(svect,
                      svect,
                      m,
                      n,
                      hA[b],
                      lda,
                      hS[b],
                      hU[b],
                      ldu,
                      hV[b],
                      ldv_trans,
                      hWork.data(),
                      size_W,
                      hE.data(),
                      hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gesvdj_initData<true, false, T>(handle, jobz, m, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gesvdj_initData<false, true, T>(handle, jobz, m, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(hipsolver_gesvdj(API,
                                             STRIDED,
                                             handle,
                                             jobz,
                                             econ,
                                             m,
                                             n,
                                             dA.data(),
                                             lda,
                                             stA,
                                             dS.data(),
                                             stS,
                                             dU.data(),
                                             ldu,
                                             stU,
                                             dV.data(),
                                             ldv,
                                             stV,
                                             dWork.data(),
                                             lwork,
                                             dinfo.data(),
                                             params,
                                             bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        gesvdj_initData<false, true, T>(handle, jobz, m, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        hipsolver_gesvdj(API,
                         STRIDED,
                         handle,
                         jobz,
                         econ,
                         m,
                         n,
                         dA.data(),
                         lda,
                         stA,
                         dS.data(),
                         stS,
                         dU.data(),
                         ldu,
                         stU,
                         dV.data(),
                         ldv,
                         stV,
                         dWork.data(),
                         lwork,
                         dinfo.data(),
                         params,
                         bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_gesvdj(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle      handle;
    hipsolver_local_gesvdj_info params;
    char                        jobzC = argus.get<char>("jobz");
    int                         econ  = !STRIDED ? argus.get<int>("econ", 0) : 0;
    int                         m     = argus.get<int>("m");
    int                         n     = argus.get<int>("n", m);
    int                         lda   = argus.get<int>("lda", m);
    int                         ldu   = argus.get<int>("ldu", m);
    int                         ldv   = argus.get<int>("ldv", n);
    int                         stA   = lda * n;
    int                         stS   = min(m, n);
    int                         stU   = ldu * (econ ? min(m, n) : m);
    int                         stV   = ldv * (econ ? min(m, n) : n);

    double      abstol     = argus.get<double>("tolerance", 2 * get_epsilon<T>());
    rocblas_int max_sweeps = argus.get<int>("max_sweeps", 100);
    rocblas_int sort_eig   = argus.get<int>("sort_eig", 1);
    hipsolverXgesvdjSetTolerance(params, abstol);
    hipsolverXgesvdjSetMaxSweeps(params, max_sweeps);
    hipsolverXgesvdjSetSortEig(params, sort_eig);

    hipsolverEigMode_t jobz      = char2hipsolver_evect(jobzC);
    int                bc        = argus.batch_count;
    int                hot_calls = argus.iters;

    rocblas_stride stUres = 0;
    rocblas_stride stVres = 0;

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_S = size_t(min(m, n));
    size_t size_V = 0;
    size_t size_U = 0;

    size_t size_Sres = 0;
    size_t size_Ures = 0;
    size_t size_Vres = 0;

    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        size_U = size_t(ldu) * (econ ? min(m, n) : m);
        size_V = size_t(ldv) * (econ ? min(m, n) : n);
    }

    if(argus.unit_check || argus.norm_check)
    {
        size_Sres = size_S;
        size_Ures = size_U;
        size_Vres = size_V;
        stUres    = stU;
        stVres    = stV;
    }

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0, max_errorv = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || bc < 0)
                        || (jobz != HIPSOLVER_EIG_MODE_NOVECTOR && (ldu < m || ldv < n));

    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
            //                                       STRIDED,
            //                                       handle,
            //                                       jobz,
            //                                       econ,
            //                                       m,
            //                                       n,
            //                                       (T* const*)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (S*)nullptr,
            //                                       stS,
            //                                       (T*)nullptr,
            //                                       ldu,
            //                                       stU,
            //                                       (T*)nullptr,
            //                                       ldv,
            //                                       stV,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (int*)nullptr,
            //                                       params,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_gesvdj(API,
                                                   STRIDED,
                                                   handle,
                                                   jobz,
                                                   econ,
                                                   m,
                                                   n,
                                                   (T*)nullptr,
                                                   lda,
                                                   stA,
                                                   (S*)nullptr,
                                                   stS,
                                                   (T*)nullptr,
                                                   ldu,
                                                   stU,
                                                   (T*)nullptr,
                                                   ldv,
                                                   stV,
                                                   (T*)nullptr,
                                                   0,
                                                   (int*)nullptr,
                                                   params,
                                                   bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_gesvdj_bufferSize(API,
                                STRIDED,
                                handle,
                                jobz,
                                econ,
                                m,
                                n,
                                (T*)nullptr,
                                lda,
                                (S*)nullptr,
                                (T*)nullptr,
                                ldu,
                                (T*)nullptr,
                                ldv,
                                &size_W,
                                params,
                                bc);
    size_t bytes_W
        = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_W);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<double> hResidualRes(1, 1, 1, bc);
    host_strided_batch_vector<int>    hSweepsRes(1, 1, 1, bc);
    host_strided_batch_vector<S>      hS(size_S, 1, stS, bc);
    host_strided_batch_vector<T>      hV(size_V, 1, stV, bc);
    host_strided_batch_vector<T>      hU(size_U, 1, stU, bc);
    host_strided_batch_vector<int>    hinfo(1, 1, 1, bc);
    host_strided_batch_vector<int>    hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S>      hSres(size_Sres, 1, stS, bc);
    host_strided_batch_vector<T>      Vres(size_Vres, 1, stVres, bc);
    host_strided_batch_vector<T>      Ures(size_Ures, 1, stUres, bc);
    // device
    device_strided_batch_vector<S>   dS(size_S, 1, stS, bc);
    device_strided_batch_vector<T>   dV(size_V, 1, stV, bc);
    device_strided_batch_vector<T>   dU(size_U, 1, stU, bc);
    device_strided_batch_vector<int> dinfo(1, 1, 1, bc);
    device_strided_batch_vector<T>   dWork(bytes_W, 1, bytes_W, 1); // bytes_W accounts for bc
    if(size_S)
        CHECK_HIP_ERROR(dS.memcheck());
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_U)
        CHECK_HIP_ERROR(dU.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>           hA(size_A, 1, bc);
        // device_batch_vector<T>         dA(size_A, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        // {
        //     gesvdj_getError<API, STRIDED, T>(handle,
        //                                      jobz,
        //                                      econ,
        //                                      m,
        //                                      n,
        //                                      dA,
        //                                      lda,
        //                                      stA,
        //                                      dS,
        //                                      stS,
        //                                      dU,
        //                                      ldu,
        //                                      stU,
        //                                      dV,
        //                                      ldv,
        //                                      stV,
        //                                      dWork,
        //                                      size_W,
        //                                      dinfo,
        //                                      params,
        //                                      abstol,
        //                                      max_sweeps,
        //                                      sort_eig,
        //                                      bc,
        //                                      hA,
        //                                      hS,
        //                                      hSres,
        //                                      Ures,
        //                                      Vres,
        //                                      hinfo,
        //                                      hinfoRes,
        //                                      hResidualRes,
        //                                      hSweepsRes,
        //                                      &max_error,
        //                                      &max_errorv);
        // }

        // // collect performance data
        // if(argus.timing)
        // {
        //     gesvdj_getPerfData<API, STRIDED, T>(handle,
        //                                         jobz,
        //                                         econ,
        //                                         m,
        //                                         n,
        //                                         dA,
        //                                         lda,
        //                                         stA,
        //                                         dS,
        //                                         stS,
        //                                         dU,
        //                                         ldu,
        //                                         stU,
        //                                         dV,
        //                                         ldv,
        //                                         stV,
        //                                         dWork,
        //                                         size_W,
        //                                         dinfo,
        //                                         params,
        //                                         bc,
        //                                         hA,
        //                                         hS,
        //                                         hU,
        //                                         hV,
        //                                         hinfo,
        //                                         &gpu_time_used,
        //                                         &cpu_time_used,
        //                                         hot_calls,
        //                                         argus.perf);
        // }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>   hA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            gesvdj_getError<API, STRIDED, T>(handle,
                                             jobz,
                                             econ,
                                             m,
                                             n,
                                             dA,
                                             lda,
                                             stA,
                                             dS,
                                             stS,
                                             dU,
                                             ldu,
                                             stU,
                                             dV,
                                             ldv,
                                             stV,
                                             dWork,
                                             size_W,
                                             dinfo,
                                             params,
                                             abstol,
                                             max_sweeps,
                                             sort_eig,
                                             bc,
                                             hA,
                                             hS,
                                             hSres,
                                             Ures,
                                             Vres,
                                             hinfo,
                                             hinfoRes,
                                             hResidualRes,
                                             hSweepsRes,
                                             &max_error,
                                             &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvdj_getPerfData<API, STRIDED, T>(handle,
                                                jobz,
                                                econ,
                                                m,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dS,
                                                stS,
                                                dU,
                                                ldu,
                                                stU,
                                                dV,
                                                ldv,
                                                stV,
                                                dWork,
                                                size_W,
                                                dinfo,
                                                params,
                                                bc,
                                                hA,
                                                hS,
                                                hU,
                                                hV,
                                                hinfo,
                                                &gpu_time_used,
                                                &cpu_time_used,
                                                hot_calls,
                                                argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 3 * max(m, n) * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, 3 * max(m, n));
        if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
            ROCSOLVER_TEST_CHECK(T, max_errorv, 3 * max(m, n));
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
            max_error = (max_error >= max_errorv) ? max_error : max_errorv;

        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("jobz",
                                       "econ",
                                       "m",
                                       "n",
                                       "lda",
                                       "strideS",
                                       "ldu",
                                       "strideU",
                                       "ldv",
                                       "strideV",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig",
                                       "batch_c");
                rocsolver_bench_output(jobz,
                                       econ,
                                       m,
                                       n,
                                       lda,
                                       stS,
                                       ldu,
                                       stU,
                                       ldv,
                                       stV,
                                       abstol,
                                       max_sweeps,
                                       sort_eig,
                                       bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("jobz",
                                       "m",
                                       "n",
                                       "lda",
                                       "ldu",
                                       "ldv",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig",
                                       "batch_c");
                rocsolver_bench_output(jobz, m, n, lda, ldu, ldv, abstol, max_sweeps, sort_eig, bc);
            }
            else
            {
                rocsolver_bench_output("jobz",
                                       "econ",
                                       "m",
                                       "n",
                                       "lda",
                                       "ldu",
                                       "ldv",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig");
                rocsolver_bench_output(
                    jobz, econ, m, n, lda, ldu, ldv, abstol, max_sweeps, sort_eig);
            }
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
