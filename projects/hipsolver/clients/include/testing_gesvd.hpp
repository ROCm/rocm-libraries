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

template <testAPI_t API, typename T, typename TT, typename W, typename U>
void gesvd_checkBadArgs(const hipsolverHandle_t handle,
                        const char              left_svect,
                        const char              right_svect,
                        const int               m,
                        const int               n,
                        W                       dA,
                        const int               lda,
                        const int               stA,
                        TT                      dS,
                        const int               stS,
                        T                       dU,
                        const int               ldu,
                        const int               stU,
                        T                       dV,
                        const int               ldv,
                        const int               stV,
                        T                       dWork,
                        const int               lwork,
                        TT                      dE,
                        const int               stE,
                        U                       dinfo,
                        const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          nullptr,
                                          left_svect,
                                          right_svect,
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          '\0',
                                          right_svect,
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          left_svect,
                                          '\0',
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          'O',
                                          'O',
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          left_svect,
                                          right_svect,
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          left_svect,
                                          right_svect,
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          left_svect,
                                          right_svect,
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          left_svect,
                                          right_svect,
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
                                          dE,
                                          stE,
                                          dinfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                          false,
                                          handle,
                                          left_svect,
                                          right_svect,
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
                                          dE,
                                          stE,
                                          (U) nullptr,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_gesvd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle handle;
    char                   left_svect  = 'A';
    char                   right_svect = 'A';
    int                    m           = 2;
    int                    n           = 2;
    int                    lda         = 2;
    int                    ldu         = 2;
    int                    ldv         = 2;
    int                    stA         = 2;
    int                    stS         = 2;
    int                    stU         = 2;
    int                    stV         = 2;
    int                    stE         = 2;
    int                    bc          = 1;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T> dA(1, 1, 1);
        // device_strided_batch_vector<S> dS(1, 1, 1, 1);
        // device_strided_batch_vector<T> dU(1, 1, 1, 1);
        // device_strided_batch_vector<T> dV(1, 1, 1, 1);
        // device_strided_batch_vector<S> dE(1, 1, 1, 1);
        // device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dS.memcheck());
        // CHECK_HIP_ERROR(dU.memcheck());
        // CHECK_HIP_ERROR(dV.memcheck());
        // CHECK_HIP_ERROR(dE.memcheck());
        // CHECK_HIP_ERROR(dinfo.memcheck());

        // int size_W;
        // hipsolver_gesvd_bufferSize(API, handle, left_svect, right_svect, m, n, dA.data(), lda, &size_W);
        // size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;
        // device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // gesvd_checkBadArgs<API>(handle, left_svect, right_svect, m, n, dA.data(), lda, stA,
        //                         dS.data(), stS, dU.data(), ldu, stU, dV.data(), ldv, stV,
        //                         dWork.data(), size_W, dE.data(), stE, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<S>   dS(1, 1, 1, 1);
        device_strided_batch_vector<T>   dU(1, 1, 1, 1);
        device_strided_batch_vector<T>   dV(1, 1, 1, 1);
        device_strided_batch_vector<S>   dE(1, 1, 1, 1);
        device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dS.memcheck());
        CHECK_HIP_ERROR(dU.memcheck());
        CHECK_HIP_ERROR(dV.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        int size_W;
        hipsolver_gesvd_bufferSize(
            API, handle, left_svect, right_svect, m, n, dA.data(), lda, &size_W);
        size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr
                             ? size_W
                             : sizeof(T) * size_W;
        device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        gesvd_checkBadArgs<API>(handle,
                                left_svect,
                                right_svect,
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
                                dE.data(),
                                stE,
                                dinfo.data(),
                                bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void gesvd_initData(const hipsolverHandle_t handle,
                    const char              left_svect,
                    const char              right_svect,
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
            if(test && (left_svect != 'N' || right_svect != 'N'))
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
          bool      NRWK,
          typename T,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Ih>
void gesvd_getError(const hipsolverHandle_t handle,
                    const char              left_svect,
                    const char              right_svect,
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
                    Td&                     dE,
                    const int               stE,
                    Id&                     dinfo,
                    const int               bc,
                    const char              left_svectT,
                    const char              right_svectT,
                    const int               mT,
                    const int               nT,
                    Ud&                     dUT,
                    const int               lduT,
                    const int               stUT,
                    Ud&                     dVT,
                    const int               ldvT,
                    const int               stVT,
                    Wh&                     hA,
                    Th&                     hS,
                    Th&                     hSres,
                    Uh&                     hU,
                    Uh&                     Ures,
                    const int               ldures,
                    Uh&                     hV,
                    Uh&                     Vres,
                    const int               ldvres,
                    Th&                     hE,
                    Th&                     hEres,
                    Ih&                     hinfo,
                    Ih&                     hinfoRes,
                    double*                 max_err,
                    double*                 max_errv)
{
    int            size_W = 5 * max(m, n);
    std::vector<T> hWork(size_W);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    gesvd_initData<true, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // execute computations:
    // complementary execution to compute all singular vectors if needed (always in-place to ensure
    // we don't combine results computed by gemm_batched with results computed by gemm_strided_batched)
    CHECK_ROCBLAS_ERROR(hipsolver_gesvd(API,
                                        NRWK,
                                        handle,
                                        left_svectT,
                                        right_svectT,
                                        mT,
                                        nT,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dS.data(),
                                        stS,
                                        dUT.data(),
                                        lduT,
                                        stUT,
                                        dVT.data(),
                                        ldvT,
                                        stVT,
                                        dWork.data(),
                                        lwork,
                                        dE.data(),
                                        stE,
                                        dinfo.data(),
                                        bc));

    if(left_svect == 'N' && right_svect != 'N')
        CHECK_HIP_ERROR(Ures.transfer_from(dUT));
    if(right_svect == 'N' && left_svect != 'N')
        CHECK_HIP_ERROR(Vres.transfer_from(dVT));

    gesvd_initData<false, true, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A);

    // CPU lapack
    for(int b = 0; b < bc; ++b)
        cpu_gesvd(left_svect,
                  right_svect,
                  m,
                  n,
                  hA[b],
                  lda,
                  hS[b],
                  hU[b],
                  ldu,
                  hV[b],
                  ldv,
                  hWork.data(),
                  size_W,
                  hE[b],
                  hinfo[b]);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_gesvd(API,
                                        NRWK,
                                        handle,
                                        left_svect,
                                        right_svect,
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
                                        dE.data(),
                                        stE,
                                        dinfo.data(),
                                        bc));

    CHECK_HIP_ERROR(hSres.transfer_from(dS));
    CHECK_HIP_ERROR(hEres.transfer_from(dE));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    if(left_svect == 'S' || left_svect == 'A')
        CHECK_HIP_ERROR(Ures.transfer_from(dU));
    if(right_svect == 'S' || right_svect == 'A')
        CHECK_HIP_ERROR(Vres.transfer_from(dV));

    if(left_svect == 'O')
    {
        CHECK_HIP_ERROR(hA.transfer_from(dA));
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < min(m, n); j++)
                    Ures[b][i + j * ldures] = hA[b][i + j * lda];
            }
        }
    }
    if(right_svect == 'O')
    {
        CHECK_HIP_ERROR(hA.transfer_from(dA));
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < min(m, n); i++)
            {
                for(int j = 0; j < n; j++)
                    Vres[b][i + j * ldvres] = hA[b][i + j * lda];
            }
        }
    }

    // Check info for non-convergence
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    double err;
    *max_errv = 0;

    for(int b = 0; b < bc; ++b)
    {
        // error is ||hS - hSres||
        err      = norm_error('F', 1, min(m, n), 1, hS[b], hSres[b]);
        *max_err = err > *max_err ? err : *max_err;

        // Check the singular vectors if required
        if(hinfo[b][0] == 0 && (left_svect != 'N' || right_svect != 'N'))
        {
            err = 0;
            // check singular vectors implicitly (A*v_k = s_k*u_k)
            for(int k = 0; k < min(m, n); ++k)
            {
                for(int i = 0; i < m; ++i)
                {
                    T tmp = 0;
                    for(int j = 0; j < n; ++j)
                        tmp += A[b * lda * n + i + j * lda] * std::conj(Vres[b][k + j * ldvres]);
                    tmp -= hSres[b][k] * Ures[b][i + k * ldures];
                    err += std::abs(tmp) * std::abs(tmp);
                }
            }
            err       = std::sqrt(err) / double(snorm('F', m, n, A.data() + b * lda * n, lda));
            *max_errv = err > *max_errv ? err : *max_errv;
        }
    }
}

template <testAPI_t API,
          bool      NRWK,
          typename T,
          typename Wd,
          typename Td,
          typename Ud,
          typename Id,
          typename Wh,
          typename Th,
          typename Uh,
          typename Ih>
void gesvd_getPerfData(const hipsolverHandle_t handle,
                       const char              left_svect,
                       const char              right_svect,
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
                       Td&                     dE,
                       const int               stE,
                       Id&                     dinfo,
                       const int               bc,
                       Wh&                     hA,
                       Th&                     hS,
                       Uh&                     hU,
                       Uh&                     hV,
                       Th&                     hE,
                       Ih&                     hinfo,
                       double*                 gpu_time_used,
                       double*                 cpu_time_used,
                       const int               hot_calls,
                       const bool              perf)
{
    int            size_W = 5 * max(m, n);
    std::vector<T> hWork(size_W);
    std::vector<T> A;

    if(!perf)
    {
        gesvd_initData<true, false, T>(
            handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_gesvd(left_svect,
                      right_svect,
                      m,
                      n,
                      hA[b],
                      lda,
                      hS[b],
                      hU[b],
                      ldu,
                      hV[b],
                      ldv,
                      hWork.data(),
                      size_W,
                      hE[b],
                      hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gesvd_initData<true, false, T>(handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gesvd_initData<false, true, T>(
            handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(hipsolver_gesvd(API,
                                            NRWK,
                                            handle,
                                            left_svect,
                                            right_svect,
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
                                            dE.data(),
                                            stE,
                                            dinfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        gesvd_initData<false, true, T>(
            handle, left_svect, right_svect, m, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        hipsolver_gesvd(API,
                        NRWK,
                        handle,
                        left_svect,
                        right_svect,
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
                        dE.data(),
                        stE,
                        dinfo.data(),
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, bool NRWK, typename T>
void testing_gesvd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle handle;
    char                   leftv  = argus.get<char>("jobu");
    char                   rightv = argus.get<char>("jobv");
    int                    m      = argus.get<int>("m");
    int                    n      = argus.get<int>("n", m);
    int                    lda    = argus.get<int>("lda", m);
    int                    ldu    = argus.get<int>("ldu", m);
    int                    ldv    = argus.get<int>("ldv", (rightv == 'A' ? n : min(m, n)));
    int                    stA    = argus.get<int>("strideA", lda * n);
    int                    stS    = argus.get<int>("strideS", min(m, n));
    int                    stU    = argus.get<int>("strideU", ldu * m);
    int                    stV    = argus.get<int>("strideV", ldv * n);
    int                    stE    = argus.get<int>("strideE", min(m, n) - 1);

    int bc        = argus.batch_count;
    int hot_calls = argus.iters;

    // check non-supported values
    if(rightv == 'O' && leftv == 'O')
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
            //                                       NRWK,
            //                                       handle,
            //                                       leftv,
            //                                       rightv,
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
            //                                       (S*)nullptr,
            //                                       stE,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                                  NRWK,
                                                  handle,
                                                  leftv,
                                                  rightv,
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
                                                  (S*)nullptr,
                                                  stE,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    /** TESTING OF SINGULAR VECTORS IS DONE IMPLICITLY, NOT EXPLICITLY COMPARING
        WITH LAPACK. SO, WE ALWAYS NEED TO COMPUTE THE SAME NUMBER OF ELEMENTS OF
        THE RIGHT AND LEFT VECTORS. WHILE DOING THIS, IF MORE VECTORS THAN THE
        SPECIFIED IN THE MAIN CALL NEED TO BE COMPUTED, WE DO SO WITH AN EXTRA CALL **/

    signed char leftvT  = 'N';
    signed char rightvT = 'N';
    int         ldvT    = 1;
    int         lduT    = 1;
    int         mT      = 0;
    int         nT      = 0;
    bool        svects  = (leftv != 'N' || rightv != 'N');

    if(svects)
    {
        if(leftv == 'N')
        {
            leftvT = 'A';
            lduT   = m;
            mT     = m;
            nT     = n;
            // if((n > m && fa == rocblas_outofplace) || (n > m && rightv == 'O'))
            //     rightvT = 'O';
        }
        if(rightv == 'N')
        {
            rightvT = 'A';
            ldvT    = n;
            mT      = m;
            nT      = n;
            // if((m >= n && fa == rocblas_outofplace) || (m >= n && leftv == 'O'))
            //     leftvT = 'O';
        }
    }

    // determine sizes
    int    ldures    = 1;
    int    ldvres    = 1;
    size_t size_Sres = 0;
    size_t size_Eres = 0;
    size_t size_Ures = 0;
    size_t size_Vres = 0;
    size_t size_UT   = 0;
    size_t size_VT   = 0;
    size_t size_A    = size_t(lda) * n;
    size_t size_S    = size_t(min(m, n));
    size_t size_E    = size_t(min(m, n) - 1);
    size_t size_V    = size_t(ldv) * n;
    size_t size_U    = size_t(ldu) * m;
    if(argus.unit_check || argus.norm_check)
    {
        size_VT   = size_t(ldvT) * nT;
        size_UT   = size_t(lduT) * mT;
        size_Sres = size_S;
        size_Eres = size_E;
        if(svects)
        {
            if(leftv == 'N')
            {
                size_Ures = size_UT;
                ldures    = lduT;
            }
            else if(leftv == 'S' || leftv == 'A')
            {
                size_Ures = size_U;
                ldures    = ldu;
            }
            else
            {
                size_Ures = m * m;
                ldures    = m;
            }

            if(rightv == 'N')
            {
                size_Vres = size_VT;
                ldvres    = ldvT;
            }
            else if(rightv == 'S' || rightv == 'A')
            {
                size_Vres = size_V;
                ldvres    = ldv;
            }
            else
            {
                size_Vres = n * n;
                ldvres    = n;
            }
        }
    }
    int stUT   = size_UT;
    int stVT   = size_VT;
    int stUres = size_Ures;
    int stVres = size_Vres;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0, max_errorv = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || m < 0 || lda < m || ldu < 1 || ldv < 1 || bc < 0)
                        || ((leftv == 'A' || leftv == 'S') && ldu < m)
                        || ((rightv == 'A' && ldv < n) || (rightv == 'S' && ldv < min(m, n)));

    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
            //                                       NRWK,
            //                                       handle,
            //                                       leftv,
            //                                       rightv,
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
            //                                       (S*)nullptr,
            //                                       stE,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_gesvd(API,
                                                  NRWK,
                                                  handle,
                                                  leftv,
                                                  rightv,
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
                                                  (S*)nullptr,
                                                  stE,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W, w1, w2;
    hipsolver_gesvd_bufferSize(API, handle, leftv, rightv, m, n, (T*)nullptr, lda, &w1);
    hipsolver_gesvd_bufferSize(API, handle, leftvT, rightvT, mT, nT, (T*)nullptr, lda, &w2);
    size_W = max(w1, w2);
    size_t bytes_W
        = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_W);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S>   hE(5 * max(m, n), 1, 5 * max(m, n), bc);
    host_strided_batch_vector<S>   hS(size_S, 1, stS, bc);
    host_strided_batch_vector<T>   hV(size_V, 1, stV, bc);
    host_strided_batch_vector<T>   hU(size_U, 1, stU, bc);
    host_strided_batch_vector<int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S>   hSres(size_Sres, 1, stS, bc);
    host_strided_batch_vector<S>   hEres(size_Eres, 1, stE, bc);
    host_strided_batch_vector<T>   Vres(size_Vres, 1, stVres, bc);
    host_strided_batch_vector<T>   Ures(size_Ures, 1, stUres, bc);
    // device
    device_strided_batch_vector<S>   dE(size_E, 1, stE, bc);
    device_strided_batch_vector<S>   dS(size_S, 1, stS, bc);
    device_strided_batch_vector<T>   dV(size_V, 1, stV, bc);
    device_strided_batch_vector<T>   dU(size_U, 1, stU, bc);
    device_strided_batch_vector<int> dinfo(1, 1, 1, bc);
    device_strided_batch_vector<T>   dVT(size_VT, 1, stVT, bc);
    device_strided_batch_vector<T>   dUT(size_UT, 1, stUT, bc);
    device_strided_batch_vector<T>   dWork(bytes_W, 1, bytes_W, 1); // bytes_W accounts for bc
    if(size_VT)
        CHECK_HIP_ERROR(dVT.memcheck());
    if(size_UT)
        CHECK_HIP_ERROR(dUT.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
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
        // host_batch_vector<T>   hA(size_A, 1, bc);
        // device_batch_vector<T> dA(size_A, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        // {
        //     gesvd_getError<API, NRWK, T>(handle,
        //                                  leftv,
        //                                  rightv,
        //                                  m,
        //                                  n,
        //                                  dA,
        //                                  lda,
        //                                  stA,
        //                                  dS,
        //                                  stS,
        //                                  dU,
        //                                  ldu,
        //                                  stU,
        //                                  dV,
        //                                  ldv,
        //                                  stV,
        //                                  dWork,
        //                                  size_W,
        //                                  dE,
        //                                  stE,
        //                                  dinfo,
        //                                  bc,
        //                                  leftvT,
        //                                  rightvT,
        //                                  mT,
        //                                  nT,
        //                                  dUT,
        //                                  lduT,
        //                                  stUT,
        //                                  dVT,
        //                                  ldvT,
        //                                  stVT,
        //                                  hA,
        //                                  hS,
        //                                  hSres,
        //                                  hU,
        //                                  Ures,
        //                                  ldures,
        //                                  hV,
        //                                  Vres,
        //                                  ldvres,
        //                                  hE,
        //                                  hEres,
        //                                  hinfo,
        //                                  hinfoRes,
        //                                  &max_error,
        //                                  &max_errorv);
        // }

        // // collect performance data
        // if(argus.timing)
        // {
        //     gesvd_getPerfData<API, NRWK, T>(handle,
        //                                     leftv,
        //                                     rightv,
        //                                     m,
        //                                     n,
        //                                     dA,
        //                                     lda,
        //                                     stA,
        //                                     dS,
        //                                     stS,
        //                                     dU,
        //                                     ldu,
        //                                     stU,
        //                                     dV,
        //                                     ldv,
        //                                     stV,
        //                                     dWork,
        //                                     size_W,
        //                                     dE,
        //                                     stE,
        //                                     dinfo,
        //                                     bc,
        //                                     hA,
        //                                     hS,
        //                                     hU,
        //                                     hV,
        //                                     hE,
        //                                     hinfo,
        //                                     &gpu_time_used,
        //                                     &cpu_time_used,
        //                                     hot_calls,
        //                                     argus.perf);
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
            gesvd_getError<API, NRWK, T>(handle,
                                         leftv,
                                         rightv,
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
                                         dE,
                                         stE,
                                         dinfo,
                                         bc,
                                         leftvT,
                                         rightvT,
                                         mT,
                                         nT,
                                         dUT,
                                         lduT,
                                         stUT,
                                         dVT,
                                         ldvT,
                                         stVT,
                                         hA,
                                         hS,
                                         hSres,
                                         hU,
                                         Ures,
                                         ldures,
                                         hV,
                                         Vres,
                                         ldvres,
                                         hE,
                                         hEres,
                                         hinfo,
                                         hinfoRes,
                                         &max_error,
                                         &max_errorv);
        }

        // collect performance data
        if(argus.timing)
        {
            gesvd_getPerfData<API, NRWK, T>(handle,
                                            leftv,
                                            rightv,
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
                                            dE,
                                            stE,
                                            dinfo,
                                            bc,
                                            hA,
                                            hS,
                                            hU,
                                            hV,
                                            hE,
                                            hinfo,
                                            &gpu_time_used,
                                            &cpu_time_used,
                                            hot_calls,
                                            argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 2 * min(m, n) * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * min(m, n));
        if(svects)
            ROCSOLVER_TEST_CHECK(T, max_errorv, 2 * min(m, n));
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(svects)
            max_error = (max_error >= max_errorv) ? max_error : max_errorv;

        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("jobu",
                                       "jobv",
                                       "m",
                                       "n",
                                       "lda",
                                       "strideS",
                                       "ldu",
                                       "strideU",
                                       "ldv",
                                       "strideV",
                                       "strideE",
                                       "batch_c");
                rocsolver_bench_output(leftv, rightv, m, n, lda, stS, ldu, stU, ldv, stV, stE, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("jobu",
                                       "jobv",
                                       "m",
                                       "n",
                                       "lda",
                                       "strideA",
                                       "strideS",
                                       "ldu",
                                       "strideU",
                                       "ldv",
                                       "strideV",
                                       "strideE",
                                       "batch_c");
                rocsolver_bench_output(
                    leftv, rightv, m, n, lda, stA, stS, ldu, stU, ldv, stV, stE, bc);
            }
            else
            {
                rocsolver_bench_output("jobu", "jobv", "m", "n", "lda", "ldu", "ldv");
                rocsolver_bench_output(leftv, rightv, m, n, lda, ldu, ldv);
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
