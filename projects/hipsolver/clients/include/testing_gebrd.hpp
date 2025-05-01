/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <testAPI_t API, typename S, typename T, typename U, typename V>
void gebrd_checkBadArgs(const hipsolverHandle_t handle,
                        const int               m,
                        const int               n,
                        T                       dA,
                        const int               lda,
                        const int               stA,
                        S                       dD,
                        const int               stD,
                        S                       dE,
                        const int               stE,
                        U                       dTauq,
                        const int               stQ,
                        U                       dTaup,
                        const int               stP,
                        U                       dWork,
                        const int               lwork,
                        V                       dInfo,
                        const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                          nullptr,
                                          m,
                                          n,
                                          dA,
                                          lda,
                                          stA,
                                          dD,
                                          stD,
                                          dE,
                                          stE,
                                          dTauq,
                                          stQ,
                                          dTaup,
                                          stP,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                          handle,
                                          m,
                                          n,
                                          (T) nullptr,
                                          lda,
                                          stA,
                                          dD,
                                          stD,
                                          dE,
                                          stE,
                                          dTauq,
                                          stQ,
                                          dTaup,
                                          stP,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                          handle,
                                          m,
                                          n,
                                          dA,
                                          lda,
                                          stA,
                                          (S) nullptr,
                                          stD,
                                          dE,
                                          stE,
                                          dTauq,
                                          stQ,
                                          dTaup,
                                          stP,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                          handle,
                                          m,
                                          n,
                                          dA,
                                          lda,
                                          stA,
                                          dD,
                                          stD,
                                          (S) nullptr,
                                          stE,
                                          dTauq,
                                          stQ,
                                          dTaup,
                                          stP,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                          handle,
                                          m,
                                          n,
                                          dA,
                                          lda,
                                          stA,
                                          dD,
                                          stD,
                                          dE,
                                          stE,
                                          (U) nullptr,
                                          stQ,
                                          dTaup,
                                          stP,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                          handle,
                                          m,
                                          n,
                                          dA,
                                          lda,
                                          stA,
                                          dD,
                                          stD,
                                          dE,
                                          stE,
                                          dTauq,
                                          stQ,
                                          (U) nullptr,
                                          stP,
                                          dWork,
                                          lwork,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                          handle,
                                          m,
                                          n,
                                          dA,
                                          lda,
                                          stA,
                                          dD,
                                          stD,
                                          dE,
                                          stE,
                                          dTauq,
                                          stQ,
                                          dTaup,
                                          stP,
                                          dWork,
                                          lwork,
                                          (V) nullptr,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_gebrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle handle;
    int                    m   = 2;
    int                    n   = 2;
    int                    lda = 2;
    int                    stA = 1;
    int                    stD = 1;
    int                    stE = 1;
    int                    stQ = 1;
    int                    stP = 1;
    int                    bc  = 1;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        // device_strided_batch_vector<S>   dE(1, 1, 1, 1);
        // device_strided_batch_vector<T>   dTauq(1, 1, 1, 1);
        // device_strided_batch_vector<T>   dTaup(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dD.memcheck());
        // CHECK_HIP_ERROR(dE.memcheck());
        // CHECK_HIP_ERROR(dTauq.memcheck());
        // CHECK_HIP_ERROR(dTaup.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // int size_W;
        // hipsolver_gebrd_bufferSize(API, handle, m, n, dA.data(), lda, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // gebrd_checkBadArgs<API>(handle,
        //                             m,
        //                             n,
        //                             dA.data(),
        //                             lda,
        //                             stA,
        //                             dD.data(),
        //                             stD,
        //                             dE.data(),
        //                             stE,
        //                             dTauq.data(),
        //                             stQ,
        //                             dTaup.data(),
        //                             stP,
        //                             dWork.data(),
        //                             size_W,
        //                             dInfo.data(),
        //                             bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        device_strided_batch_vector<S>   dE(1, 1, 1, 1);
        device_strided_batch_vector<T>   dTauq(1, 1, 1, 1);
        device_strided_batch_vector<T>   dTaup(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTauq.memcheck());
        CHECK_HIP_ERROR(dTaup.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_gebrd_bufferSize(API, handle, m, n, dA.data(), lda, &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        gebrd_checkBadArgs<API>(handle,
                                m,
                                n,
                                dA.data(),
                                lda,
                                stA,
                                dD.data(),
                                stD,
                                dE.data(),
                                stE,
                                dTauq.data(),
                                stQ,
                                dTaup.data(),
                                stP,
                                dWork.data(),
                                size_W,
                                dInfo.data(),
                                bc);
    }
}

template <bool CPU,
          bool GPU,
          typename T,
          typename Sd,
          typename Td,
          typename Ud,
          typename Sh,
          typename Th,
          typename Uh>
void gebrd_initData(const hipsolverHandle_t handle,
                    const int               m,
                    const int               n,
                    Td&                     dA,
                    const int               lda,
                    const int               stA,
                    Sd&                     dD,
                    const int               stD,
                    Sd&                     dE,
                    const int               stE,
                    Ud&                     dTauq,
                    const int               stQ,
                    Ud&                     dTaup,
                    const int               stP,
                    const int               bc,
                    Th&                     hA,
                    Sh&                     hD,
                    Sh&                     hE,
                    Uh&                     hTauq,
                    Uh&                     hTaup)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j || (m >= n && j == i + 1) || (m < n && i == j + 1))
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
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
          typename T,
          typename Sd,
          typename Td,
          typename Ud,
          typename Vd,
          typename Sh,
          typename Th,
          typename Uh,
          typename Vh>
void gebrd_getError(const hipsolverHandle_t handle,
                    const int               m,
                    const int               n,
                    Td&                     dA,
                    const int               lda,
                    const int               stA,
                    Sd&                     dD,
                    const int               stD,
                    Sd&                     dE,
                    const int               stE,
                    Ud&                     dTauq,
                    const int               stQ,
                    Ud&                     dTaup,
                    const int               stP,
                    Ud&                     dWork,
                    const int               lwork,
                    Vd&                     dInfo,
                    const int               bc,
                    Th&                     hA,
                    Th&                     hARes,
                    Sh&                     hD,
                    Sh&                     hE,
                    Uh&                     hTauq,
                    Uh&                     hTaup,
                    Vh&                     hInfo,
                    Vh&                     hInfoRes,
                    double*                 max_err)
{
    constexpr bool COMPLEX              = is_complex<T>;
    constexpr bool VERIFY_IMPLICIT_TEST = false;

    std::vector<T> hW(max(m, n));

    // input data initialization
    gebrd_initData<true, true, T>(handle,
                                  m,
                                  n,
                                  dA,
                                  lda,
                                  stA,
                                  dD,
                                  stD,
                                  dE,
                                  stE,
                                  dTauq,
                                  stQ,
                                  dTaup,
                                  stP,
                                  bc,
                                  hA,
                                  hD,
                                  hE,
                                  hTauq,
                                  hTaup);

    // execute computations
    // use verify_implicit_test to check correctness of the implicit test using
    // CPU lapack
    if(!VERIFY_IMPLICIT_TEST)
    {
        // GPU lapack
        CHECK_ROCBLAS_ERROR(hipsolver_gebrd(API,
                                            handle,
                                            m,
                                            n,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dD.data(),
                                            stD,
                                            dE.data(),
                                            stE,
                                            dTauq.data(),
                                            stQ,
                                            dTaup.data(),
                                            stP,
                                            dWork.data(),
                                            lwork,
                                            dInfo.data(),
                                            bc));
        CHECK_HIP_ERROR(hARes.transfer_from(dA));
        CHECK_HIP_ERROR(hTauq.transfer_from(dTauq));
        CHECK_HIP_ERROR(hTaup.transfer_from(dTaup));
        CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    }
    else
    {
        // CPU lapack
        for(int b = 0; b < bc; ++b)
        {
            memcpy(hARes[b], hA[b], lda * n * sizeof(T));
            cpu_gebrd(m,
                      n,
                      hARes[b],
                      lda,
                      hD[b],
                      hE[b],
                      hTauq[b],
                      hTaup[b],
                      hW.data(),
                      max(m, n),
                      hInfoRes[b]);
        }
    }

    // reconstruct A from the factorization for implicit testing
    std::vector<T> vec(max(m, n));
    vec[0] = 1;
    for(int b = 0; b < bc; ++b)
    {
        T* a    = hARes[b];
        T* tauq = hTauq[b];
        T* taup = hTaup[b];

        if(m >= n)
        {
            for(int j = n - 1; j >= 0; j--)
            {
                if(j < n - 1)
                {
                    if(COMPLEX)
                    {
                        cpu_lacgv(1, taup + j, 1);
                        cpu_lacgv(n - j - 1, a + j + (j + 1) * lda, lda);
                    }
                    for(int i = 1; i < n - j - 1; i++)
                    {
                        vec[i]                   = a[j + (j + i + 1) * lda];
                        a[j + (j + i + 1) * lda] = 0;
                    }
                    cpu_larf(HIPSOLVER_SIDE_RIGHT,
                             m - j,
                             n - j - 1,
                             vec.data(),
                             1,
                             taup + j,
                             a + j + (j + 1) * lda,
                             lda,
                             hW.data());
                    if(COMPLEX)
                        cpu_lacgv(1, taup + j, 1);
                }

                for(int i = 1; i < m - j; i++)
                {
                    vec[i]               = a[(j + i) + j * lda];
                    a[(j + i) + j * lda] = 0;
                }
                cpu_larf(HIPSOLVER_SIDE_LEFT,
                         m - j,
                         n - j,
                         vec.data(),
                         1,
                         tauq + j,
                         a + j + j * lda,
                         lda,
                         hW.data());
            }
        }
        else
        {
            for(int j = m - 1; j >= 0; j--)
            {
                if(j < m - 1)
                {
                    for(int i = 1; i < m - j - 1; i++)
                    {
                        vec[i]                   = a[(j + i + 1) + j * lda];
                        a[(j + i + 1) + j * lda] = 0;
                    }
                    cpu_larf(HIPSOLVER_SIDE_LEFT,
                             m - j - 1,
                             n - j,
                             vec.data(),
                             1,
                             tauq + j,
                             a + (j + 1) + j * lda,
                             lda,
                             hW.data());
                }

                if(COMPLEX)
                {
                    cpu_lacgv(1, taup + j, 1);
                    cpu_lacgv(n - j, a + j + j * lda, lda);
                }
                for(int i = 1; i < n - j; i++)
                {
                    vec[i]               = a[j + (j + i) * lda];
                    a[j + (j + i) * lda] = 0;
                }
                cpu_larf(HIPSOLVER_SIDE_RIGHT,
                         m - j,
                         n - j,
                         vec.data(),
                         1,
                         taup + j,
                         a + j + j * lda,
                         lda,
                         hW.data());
                if(COMPLEX)
                    cpu_lacgv(1, taup + j, 1);
            }
        }
    }

    // error is ||hA - hARes|| / ||hA||
    // using frobenius norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        err      = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }

    // check info
    err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
        if(hInfoRes[b][0] != 0)
            err++;
    }
    *max_err += err;
}

template <testAPI_t API,
          typename T,
          typename Sd,
          typename Td,
          typename Ud,
          typename Vd,
          typename Sh,
          typename Th,
          typename Uh,
          typename Vh>
void gebrd_getPerfData(const hipsolverHandle_t handle,
                       const int               m,
                       const int               n,
                       Td&                     dA,
                       const int               lda,
                       const int               stA,
                       Sd&                     dD,
                       const int               stD,
                       Sd&                     dE,
                       const int               stE,
                       Ud&                     dTauq,
                       const int               stQ,
                       Ud&                     dTaup,
                       const int               stP,
                       Ud&                     dWork,
                       const int               lwork,
                       Vd&                     dInfo,
                       const int               bc,
                       Th&                     hA,
                       Sh&                     hD,
                       Sh&                     hE,
                       Uh&                     hTauq,
                       Uh&                     hTaup,
                       Vh&                     hInfo,
                       double*                 gpu_time_used,
                       double*                 cpu_time_used,
                       const int               hot_calls,
                       const bool              perf)
{
    std::vector<T> hW(max(m, n));

    if(!perf)
    {
        gebrd_initData<true, false, T>(handle,
                                       m,
                                       n,
                                       dA,
                                       lda,
                                       stA,
                                       dD,
                                       stD,
                                       dE,
                                       stE,
                                       dTauq,
                                       stQ,
                                       dTaup,
                                       stP,
                                       bc,
                                       hA,
                                       hD,
                                       hE,
                                       hTauq,
                                       hTaup);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_gebrd(
                m, n, hA[b], lda, hD[b], hE[b], hTauq[b], hTaup[b], hW.data(), max(m, n), hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gebrd_initData<true, false, T>(handle,
                                   m,
                                   n,
                                   dA,
                                   lda,
                                   stA,
                                   dD,
                                   stD,
                                   dE,
                                   stE,
                                   dTauq,
                                   stQ,
                                   dTaup,
                                   stP,
                                   bc,
                                   hA,
                                   hD,
                                   hE,
                                   hTauq,
                                   hTaup);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gebrd_initData<false, true, T>(handle,
                                       m,
                                       n,
                                       dA,
                                       lda,
                                       stA,
                                       dD,
                                       stD,
                                       dE,
                                       stE,
                                       dTauq,
                                       stQ,
                                       dTaup,
                                       stP,
                                       bc,
                                       hA,
                                       hD,
                                       hE,
                                       hTauq,
                                       hTaup);

        CHECK_ROCBLAS_ERROR(hipsolver_gebrd(API,
                                            handle,
                                            m,
                                            n,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dD.data(),
                                            stD,
                                            dE.data(),
                                            stE,
                                            dTauq.data(),
                                            stQ,
                                            dTaup.data(),
                                            stP,
                                            dWork.data(),
                                            lwork,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        gebrd_initData<false, true, T>(handle,
                                       m,
                                       n,
                                       dA,
                                       lda,
                                       stA,
                                       dD,
                                       stD,
                                       dE,
                                       stE,
                                       dTauq,
                                       stQ,
                                       dTaup,
                                       stP,
                                       bc,
                                       hA,
                                       hD,
                                       hE,
                                       hTauq,
                                       hTaup);

        start = get_time_us_sync(stream);
        hipsolver_gebrd(API,
                        handle,
                        m,
                        n,
                        dA.data(),
                        lda,
                        stA,
                        dD.data(),
                        stD,
                        dE.data(),
                        stE,
                        dTauq.data(),
                        stQ,
                        dTaup.data(),
                        stP,
                        dWork.data(),
                        lwork,
                        dInfo.data(),
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_gebrd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle handle;
    int                    m   = argus.get<int>("m");
    int                    n   = argus.get<int>("n", m);
    int                    lda = argus.get<int>("lda", m);
    int                    stA = argus.get<int>("strideA", lda * n);
    int                    stD = argus.get<int>("strideD", min(m, n));
    int                    stE = argus.get<int>("strideE", min(m, n) - 1);
    int                    stQ = argus.get<int>("strideQ", min(m, n));
    int                    stP = argus.get<int>("strideP", min(m, n));

    int bc        = argus.batch_count;
    int hot_calls = argus.iters;

    int stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = lda * n;
    size_t size_D    = min(m, n);
    size_t size_E    = min(m, n) - 1;
    size_t size_Q    = min(m, n);
    size_t size_P    = min(m, n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
            //                                       handle,
            //                                       m,
            //                                       n,
            //                                       (T* const*)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (S*)nullptr,
            //                                       stD,
            //                                       (S*)nullptr,
            //                                       stE,
            //                                       (T*)nullptr,
            //                                       stQ,
            //                                       (T*)nullptr,
            //                                       stP,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_gebrd(API,
                                                  handle,
                                                  m,
                                                  n,
                                                  (T*)nullptr,
                                                  lda,
                                                  stA,
                                                  (S*)nullptr,
                                                  stD,
                                                  (S*)nullptr,
                                                  stE,
                                                  (T*)nullptr,
                                                  stQ,
                                                  (T*)nullptr,
                                                  stP,
                                                  (T*)nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_gebrd_bufferSize(API, handle, m, n, (T*)nullptr, lda, &size_W);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_W);
        return;
    }

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>             hA(size_A, 1, bc);
        // host_batch_vector<T>             hARes(size_ARes, 1, bc);
        // host_strided_batch_vector<S>     hD(size_D, 1, stD, bc);
        // host_strided_batch_vector<S>     hE(size_E, 1, stE, bc);
        // host_strided_batch_vector<T>     hTaup(size_P, 1, stP, bc);
        // host_strided_batch_vector<T>     hTauq(size_Q, 1, stQ, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_strided_batch_vector<S>   dD(size_D, 1, stD, bc);
        // device_strided_batch_vector<S>   dE(size_E, 1, stE, bc);
        // device_strided_batch_vector<T>   dTauq(size_Q, 1, stQ, bc);
        // device_strided_batch_vector<T>   dTaup(size_P, 1, stP, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_D)
        //     CHECK_HIP_ERROR(dD.memcheck());
        // if(size_E)
        //     CHECK_HIP_ERROR(dE.memcheck());
        // if(size_Q)
        //     CHECK_HIP_ERROR(dTauq.memcheck());
        // if(size_P)
        //     CHECK_HIP_ERROR(dTaup.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     gebrd_getError<API, T>(handle,
        //                                m,
        //                                n,
        //                                dA,
        //                                lda,
        //                                stA,
        //                                dD,
        //                                stD,
        //                                dE,
        //                                stE,
        //                                dTauq,
        //                                stQ,
        //                                dTaup,
        //                                stP,
        //                                dWork,
        //                                size_W,
        //                                dInfo,
        //                                bc,
        //                                hA,
        //                                hARes,
        //                                hD,
        //                                hE,
        //                                hTauq,
        //                                hTaup,
        //                                hInfo,
        //                                hInfoRes,
        //                                &max_error);

        // // collect performance data
        // if(argus.timing)
        //     gebrd_getPerfData<API, T>(handle,
        //                                   m,
        //                                   n,
        //                                   dA,
        //                                   lda,
        //                                   stA,
        //                                   dD,
        //                                   stD,
        //                                   dE,
        //                                   stE,
        //                                   dTauq,
        //                                   stQ,
        //                                   dTaup,
        //                                   stP,
        //                                   dWork,
        //                                   size_W,
        //                                   dInfo,
        //                                   bc,
        //                                   hA,
        //                                   hD,
        //                                   hE,
        //                                   hTauq,
        //                                   hTaup,
        //                                   hInfo,
        //                                   &gpu_time_used,
        //                                   &cpu_time_used,
        //                                   hot_calls,
        //                                   argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>     hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<S>     hD(size_D, 1, stD, bc);
        host_strided_batch_vector<S>     hE(size_E, 1, stE, bc);
        host_strided_batch_vector<T>     hTaup(size_P, 1, stP, bc);
        host_strided_batch_vector<T>     hTauq(size_Q, 1, stQ, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<S>   dD(size_D, 1, stD, bc);
        device_strided_batch_vector<S>   dE(size_E, 1, stE, bc);
        device_strided_batch_vector<T>   dTauq(size_Q, 1, stQ, bc);
        device_strided_batch_vector<T>   dTaup(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, 1); // size_W accounts for bc
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_D)
            CHECK_HIP_ERROR(dD.memcheck());
        if(size_E)
            CHECK_HIP_ERROR(dE.memcheck());
        if(size_Q)
            CHECK_HIP_ERROR(dTauq.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dTaup.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebrd_getError<API, T>(handle,
                                   m,
                                   n,
                                   dA,
                                   lda,
                                   stA,
                                   dD,
                                   stD,
                                   dE,
                                   stE,
                                   dTauq,
                                   stQ,
                                   dTaup,
                                   stP,
                                   dWork,
                                   size_W,
                                   dInfo,
                                   bc,
                                   hA,
                                   hARes,
                                   hD,
                                   hE,
                                   hTauq,
                                   hTaup,
                                   hInfo,
                                   hInfoRes,
                                   &max_error);

        // collect performance data
        if(argus.timing)
            gebrd_getPerfData<API, T>(handle,
                                      m,
                                      n,
                                      dA,
                                      lda,
                                      stA,
                                      dD,
                                      stD,
                                      dE,
                                      stE,
                                      dTauq,
                                      stQ,
                                      dTaup,
                                      stP,
                                      dWork,
                                      size_W,
                                      dInfo,
                                      bc,
                                      hA,
                                      hD,
                                      hE,
                                      hTauq,
                                      hTaup,
                                      hInfo,
                                      &gpu_time_used,
                                      &cpu_time_used,
                                      hot_calls,
                                      argus.perf);
    }

    // validate results for rocsolver-test
    // using m*n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, m * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
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
