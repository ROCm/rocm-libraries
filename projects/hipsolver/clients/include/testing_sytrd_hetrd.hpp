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

template <testAPI_t API, typename S, typename T, typename U, typename V>
void sytrd_hetrd_checkBadArgs(const hipsolverHandle_t   handle,
                              const hipsolverFillMode_t uplo,
                              const int                 n,
                              T                         dA,
                              const int                 lda,
                              const int                 stA,
                              S                         dD,
                              const int                 stD,
                              S                         dE,
                              const int                 stE,
                              U                         dTau,
                              const int                 stP,
                              U                         dWork,
                              const int                 lwork,
                              V                         dInfo,
                              const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                nullptr,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dE,
                                                stE,
                                                dTau,
                                                stP,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                handle,
                                                hipsolverFillMode_t(-1),
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dE,
                                                stE,
                                                dTau,
                                                stP,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                handle,
                                                uplo,
                                                n,
                                                (T) nullptr,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dE,
                                                stE,
                                                dTau,
                                                stP,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                handle,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                (S) nullptr,
                                                stD,
                                                dE,
                                                stE,
                                                dTau,
                                                stP,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                handle,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                (S) nullptr,
                                                stE,
                                                dTau,
                                                stP,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                handle,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dE,
                                                stE,
                                                (U) nullptr,
                                                stP,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                handle,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dD,
                                                stD,
                                                dE,
                                                stE,
                                                dTau,
                                                stP,
                                                dWork,
                                                lwork,
                                                (V) nullptr,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_sytrd_hetrd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle handle;
    hipsolverFillMode_t    uplo = HIPSOLVER_FILL_MODE_UPPER;
    int                    n    = 2;
    int                    lda  = 2;
    int                    stA  = 1;
    int                    stD  = 1;
    int                    stE  = 1;
    int                    stP  = 1;
    int                    bc   = 1;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        // device_strided_batch_vector<S>   dE(1, 1, 1, 1);
        // device_strided_batch_vector<T>   dTau(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dD.memcheck());
        // CHECK_HIP_ERROR(dE.memcheck());
        // CHECK_HIP_ERROR(dTau.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // int size_W;
        // hipsolver_sytrd_hetrd_bufferSize(
        //     API, handle, uplo, n, dA.data(), lda, dD.data(), dE.data(), dTau.data(), &size_W);
        // size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;
        // device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // sytrd_hetrd_checkBadArgs<API>(handle,
        //                                   uplo,
        //                                   n,
        //                                   dA.data(),
        //                                   lda,
        //                                   stA,
        //                                   dD.data(),
        //                                   stD,
        //                                   dE.data(),
        //                                   stE,
        //                                   dTau.data(),
        //                                   stP,
        //                                   dWork.data(),
        //                                   size_W,
        //                                   dInfo.data(),
        //                                   bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        device_strided_batch_vector<S>   dE(1, 1, 1, 1);
        device_strided_batch_vector<T>   dTau(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dTau.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_sytrd_hetrd_bufferSize(
            API, handle, uplo, n, dA.data(), lda, dD.data(), dE.data(), dTau.data(), &size_W);
        size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr
                             ? size_W
                             : sizeof(T) * size_W;
        device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        sytrd_hetrd_checkBadArgs<API>(handle,
                                      uplo,
                                      n,
                                      dA.data(),
                                      lda,
                                      stA,
                                      dD.data(),
                                      stD,
                                      dE.data(),
                                      stE,
                                      dTau.data(),
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
          typename Td,
          typename Th,
          std::enable_if_t<!is_complex<T>, int> = 0>
void sytrd_hetrd_initData(
    const hipsolverHandle_t handle, const int n, Td& dA, const int lda, const int bc, Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j || i == j + 1 || i == j - 1)
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

template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Th,
          std::enable_if_t<is_complex<T>, int> = 0>
void sytrd_hetrd_initData(
    const hipsolverHandle_t handle, const int n, Td& dA, const int lda, const int bc, Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = hA[b][i + j * lda].real() + 400;
                    else if(i == j + 1 || i == j - 1)
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
void sytrd_hetrd_getError(const hipsolverHandle_t   handle,
                          const hipsolverFillMode_t uplo,
                          const int                 n,
                          Td&                       dA,
                          const int                 lda,
                          const int                 stA,
                          Sd&                       dD,
                          const int                 stD,
                          Sd&                       dE,
                          const int                 stE,
                          Ud&                       dTau,
                          const int                 stP,
                          Ud&                       dWork,
                          const int                 lwork,
                          Vd&                       dInfo,
                          const int                 bc,
                          Th&                       hA,
                          Th&                       hARes,
                          Sh&                       hD,
                          Sh&                       hE,
                          Uh&                       hTau,
                          Vh&                       hInfo,
                          Vh&                       hInfoRes,
                          double*                   max_err)
{
    using S                = decltype(std::real(T{}));
    constexpr bool COMPLEX = is_complex<T>;

    std::vector<T> hW(32 * n);

    // input data initialization
    sytrd_hetrd_initData<true, true, T>(handle, n, dA, lda, bc, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_sytrd_hetrd(API,
                                              handle,
                                              uplo,
                                              n,
                                              dA.data(),
                                              lda,
                                              stA,
                                              dD.data(),
                                              stD,
                                              dE.data(),
                                              stE,
                                              dTau.data(),
                                              stP,
                                              dWork.data(),
                                              lwork,
                                              dInfo.data(),
                                              bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hTau.transfer_from(dTau));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // Reconstruct matrix A from the factorization for implicit testing
    // A = H(n-1)...H(2)H(1)*T*H(1)'H(2)'...H(n-1)' if upper
    // A = H(1)H(2)...H(n-1)*T*H(n-1)'...H(2)'H(1)' if lower
    std::vector<T> v(n);
    for(int b = 0; b < bc; ++b)
    {
        T* a = hARes[b];
        T* t = hTau[b];

        if(uplo == HIPSOLVER_FILL_MODE_LOWER)
        {
            for(int i = 0; i < n - 2; ++i)
                a[i + (n - 1) * lda] = 0;
            a[(n - 2) + (n - 1) * lda] = a[(n - 1) + (n - 2) * lda];

            // for each column
            for(int j = n - 2; j >= 0; --j)
            {
                // prepare T and v
                for(int i = 0; i < j - 1; ++i)
                    a[i + j * lda] = 0;
                if(j > 0)
                    a[(j - 1) + j * lda] = a[j + (j - 1) * lda];
                for(int i = j + 2; i < n; ++i)
                {
                    v[i - j - 1]   = a[i + j * lda];
                    a[i + j * lda] = 0;
                }
                v[0] = 1;

                // apply householder reflector
                cpu_larf(HIPSOLVER_SIDE_LEFT,
                         n - 1 - j,
                         n - j,
                         v.data(),
                         1,
                         t + j,
                         a + (j + 1) + j * lda,
                         lda,
                         hW.data());
                if(COMPLEX)
                    cpu_lacgv(1, t + j, 1);
                cpu_larf(HIPSOLVER_SIDE_RIGHT,
                         n - j,
                         n - 1 - j,
                         v.data(),
                         1,
                         t + j,
                         a + j + (j + 1) * lda,
                         lda,
                         hW.data());
            }
        }

        else
        {
            a[1] = a[lda];
            for(int i = 2; i < n; ++i)
                a[i] = 0;

            // for each column
            for(int j = 1; j <= n - 1; ++j)
            {
                // prepare T and v
                for(int i = 0; i < j - 1; ++i)
                {
                    v[i]           = a[i + j * lda];
                    a[i + j * lda] = 0;
                }
                v[j - 1] = 1;
                if(j < n - 1)
                    a[(j + 1) + j * lda] = a[j + (j + 1) * lda];
                for(int i = j + 2; i < n; ++i)
                    a[i + j * lda] = 0;

                // apply householder reflector
                cpu_larf(HIPSOLVER_SIDE_LEFT, j, j + 1, v.data(), 1, t + j - 1, a, lda, hW.data());
                if(COMPLEX)
                    cpu_lacgv(1, t + j - 1, 1);
                cpu_larf(HIPSOLVER_SIDE_RIGHT, j + 1, j, v.data(), 1, t + j - 1, a, lda, hW.data());
            }
        }
    }

    // error is ||hA - hARes|| / ||hA||
    // using frobenius norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        *max_err = (uplo == HIPSOLVER_FILL_MODE_LOWER)
                       ? norm_error_lowerTr('F', n, n, lda, hA[b], hARes[b])
                       : norm_error_upperTr('F', n, n, lda, hA[b], hARes[b]);
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
void sytrd_hetrd_getPerfData(const hipsolverHandle_t   handle,
                             const hipsolverFillMode_t uplo,
                             const int                 n,
                             Td&                       dA,
                             const int                 lda,
                             const int                 stA,
                             Sd&                       dD,
                             const int                 stD,
                             Sd&                       dE,
                             const int                 stE,
                             Ud&                       dTau,
                             const int                 stP,
                             Ud&                       dWork,
                             const int                 lwork,
                             Vd&                       dInfo,
                             const int                 bc,
                             Th&                       hA,
                             Sh&                       hD,
                             Sh&                       hE,
                             Uh&                       hTau,
                             Vh&                       hInfo,
                             double*                   gpu_time_used,
                             double*                   cpu_time_used,
                             const int                 hot_calls,
                             const bool                perf)
{
    using S = decltype(std::real(T{}));

    std::vector<T> hW(32 * n);

    if(!perf)
    {
        sytrd_hetrd_initData<true, false, T>(handle, n, dA, lda, bc, hA);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_sytrd_hetrd(uplo, n, hA[b], lda, hD[b], hE[b], hTau[b], hW.data(), 32 * n);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sytrd_hetrd_initData<true, false, T>(handle, n, dA, lda, bc, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sytrd_hetrd_initData<false, true, T>(handle, n, dA, lda, bc, hA);

        CHECK_ROCBLAS_ERROR(hipsolver_sytrd_hetrd(API,
                                                  handle,
                                                  uplo,
                                                  n,
                                                  dA.data(),
                                                  lda,
                                                  stA,
                                                  dD.data(),
                                                  stD,
                                                  dE.data(),
                                                  stE,
                                                  dTau.data(),
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
        sytrd_hetrd_initData<false, true, T>(handle, n, dA, lda, bc, hA);

        start = get_time_us_sync(stream);
        hipsolver_sytrd_hetrd(API,
                              handle,
                              uplo,
                              n,
                              dA.data(),
                              lda,
                              stA,
                              dD.data(),
                              stD,
                              dE.data(),
                              stE,
                              dTau.data(),
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
void testing_sytrd_hetrd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle handle;
    char                   uploC = argus.get<char>("uplo");
    int                    n     = argus.get<int>("n");
    int                    lda   = argus.get<int>("lda", n);
    int                    stA   = argus.get<int>("strideA", lda * n);
    int                    stD   = argus.get<int>("strideD", n);
    int                    stE   = argus.get<int>("strideE", n - 1);
    int                    stP   = argus.get<int>("strideP", n - 1);

    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    int stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    if(uplo != HIPSOLVER_FILL_MODE_UPPER && uplo != HIPSOLVER_FILL_MODE_LOWER)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
            //                                             handle,
            //                                             uplo,
            //                                             n,
            //                                             (T* const*)nullptr,
            //                                             lda,
            //                                             stA,
            //                                             (S*)nullptr,
            //                                             stD,
            //                                             (S*)nullptr,
            //                                             stE,
            //                                             (T*)nullptr,
            //                                             stP,
            //                                             (T*)nullptr,
            //                                             0,
            //                                             (int*)nullptr,
            //                                             bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                        handle,
                                                        uplo,
                                                        n,
                                                        (T*)nullptr,
                                                        lda,
                                                        stA,
                                                        (S*)nullptr,
                                                        stD,
                                                        (S*)nullptr,
                                                        stE,
                                                        (T*)nullptr,
                                                        stP,
                                                        (T*)nullptr,
                                                        0,
                                                        (int*)nullptr,
                                                        bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A    = lda * n;
    size_t size_D    = n;
    size_t size_E    = n - 1;
    size_t size_tau  = n - 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
            //                                             handle,
            //                                             uplo,
            //                                             n,
            //                                             (T* const*)nullptr,
            //                                             lda,
            //                                             stA,
            //                                             (S*)nullptr,
            //                                             stD,
            //                                             (S*)nullptr,
            //                                             stE,
            //                                             (T*)nullptr,
            //                                             stP,
            //                                             (T*)nullptr,
            //                                             0,
            //                                             (int*)nullptr,
            //                                             bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_sytrd_hetrd(API,
                                                        handle,
                                                        uplo,
                                                        n,
                                                        (T*)nullptr,
                                                        lda,
                                                        stA,
                                                        (S*)nullptr,
                                                        stD,
                                                        (S*)nullptr,
                                                        stE,
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
    hipsolver_sytrd_hetrd_bufferSize(
        API, handle, uplo, n, (T*)nullptr, lda, (S*)nullptr, (S*)nullptr, (T*)nullptr, &size_W);
    size_t bytes_W
        = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_W);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S>   hD(size_D, 1, stD, bc);
    host_strided_batch_vector<S>   hE(size_E, 1, stE, bc);
    host_strided_batch_vector<T>   hTau(size_tau, 1, stP, bc);
    host_strided_batch_vector<int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<int> hInfoRes(1, 1, 1, bc);
    // device
    device_strided_batch_vector<S>   dD(size_D, 1, stD, bc);
    device_strided_batch_vector<S>   dE(size_E, 1, stE, bc);
    device_strided_batch_vector<T>   dTau(size_tau, 1, stP, bc);
    device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
    device_strided_batch_vector<T>   dWork(bytes_W, 1, bytes_W, 1); // bytes_W accounts for bc
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>           hA(size_A, 1, bc);
        // host_batch_vector<T>           hARes(size_ARes, 1, bc);
        // device_batch_vector<T>         dA(size_A, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     sytrd_hetrd_getError<API, T>(handle,
        //                                      uplo,
        //                                      n,
        //                                      dA,
        //                                      lda,
        //                                      stA,
        //                                      dD,
        //                                      stD,
        //                                      dE,
        //                                      stE,
        //                                      dTau,
        //                                      stP,
        //                                      dWork,
        //                                      size_W,
        //                                      dInfo,
        //                                      bc,
        //                                      hA,
        //                                      hARes,
        //                                      hD,
        //                                      hE,
        //                                      hTau,
        //                                      hInfo,
        //                                      hInfoRes,
        //                                      &max_error);

        // // collect performance data
        // if(argus.timing)
        //     sytrd_hetrd_getPerfData<API, T>(handle,
        //                                         uplo,
        //                                         n,
        //                                         dA,
        //                                         lda,
        //                                         stA,
        //                                         dD,
        //                                         stD,
        //                                         dE,
        //                                         stE,
        //                                         dTau,
        //                                         stP,
        //                                         dWork,
        //                                         size_W,
        //                                         dInfo,
        //                                         bc,
        //                                         hA,
        //                                         hD,
        //                                         hE,
        //                                         hTau,
        //                                         hInfo,
        //                                         &gpu_time_used,
        //                                         &cpu_time_used,
        //                                         hot_calls,
        //                                         argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>   hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>   hARes(size_ARes, 1, stARes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytrd_hetrd_getError<API, T>(handle,
                                         uplo,
                                         n,
                                         dA,
                                         lda,
                                         stA,
                                         dD,
                                         stD,
                                         dE,
                                         stE,
                                         dTau,
                                         stP,
                                         dWork,
                                         size_W,
                                         dInfo,
                                         bc,
                                         hA,
                                         hARes,
                                         hD,
                                         hE,
                                         hTau,
                                         hInfo,
                                         hInfoRes,
                                         &max_error);

        // collect performance data
        if(argus.timing)
            sytrd_hetrd_getPerfData<API, T>(handle,
                                            uplo,
                                            n,
                                            dA,
                                            lda,
                                            stA,
                                            dD,
                                            stD,
                                            dE,
                                            stE,
                                            dTau,
                                            stP,
                                            dWork,
                                            size_W,
                                            dInfo,
                                            bc,
                                            hA,
                                            hD,
                                            hE,
                                            hTau,
                                            hInfo,
                                            &gpu_time_used,
                                            &cpu_time_used,
                                            hot_calls,
                                            argus.perf);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

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
                rocsolver_bench_output(
                    "uplo", "n", "lda", "strideD", "strideE", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stD, stE, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output(
                    "uplo", "n", "lda", "strideA", "strideD", "strideE", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, lda, stA, stD, stE, stP, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "lda");
                rocsolver_bench_output(uploC, n, lda);
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
