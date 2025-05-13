/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

template <testAPI_t API, typename T, typename S, typename SS, typename U>
void syevdx_heevdx_checkBadArgs(const hipsolverHandle_t   handle,
                                const hipsolverEigMode_t  evect,
                                const hipsolverEigRange_t erange,
                                const hipsolverFillMode_t uplo,
                                const int                 n,
                                T                         dA,
                                const int                 lda,
                                const int                 stA,
                                const SS                  vl,
                                const SS                  vu,
                                const int                 il,
                                const int                 iu,
                                U                         hNev,
                                S                         dW,
                                const int                 stW,
                                T                         dWork,
                                const int                 lwork,
                                U                         dinfo,
                                const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  nullptr,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dinfo,
                                                  bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  handle,
                                                  hipsolverEigMode_t(-1),
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dinfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  handle,
                                                  evect,
                                                  hipsolverEigRange_t(-1),
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dinfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  handle,
                                                  evect,
                                                  erange,
                                                  hipsolverFillMode_t(-1),
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dinfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  handle,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  (T) nullptr,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dinfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  handle,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  (U) nullptr,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dinfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  handle,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  (S) nullptr,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dinfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                  handle,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  (U) nullptr,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_syevdx_heevdx_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle handle;
    hipsolverEigMode_t     evect  = HIPSOLVER_EIG_MODE_VECTOR;
    hipsolverEigRange_t    erange = HIPSOLVER_EIG_RANGE_V;
    hipsolverFillMode_t    uplo   = HIPSOLVER_FILL_MODE_LOWER;
    int                    n      = 1;
    int                    lda    = 1;
    int                    stA    = 1;
    int                    stW    = 1;
    int                    bc     = 1;

    S   vl = 0.0;
    S   vu = 1.0;
    int il = 0;
    int iu = 0;

    if(BATCHED)
    {
        // // memory allocations
        // host_strided_batch_vector<int>   hNev(1, 1, 1, 1);
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_strided_batch_vector<S>   dW(1, 1, 1, 1);
        // device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dW.memcheck());
        // CHECK_HIP_ERROR(dinfo.memcheck());

        // int size_W;
        // hipsolver_syevdx_heevdx_bufferSize(API,
        //                                    handle,
        //                                    evect,
        //                                    erange,
        //                                    uplo,
        //                                    n,
        //                                    dA.data(),
        //                                    lda,
        //                                    vl,
        //                                    vu,
        //                                    il,
        //                                    iu,
        //                                    hNev.data(),
        //                                    dW.data(),
        //                                    &size_W);
        // size_t bytes_W = sizeof(T) * size_W;
        // device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // syevdx_heevdx_checkBadArgs<API>(handle,
        //                                     evect,
        //                                     erange,
        //                                     uplo,
        //                                     n,
        //                                     dA.data(),
        //                                     lda,
        //                                     stA,
        //                                     vl,
        //                                     vu,
        //                                     il,
        //                                     iu,
        //                                     hNev.data(),
        //                                     dW.data(),
        //                                     stW,
        //                                     dWork.data(),
        //                                     size_W,
        //                                     dinfo.data(),
        //                                     bc);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<int>   hNev(1, 1, 1, 1);
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<S>   dW(1, 1, 1, 1);
        device_strided_batch_vector<int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        int size_W;
        hipsolver_syevdx_heevdx_bufferSize(API,
                                           handle,
                                           evect,
                                           erange,
                                           uplo,
                                           n,
                                           dA.data(),
                                           lda,
                                           vl,
                                           vu,
                                           il,
                                           iu,
                                           hNev.data(),
                                           dW.data(),
                                           &size_W);
        size_t                         bytes_W = sizeof(T) * size_W;
        device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        syevdx_heevdx_checkBadArgs<API>(handle,
                                        evect,
                                        erange,
                                        uplo,
                                        n,
                                        dA.data(),
                                        lda,
                                        stA,
                                        vl,
                                        vu,
                                        il,
                                        iu,
                                        hNev.data(),
                                        dW.data(),
                                        stW,
                                        dWork.data(),
                                        size_W,
                                        dinfo.data(),
                                        bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevdx_heevdx_initData(const hipsolverHandle_t  handle,
                            const hipsolverEigMode_t evect,
                            const int                n,
                            Td&                      dA,
                            const int                lda,
                            const int                bc,
                            Th&                      hA,
                            std::vector<T>&          A,
                            bool                     test = true)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < n; i++)
            {
                for(int j = i; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 10;
                    else
                    {
                        if(j == i + 1)
                        {
                            hA[b][i + j * lda] = (hA[b][i + j * lda] - 5) / 10;
                            hA[b][j + i * lda] = conj(hA[b][i + j * lda]);
                        }
                        else
                            hA[b][j + i * lda] = hA[b][i + j * lda] = 0;
                    }
                }
                if(i == n / 4 || i == n / 2 || i == n - 1 || i == n / 7 || i == n / 5 || i == n / 3)
                    hA[b][i + i * lda] *= -1;
            }

            // make copy of original data to test vectors if required
            if(test && evect == HIPSOLVER_EIG_MODE_VECTOR)
            {
                for(int i = 0; i < n; i++)
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
          typename T,
          typename S,
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Ih>
void syevdx_heevdx_getError(const hipsolverHandle_t   handle,
                            const hipsolverEigMode_t  evect,
                            const hipsolverEigRange_t erange,
                            const hipsolverFillMode_t uplo,
                            const int                 n,
                            Td&                       dA,
                            const int                 lda,
                            const int                 stA,
                            const S                   vl,
                            const S                   vu,
                            const int                 il,
                            const int                 iu,
                            Ih&                       hNevRes,
                            Sd&                       dW,
                            const int                 stW,
                            Td&                       dWork,
                            const int                 lwork,
                            Id&                       dinfo,
                            const int                 bc,
                            Th&                       hA,
                            Th&                       hARes,
                            Ih&                       hNev,
                            Sh&                       hW,
                            Sh&                       hWRes,
                            Ih&                       hinfo,
                            Ih&                       hinfoRes,
                            double*                   max_err)
{
    constexpr bool COMPLEX = is_complex<T>;

    int size_work  = !COMPLEX ? 35 * n : 33 * n;
    int size_rwork = !COMPLEX ? 0 : 7 * n;
    int size_iwork = 5 * n;

    std::vector<T>   work(size_work);
    std::vector<S>   rwork(size_rwork);
    std::vector<int> iwork(size_iwork);
    std::vector<T>   A(lda * n * bc);
    std::vector<T>   Z(lda * n);
    std::vector<int> ifail(n);

    S abstol = 2 * get_safemin<S>();

    // input data initialization
    syevdx_heevdx_initData<true, true, T>(handle, evect, n, dA, lda, bc, hA, A);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_syevdx_heevdx(API,
                                                handle,
                                                evect,
                                                erange,
                                                uplo,
                                                n,
                                                dA.data(),
                                                lda,
                                                stA,
                                                vl,
                                                vu,
                                                il,
                                                iu,
                                                hNevRes.data(),
                                                dW.data(),
                                                stW,
                                                dWork.data(),
                                                lwork,
                                                dinfo.data(),
                                                bc));

    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
    if(evect == HIPSOLVER_EIG_MODE_VECTOR)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
        cpu_syevx_heevx(evect,
                        erange,
                        uplo,
                        n,
                        hA[b],
                        lda,
                        vl,
                        vu,
                        il,
                        iu,
                        abstol,
                        hNev[b],
                        hW[b],
                        Z.data(),
                        lda,
                        work.data(),
                        size_work,
                        rwork.data(),
                        iwork.data(),
                        ifail.data(),
                        hinfo[b]);

    // Check info for non-convergence
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;
    }

    // Check number of returned eigenvalues
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hNev[b][0], hNevRes[b][0]) << "where b = " << b;
        if(hNev[b][0] != hNevRes[b][0])
            *max_err += 1;
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    double err = 0;

    for(int b = 0; b < bc; ++b)
    {
        if(evect != HIPSOLVER_EIG_MODE_VECTOR)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hW - hWRes|| / ||hW||
            // using frobenius norm
            if(hinfo[b][0] == 0)
                err = norm_error('F', 1, hNev[b][0], 1, hW[b], hWRes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hinfo[b][0] == 0)
            {
                // multiply A with each of the m eigenvectors and divide by corresponding
                // eigenvalues
                T alpha;
                T beta = 0;
                for(int j = 0; j < hNev[b][0]; j++)
                {
                    alpha = T(1) / hWRes[b][j];
                    cpu_symv_hemv(uplo,
                                  n,
                                  alpha,
                                  A.data() + b * lda * n,
                                  lda,
                                  hARes[b] + j * lda,
                                  1,
                                  beta,
                                  hA[b] + j * lda,
                                  1);
                }

                // error is ||hA - hARes|| / ||hA||
                // using frobenius norm
                err      = norm_error('F', n, hNev[b][0], lda, hA[b], hARes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <testAPI_t API,
          typename T,
          typename S,
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Ih>
void syevdx_heevdx_getPerfData(const hipsolverHandle_t   handle,
                               const hipsolverEigMode_t  evect,
                               const hipsolverEigRange_t erange,
                               const hipsolverFillMode_t uplo,
                               const int                 n,
                               Td&                       dA,
                               const int                 lda,
                               const int                 stA,
                               const S                   vl,
                               const S                   vu,
                               const int                 il,
                               const int                 iu,
                               Ih&                       hNevRes,
                               Sd&                       dW,
                               const int                 stW,
                               Td&                       dWork,
                               const int                 lwork,
                               Id&                       dinfo,
                               const int                 bc,
                               Th&                       hA,
                               Ih&                       hNev,
                               Sh&                       hW,
                               Ih&                       hinfo,
                               double*                   gpu_time_used,
                               double*                   cpu_time_used,
                               const int                 hot_calls,
                               const bool                perf)
{
    constexpr bool COMPLEX = is_complex<T>;

    int size_work  = !COMPLEX ? 35 * n : 33 * n;
    int size_rwork = !COMPLEX ? 0 : 7 * n;
    int size_iwork = 5 * n;

    std::vector<T>   work(size_work);
    std::vector<S>   rwork(size_rwork);
    std::vector<int> iwork(size_iwork);
    std::vector<T>   A;
    std::vector<T>   Z(lda * n);
    std::vector<int> ifail(n);

    S abstol = 2 * get_safemin<S>();

    if(!perf)
    {
        syevdx_heevdx_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_syevx_heevx(evect,
                            erange,
                            uplo,
                            n,
                            hA[b],
                            lda,
                            vl,
                            vu,
                            il,
                            iu,
                            abstol,
                            hNev[b],
                            hW[b],
                            Z.data(),
                            lda,
                            work.data(),
                            size_work,
                            rwork.data(),
                            iwork.data(),
                            ifail.data(),
                            hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    syevdx_heevdx_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        syevdx_heevdx_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(hipsolver_syevdx_heevdx(API,
                                                    handle,
                                                    evect,
                                                    erange,
                                                    uplo,
                                                    n,
                                                    dA.data(),
                                                    lda,
                                                    stA,
                                                    vl,
                                                    vu,
                                                    il,
                                                    iu,
                                                    hNevRes.data(),
                                                    dW.data(),
                                                    stW,
                                                    dWork.data(),
                                                    lwork,
                                                    dinfo.data(),
                                                    bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        syevdx_heevdx_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        hipsolver_syevdx_heevdx(API,
                                handle,
                                evect,
                                erange,
                                uplo,
                                n,
                                dA.data(),
                                lda,
                                stA,
                                vl,
                                vu,
                                il,
                                iu,
                                hNevRes.data(),
                                dW.data(),
                                stW,
                                dWork.data(),
                                lwork,
                                dinfo.data(),
                                bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_syevdx_heevdx(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle handle;
    char                   evectC  = argus.get<char>("jobz");
    char                   erangeC = argus.get<char>("range");
    char                   uploC   = argus.get<char>("uplo");
    int                    n       = argus.get<int>("n");
    int                    lda     = argus.get<int>("lda", n);
    int                    stA     = argus.get<int>("strideA", lda * n);
    int                    stW     = argus.get<int>("strideW", n);

    S   vl = S(argus.get<double>("vl", 0));
    S   vu = S(argus.get<double>("vu", erangeC == 'V' ? 1 : 0));
    int il = argus.get<int>("il", erangeC == 'I' ? 1 : 0);
    int iu = argus.get<int>("iu", erangeC == 'I' ? 1 : 0);

    hipsolverEigMode_t  evect     = char2hipsolver_evect(evectC);
    hipsolverEigRange_t erange    = char2hipsolver_erange(erangeC);
    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_W    = n;
    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0 || (erange == HIPSOLVER_EIG_RANGE_V && vl >= vu)
                         || (erange == HIPSOLVER_EIG_RANGE_I && (il < 1 || iu < 0))
                         || (erange == HIPSOLVER_EIG_RANGE_I && (iu > n || (n > 0 && il > iu))));
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
            //                                               handle,
            //                                               evect,
            //                                               erange,
            //                                               uplo,
            //                                               n,
            //                                               (T* const*)nullptr,
            //                                               lda,
            //                                               stA,
            //                                               vl,
            //                                               vu,
            //                                               il,
            //                                               iu,
            //                                               (int*)nullptr,
            //                                               (S*)nullptr,
            //                                               stW,
            //                                               (T*)nullptr,
            //                                               0,
            //                                               (int*)nullptr,
            //                                               bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_syevdx_heevdx(API,
                                                          handle,
                                                          evect,
                                                          erange,
                                                          uplo,
                                                          n,
                                                          (T*)nullptr,
                                                          lda,
                                                          stA,
                                                          vl,
                                                          vu,
                                                          il,
                                                          iu,
                                                          (int*)nullptr,
                                                          (S*)nullptr,
                                                          stW,
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
    int size_Work;
    hipsolver_syevdx_heevdx_bufferSize(API,
                                       handle,
                                       evect,
                                       erange,
                                       uplo,
                                       n,
                                       (T*)nullptr,
                                       lda,
                                       vl,
                                       vu,
                                       il,
                                       iu,
                                       (int*)nullptr,
                                       (S*)nullptr,
                                       &size_Work);
    size_t bytes_Work = sizeof(T) * size_Work;

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_Work);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<int> hNev(1, 1, 1, bc);
    host_strided_batch_vector<int> hNevRes(1, 1, 1, bc);
    host_strided_batch_vector<S>   hW(size_W, 1, stW, bc);
    host_strided_batch_vector<int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S>   hWres(size_WRes, 1, stW, bc);
    // device
    device_strided_batch_vector<S>   dW(size_W, 1, stW, bc);
    device_strided_batch_vector<int> dinfo(1, 1, 1, bc);
    device_strided_batch_vector<T>   dWork(
        bytes_Work, 1, bytes_Work, 1); // bytes_Work accounts for bc
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());
    if(size_Work)
        CHECK_HIP_ERROR(dWork.memcheck());

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>   hA(size_A, 1, bc);
        // host_batch_vector<T>   hARes(size_ARes, 1, bc);
        // device_batch_vector<T> dA(size_A, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        // {
        //     syevdx_heevdx_getError<API, T>(handle,
        //                                        evect,
        //                                        erange,
        //                                        uplo,
        //                                        n,
        //                                        dA,
        //                                        lda,
        //                                        stA,
        //                                        vl,
        //                                        vu,
        //                                        il,
        //                                        iu,
        //                                        hNevRes,
        //                                        dW,
        //                                        stW,
        //                                        dWork,
        //                                        size_Work,
        //                                        dinfo,
        //                                        bc,
        //                                        hA,
        //                                        hARes,
        //                                        hNev,
        //                                        hW,
        //                                        hWres,
        //                                        hinfo,
        //                                        hinfoRes,
        //                                        &max_error);
        // }

        // // collect performance data
        // if(argus.timing)
        // {
        //     syevdx_heevdx_getPerfData<API, T>(handle,
        //                                           evect,
        //                                           erange,
        //                                           uplo,
        //                                           n,
        //                                           dA,
        //                                           lda,
        //                                           stA,
        //                                           vl,
        //                                           vu,
        //                                           il,
        //                                           iu,
        //                                           hNevRes,
        //                                           dW,
        //                                           stW,
        //                                           dWork,
        //                                           size_Work,
        //                                           dinfo,
        //                                           bc,
        //                                           hA,
        //                                           hNev,
        //                                           hW,
        //                                           hinfo,
        //                                           &gpu_time_used,
        //                                           &cpu_time_used,
        //                                           hot_calls,
        //                                           argus.perf);
        // }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>   hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>   hARes(size_ARes, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevdx_heevdx_getError<API, T>(handle,
                                           evect,
                                           erange,
                                           uplo,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           vl,
                                           vu,
                                           il,
                                           iu,
                                           hNevRes,
                                           dW,
                                           stW,
                                           dWork,
                                           size_Work,
                                           dinfo,
                                           bc,
                                           hA,
                                           hARes,
                                           hNev,
                                           hW,
                                           hWres,
                                           hinfo,
                                           hinfoRes,
                                           &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevdx_heevdx_getPerfData<API, T>(handle,
                                              evect,
                                              erange,
                                              uplo,
                                              n,
                                              dA,
                                              lda,
                                              stA,
                                              vl,
                                              vu,
                                              il,
                                              iu,
                                              hNevRes,
                                              dW,
                                              stW,
                                              dWork,
                                              size_Work,
                                              dinfo,
                                              bc,
                                              hA,
                                              hNev,
                                              hW,
                                              hinfo,
                                              &gpu_time_used,
                                              &cpu_time_used,
                                              hot_calls,
                                              argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 3 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 3 * n);

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
                rocsolver_bench_output("jobz",
                                       "range",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "vl",
                                       "vu",
                                       "il",
                                       "iu",
                                       "strideW",
                                       "batch_c");
                rocsolver_bench_output(evectC, erangeC, uploC, n, lda, vl, vu, il, iu, stW, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("jobz",
                                       "range",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "strideA",
                                       "vl",
                                       "vu",
                                       "il",
                                       "iu",
                                       "strideW",
                                       "batch_c");
                rocsolver_bench_output(
                    evectC, erangeC, uploC, n, lda, stA, vl, vu, il, iu, stW, bc);
            }
            else
            {
                rocsolver_bench_output("jobz", "range", "uplo", "n", "lda", "vl", "vu", "il", "iu");
                rocsolver_bench_output(evectC, erangeC, uploC, n, lda, vl, vu, il, iu);
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
