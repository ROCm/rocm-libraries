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

template <testAPI_t API, typename T, typename S, typename U>
void sygvdx_hegvdx_checkBadArgs(const hipsolverHandle_t   handle,
                                const hipsolverEigType_t  itype,
                                const hipsolverEigMode_t  evect,
                                const hipsolverEigRange_t erange,
                                const hipsolverFillMode_t uplo,
                                const int                 n,
                                T                         dA,
                                const int                 lda,
                                const int                 stA,
                                T                         dB,
                                const int                 ldb,
                                const int                 stB,
                                const S                   vl,
                                const S                   vu,
                                const int                 il,
                                const int                 iu,
                                int*                      hNev,
                                U                         dW,
                                const int                 stW,
                                T                         dWork,
                                const int                 lwork,
                                int*                      dInfo,
                                const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  nullptr,
                                                  itype,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  hipsolverEigType_t(-1),
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  hipsolverEigMode_t(-1),
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  hipsolverEigRange_t(-1),
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  erange,
                                                  hipsolverFillMode_t(-1),
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  (T) nullptr,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  (T) nullptr,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  (int*)nullptr,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  (U) nullptr,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  dInfo,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  erange,
                                                  uplo,
                                                  n,
                                                  dA,
                                                  lda,
                                                  stA,
                                                  dB,
                                                  ldb,
                                                  stB,
                                                  vl,
                                                  vu,
                                                  il,
                                                  iu,
                                                  hNev,
                                                  dW,
                                                  stW,
                                                  dWork,
                                                  lwork,
                                                  (int*)nullptr,
                                                  bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_sygvdx_hegvdx_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle handle;
    int                    n      = 1;
    int                    lda    = 1;
    int                    ldb    = 1;
    int                    ldz    = 1;
    int                    stA    = 1;
    int                    stB    = 1;
    int                    stW    = 1;
    int                    stE    = 1;
    int                    stZ    = 1;
    int                    bc     = 1;
    hipsolverEigType_t     itype  = HIPSOLVER_EIG_TYPE_1;
    hipsolverEigMode_t     evect  = HIPSOLVER_EIG_MODE_VECTOR;
    hipsolverEigRange_t    erange = HIPSOLVER_EIG_RANGE_V;
    hipsolverFillMode_t    uplo   = HIPSOLVER_FILL_MODE_UPPER;

    S   vl = 0.0;
    S   vu = 1.0;
    int il = 0;
    int iu = 0;

    if(BATCHED)
    {
        // // memory allocations
        // host_strided_batch_vector<int>   hNev(1, 1, 1, 1);
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_batch_vector<T>           dB(1, 1, 1);
        // device_batch_vector<T>           dZ(1, 1, 1);
        // device_strided_batch_vector<S>   dW(1, 1, 1, 1);
        // device_strided_batch_vector<S>   dE(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dB.memcheck());
        // CHECK_HIP_ERROR(dZ.memcheck());
        // CHECK_HIP_ERROR(dW.memcheck());
        // CHECK_HIP_ERROR(dE.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // int size_W;
        // hipsolver_sygvdx_hegvdx_bufferSize(API,
        //                                    handle,
        //                                    itype,
        //                                    evect,
        //                                    erange,
        //                                    uplo,
        //                                    n,
        //                                    dA.data(),
        //                                    lda,
        //                                    dB.data(),
        //                                    ldb,
        //                                    vl,
        //                                    vu,
        //                                    il,
        //                                    iu,
        //                                    hNev.data(),
        //                                    dW.data(),
        //                                    &size_W);
        // size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr ? size_W : sizeof(T) * size_W;
        // device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // sygvdx_hegvdx_checkBadArgs<API>(handle,
        //                                     itype,
        //                                     evect,
        //                                     erange,
        //                                     uplo,
        //                                     n,
        //                                     dA.data(),
        //                                     lda,
        //                                     stA,
        //                                     dB.data(),
        //                                     ldb,
        //                                     stB,
        //                                     vl,
        //                                     vu,
        //                                     il,
        //                                     iu,
        //                                     hNev.data(),
        //                                     dW.data(),
        //                                     stW,
        //                                     dWork.data(),
        //                                     size_W,
        //                                     dInfo.data(),
        //                                     bc);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<int>   hNev(1, 1, 1, 1);
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dB(1, 1, 1, 1);
        device_strided_batch_vector<T>   dZ(1, 1, 1, 1);
        device_strided_batch_vector<S>   dW(1, 1, 1, 1);
        device_strided_batch_vector<S>   dE(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_sygvdx_hegvdx_bufferSize(API,
                                           handle,
                                           itype,
                                           evect,
                                           erange,
                                           uplo,
                                           n,
                                           dA.data(),
                                           lda,
                                           dB.data(),
                                           ldb,
                                           vl,
                                           vu,
                                           il,
                                           iu,
                                           hNev.data(),
                                           dW.data(),
                                           &size_W);
        size_t bytes_W = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr
                             ? size_W
                             : sizeof(T) * size_W;
        device_strided_batch_vector<T> dWork(bytes_W, 1, bytes_W, 1);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        sygvdx_hegvdx_checkBadArgs<API>(handle,
                                        itype,
                                        evect,
                                        erange,
                                        uplo,
                                        n,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dB.data(),
                                        ldb,
                                        stB,
                                        vl,
                                        vu,
                                        il,
                                        iu,
                                        hNev.data(),
                                        dW.data(),
                                        stW,
                                        dWork.data(),
                                        size_W,
                                        dInfo.data(),
                                        bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sygvdx_hegvdx_initData(const hipsolverHandle_t       handle,
                            const hipsolverEigType_t      itype,
                            const hipsolverEigMode_t      evect,
                            const int                     n,
                            Td&                           dA,
                            const int                     lda,
                            const int                     stA,
                            Td&                           dB,
                            const int                     ldb,
                            const int                     stB,
                            const int                     bc,
                            Th&                           hA,
                            Th&                           hB,
                            host_strided_batch_vector<T>& A,
                            host_strided_batch_vector<T>& B,
                            const bool                    test)
{
    if(CPU)
    {
        int                          info;
        int                          ldu = n;
        host_strided_batch_vector<T> U(n * n, 1, n * n, bc);
        rocblas_init<T>(hA, true);
        rocblas_init<T>(U, true);

        for(int b = 0; b < bc; ++b)
        {
            // for testing purposes, we start with a reduced matrix M for the standard equivalent problem
            // with spectrum in a desired range (-20, 20). Then we construct the generalized pair
            // (A, B) from there.
            for(int i = 0; i < n; i++)
            {
                // scale matrices and set hA = M (symmetric/hermitian), hB = U (upper triangular)
                for(int j = i; j < n; j++)
                {
                    if(i == j)
                    {
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 10;
                        U[b][i + j * ldu]  = std::real(U[b][i + j * ldu]) / 100 + 1;
                        hB[b][i + j * ldb] = U[b][i + j * ldu];
                    }
                    else
                    {
                        if(j == i + 1)
                        {
                            hA[b][i + j * lda] = (hA[b][i + j * lda] - 5) / 10;
                            hA[b][j + i * lda] = conj(hA[b][i + j * lda]);
                        }
                        else
                            hA[b][j + i * lda] = hA[b][i + j * lda] = 0;

                        U[b][i + j * ldu]  = (U[b][i + j * ldu] - 5) / 100;
                        hB[b][i + j * ldb] = U[b][i + j * ldu];
                        hB[b][j + i * ldb] = 0;
                        U[b][j + i * ldu]  = 0;
                    }
                }
                if(i == n / 4 || i == n / 2 || i == n - 1 || i == n / 7 || i == n / 5 || i == n / 3)
                    hA[b][i + i * lda] *= -1;
            }

            // form B = U' U
            T one = T(1);
            cpu_trmm(HIPSOLVER_SIDE_LEFT,
                     HIPSOLVER_FILL_MODE_UPPER,
                     HIPSOLVER_OP_C,
                     'N',
                     n,
                     n,
                     one,
                     U[b],
                     ldu,
                     hB[b],
                     ldb);

            if(itype == HIPSOLVER_EIG_TYPE_1)
            {
                // form A = U' M U
                cpu_trmm(HIPSOLVER_SIDE_LEFT,
                         HIPSOLVER_FILL_MODE_UPPER,
                         HIPSOLVER_OP_C,
                         'N',
                         n,
                         n,
                         one,
                         U[b],
                         ldu,
                         hA[b],
                         lda);
                cpu_trmm(HIPSOLVER_SIDE_RIGHT,
                         HIPSOLVER_FILL_MODE_UPPER,
                         HIPSOLVER_OP_N,
                         'N',
                         n,
                         n,
                         one,
                         U[b],
                         ldu,
                         hA[b],
                         lda);
            }
            else
            {
                // form A = inv(U) M inv(U')
                cpu_trsm(HIPSOLVER_SIDE_LEFT,
                         HIPSOLVER_FILL_MODE_UPPER,
                         HIPSOLVER_OP_N,
                         'N',
                         n,
                         n,
                         one,
                         U[b],
                         ldu,
                         hA[b],
                         lda);
                cpu_trsm(HIPSOLVER_SIDE_RIGHT,
                         HIPSOLVER_FILL_MODE_UPPER,
                         HIPSOLVER_OP_C,
                         'N',
                         n,
                         n,
                         one,
                         U[b],
                         ldu,
                         hA[b],
                         lda);
            }

            // store A and B for testing purposes
            if(test && evect != HIPSOLVER_EIG_MODE_NOVECTOR)
            {
                for(int i = 0; i < n; i++)
                {
                    for(int j = 0; j < n; j++)
                    {
                        if(itype != HIPSOLVER_EIG_TYPE_3)
                        {
                            A[b][i + j * lda] = hA[b][i + j * lda];
                            B[b][i + j * ldb] = hB[b][i + j * ldb];
                        }
                        else
                        {
                            A[b][i + j * lda] = hB[b][i + j * ldb];
                            B[b][i + j * ldb] = hA[b][i + j * lda];
                        }
                    }
                }
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <testAPI_t API,
          typename T,
          typename S,
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh,
          typename Vh>
void sygvdx_hegvdx_getError(const hipsolverHandle_t   handle,
                            const hipsolverEigType_t  itype,
                            const hipsolverEigMode_t  evect,
                            const hipsolverEigRange_t erange,
                            const hipsolverFillMode_t uplo,
                            const int                 n,
                            Td&                       dA,
                            const int                 lda,
                            const int                 stA,
                            Td&                       dB,
                            const int                 ldb,
                            const int                 stB,
                            const S                   vl,
                            const S                   vu,
                            const int                 il,
                            const int                 iu,
                            Vh&                       hNevRes,
                            Ud&                       dW,
                            const int                 stW,
                            Td&                       dWork,
                            const int                 lwork,
                            Vd&                       dInfo,
                            const int                 bc,
                            Th&                       hA,
                            Th&                       hARes,
                            Th&                       hB,
                            Vh&                       hNev,
                            Uh&                       hW,
                            Uh&                       hWRes,
                            Vh&                       hInfo,
                            Vh&                       hInfoRes,
                            double*                   max_err)
{
    constexpr bool COMPLEX = is_complex<T>;

    int size_work  = (COMPLEX ? 2 * n : 8 * n);
    int size_rwork = (COMPLEX ? 7 * n : 0);
    int size_iwork = 5 * n;

    std::vector<T>               work(size_work);
    std::vector<S>               rwork(size_rwork);
    std::vector<int>             iwork(size_iwork);
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);
    std::vector<T>               Z(n * n);
    std::vector<int>             ifail(n);

    // input data initialization
    sygvdx_hegvdx_initData<true, true, T>(
        handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, true);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_sygvdx_hegvdx(API,
                                                handle,
                                                itype,
                                                evect,
                                                erange,
                                                uplo,
                                                n,
                                                dA.data(),
                                                lda,
                                                stA,
                                                dB.data(),
                                                ldb,
                                                stB,
                                                vl,
                                                vu,
                                                il,
                                                iu,
                                                hNevRes.data(),
                                                dW.data(),
                                                stW,
                                                dWork.data(),
                                                lwork,
                                                dInfo.data(),
                                                bc));

    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect != HIPSOLVER_EIG_MODE_NOVECTOR)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    S abstol = 2 * get_safemin<S>();
    for(int b = 0; b < bc; ++b)
    {
        cpu_sygvx_hegvx(itype,
                        evect,
                        erange,
                        uplo,
                        n,
                        hA[b],
                        lda,
                        hB[b],
                        ldb,
                        vl,
                        vu,
                        il,
                        iu,
                        abstol,
                        hNev[b],
                        hW[b],
                        Z.data(),
                        n,
                        work.data(),
                        size_work,
                        rwork.data(),
                        iwork.data(),
                        ifail.data(),
                        hInfo[b]);
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved.
    // We do test with indefinite matrices B).

    // check info for non-convergence and/or positive-definiteness
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
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

    double err;

    for(int b = 0; b < bc; ++b)
    {
        if(evect == HIPSOLVER_EIG_MODE_NOVECTOR)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hW - hWRes|| / ||hW||
            // using frobenius norm
            if(hInfoRes[b][0] == 0)
            {
                err      = norm_error('F', 1, hNev[b][0], 1, hW[b], hWRes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hInfoRes[b][0] == 0)
            {
                T alpha = 1;
                T beta  = 0;

                // hARes contains eigenvectors x
                // compute B*x (or A*x) and store in hB
                cpu_symm_hemm(HIPSOLVER_SIDE_LEFT,
                              uplo,
                              n,
                              hNev[b][0],
                              alpha,
                              B[b],
                              ldb,
                              hARes[b],
                              lda,
                              beta,
                              hB[b],
                              ldb);

                if(itype == HIPSOLVER_EIG_TYPE_1)
                {
                    // problem is A*x = (lambda)*B*x

                    // compute (1/lambda)*A*x and store in hA
                    for(int j = 0; j < hNev[b][0]; j++)
                    {
                        alpha = T(1) / hWRes[b][j];
                        cpu_symv_hemv(uplo,
                                      n,
                                      alpha,
                                      A[b],
                                      lda,
                                      hARes[b] + j * lda,
                                      1,
                                      beta,
                                      hA[b] + j * lda,
                                      1);
                    }

                    // move B*x into hARes
                    for(int i = 0; i < n; i++)
                        for(int j = 0; j < hNev[b][0]; j++)
                            hARes[b][i + j * lda] = hB[b][i + j * ldb];
                }
                else
                {
                    // problem is A*B*x = (lambda)*x or B*A*x = (lambda)*x

                    // compute (1/lambda)*A*B*x or (1/lambda)*B*A*x and store in hA
                    for(int j = 0; j < hNev[b][0]; j++)
                    {
                        alpha = T(1) / hWRes[b][j];
                        cpu_symv_hemv(uplo,
                                      n,
                                      alpha,
                                      A[b],
                                      lda,
                                      hB[b] + j * ldb,
                                      1,
                                      beta,
                                      hA[b] + j * lda,
                                      1);
                    }
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
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh,
          typename Vh>
void sygvdx_hegvdx_getPerfData(const hipsolverHandle_t   handle,
                               const hipsolverEigType_t  itype,
                               const hipsolverEigMode_t  evect,
                               const hipsolverEigRange_t erange,
                               const hipsolverFillMode_t uplo,
                               const int                 n,
                               Td&                       dA,
                               const int                 lda,
                               const int                 stA,
                               Td&                       dB,
                               const int                 ldb,
                               const int                 stB,
                               const S                   vl,
                               const S                   vu,
                               const int                 il,
                               const int                 iu,
                               Vh&                       hNevRes,
                               Ud&                       dW,
                               const int                 stW,
                               Td&                       dWork,
                               const int                 lwork,
                               Vd&                       dInfo,
                               const int                 bc,
                               Th&                       hA,
                               Th&                       hB,
                               Vh&                       hNev,
                               Uh&                       hW,
                               Vh&                       hInfo,
                               double*                   gpu_time_used,
                               double*                   cpu_time_used,
                               const int                 hot_calls,
                               const bool                perf)
{
    constexpr bool COMPLEX = is_complex<T>;

    int size_work  = (COMPLEX ? 2 * n : 8 * n);
    int size_rwork = (COMPLEX ? 7 * n : 0);
    int size_iwork = 5 * n;

    std::vector<T>               work(size_work);
    std::vector<S>               rwork(size_rwork);
    std::vector<int>             iwork(size_iwork);
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);
    std::vector<T>               Z(n * n);
    std::vector<int>             ifail(n);

    S abstol = 2 * get_safemin<S>();

    if(!perf)
    {
        sygvdx_hegvdx_initData<true, false, T>(
            handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cpu_sygvx_hegvx(itype,
                            evect,
                            erange,
                            uplo,
                            n,
                            hA[b],
                            lda,
                            hB[b],
                            ldb,
                            vl,
                            vu,
                            il,
                            iu,
                            abstol,
                            hNev[b],
                            hW[b],
                            Z.data(),
                            n,
                            work.data(),
                            size_work,
                            rwork.data(),
                            iwork.data(),
                            ifail.data(),
                            hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sygvdx_hegvdx_initData<true, false, T>(
        handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygvdx_hegvdx_initData<false, true, T>(
            handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false);

        CHECK_ROCBLAS_ERROR(hipsolver_sygvdx_hegvdx(API,
                                                    handle,
                                                    itype,
                                                    evect,
                                                    erange,
                                                    uplo,
                                                    n,
                                                    dA.data(),
                                                    lda,
                                                    stA,
                                                    dB.data(),
                                                    ldb,
                                                    stB,
                                                    vl,
                                                    vu,
                                                    il,
                                                    iu,
                                                    hNevRes.data(),
                                                    dW.data(),
                                                    stW,
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
        sygvdx_hegvdx_initData<false, true, T>(
            handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false);

        start = get_time_us_sync(stream);
        hipsolver_sygvdx_hegvdx(API,
                                handle,
                                itype,
                                evect,
                                erange,
                                uplo,
                                n,
                                dA.data(),
                                lda,
                                stA,
                                dB.data(),
                                ldb,
                                stB,
                                vl,
                                vu,
                                il,
                                iu,
                                hNevRes.data(),
                                dW.data(),
                                stW,
                                dWork.data(),
                                lwork,
                                dInfo.data(),
                                bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_sygvdx_hegvdx(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle handle;
    char                   itypeC  = argus.get<char>("itype");
    char                   evectC  = argus.get<char>("jobz");
    char                   erangeC = argus.get<char>("range");
    char                   uploC   = argus.get<char>("uplo");
    int                    n       = argus.get<int>("n");
    int                    lda     = argus.get<int>("lda", n);
    int                    ldb     = argus.get<int>("ldb", n);
    int                    stA     = argus.get<int>("strideA", lda * n);
    int                    stB     = argus.get<int>("strideB", ldb * n);
    int                    stW     = argus.get<int>("strideW", n);

    S   vl = S(argus.get<double>("vl", 0));
    S   vu = S(argus.get<double>("vu", erangeC == 'V' ? 1 : 0));
    int il = argus.get<int>("il", erangeC == 'I' ? 1 : 0);
    int iu = argus.get<int>("iu", erangeC == 'I' ? 1 : 0);

    hipsolverEigType_t  itype     = char2hipsolver_eform(itypeC);
    hipsolverEigMode_t  evect     = char2hipsolver_evect(evectC);
    hipsolverEigRange_t erange    = char2hipsolver_erange(erangeC);
    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    int stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    int stWRes = (argus.unit_check || argus.norm_check) ? stW : 0;

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_B    = size_t(ldb) * n;
    size_t size_W    = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size
        = (n < 0 || lda < n || ldb < n || bc < 0 || (erange == HIPSOLVER_EIG_RANGE_V && vl >= vu)
           || (erange == HIPSOLVER_EIG_RANGE_I && (il < 1 || iu < 0))
           || (erange == HIPSOLVER_EIG_RANGE_I && (iu > n || (n > 0 && il > iu))));
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
            //                                               handle,
            //                                               itype,
            //                                               evect,
            //                                               erange,
            //                                               uplo,
            //                                               n,
            //                                               (T* const*)nullptr,
            //                                               lda,
            //                                               stA,
            //                                               (T* const*)nullptr,
            //                                               ldb,
            //                                               stB,
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
            EXPECT_ROCBLAS_STATUS(hipsolver_sygvdx_hegvdx(API,
                                                          handle,
                                                          itype,
                                                          evect,
                                                          erange,
                                                          uplo,
                                                          n,
                                                          (T*)nullptr,
                                                          lda,
                                                          stA,
                                                          (T*)nullptr,
                                                          ldb,
                                                          stB,
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
    hipsolver_sygvdx_hegvdx_bufferSize(API,
                                       handle,
                                       itype,
                                       evect,
                                       erange,
                                       uplo,
                                       n,
                                       (T*)nullptr,
                                       lda,
                                       (T*)nullptr,
                                       ldb,
                                       vl,
                                       vu,
                                       il,
                                       iu,
                                       (int*)nullptr,
                                       (S*)nullptr,
                                       &size_Work);
    size_t bytes_Work = std::getenv("HIPSOLVER_BUFFERSIZE_RETURN_BYTES") != nullptr
                            ? size_Work
                            : sizeof(T) * size_Work;

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
    host_strided_batch_vector<S>   hWRes(size_WRes, 1, stWRes, bc);
    host_strided_batch_vector<int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<int> hInfoRes(1, 1, 1, bc);
    // device
    device_strided_batch_vector<S>   dW(size_W, 1, stW, bc);
    device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
    device_strided_batch_vector<T>   dWork(
        bytes_Work, 1, bytes_Work, 1); // bytes_Work accounts for bc
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_Work)
        CHECK_HIP_ERROR(dWork.memcheck());

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>   hA(size_A, 1, bc);
        // host_batch_vector<T>   hARes(size_ARes, 1, bc);
        // host_batch_vector<T>   hB(size_B, 1, bc);
        // device_batch_vector<T> dA(size_A, 1, bc);
        // device_batch_vector<T> dB(size_B, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_B)
        //     CHECK_HIP_ERROR(dB.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     sygvdx_hegvdx_getError<API, T>(handle,
        //                                        itype,
        //                                        evect,
        //                                        erange,
        //                                        uplo,
        //                                        n,
        //                                        dA,
        //                                        lda,
        //                                        stA,
        //                                        dB,
        //                                        ldb,
        //                                        stB,
        //                                        vl,
        //                                        vu,
        //                                        il,
        //                                        iu,
        //                                        hNevRes,
        //                                        dW,
        //                                        stW,
        //                                        dWork,
        //                                        size_Work,
        //                                        dInfo,
        //                                        bc,
        //                                        hA,
        //                                        hARes,
        //                                        hB,
        //                                        hNev,
        //                                        hW,
        //                                        hWRes,
        //                                        hInfo,
        //                                        hInfoRes,
        //                                        &max_error);

        // // collect performance data
        // if(argus.timing)
        //     sygvdx_hegvdx_getPerfData<API, T>(handle,
        //                                           itype,
        //                                           evect,
        //                                           erange,
        //                                           uplo,
        //                                           n,
        //                                           dA,
        //                                           lda,
        //                                           stA,
        //                                           dB,
        //                                           ldb,
        //                                           stB,
        //                                           vl,
        //                                           vu,
        //                                           il,
        //                                           iu,
        //                                           hNevRes,
        //                                           dW,
        //                                           stW,
        //                                           dWork,
        //                                           size_Work,
        //                                           dInfo,
        //                                           bc,
        //                                           hA,
        //                                           hB,
        //                                           hNev,
        //                                           hW,
        //                                           hInfo,
        //                                           &gpu_time_used,
        //                                           &cpu_time_used,
        //                                           hot_calls,
        //                                           argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>   hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>   hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T>   hB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygvdx_hegvdx_getError<API, T>(handle,
                                           itype,
                                           evect,
                                           erange,
                                           uplo,
                                           n,
                                           dA,
                                           lda,
                                           stA,
                                           dB,
                                           ldb,
                                           stB,
                                           vl,
                                           vu,
                                           il,
                                           iu,
                                           hNevRes,
                                           dW,
                                           stW,
                                           dWork,
                                           size_Work,
                                           dInfo,
                                           bc,
                                           hA,
                                           hARes,
                                           hB,
                                           hNev,
                                           hW,
                                           hWRes,
                                           hInfo,
                                           hInfoRes,
                                           &max_error);

        // collect performance data
        if(argus.timing)
            sygvdx_hegvdx_getPerfData<API, T>(handle,
                                              itype,
                                              evect,
                                              erange,
                                              uplo,
                                              n,
                                              dA,
                                              lda,
                                              stA,
                                              dB,
                                              ldb,
                                              stB,
                                              vl,
                                              vu,
                                              il,
                                              iu,
                                              hNevRes,
                                              dW,
                                              stW,
                                              dWork,
                                              size_Work,
                                              dInfo,
                                              bc,
                                              hA,
                                              hB,
                                              hNev,
                                              hW,
                                              hInfo,
                                              &gpu_time_used,
                                              &cpu_time_used,
                                              hot_calls,
                                              argus.perf);
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
                rocsolver_bench_output(
                    "itype", "evect", "uplo", "n", "lda", "ldb", "strideW", "batch_c");
                rocsolver_bench_output(itypeC, evectC, uploC, n, lda, ldb, stW, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype",
                                       "evect",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "ldb",
                                       "strideA",
                                       "strideB",
                                       "strideW",
                                       "batch_c");
                rocsolver_bench_output(itypeC, evectC, uploC, n, lda, ldb, stA, stB, stW, bc);
            }
            else
            {
                rocsolver_bench_output("itype", "evect", "uplo", "n", "lda", "ldb");
                rocsolver_bench_output(itypeC, evectC, uploC, n, lda, ldb);
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
