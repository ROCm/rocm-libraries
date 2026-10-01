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

#include "../rocsolvercommon/matrix_utils.hpp"
#include "clientcommon.hpp"
#include "hipsolver_timer.hpp"

template <testAPI_t API, typename I, typename SIZE, typename Sd, typename Td, typename Ud>
void stedc_checkBadArgs(const hipsolverHandle_t   handle,
                        const hipsolverDnParams_t params,
                        const hipsolverEigComp_t  compz,
                        const I                   n,
                        Sd                        dD,
                        Sd                        dE,
                        Td                        dC,
                        const I                   ldc,
                        Td                        dWork,
                        const SIZE                bytes_dW,
                        Td                        hWork,
                        const SIZE                bytes_hW,
                        Ud                        dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                          nullptr,
                                          params,
                                          compz,
                                          n,
                                          dD,
                                          dE,
                                          dC,
                                          ldc,
                                          dWork,
                                          bytes_dW,
                                          hWork,
                                          bytes_hW,
                                          dInfo),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // params
    EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                          handle,
                                          (hipsolverDnParams_t) nullptr,
                                          compz,
                                          n,
                                          dD,
                                          dE,
                                          dC,
                                          ldc,
                                          dWork,
                                          bytes_dW,
                                          hWork,
                                          bytes_hW,
                                          dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                          handle,
                                          params,
                                          hipsolverEigComp_t(-1),
                                          n,
                                          dD,
                                          dE,
                                          dC,
                                          ldc,
                                          dWork,
                                          bytes_dW,
                                          hWork,
                                          bytes_hW,
                                          dInfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                          handle,
                                          params,
                                          compz,
                                          n,
                                          (Sd) nullptr,
                                          dE,
                                          dC,
                                          ldc,
                                          dWork,
                                          bytes_dW,
                                          hWork,
                                          bytes_hW,
                                          dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                          handle,
                                          params,
                                          compz,
                                          n,
                                          dD,
                                          (Sd) nullptr,
                                          dC,
                                          ldc,
                                          dWork,
                                          bytes_dW,
                                          hWork,
                                          bytes_hW,
                                          dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                          handle,
                                          params,
                                          compz,
                                          n,
                                          dD,
                                          dE,
                                          (Td) nullptr,
                                          ldc,
                                          dWork,
                                          bytes_dW,
                                          hWork,
                                          bytes_hW,
                                          dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                          handle,
                                          params,
                                          compz,
                                          n,
                                          dD,
                                          dE,
                                          dC,
                                          ldc,
                                          dWork,
                                          bytes_dW,
                                          hWork,
                                          bytes_hW,
                                          (Ud) nullptr),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_stedc_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    hipsolverEigComp_t     compz = HIPSOLVER_EIG_COMP_I;
    I                      n     = 2;
    I                      ldc   = 2;

    // memory allocations
    device_strided_batch_vector<S>   dD(1, 1, 1, 1);
    device_strided_batch_vector<S>   dE(1, 1, 1, 1);
    device_strided_batch_vector<T>   dC(1, 1, 1, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    SIZE bytes_dW, bytes_hW;
    hipsolver_stedc_bufferSize(API,
                               handle,
                               params,
                               compz,
                               n,
                               (S*)nullptr,
                               (S*)nullptr,
                               (T*)nullptr,
                               ldc,
                               &bytes_dW,
                               &bytes_hW);
    SIZE size_dW = (bytes_dW + sizeof(T) - 1) / sizeof(T);
    SIZE size_hW = (bytes_hW + sizeof(T) - 1) / sizeof(T);

    host_strided_batch_vector<T>   hWork(size_hW, 1, size_hW, 1);
    device_strided_batch_vector<T> dWork(size_dW, 1, size_dW, 1);
    if(size_dW)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    stedc_checkBadArgs<API>(handle,
                            params,
                            compz,
                            n,
                            dD.data(),
                            dE.data(),
                            dC.data(),
                            ldc,
                            dWork.data(),
                            bytes_dW,
                            hWork.data(),
                            bytes_hW,
                            dInfo.data());
}

template <bool CPU,
          bool GPU,
          typename T,
          typename I,
          typename Sd,
          typename Td,
          typename Ud,
          typename Sh,
          typename Th,
          typename Uh>
void stedc_initData(const hipsolverHandle_t  handle,
                    const hipsolverEigComp_t compz,
                    const I                  n,
                    Sd&                      dD,
                    Sd&                      dE,
                    Td&                      dC,
                    const I                  ldc,
                    Ud& /* dInfo */,
                    Sh& hD,
                    Sh& hE,
                    Th& hC,
                    Uh& /* hInfo */)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));

        // if the matrix is too small (n < 4), simply initialize D and E
        if(n < 4)
        {
            rocblas_init<S>(hD, true);
            rocblas_init<S>(hE, true);
        }

        // otherwise, the marix will be divided in exactly 2 independent blocks, if the size is even,
        // or 3 if the size is odd. The 2 main independent blocks will have the same eigenvalues.
        // The last block, when the size is odd, will have eigenvalue equal 1.
        else
        {
            I N1 = n / 2;
            I E  = n - 2 * N1;

            // a. initialize the eigenvalues for the uppermost sub-blocks of the main independent blocks.
            // The second sub-block will have some repeated eigenvalues in order to test the deflation process
            S              d;
            I              NN1 = N1 / 2;
            I              NN2 = N1 - NN1;
            I              s1  = NN1 * NN1;
            I              s2  = NN2 * NN2;
            I              sw  = NN2 * 32;
            std::vector<S> A1(s1);
            std::vector<S> A2(s2);
            for(I i = 0; i < NN1; ++i)
            {
                for(I j = 0; j < NN1; ++j)
                {
                    if(i == j)
                    {
                        d               = (i + 1) / S(NN1);
                        A1[i + i * NN1] = d;
                        A2[i + i * NN2] = (i % 2 == 0) ? d : -d;
                    }
                    else
                    {
                        A1[i + j * NN1] = 0;
                        A2[i + j * NN2] = 0;
                    }
                }
            }
            if(NN2 > NN1)
            {
                for(I i = 0; i < NN1; ++i)
                {
                    A2[NN1 + i * NN2] = 0;
                    A2[i + NN1 * NN2] = 0;
                }
                A2[NN1 + NN1 * NN2] = 0;
            }

            // b. find the corresponding tridiagonal matrices containing the setup eigenvalues of each sub-block
            // first find random orthogonal matrices Q1 and Q2
            int info;
            Sh  Q1(s1, 1, s1, 1);
            Sh  Q2(s2, 1, s2, 1);
            rocblas_init<S>(Q1, true);
            rocblas_init<S>(Q2, true);
            std::vector<S> hW(sw);
            std::vector<S> ipiv1(NN1);
            std::vector<S> ipiv2(NN2);
            cpu_geqrf<S>(NN1, NN1, Q1.data(), NN1, ipiv1.data(), hW.data(), sw, &info);
            cpu_geqrf<S>(NN2, NN2, Q2.data(), NN2, ipiv2.data(), hW.data(), sw, &info);
            // now multiply the orthogonal matrices by the diagonals A1 and A2 to hide the eigenvalues
            cpu_ormqr_unmqr<S>(HIPSOLVER_SIDE_LEFT,
                               HIPSOLVER_OP_T,
                               NN1,
                               NN1,
                               NN1,
                               Q1.data(),
                               NN1,
                               ipiv1.data(),
                               A1.data(),
                               NN1,
                               hW.data(),
                               sw,
                               &info);
            cpu_ormqr_unmqr<S>(HIPSOLVER_SIDE_RIGHT,
                               HIPSOLVER_OP_N,
                               NN1,
                               NN1,
                               NN1,
                               Q1.data(),
                               NN1,
                               ipiv1.data(),
                               A1.data(),
                               NN1,
                               hW.data(),
                               sw,
                               &info);
            cpu_ormqr_unmqr<S>(HIPSOLVER_SIDE_LEFT,
                               HIPSOLVER_OP_T,
                               NN2,
                               NN2,
                               NN2,
                               Q2.data(),
                               NN2,
                               ipiv2.data(),
                               A2.data(),
                               NN2,
                               hW.data(),
                               sw,
                               &info);
            cpu_ormqr_unmqr<S>(HIPSOLVER_SIDE_RIGHT,
                               HIPSOLVER_OP_N,
                               NN2,
                               NN2,
                               NN2,
                               Q2.data(),
                               NN2,
                               ipiv2.data(),
                               A2.data(),
                               NN2,
                               hW.data(),
                               sw,
                               &info);
            // finally, perform tridiagonalization
            cpu_sytrd_hetrd<S>(HIPSOLVER_FILL_MODE_UPPER,
                               NN1,
                               A1.data(),
                               NN1,
                               hD[0],
                               hE[0],
                               ipiv1.data(),
                               hW.data(),
                               sw);
            cpu_sytrd_hetrd<S>(HIPSOLVER_FILL_MODE_UPPER,
                               NN2,
                               A2.data(),
                               NN2,
                               hD[0] + NN1,
                               hE[0] + NN1,
                               ipiv2.data(),
                               hW.data(),
                               sw);

            // c. integrate blocks into final matrix
            // integrate the 2 sub-blocks into the first independent block
            hE[0][NN1 - 1] = 1;
            hD[0][NN1 - 1] += 1;
            hD[0][NN1] += 1;
            // copy the independent block over
            for(I i = 0; i < N1; ++i)
            {
                hD[0][N1 + i] = hD[0][i];
                hE[0][N1 + i] = hE[0][i];
            }
            hE[0][N1 - 1]     = 0;
            hE[0][2 * N1 - 1] = 0;
            // integrate the 2 sub-blocks into the second independent block
            // (using negative p to test secular eqn algorithm)
            hE[0][N1 + NN1 - 1] = -1;
            hD[0][N1 + NN1 - 1] -= 2;
            hD[0][N1 + NN1] -= 2;
            // if there is a third independent block, initialize it with 1
            if(E == 1)
                hD[0][n - 1] = 1;
        }

        // initialize C to the identity matrix
        if(compz == HIPSOLVER_EIG_COMP_V)
        {
            for(I j = 0; j < n; j++)
            {
                for(I i = 0; i < n; i++)
                {
                    if(i == j)
                        hC[0][i + j * ldc] = 1;
                    else
                        hC[0][i + j * ldc] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dD.transfer_from(hD));
        CHECK_HIP_ERROR(dE.transfer_from(hE));

        if(compz == HIPSOLVER_EIG_COMP_V)
            CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <testAPI_t API,
          typename T,
          typename I,
          typename SIZE,
          typename Sd,
          typename Td,
          typename Ud,
          typename Sh,
          typename Th,
          typename Uh>
void stedc_getError(const hipsolverHandle_t   handle,
                    const hipsolverDnParams_t params,
                    const hipsolverEigComp_t  compz,
                    const I                   n,
                    Sd&                       dD,
                    Sd&                       dE,
                    Td&                       dC,
                    const I                   ldc,
                    Td&                       dWork,
                    const SIZE                bytes_dW,
                    Th&                       hWork,
                    const SIZE                bytes_hW,
                    Ud&                       dInfo,
                    Sh&                       hD,
                    Sh&                       hDRes,
                    Sh&                       hE,
                    Sh&                       hERes,
                    Th&                       hC,
                    Th&                       hCRes,
                    Uh&                       hInfo,
                    Uh&                       hInfoRes,
                    double*                   max_err,
                    double*                   max_errv)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));

    using HMatT  = HostMatrix<T, int>;
    using HMatS  = HostMatrix<S, int>;
    using BDescT = typename HMatT::BlockDescriptor;
    using BDescS = typename HMatS::BlockDescriptor;

    int    lgn   = floor(log(n - 1) / log(2)) + 1;
    size_t lwork = (COMPLEX) ? n * n : 0;
    size_t lrwork
        = (compz == HIPSOLVER_EIG_COMP_N || n <= 1) ? 1 : 1 + 3 * n + 4 * n * n + 2 * n * lgn;
    size_t         liwork = (compz == HIPSOLVER_EIG_COMP_N || n <= 1) ? 1 : 6 + 6 * n + 5 * n * lgn;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<int> iwork(liwork);

    // input data initialization
    stedc_initData<true, true, T>(handle, compz, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_stedc(API,
                                        handle,
                                        params,
                                        compz,
                                        n,
                                        dD.data(),
                                        dE.data(),
                                        dC.data(),
                                        ldc,
                                        dWork.data(),
                                        bytes_dW,
                                        hWork.data(),
                                        bytes_hW,
                                        dInfo.data()));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(compz != HIPSOLVER_EIG_COMP_N)
        CHECK_HIP_ERROR(hCRes.transfer_from(dC));

    // if eigenvectors were required, prepare matrix A (upper triangular) for implicit tests
    I                            lda    = n;
    size_t                       size_A = lda * n;
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    if(compz != HIPSOLVER_EIG_COMP_N)
    {
        for(I i = 0; i < n; i++)
        {
            for(I j = i; j < n; j++)
            {
                if(i == j)
                    hA[0][i + j * lda] = hD[0][i];
                else if(i + 1 == j)
                    hA[0][i + j * lda] = hE[0][i];
                else
                    hA[0][i + j * lda] = 0;
            }
        }
    }

    // CPU lapack
    cpu_stedc(compz,
              n,
              hD[0],
              hE[0],
              hC[0],
              ldc,
              work.data(),
              lwork,
              rwork.data(),
              lrwork,
              iwork.data(),
              liwork,
              hInfo[0]);

    // Depending on `compz`, stedc can return the eigenvectors of the original
    // matrix (A) or the eigenvectors of the tridiagonal matrix (T).  Thus,
    // given a pair U, D of eigenvectors and eigenvalues computed by stedc the
    // reconstructed matrix
    //
    // U * D * adjoint(U)
    //
    // can be either A or T.  The following code uses lapack to compute
    //
    // AorT = U * D * adjoint(U),
    //
    // which will be later used to compare with the eigenvectors and
    // eigenvalues computed by rocSOLVER.
    //
    auto AorT = HMatT::Empty();
    if((compz != HIPSOLVER_EIG_COMP_N) && (n > 0))
    {
        auto C = HMatT::Wrap(hC[0], ldc, n)->block(BDescT().nrows(n).ncols(n));
        auto d = HMatT::Convert(hD[0], 1, n)->block(BDescT().nrows(1).ncols(n));
        auto D = HMatT::Zeros(n, n).diag(d);
        AorT   = C * D * adjoint(C);
    }

    // check info
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        *max_err = 1;
    else
        *max_err = 0;

    double err;
    *max_errv = 0;

    if((hInfo[0][0] == 0) && (n > 0))
    {
        // check that eigenvalues are correct and in order
        // error is ||hD - hDRes|| / ||hD||
        // using frobenius norm
        err      = norm_error('F', 1, n, 1, hD[0], hDRes[0]);
        *max_err = err > *max_err ? err : *max_err;

        // check eigenvectors if required
        if(compz != HIPSOLVER_EIG_COMP_N)
        {
            // Input matrix
            auto C = HMatT::Wrap(hCRes[0], ldc, n)->block(BDescT().nrows(n).ncols(n));
            // Computed eigenvalues
            auto d = HMatT::Convert(hDRes[0], 1, n)->block(BDescT().nrows(1).ncols(n));
            // Diagonal matrix of size n by n with computed eigenvalues
            auto D = HMatT::Zeros(n, n).diag(d);

            // Orthogonal error
            auto OE   = C * adjoint(C) - HMatT::Eye(n, n);
            err       = OE.max_col_norm();
            *max_errv = err > *max_err ? err : *max_err;

            // Residual error
            auto RE  = AorT - C * D * adjoint(C);
            err      = RE.norm() / AorT.norm();
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <testAPI_t API,
          typename T,
          typename I,
          typename SIZE,
          typename Sd,
          typename Td,
          typename Ud,
          typename Sh,
          typename Th,
          typename Uh>
void stedc_getPerfData(const hipsolverHandle_t   handle,
                       const hipsolverDnParams_t params,
                       const hipsolverEigComp_t  compz,
                       const I                   n,
                       Sd&                       dD,
                       Sd&                       dE,
                       Td&                       dC,
                       const I                   ldc,
                       Td&                       dWork,
                       const SIZE                bytes_dW,
                       Th&                       hWork,
                       const SIZE                bytes_hW,
                       Ud&                       dInfo,
                       Sh&                       hD,
                       Sh&                       hE,
                       Th&                       hC,
                       Uh&                       hInfo,
                       double*                   gpu_time_used,
                       double*                   cpu_time_used,
                       const int                 hot_calls,
                       const bool                perf)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));

    int    lgn   = floor(log(n - 1) / log(2)) + 1;
    size_t lwork = (COMPLEX) ? n * n : 0;
    size_t lrwork
        = (compz == HIPSOLVER_EIG_COMP_N || n <= 1) ? 1 : 1 + 3 * n + 4 * n * n + 2 * n * lgn;
    size_t         liwork = (compz == HIPSOLVER_EIG_COMP_N || n <= 1) ? 1 : 6 + 6 * n + 5 * n * lgn;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<int> iwork(liwork);

    if(!perf)
    {
        stedc_initData<true, false, T>(handle, compz, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_stedc(compz,
                  n,
                  hD[0],
                  hE[0],
                  hC[0],
                  ldc,
                  work.data(),
                  lwork,
                  rwork.data(),
                  lrwork,
                  iwork.data(),
                  liwork,
                  hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    stedc_initData<true, false, T>(handle, compz, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        stedc_initData<false, true, T>(handle, compz, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

        CHECK_ROCBLAS_ERROR(hipsolver_stedc(API,
                                            handle,
                                            params,
                                            compz,
                                            n,
                                            dD.data(),
                                            dE.data(),
                                            dC.data(),
                                            ldc,
                                            dWork.data(),
                                            bytes_dW,
                                            hWork.data(),
                                            bytes_hW,
                                            dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    hipsolver_timer timer;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        stedc_initData<false, true, T>(handle, compz, n, dD, dE, dC, ldc, dInfo, hD, hE, hC, hInfo);

        timer.start(stream);
        hipsolver_stedc(API,
                        handle,
                        params,
                        compz,
                        n,
                        dD.data(),
                        dE.data(),
                        dC.data(),
                        ldc,
                        dWork.data(),
                        bytes_dW,
                        hWork.data(),
                        bytes_hW,
                        dInfo.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T, typename I, typename SIZE>
void testing_stedc(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle handle;
    hipsolver_local_params params;
    char                   compzC = argus.get<char>("jobz");
    I                      n      = argus.get<int>("n");
    I                      ldc    = argus.get<int>("ldc", n);

    hipsolverEigComp_t compz     = char2hipsolver_evect_comp(compzC);
    int                hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_D  = n;
    size_t size_E  = n;
    size_t size_C  = ldc * n;
    double max_err = 0, max_errv = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_ERes = (argus.unit_check || argus.norm_check) ? size_E : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || (compz != HIPSOLVER_EIG_COMP_N && ldc < n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                              handle,
                                              params,
                                              compz,
                                              n,
                                              (S*)nullptr,
                                              (S*)nullptr,
                                              (T*)nullptr,
                                              ldc,
                                              nullptr,
                                              (SIZE)0,
                                              nullptr,
                                              (SIZE)0,
                                              (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    SIZE bytes_dW, bytes_hW;
    hipsolver_stedc_bufferSize(API,
                               handle,
                               params,
                               compz,
                               n,
                               (S*)nullptr,
                               (S*)nullptr,
                               (T*)nullptr,
                               ldc,
                               &bytes_dW,
                               &bytes_hW);
    SIZE size_dW = (bytes_dW + sizeof(T) - 1) / sizeof(T);
    SIZE size_hW = (bytes_hW + sizeof(T) - 1) / sizeof(T);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, bytes_dW);
        return;
    }

    // memory allocations
    host_strided_batch_vector<S>     hD(size_D, 1, size_D, 1);
    host_strided_batch_vector<S>     hDRes(size_DRes, 1, size_DRes, 1);
    host_strided_batch_vector<S>     hE(size_E, 1, size_E, 1);
    host_strided_batch_vector<S>     hERes(size_ERes, 1, size_ERes, 1);
    host_strided_batch_vector<T>     hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T>     hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<int>   hInfo(1, 1, 1, 1);
    host_strided_batch_vector<int>   hInfoRes(1, 1, 1, 1);
    host_strided_batch_vector<T>     hWork(size_hW, 1, size_hW, 1);
    device_strided_batch_vector<S>   dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S>   dE(size_E, 1, size_E, 1);
    device_strided_batch_vector<T>   dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    device_strided_batch_vector<T>   dWork(size_dW, 1, size_dW, 1);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_dW)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_stedc(API,
                                              handle,
                                              params,
                                              compz,
                                              n,
                                              dD.data(),
                                              dE.data(),
                                              dC.data(),
                                              ldc,
                                              dWork.data(),
                                              bytes_dW,
                                              hWork.data(),
                                              bytes_hW,
                                              dInfo.data()),
                              HIPSOLVER_STATUS_SUCCESS);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        stedc_getError<API, T>(handle,
                               params,
                               compz,
                               n,
                               dD,
                               dE,
                               dC,
                               ldc,
                               dWork,
                               bytes_dW,
                               hWork,
                               bytes_hW,
                               dInfo,
                               hD,
                               hDRes,
                               hE,
                               hERes,
                               hC,
                               hCRes,
                               hInfo,
                               hInfoRes,
                               &max_err,
                               &max_errv);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        stedc_getPerfData<API, T>(handle,
                                  params,
                                  compz,
                                  n,
                                  dD,
                                  dE,
                                  dC,
                                  ldc,
                                  dWork,
                                  bytes_dW,
                                  hWork,
                                  bytes_hW,
                                  dInfo,
                                  hD,
                                  hE,
                                  hC,
                                  hInfo,
                                  &gpu_time_used,
                                  &cpu_time_used,
                                  hot_calls,
                                  argus.perf);

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_err, n);
        if(compz != HIPSOLVER_EIG_COMP_N)
            ROCSOLVER_TEST_CHECK(T, max_errv, n);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("jobz", "n", "ldc");
            rocsolver_bench_output(compzC, n, ldc);

            std::cerr << "\n============================================\n";
            std::cerr << "Results:\n";
            std::cerr << "============================================\n";
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, std::max(max_err, max_errv));
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            std::cerr << std::endl;
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, std::max(max_err, max_errv));
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}
