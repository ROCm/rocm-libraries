/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename U>
void sygvdx_hegvdx_inplace_checkBadArgs(const rocblas_handle handle,
                                        const rocblas_eform itype,
                                        const rocblas_evect evect,
                                        const rocblas_erange erange,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        T dA,
                                        const rocblas_int lda,
                                        const rocblas_stride stA,
                                        T dB,
                                        const rocblas_int ldb,
                                        const rocblas_stride stB,
                                        const S vl,
                                        const S vu,
                                        const rocblas_int il,
                                        const rocblas_int iu,
                                        const S abstol,
                                        rocblas_int* hNev,
                                        U dW,
                                        const rocblas_stride stW,
                                        rocblas_int* dInfo,
                                        const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(STRIDED, nullptr, itype, evect, erange,
                                                          uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu,
                                                          il, iu, abstol, hNev, dW, stW, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                              STRIDED, handle, rocblas_eform(0), evect, erange, uplo, n, dA, lda,
                              stA, dB, ldb, stB, vl, vu, il, iu, abstol, hNev, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(STRIDED, handle, itype,
                                                          rocblas_evect_tridiagonal, erange, uplo,
                                                          n, dA, lda, stA, dB, ldb, stB, vl, vu, il,
                                                          iu, abstol, hNev, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                              STRIDED, handle, itype, evect, rocblas_erange(0), uplo, n, dA, lda,
                              stA, dB, ldb, stB, vl, vu, il, iu, abstol, hNev, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                              STRIDED, handle, itype, evect, erange, rocblas_fill_full, n, dA, lda,
                              stA, dB, ldb, stB, vl, vu, il, iu, abstol, hNev, dW, stW, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                                  STRIDED, handle, itype, evect, erange, uplo, n, dA, lda, stA, dB,
                                  ldb, stB, vl, vu, il, iu, abstol, hNev, dW, stW, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                              STRIDED, handle, itype, evect, erange, uplo, n, (T) nullptr, lda, stA,
                              dB, ldb, stB, vl, vu, il, iu, abstol, hNev, dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(STRIDED, handle, itype, evect, erange,
                                                          uplo, n, dA, lda, stA, (T) nullptr, ldb,
                                                          stB, vl, vu, il, iu, abstol, hNev, dW,
                                                          stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(STRIDED, handle, itype, evect, erange,
                                                          uplo, n, dA, lda, stA, dB, ldb, stB, vl,
                                                          vu, il, iu, abstol, (rocblas_int*)nullptr,
                                                          dW, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                              STRIDED, handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb,
                              stB, vl, vu, il, iu, abstol, hNev, (U) nullptr, stW, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                              STRIDED, handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb,
                              stB, vl, vu, il, iu, abstol, hNev, dW, stW, (rocblas_int*)nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(STRIDED, handle, itype, evect, erange,
                                                          uplo, 0, (T) nullptr, lda, stA,
                                                          (T) nullptr, ldb, stB, vl, vu, il, iu,
                                                          abstol, hNev, (U) nullptr, stW, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                                  STRIDED, handle, itype, evect, erange, uplo, n, dA, lda, stA, dB,
                                  ldb, stB, vl, vu, il, iu, abstol, (rocblas_int*)nullptr, dW, stW,
                                  (rocblas_int*)nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygvdx_hegvdx_inplace_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_stride stA = 1;
    rocblas_stride stB = 1;
    rocblas_stride stW = 1;
    rocblas_int bc = 1;
    rocblas_eform itype = rocblas_eform_ax;
    rocblas_evect evect = rocblas_evect_original;
    rocblas_erange erange = rocblas_erange_value;
    rocblas_fill uplo = rocblas_fill_upper;

    S vl = 0.0;
    S vu = 1.0;
    rocblas_int il = 0;
    rocblas_int iu = 0;
    S abstol = 0;

    if(BATCHED)
    {
        // memory allocations
        host_strided_batch_vector<rocblas_int> hNev(1, 1, 1, 1);
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygvdx_hegvdx_inplace_checkBadArgs<STRIDED>(
            handle, itype, evect, erange, uplo, n, dA.data(), lda, stA, dB.data(), ldb, stB, vl, vu,
            il, iu, abstol, hNev.data(), dW.data(), stW, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<rocblas_int> hNev(1, 1, 1, 1);
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygvdx_hegvdx_inplace_checkBadArgs<STRIDED>(
            handle, itype, evect, erange, uplo, n, dA.data(), lda, stA, dB.data(), ldb, stB, vl, vu,
            il, iu, abstol, hNev.data(), dW.data(), stW, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sygvdx_hegvdx_inplace_initData(const rocblas_handle handle,
                                    const rocblas_eform itype,
                                    const rocblas_evect evect,
                                    const rocblas_int n,
                                    Td& dA,
                                    const rocblas_int lda,
                                    const rocblas_stride stA,
                                    Td& dB,
                                    const rocblas_int ldb,
                                    const rocblas_stride stB,
                                    const rocblas_int bc,
                                    Th& hA,
                                    Th& hB,
                                    host_strided_batch_vector<T>& A,
                                    host_strided_batch_vector<T>& B,
                                    const bool test,
                                    const bool singular)
{
    if(CPU)
    {
        rocblas_int info;
        rocblas_int ldu = n;
        host_strided_batch_vector<T> U(n * n, 1, n * n, bc);
        rocblas_init<T>(hA, true);
        rocblas_init<T>(U, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // for testing purposes, we start with a reduced matrix M for the standard equivalent problem
            // with spectrum in a desired range (-20, 20). Then we construct the generalized pair
            // (A, B) from there.
            for(rocblas_int i = 0; i < n; i++)
            {
                // scale matrices and set hA = M (symmetric/hermitian), hB = U (upper triangular)
                for(rocblas_int j = i; j < n; j++)
                {
                    if(i == j)
                    {
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 10;
                        U[b][i + j * ldu] = std::real(U[b][i + j * ldu]) / 100 + 1;
                        hB[b][i + j * ldb] = U[b][i + j * ldu];
                    }
                    else
                    {
                        if(j == i + 1)
                        {
                            hA[b][i + j * lda] = (hA[b][i + j * lda] - 5) / 10;
                            hA[b][j + i * lda] = sconj(hA[b][i + j * lda]);
                        }
                        else
                            hA[b][j + i * lda] = hA[b][i + j * lda] = 0;

                        U[b][i + j * ldu] = (U[b][i + j * ldu] - 5) / 100;
                        hB[b][i + j * ldb] = U[b][i + j * ldu];
                        hB[b][j + i * ldb] = 0;
                        U[b][j + i * ldu] = 0;
                    }
                }
                if(i == n / 4 || i == n / 2 || i == n - 1 || i == n / 7 || i == n / 5 || i == n / 3)
                    hA[b][i + i * lda] *= -1;
            }

            // form B = U' U
            T one = T(1);
            cpu_trmm(rocblas_side_left, rocblas_fill_upper, rocblas_operation_conjugate_transpose,
                     rocblas_diagonal_non_unit, n, n, one, U[b], ldu, hB[b], ldb);

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // make some matrices B not positive definite
                // always the same elements for debugging purposes
                // the algorithm must detect the lower order of the principal minors <= 0
                // in those matrices in the batch that are non positive definite
                rocblas_int i = n / 4 + b;
                i -= (i / n) * n;
                hB[b][i + i * ldb] = 0;
                i = n / 2 + b;
                i -= (i / n) * n;
                hB[b][i + i * ldb] = 0;
                i = n - 1 + b;
                i -= (i / n) * n;
                hB[b][i + i * ldb] = 0;
            }

            if(itype == rocblas_eform_ax)
            {
                // form A = U' M U
                cpu_trmm(rocblas_side_left, rocblas_fill_upper, rocblas_operation_conjugate_transpose,
                         rocblas_diagonal_non_unit, n, n, one, U[b], ldu, hA[b], lda);
                cpu_trmm(rocblas_side_right, rocblas_fill_upper, rocblas_operation_none,
                         rocblas_diagonal_non_unit, n, n, one, U[b], ldu, hA[b], lda);
            }
            else
            {
                // form A = inv(U) M inv(U')
                cpu_trsm(rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                         rocblas_diagonal_non_unit, n, n, one, U[b], ldu, hA[b], lda);
                cpu_trsm(rocblas_side_right, rocblas_fill_upper,
                         rocblas_operation_conjugate_transpose, rocblas_diagonal_non_unit, n, n,
                         one, U[b], ldu, hA[b], lda);
            }

            // store A and B for testing purposes
            if(test && evect != rocblas_evect_none)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        if(itype != rocblas_eform_bax)
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

template <bool STRIDED, typename T, typename S, typename Td, typename Ud, typename Vd, typename Th, typename Uh, typename Vh>
void sygvdx_hegvdx_inplace_getError(const rocblas_handle handle,
                                    const rocblas_eform itype,
                                    const rocblas_evect evect,
                                    const rocblas_erange erange,
                                    const rocblas_fill uplo,
                                    const rocblas_int n,
                                    Td& dA,
                                    const rocblas_int lda,
                                    const rocblas_stride stA,
                                    Td& dB,
                                    const rocblas_int ldb,
                                    const rocblas_stride stB,
                                    const S vl,
                                    const S vu,
                                    const rocblas_int il,
                                    const rocblas_int iu,
                                    const S abstol,
                                    Vh& hNevRes,
                                    Ud& dW,
                                    const rocblas_stride stW,
                                    Vd& dInfo,
                                    const rocblas_int bc,
                                    Th& hA,
                                    Th& hARes,
                                    Th& hB,
                                    Vh& hNev,
                                    Uh& hW,
                                    Uh& hWRes,
                                    Vh& hInfo,
                                    Vh& hInfoRes,
                                    double* max_err,
                                    const bool singular)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    int lwork = (COMPLEX ? 2 * n : 8 * n);
    int lrwork = (COMPLEX ? 7 * n : 0);
    int liwork = 5 * n;
    int ldz = n;

    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<int> iwork(liwork);
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);
    std::vector<T> Z(ldz * n);
    std::vector<int> ifail(n);

    // input data initialization
    sygvdx_hegvdx_inplace_initData<true, true, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb,
                                                  stB, bc, hA, hB, A, B, true, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sygvdx_hegvdx_inplace(
        STRIDED, handle, itype, evect, erange, uplo, n, dA.data(), lda, stA, dB.data(), ldb, stB,
        vl, vu, il, iu, abstol, hNevRes.data(), dW.data(), stW, dInfo.data(), bc));

    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect != rocblas_evect_none)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
    S atol = (abstol == 0) ? 2 * get_safemin<S>() : abstol;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cpu_sygvx_hegvx(itype, evect, erange, uplo, n, hA[b], lda, hB[b], ldb, vl, vu, il, iu, atol,
                        hNev[b], hW[b], Z.data(), ldz, work.data(), lwork, rwork.data(),
                        iwork.data(), ifail.data(), hInfo[b]);
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved.
    // We do test with indefinite matrices B).

    // check info for non-convergence and/or positive-definiteness
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            *max_err += 1;
    }

    // Check number of returned eigenvalues
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hNev[b][0], hNevRes[b][0]) << "where b = " << b;
        if(hNev[b][0] != hNevRes[b][0])
            *max_err += 1;
    }

    double err;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(evect == rocblas_evect_none)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hW - hWRes|| / ||hW||
            // using frobenius norm
            if(hInfo[b][0] == 0)
            {
                err = norm_error('F', 1, hNev[b][0], 1, hW[b], hWRes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hInfo[b][0] == 0)
            {
                T alpha = 1;
                T beta = 0;

                // hARes contains eigenvectors x
                // compute B*x (or A*x) and store in hB
                cpu_symm_hemm(rocblas_side_left, uplo, n, hNev[b][0], alpha, B[b], ldb, hARes[b],
                              lda, beta, hB[b], ldb);

                if(itype == rocblas_eform_ax)
                {
                    // problem is A*x = (lambda)*B*x

                    // compute (1/lambda)*A*x and store in hA
                    for(int j = 0; j < hNev[b][0]; j++)
                    {
                        alpha = T(1) / hWRes[b][j];
                        cpu_symv_hemv(uplo, n, alpha, A[b], lda, hARes[b] + j * lda, 1, beta,
                                      hA[b] + j * lda, 1);
                    }

                    // move B*x into hARes
                    for(rocblas_int i = 0; i < n; i++)
                        for(rocblas_int j = 0; j < hNev[b][0]; j++)
                            hARes[b][i + j * lda] = hB[b][i + j * ldb];
                }
                else
                {
                    // problem is A*B*x = (lambda)*x or B*A*x = (lambda)*x

                    // compute (1/lambda)*A*B*x or (1/lambda)*B*A*x and store in hA
                    for(int j = 0; j < hNev[b][0]; j++)
                    {
                        alpha = T(1) / hWRes[b][j];
                        cpu_symv_hemv(uplo, n, alpha, A[b], lda, hB[b] + j * ldb, 1, beta,
                                      hA[b] + j * lda, 1);
                    }
                }

                // error is ||hA - hARes|| / ||hA||
                // using frobenius norm
                err = norm_error('F', n, hNev[b][0], lda, hA[b], hARes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <bool STRIDED, typename T, typename S, typename Td, typename Ud, typename Vd, typename Th, typename Uh, typename Vh>
void sygvdx_hegvdx_inplace_getPerfData(const rocblas_handle handle,
                                       const rocblas_eform itype,
                                       const rocblas_evect evect,
                                       const rocblas_erange erange,
                                       const rocblas_fill uplo,
                                       const rocblas_int n,
                                       Td& dA,
                                       const rocblas_int lda,
                                       const rocblas_stride stA,
                                       Td& dB,
                                       const rocblas_int ldb,
                                       const rocblas_stride stB,
                                       const S vl,
                                       const S vu,
                                       const rocblas_int il,
                                       const rocblas_int iu,
                                       const S abstol,
                                       Vh& hNevRes,
                                       Ud& dW,
                                       const rocblas_stride stW,
                                       Vd& dInfo,
                                       const rocblas_int bc,
                                       Th& hA,
                                       Th& hB,
                                       Vh& hNev,
                                       Uh& hW,
                                       Vh& hInfo,
                                       double* gpu_time_used,
                                       double* cpu_time_used,
                                       const rocblas_int hot_calls,
                                       const int profile,
                                       const bool profile_kernels,
                                       const bool perf,
                                       const bool singular)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    int lwork = (COMPLEX ? 2 * n : 8 * n);
    int lrwork = (COMPLEX ? 7 * n : 0);
    int liwork = 5 * n;
    int ldz = n;

    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<int> iwork(liwork);
    host_strided_batch_vector<T> A(1, 1, 1, 1);
    host_strided_batch_vector<T> B(1, 1, 1, 1);
    std::vector<T> Z(ldz * n);
    std::vector<int> ifail(n);

    // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
    S atol = (abstol == 0) ? 2 * get_safemin<S>() : abstol;

    if(!perf)
    {
        sygvdx_hegvdx_inplace_initData<true, false, T>(handle, itype, evect, n, dA, lda, stA, dB,
                                                       ldb, stB, bc, hA, hB, A, B, false, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cpu_sygvx_hegvx(itype, evect, erange, uplo, n, hA[b], lda, hB[b], ldb, vl, vu, il, iu,
                            atol, hNev[b], hW[b], Z.data(), ldz, work.data(), lwork, rwork.data(),
                            iwork.data(), ifail.data(), hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sygvdx_hegvdx_inplace_initData<true, false, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb,
                                                   stB, bc, hA, hB, A, B, false, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygvdx_hegvdx_inplace_initData<false, true, T>(handle, itype, evect, n, dA, lda, stA, dB,
                                                       ldb, stB, bc, hA, hB, A, B, false, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_sygvdx_hegvdx_inplace(
            STRIDED, handle, itype, evect, erange, uplo, n, dA.data(), lda, stA, dB.data(), ldb,
            stB, vl, vu, il, iu, abstol, hNevRes.data(), dW.data(), stW, dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        sygvdx_hegvdx_inplace_initData<false, true, T>(handle, itype, evect, n, dA, lda, stA, dB,
                                                       ldb, stB, bc, hA, hB, A, B, false, singular);

        start = get_time_us_sync(stream);
        rocsolver_sygvdx_hegvdx_inplace(STRIDED, handle, itype, evect, erange, uplo, n, dA.data(),
                                        lda, stA, dB.data(), ldb, stB, vl, vu, il, iu, abstol,
                                        hNevRes.data(), dW.data(), stW, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygvdx_hegvdx_inplace(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char itypeC = argus.get<char>("itype");
    char evectC = argus.get<char>("evect");
    char erangeC = argus.get<char>("erange");
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideW", n);

    S vl = S(argus.get<double>("vl", 0));
    S vu = S(argus.get<double>("vu", erangeC == 'V' ? 1 : 0));
    rocblas_int il = argus.get<rocblas_int>("il", erangeC == 'I' ? 1 : 0);
    rocblas_int iu = argus.get<rocblas_int>("iu", erangeC == 'I' ? 1 : 0);
    S abstol = S(argus.get<double>("abstol", 0));

    rocblas_eform itype = char2rocblas_eform(itypeC);
    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_erange erange = char2rocblas_erange(erangeC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stWRes = (argus.unit_check || argus.norm_check) ? stW : 0;

    // check non-supported values
    if(uplo == rocblas_fill_full || evect == rocblas_evect_tridiagonal)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                                      STRIDED, handle, itype, evect, erange, uplo, n,
                                      (T* const*)nullptr, lda, stA, (T* const*)nullptr, ldb, stB,
                                      vl, vu, il, iu, abstol, (rocblas_int*)nullptr, (S*)nullptr,
                                      stW, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx_inplace(STRIDED, handle, itype, evect, erange, uplo, n,
                                                (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB, vl,
                                                vu, il, iu, abstol, (rocblas_int*)nullptr,
                                                (S*)nullptr, stW, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * n;
    size_t size_W = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_WRes = (argus.unit_check || argus.norm_check) ? size_W : 0;

    // check invalid sizes
    bool invalid_size
        = (n < 0 || lda < n || ldb < n || bc < 0 || (erange == rocblas_erange_value && vl >= vu)
           || (erange == rocblas_erange_index && (il < 1 || iu < 0))
           || (erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu))));
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                                      STRIDED, handle, itype, evect, erange, uplo, n,
                                      (T* const*)nullptr, lda, stA, (T* const*)nullptr, ldb, stB,
                                      vl, vu, il, iu, abstol, (rocblas_int*)nullptr, (S*)nullptr,
                                      stW, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx_inplace(STRIDED, handle, itype, evect, erange, uplo, n,
                                                (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB, vl,
                                                vu, il, iu, abstol, (rocblas_int*)nullptr,
                                                (S*)nullptr, stW, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_sygvdx_hegvdx_inplace(
                STRIDED, handle, itype, evect, erange, uplo, n, (T* const*)nullptr, lda, stA,
                (T* const*)nullptr, ldb, stB, vl, vu, il, iu, abstol, (rocblas_int*)nullptr,
                (S*)nullptr, stW, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sygvdx_hegvdx_inplace(
                STRIDED, handle, itype, evect, erange, uplo, n, (T*)nullptr, lda, stA, (T*)nullptr,
                ldb, stB, vl, vu, il, iu, abstol, (rocblas_int*)nullptr, (S*)nullptr, stW,
                (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<rocblas_int> hNev(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hNevRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hW(size_W, 1, stW, bc);
    host_strided_batch_vector<S> hWRes(size_WRes, 1, stWRes, bc);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    // device
    device_strided_batch_vector<S> dW(size_W, 1, stW, bc);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                                      STRIDED, handle, itype, evect, erange, uplo, n, dA.data(),
                                      lda, stA, dB.data(), ldb, stB, vl, vu, il, iu, abstol,
                                      hNevRes.data(), dW.data(), stW, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygvdx_hegvdx_inplace_getError<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                abstol, hNevRes, dW, stW, dInfo, bc, hA, hARes, hB, hNev, hW, hWRes, hInfo,
                hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sygvdx_hegvdx_inplace_getPerfData<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                abstol, hNevRes, dW, stW, dInfo, bc, hA, hB, hNev, hW, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx_inplace(
                                      STRIDED, handle, itype, evect, erange, uplo, n, dA.data(),
                                      lda, stA, dB.data(), ldb, stB, vl, vu, il, iu, abstol,
                                      hNevRes.data(), dW.data(), stW, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygvdx_hegvdx_inplace_getError<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                abstol, hNevRes, dW, stW, dInfo, bc, hA, hARes, hB, hNev, hW, hWRes, hInfo,
                hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            sygvdx_hegvdx_inplace_getPerfData<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                abstol, hNevRes, dW, stW, dInfo, bc, hA, hB, hNev, hW, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
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
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("itype", "evect", "erange", "uplo", "n", "lda", "ldb", "vl",
                                       "vu", "il", "iu", "abstol", "strideW", "batch_c");
                rocsolver_bench_output(itypeC, evectC, erangeC, uploC, n, lda, ldb, vl, vu, il, iu,
                                       abstol, stW, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype", "evect", "erange", "uplo", "n", "lda", "ldb",
                                       "strideA", "strideB", "vl", "vu", "il", "iu", "abstol",
                                       "strideW", "batch_c");
                rocsolver_bench_output(itypeC, evectC, erangeC, uploC, n, lda, ldb, stA, stB, vl,
                                       vu, il, iu, abstol, stW, bc);
            }
            else
            {
                rocsolver_bench_output("itype", "evect", "erange", "uplo", "n", "lda", "ldb", "vl",
                                       "vu", "il", "iu", "abstol");
                rocsolver_bench_output(itypeC, evectC, erangeC, uploC, n, lda, ldb, vl, vu, il, iu,
                                       abstol);
            }
            rocsolver_bench_header("Results:");
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
            rocsolver_bench_endl();
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
