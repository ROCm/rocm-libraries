/* **************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/matrix_utils/matrix_utils.hpp"
#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename U>
void syevd_heevd_checkBadArgs(const rocblas_handle handle,
                              const rocblas_evect evect,
                              const rocblas_fill uplo,
                              const rocblas_int n,
                              T dA,
                              const rocblas_int lda,
                              const rocblas_stride stA,
                              S dD,
                              const rocblas_stride stD,
                              S dE,
                              const rocblas_stride stE,
                              U dinfo,
                              const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, nullptr, evect, uplo, n, dA, lda, stA, dD,
                                                stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, rocblas_evect(0), uplo, n, dA, lda,
                                                stA, dD, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, rocblas_fill_full, n, dA,
                                                lda, stA, dD, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA, lda, stA,
                                                    dD, stD, dE, stE, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, (T) nullptr, lda,
                                                stA, dD, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA, lda, stA,
                                                (S) nullptr, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA, lda, stA, dD,
                                                stD, (S) nullptr, stE, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA, lda, stA, dD,
                                                stD, dE, stE, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, 0, (T) nullptr, lda,
                                                stA, (S) nullptr, stD, (S) nullptr, stE, dinfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA, lda, stA,
                                                    dD, stD, dE, stE, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syevd_heevd_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect evect = rocblas_evect_none;
    rocblas_fill uplo = rocblas_fill_lower;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stD = 1;
    rocblas_stride stE = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        syevd_heevd_checkBadArgs<STRIDED>(handle, evect, uplo, n, dA.data(), lda, stA, dD.data(),
                                          stD, dE.data(), stE, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        syevd_heevd_checkBadArgs<STRIDED>(handle, evect, uplo, n, dA.data(), lda, stA, dD.data(),
                                          stD, dE.data(), stE, dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevd_heevd_default_initData(const rocblas_handle handle,
                                  const rocblas_evect evect,
                                  const rocblas_int n,
                                  Td& dA,
                                  const rocblas_int lda,
                                  const rocblas_int bc,
                                  Th& hA,
                                  std::vector<T>& A,
                                  bool test = true)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = i; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 400;
                    else
                    {
                        hA[b][i + j * lda] -= 4;
                        hA[b][j + i * lda] = sconj(hA[b][i + j * lda]);
                    }
                }
            }

            // make copy of original data to test vectors if required
            if(test && evect == rocblas_evect_original)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
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

// Creates an `n` by `n` matrix `A` with the following eigenvalues:
//
// spectrum(A) = { l_i = ulp * i, for 1 <= i <= n - 1; l_n = 1 }
//
// where `ulp` is the smallest floating point number such that `1 + ulp > 1`.
//
template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevd_heevd_eig7_initData(const rocblas_handle handle,
                               const rocblas_evect evect,
                               const rocblas_int n,
                               Td& dA,
                               const rocblas_int lda,
                               const rocblas_int bc,
                               Th& hA,
                               std::vector<T>& A,
                               bool test = true)
{
    using S = decltype(std::real(T{}));

    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMat = HostMatrix<T, rocblas_int>;
            using BDesc = typename HMat::BlockDescriptor;

            auto hAw = HMat::Wrap(hA[b], lda, n);
            if(hAw) // update matrix hA if n >= 1
            {
                auto eigs = std::numeric_limits<S>::epsilon() * HMat::FromRange(1, n - 1, n - 1);
                eigs = cat(eigs, HMat::Ones(1, 1));
                auto [Q, _] = qr((*hAw).block(BDesc().nrows(n).ncols(n)));
                hAw->set_to_zero();

                hAw->copy_data_from(Q * HMat::Zeros(n).diag(eigs) * adjoint(Q));
            }

            // make copy of original data to test vectors if required
            if(test && evect == rocblas_evect_original)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
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

// Creates an `n` by `n` tridiagonal, Wilkinson matrix, which is formed as follows:
//
// 1. If `n` is even:
//                      (   1          1            1          1   )
// W_{2m + 1} = tridiag ( m   (m - 1) ... 0.5 0.5 ... (m - 1)    m )
//                      (   1          1            1          1   )
//
// 2. If `n` is odd:
//                      (   1          1         1          1   )
// W_{2m + 1} = tridiag ( m   (m - 1) ... 1 0 1 ... (m - 1)   m )
//                      (   1          1         1          1   )
//
// where `n = 2m + 1`.
//
template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevd_heevd_wilkinson_initData(const rocblas_handle handle,
                                    const rocblas_evect evect,
                                    const rocblas_int n,
                                    Td& dA,
                                    const rocblas_int lda,
                                    const rocblas_int bc,
                                    Th& hA,
                                    std::vector<T>& A,
                                    bool test = true)
{
    using S = decltype(std::real(T{}));

    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMat = HostMatrix<T, rocblas_int>;
            using BDesc = typename HMat::BlockDescriptor;

            auto hAw = HMat::Wrap(hA[b], lda, n);
            if(hAw) // update matrix hA if n >= 1
            {
                S m = (n - 1) / S(2);
                auto A = HMat::Zeros(n, n);
                auto E = HMat::Ones(n - 1, 1);
                auto D = HMat::Zeros(n, 1);

                for(rocblas_int i = 0; i < n / 2; ++i)
                {
                    D[i] = m - i;
                    D[n - 1 - i] = m - i;
                }

                A.diag(D);
                A.sup_diag(E);
                A.sub_diag(E);

                hAw->set_to_zero();
                hAw->copy_data_from(A);
            }

            // make copy of original data to test vectors if required
            if(test && evect == rocblas_evect_original)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
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

// Creates an `n` by `n` tridiagonal, Toeplitz matrix T of the following form:
//
//             (   1         1         1    )
// T = tridiag ( 2   2 ... 2   2 ... 2    2 )
//             (   1         1         1    )
//
template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevd_heevd_toeplitz_initData(const rocblas_handle handle,
                                   const rocblas_evect evect,
                                   const rocblas_int n,
                                   Td& dA,
                                   const rocblas_int lda,
                                   const rocblas_int bc,
                                   Th& hA,
                                   std::vector<T>& A,
                                   bool test = true)
{
    using S = decltype(std::real(T{}));

    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMat = HostMatrix<T, rocblas_int>;
            using BDesc = typename HMat::BlockDescriptor;

            auto hAw = HMat::Wrap(hA[b], lda, n);
            if(hAw) // update matrix hA if n >= 1
            {
                auto A = HMat::Zeros(n, n);
                auto E = HMat::Ones(n - 1, 1);
                auto D = 2 * HMat::Ones(n, 1);

                A.diag(D);
                A.sup_diag(E);
                A.sub_diag(E);

                hAw->set_to_zero();
                hAw->copy_data_from(A);
            }

            // make copy of original data to test vectors if required
            if(test && evect == rocblas_evect_original)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
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

// For `n > 1`, creates a symmetrized `n` by `n` tridiagonal, Clement matrix T of the following form:
//
//             (   sqrt(n - 1)   sqrt(2(n - 2))     sqrt((n - 2)2)   sqrt(n - 1)   )
// T = tridiag ( 0             0                ...                0             0 )
//             (   sqrt(n - 1)   sqrt(2(n - 2))     sqrt((n - 2)2)   sqrt(n - 1)   )
//
// were the `i-th` off-diagonal entry is sqrt(i(n - i)), 1 <= i < n.
//
template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevd_heevd_clement_initData(const rocblas_handle handle,
                                  const rocblas_evect evect,
                                  const rocblas_int n,
                                  Td& dA,
                                  const rocblas_int lda,
                                  const rocblas_int bc,
                                  Th& hA,
                                  std::vector<T>& A,
                                  bool test = true)
{
    using S = decltype(std::real(T{}));

    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            // New matrix initialization
            using HMat = HostMatrix<T, rocblas_int>;
            using BDesc = typename HMat::BlockDescriptor;

            auto hAw = HMat::Wrap(hA[b], lda, n);
            if(hAw) // update matrix hA if n >= 1
            {
                auto A = HMat::Zeros(n, n);
                auto E = HMat::Ones(n - 1, 1);
                auto D = HMat::Zeros(n, 1);

                for(rocblas_int i = 1; i < n; ++i)
                {
                    E[i - 1] = std::sqrt(i * (n - i));
                }

                A.diag(D);
                A.sup_diag(E);
                A.sub_diag(E);

                hAw->set_to_zero();
                hAw->copy_data_from(A);
            }

            // make copy of original data to test vectors if required
            if(test && evect == rocblas_evect_original)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
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

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syevd_heevd_initData(const rocblas_handle handle,
                          const rocblas_evect evect,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_int bc,
                          Th& hA,
                          std::vector<T>& A,
                          bool test = true)
{
    if((std::getenv("TEST_EIG7") != nullptr) || (std::getenv("SYEVD_TEST_EIG7") != nullptr))
    {
        syevd_heevd_eig7_initData<CPU, GPU>(handle, evect, n, dA, lda, bc, hA, A, test);
    }
    else if((std::getenv("TEST_WILKINSON") != nullptr)
            || (std::getenv("SYEVD_TEST_WILKINSON") != nullptr))
    {
        syevd_heevd_wilkinson_initData<CPU, GPU>(handle, evect, n, dA, lda, bc, hA, A, test);
    }
    else if((std::getenv("TEST_CLEMENT") != nullptr)
            || (std::getenv("SYEVD_TEST_CLEMENT") != nullptr))
    {
        syevd_heevd_clement_initData<CPU, GPU>(handle, evect, n, dA, lda, bc, hA, A, test);
    }
    else if((std::getenv("TEST_TOEPLITZ") != nullptr)
            || (std::getenv("SYEVD_TEST_TOEPLITZ") != nullptr))
    {
        syevd_heevd_toeplitz_initData<CPU, GPU>(handle, evect, n, dA, lda, bc, hA, A, test);
    }
    else
    {
        syevd_heevd_default_initData<CPU, GPU>(handle, evect, n, dA, lda, bc, hA, A, test);
    }

    return;
}

template <bool STRIDED, typename T, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void syevd_heevd_getError(const rocblas_handle handle,
                          const rocblas_evect evect,
                          const rocblas_fill uplo,
                          const rocblas_int n,
                          Td& dA,
                          const rocblas_int lda,
                          const rocblas_stride stA,
                          Sd& dD,
                          const rocblas_stride stD,
                          Sd& dE,
                          const rocblas_stride stE,
                          Id& dinfo,
                          const rocblas_int bc,
                          Th& hA,
                          Th& hAres,
                          Sh& hD,
                          Sh& hDres,
                          Ih& hinfo,
                          Ih& hinfoRes,
                          double* max_err,
                          double* max_errv)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    using S = decltype(std::real(T{}));

    using HMat = HostMatrix<T, rocblas_int>;
    using BDesc = typename HMat::BlockDescriptor;

    int lgn = floor(log(n - 1) / log(2)) + 1;
    int sizeE, lwork;
    if(!COMPLEX)
    {
        sizeE = (evect == rocblas_evect_none ? 2 * n + 1 : 1 + 6 * n + 2 * n * n);
        lwork = 0;
    }
    else
    {
        sizeE = (evect == rocblas_evect_none ? n : 1 + 5 * n + 2 * n * n);
        lwork = (evect == rocblas_evect_none ? n + 1 : 2 * n + n * n);
    }
    int liwork = (evect == rocblas_evect_none ? 1 : 3 + 5 * n);

    std::vector<T> work(lwork);
    std::vector<S> hE(sizeE);
    std::vector<int> iwork(liwork);
    std::vector<T> A(lda * n * bc);

    // input data initialization
    syevd_heevd_initData<true, true, T>(handle, evect, n, dA, lda, bc, hA, A);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA.data(), lda, stA,
                                              dD.data(), stD, dE.data(), stE, dinfo.data(), bc));

    CHECK_HIP_ERROR(hDres.transfer_from(dD));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
    if(evect == rocblas_evect_original)
        CHECK_HIP_ERROR(hAres.transfer_from(dA));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
        cpu_syevd_heevd(evect, uplo, n, hA[b], lda, hD[b], work.data(), lwork, hE.data(), sizeE,
                        iwork.data(), liwork, hinfo[b]);

    // Check info for non-convergence
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hinfo[b][0], hinfoRes[b][0]) << "where b = " << b;
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;
    }

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    double err = 0;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(evect != rocblas_evect_original)
        {
            // only eigenvalues needed; compare with LAPACK

            // error is ||hD - hDRes|| / ||hD||
            // using frobenius norm
            if(hinfo[b][0] == 0)
                err = norm_error('F', 1, n, 1, hD[b], hDres[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
        else
        {
            // both eigenvalues and eigenvectors needed; compare with input
            // matrix
            if((hinfo[b][0] == 0) && (n > 0))
            {
                // Input matrix
                auto M = HMat::Wrap(A.data() + b * lda * n, lda, n)->block(BDesc().nrows(n).ncols(n));

                // Computed eigenvectors
                auto U = HMat::Wrap(hAres[b], lda, n)->block(BDesc().nrows(n).ncols(n));
                // Computed eigenvalues
                auto d = HMat::Convert(hDres[b], 1, n)->block(BDesc().nrows(1).ncols(n));
                // Diagonal matrix of size n by n with computed eigenvalues
                auto D = HMat::Zeros(n, n).diag(d);

                // Orthogonal error
                auto OE = U * adjoint(U) - HMat::Eye(n, n);
                err = OE.max_col_norm();
                *max_errv = err > *max_err ? err : *max_err;

                // Residual error
                auto RE = M - U * D * adjoint(U);
                err = RE.norm() / M.norm();
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <bool STRIDED, typename T, typename Sd, typename Td, typename Id, typename Sh, typename Th, typename Ih>
void syevd_heevd_getPerfData(const rocblas_handle handle,
                             const rocblas_evect evect,
                             const rocblas_fill uplo,
                             const rocblas_int n,
                             Td& dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             Sd& dD,
                             const rocblas_stride stD,
                             Sd& dE,
                             const rocblas_stride stE,
                             Id& dinfo,
                             const rocblas_int bc,
                             Th& hA,
                             Sh& hD,
                             Ih& hinfo,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;
    using S = decltype(std::real(T{}));

    int sizeE, lwork;
    if(!COMPLEX)
    {
        sizeE = (evect == rocblas_evect_none ? 2 * n + 1 : 1 + 6 * n + 2 * n * n);
        lwork = 0;
    }
    else
    {
        sizeE = (evect == rocblas_evect_none ? n : 1 + 5 * n + 2 * n * n);
        lwork = (evect == rocblas_evect_none ? n + 1 : 2 * n + n * n);
    }
    int liwork = (evect == rocblas_evect_none ? 1 : 3 + 5 * n);

    std::vector<T> work(lwork);
    std::vector<S> hE(sizeE);
    std::vector<int> iwork(liwork);
    std::vector<T> A;

    if(!perf)
    {
        syevd_heevd_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_syevd_heevd(evect, uplo, n, hA[b], lda, hD[b], work.data(), lwork, hE.data(), sizeE,
                            iwork.data(), liwork, hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    syevd_heevd_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        syevd_heevd_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA.data(), lda, stA,
                                                  dD.data(), stD, dE.data(), stE, dinfo.data(), bc));
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
        syevd_heevd_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA.data(), lda, stA, dD.data(), stD,
                              dE.data(), stE, dinfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syevd_heevd(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char evectC = argus.get<char>("evect");
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stD = argus.get<rocblas_stride>("strideD", n);
    rocblas_stride stE = argus.get<rocblas_stride>("strideE", n);

    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    if(argus.alg_mode == 1)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_set_alg_mode(handle, rocsolver_function_sterf, rocsolver_alg_mode_hybrid),
            rocblas_status_success);

        rocsolver_alg_mode alg_mode;
        EXPECT_ROCBLAS_STATUS(rocsolver_get_alg_mode(handle, rocsolver_function_sterf, &alg_mode),
                              rocblas_status_success);

        EXPECT_EQ(alg_mode, rocsolver_alg_mode_hybrid);
    }

    // check non-supported values
    if(uplo == rocblas_fill_full || evect == rocblas_evect_tridiagonal)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, (T* const*)nullptr, lda, stA,
                                      (S*)nullptr, stD, (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n,
                                                        (T*)nullptr, lda, stA, (S*)nullptr, stD,
                                                        (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_D = n;
    size_t size_E = size_D;
    size_t size_Ares = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_Dres = (argus.unit_check || argus.norm_check) ? size_D : 0;

    double max_error = 0, max_ortho_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, (T* const*)nullptr, lda, stA,
                                      (S*)nullptr, stD, (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n,
                                                        (T*)nullptr, lda, stA, (S*)nullptr, stD,
                                                        (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n,
                                                    (T* const*)nullptr, lda, stA, (S*)nullptr, stD,
                                                    (S*)nullptr, stE, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, (T*)nullptr,
                                                    lda, stA, (S*)nullptr, stD, (S*)nullptr, stE,
                                                    (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hDres(size_Dres, 1, stD, bc);
    // device
    device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
    device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hAres(size_Ares, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevd_heevd_getError<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE, stE,
                                             dinfo, bc, hA, hAres, hD, hDres, hinfo, hinfoRes,
                                             &max_error, &max_ortho_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevd_heevd_getPerfData<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE,
                                                stE, dinfo, bc, hA, hD, hinfo, &gpu_time_used,
                                                &cpu_time_used, hot_calls, argus.profile,
                                                argus.profile_kernels, argus.perf);
        }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hAres(size_Ares, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syevd_heevd(STRIDED, handle, evect, uplo, n, dA.data(),
                                                        lda, stA, dD.data(), stD, dE.data(), stE,
                                                        dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syevd_heevd_getError<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE, stE,
                                             dinfo, bc, hA, hAres, hD, hDres, hinfo, hinfoRes,
                                             &max_error, &max_ortho_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syevd_heevd_getPerfData<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE,
                                                stE, dinfo, bc, hA, hD, hinfo, &gpu_time_used,
                                                &cpu_time_used, hot_calls, argus.profile,
                                                argus.profile_kernels, argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using 10 * n * machine_precision as tolerance
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, max_error, 10 * n);
        if(evect != rocblas_evect_none)
            ROCSOLVER_TEST_CHECK(T, max_ortho_error, 10 * n);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("evect", "uplo", "n", "lda", "strideD", "strideE", "batch_c");
                rocsolver_bench_output(evectC, uploC, n, lda, stD, stE, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("evect", "uplo", "n", "lda", "strideA", "strideD", "strideE",
                                       "batch_c");
                rocsolver_bench_output(evectC, uploC, n, lda, stA, stD, stE, bc);
            }
            else
            {
                rocsolver_bench_output("evect", "uplo", "n", "lda");
                rocsolver_bench_output(evectC, uploC, n, lda);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
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

#define EXTERN_TESTING_SYEVD_HEEVD(...) \
    extern template void testing_syevd_heevd<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYEVD_HEEVD, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
