/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/misc/clss.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename U>
void sygvdx_hegvdx_checkBadArgs(const rocblas_handle handle,
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
                                rocblas_int* dNev,
                                U dW,
                                const rocblas_stride stW,
                                T dZ,
                                const rocblas_int ldz,
                                const rocblas_stride stZ,
                                rocblas_int* dInfo,
                                const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, nullptr, itype, evect, erange, uplo, n,
                                                  dA, lda, stA, dB, ldb, stB, vl, vu, il, iu, dNev,
                                                  dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, rocblas_eform(0), evect, erange,
                                                  uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il,
                                                  iu, dNev, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, rocblas_evect_tridiagonal,
                                                  erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl,
                                                  vu, il, iu, dNev, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, rocblas_erange(0),
                                                  uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il,
                                                  iu, dNev, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange,
                                                  rocblas_fill_full, n, dA, lda, stA, dB, ldb, stB,
                                                  vl, vu, il, iu, dNev, dW, stW, dZ, ldz, stZ,
                                                  dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo,
                                                      n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                                                      dNev, dW, stW, dZ, ldz, stZ, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                                  (T) nullptr, lda, stA, dB, ldb, stB, vl, vu, il,
                                                  iu, dNev, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                                  dA, lda, stA, (T) nullptr, ldb, stB, vl, vu, il,
                                                  iu, dNev, dW, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                                  dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                                                  (rocblas_int*)nullptr, dW, stW, dZ, ldz, stZ,
                                                  dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                                  dA, lda, stA, dB, ldb, stB, vl, vu, il, iu, dNev,
                                                  (U) nullptr, stW, dZ, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                                  dA, lda, stA, dB, ldb, stB, vl, vu, il, iu, dNev,
                                                  dW, stW, (T) nullptr, ldz, stZ, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                                  dA, lda, stA, dB, ldb, stB, vl, vu, il, iu, dNev,
                                                  dW, stW, dZ, ldz, stZ, (rocblas_int*)nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, 0,
                                                  (T) nullptr, lda, stA, (T) nullptr, ldb, stB, vl,
                                                  vu, il, iu, dNev, (U) nullptr, stW, (T) nullptr,
                                                  ldz, stZ, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo,
                                                      n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                                                      (rocblas_int*)nullptr, dW, stW, dZ, ldz, stZ,
                                                      (rocblas_int*)nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygvdx_hegvdx_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldz = 1;
    rocblas_stride stA = 1;
    rocblas_stride stB = 1;
    rocblas_stride stW = 1;
    rocblas_stride stZ = 1;
    rocblas_int bc = 1;
    rocblas_eform itype = rocblas_eform_ax;
    rocblas_evect evect = rocblas_evect_original;
    rocblas_erange erange = rocblas_erange_value;
    rocblas_fill uplo = rocblas_fill_upper;

    S vl = 0.0;
    S vu = 1.0;
    rocblas_int il = 0;
    rocblas_int iu = 0;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_batch_vector<T> dZ(1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dNev.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygvdx_hegvdx_checkBadArgs<STRIDED>(handle, itype, evect, erange, uplo, n, dA.data(), lda,
                                            stA, dB.data(), ldb, stB, vl, vu, il, iu, dNev.data(),
                                            dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<T> dZ(1, 1, 1, 1);
        device_strided_batch_vector<S> dW(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dZ.memcheck());
        CHECK_HIP_ERROR(dW.memcheck());
        CHECK_HIP_ERROR(dNev.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sygvdx_hegvdx_checkBadArgs<STRIDED>(handle, itype, evect, erange, uplo, n, dA.data(), lda,
                                            stA, dB.data(), ldb, stB, vl, vu, il, iu, dNev.data(),
                                            dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc);
    }
}

//
// If the environment variable:
//
// ROCSOLVER_SYGVDX_HEGVDX_USE_LEGACY_TESTS
//
// is defined, `sygvdx_hegvdx_getError` will compute errors using the
// legacy error bounds (for debugging purposes).
//
// Otherwise the new error bounds are always used.
//
static bool sygvdx_hegvdx_use_legacy_tests()
{
    bool status = false;
    if(std::getenv("ROCSOLVER_SYGVDX_HEGVDX_USE_LEGACY_TESTS") != nullptr)
    {
        status = true;
    }
    return status;
}

//
// The default behaviour of `sygvdx_hegvdx_getError()` is to check if the
// number of computed eigenvalues match the number of reference eigenvalues,
// and then to check all computed eigenvalues for their accuracy, but this
// behaviour can be relaxed.  This leads to two modes of operation: a relaxed
// check and a full (default) check.  Those are controlled by function
// `test_for_equality_of_number_of_computed_eigenvalues()`, below, in the
// following manner:
//
// a) If `ROCSOLVER_LAX_EIGENSOLVERS_TESTS` is defined, then the test suite
// will only use the subset of computed eigenvalues that match reference
// eigenvalues (up to the given tolerance); except
//
// b) If `ROCSOLVER_FULL_EIGENSOLVERS_TESTS` is defined, then the test suite
// will unconditionally check all eigenvalues for their accuracy.
//
// The relaxed tests are intended as a means to decouple the computation of
// error bounds of eigenvalues and eigenvectors, allowing tests to pass in the
// case that not all eigenvalues could be accurately computed, but all accurate
// eigenvalues have accurate eigenvectors.  If eigenvectors are not accurate,
// the corresponding tests will fail both in full mode and in relaxed mode.
//
// Note: the relaxed version of the tests is only supported when using the new
// error bounds, see also function `sygvdx_hegvdx_use_legacy_tests()`.
//
static bool test_for_equality_of_number_of_computed_eigenvalues()
{
    bool status = true;
#if defined(ROCSOLVER_LAX_EIGENSOLVERS_TESTS)
    status = false;
#else
    if(std::getenv("ROCSOLVER_LAX_EIGENSOLVERS_TESTS") != nullptr)
    {
        status = false;
    }
#endif
    if(std::getenv("ROCSOLVER_FULL_EIGENSOLVERS_TESTS") != nullptr)
    {
        status = true;
    }
    return status;
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sygvdx_hegvdx_initData(const rocblas_handle handle,
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

        bool use_legacy_tests = sygvdx_hegvdx_use_legacy_tests();

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // for testing purposes, we start with a reduced matrix M for the standard equivalent problem
            // with spectrum in a desired range (-20, 20). Then we construct the generalized pair
            // (A, B) from there.
            memset(hB[b], 0,
                   sizeof(T) * n * ldb); // since ldb >= n, make sure all entries of B are initialized
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
                        if(use_legacy_tests)
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
                        else
                        {
                            A[b][i + j * lda] = hA[b][i + j * lda];
                            B[b][i + j * ldb] = hB[b][i + j * ldb];
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
void sygvdx_hegvdx_getError(const rocblas_handle handle,
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
                            Vd& dNev,
                            Ud& dW,
                            const rocblas_stride stW,
                            Td& dZ,
                            const rocblas_int ldz,
                            const rocblas_stride stZ,
                            Vd& dInfo,
                            const rocblas_int bc,
                            Th& hA,
                            Th& hB,
                            Vh& hNev,
                            Vh& hNevRes,
                            Uh& hW,
                            Uh& hWRes,
                            Th& hZ,
                            Th& hZRes,
                            Vh& hInfo,
                            Vh& hInfoRes,
                            double* max_err,
                            const bool singular,
                            size_t& hashA,
                            size_t& hashB,
                            size_t& hashW,
                            size_t& hashZ)
{
    using HMat = HostMatrix<T, rocblas_int>;
    using BDesc = typename HMat::BlockDescriptor;
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    int lwork = (COMPLEX ? 2 * n : 8 * n);
    int lrwork = (COMPLEX ? 7 * n : 0);
    int liwork = 5 * n;

    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    std::vector<int> iwork(liwork);
    std::vector<int> hIfail(n);
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);
    std::vector<closest_largest_subsequences<S>> clss(bc);
    std::vector<bool> skip_test(bc, false);

    bool use_legacy_tests = sygvdx_hegvdx_use_legacy_tests();
    bool test_for_equality = test_for_equality_of_number_of_computed_eigenvalues();

    // input data initialization
    sygvdx_hegvdx_initData<true, true, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc,
                                          hA, hB, A, B, true, singular);

    // hash inputs
    hashA = deterministic_hash(hA, bc);
    hashB = deterministic_hash(hB, bc);

    // CPU lapack
    // abstol = 0 ensures max accuracy in rocsolver; for lapack we should use 2*safemin
    S atol = 2 * get_safemin<S>();
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cpu_sygvx_hegvx(itype, evect, erange, uplo, n, hA[b], lda, hB[b], ldb, vl, vu, il, iu, atol,
                        hNev[b], hW[b], hZ[b], ldz, work.data(), lwork, rwork.data(), iwork.data(),
                        hIfail.data(), hInfo[b]);

        // Capture failures where B is not positive definite (hInfo[b][0] > n),
        // or where the i-argument has an illegal value (hInfo[b][0] < 0).  All other LAPACK
        // failures skip the test.
        if((hInfo[b][0] > 0) && (hInfo[b][0] <= n))
        {
            skip_test[b] = true;
        }
    }

    //
    // Given an eigenvalue l_i of the symmetric matrix A and a computed
    // eigenvalue l_i^* (obtained with a backward stable method), Weyl's
    // theorem yields |l_i - l_i^*| <= K*ulp*||A||_2, where K depends on n.
    // For the sake of this test, we will set K = C * n, with C ~ 1.
    //
    // Thus, if the range to look for eigenvalues is the interval (vl, vu],
    // calls to the solver should look for computed eigenvalues in the range
    // (vl - tol, vu + tol], where `tol = C * n * ulp * ||A||`.
    //
    S C = 4;
    std::vector<S> tols(bc, 0);
    std::vector<S> norms(bc, 0);
    S tol = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hNev[b][0] > 0)
        {
            // Get lapack eigenvalues (reference to which rocSOLVER's sygvdx will be compared to)
            auto eigsLapack = *HMat::Convert(hW[b], hNev[b][0], 1);
            norms[b] = eigsLapack.max_coeff_norm();
        }
        else
        {
            norms[b] = S(0);
        }

        tols[b] = C * n * std::numeric_limits<S>::epsilon() * norms[b];
        if(std::isfinite(tols[b]) && (tols[b] > tol))
        {
            tol = tols[b];
        }
    }

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sygvdx_hegvdx(
        STRIDED, handle, itype, evect, erange, uplo, n, dA.data(), lda, stA, dB.data(), ldb, stB,
        vl, vu, il, iu, dNev.data(), dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc));

    CHECK_HIP_ERROR(hNevRes.transfer_from(dNev));
    CHECK_HIP_ERROR(hWRes.transfer_from(dW));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect != rocblas_evect_none)
        CHECK_HIP_ERROR(hZRes.transfer_from(dZ));

    // hash outputs
    hashW = deterministic_hash(hWRes, bc);
    if(evect != rocblas_evect_none)
        hashZ = deterministic_hash(hZRes, bc);

    // Except for the cases in which B is indefinite, we expect the eigensolver
    // to converge for all input matrices.

    // check info for illegal values and/or positive-definiteness
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // Capture failures where B is not positive definite (hInfo[b][0] > n),
        // or where the i-argument has an illegal value (hInfo[b][0] < 0).  All other LAPACK
        // failures skip the test.
        if(skip_test[b])
            continue;

        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            *max_err += 1;

        auto numMatchingEigs = clss[b](hW[b], hNev[b][0], hWRes[b], hNevRes[b][0], tols[b]);
        if(test_for_equality)
        {
            EXPECT_EQ(hNev[b][0], numMatchingEigs) << "where b = " << b;
            if(hNev[b][0] != numMatchingEigs)
                *max_err += 1;
        }
    }

    //
    // Compute errors
    //
    double err;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        auto [lapackEigs, rocsolverEigs] = clss[b].subseqs();
        auto [_, rocsolverEigsIds] = clss[b].subseqs_ids();
        auto numMatchingEigs = rocsolverEigs.size();

        // Number of eigenvalues computed by rocSOLVER
        auto numRocsolverEigs = hNevRes[b][0];

        // Only check accuracy for tests in which both computed and reference values exist and are well defined.
        if(skip_test[b] || (numMatchingEigs == 0) || (hInfo[b][0] != 0))
            continue;

        if(evect == rocblas_evect_none)
        {
            //
            // Only eigenvalues
            //

            if(use_legacy_tests)
            {
                err = norm_error('F', 1, numMatchingEigs, 1, lapackEigs.data(), rocsolverEigs.data());
                *max_err = err > *max_err ? err : *max_err;
            }
            else
            {
                // Get computed eigenvalues
                auto eigs
                    = *HMat::Convert(rocsolverEigs.data(), rocsolverEigs.size(),
                                     1); // convert eigenvalues from type S to type T, if required

                // Get lapack (reference) eigenvalues
                auto eigsRef
                    = *HMat::Convert(lapackEigs.data(), lapackEigs.size(),
                                     1); // convert eigenvalues from type S to type T, if required
                err = (eigs - eigsRef).norm() / eigsRef.norm();
                *max_err = err > *max_err ? err : *max_err;
            }
        }
        else
        {
            //
            // Both eigenvalues and eigenvectors
            //

            if(use_legacy_tests)
            {
                T alpha = 1;
                T beta = 0;

                // hZRes contains eigenvectors x
                // compute B*x (or A*x) and store in hB
                cpu_symm_hemm(rocblas_side_left, uplo, n, numRocsolverEigs, alpha, B[b], ldb,
                              hZRes[b], ldz, beta, hB[b], ldb);

                auto [_, hWResIds] = clss[b].subseqs_ids();
                if(itype == rocblas_eform_ax)
                {
                    // problem is A*x = (lambda)*B*x

                    // compute (1/lambda)*A*x and store in hA
                    for(int j = 0; j < numMatchingEigs; j++)
                    {
                        int jj = hWResIds[j]; // Id of rocSOLVER eigen-pair associated to j-th LAPACK eigen-pair
                        alpha = T(1) / hWRes[b][jj];
                        cpu_symv_hemv(uplo, n, alpha, A[b], lda, hZRes[b] + jj * ldz, 1, beta,
                                      hA[b] + j * lda, 1);
                    }

                    // move B*x into hZRes
                    for(rocblas_int i = 0; i < n; i++)
                    {
                        for(rocblas_int j = 0; j < numMatchingEigs; j++)
                        {
                            int jj = hWResIds[j]; // Id of rocSOLVER eigen-pair associated to j-th LAPACK eigen-pair
                            hZRes[b][i + j * ldz] = hB[b][i + jj * ldb];
                        }
                    }
                }
                else
                {
                    // problem is A*B*x = (lambda)*x or B*A*x = (lambda)*x

                    // compute (1/lambda)*A*B*x or (1/lambda)*B*A*x and store in hA
                    for(int j = 0; j < numMatchingEigs; j++)
                    {
                        int jj = hWResIds[j]; // Id of rocSOLVER eigen-pair associated to j-th LAPACK eigen-pair
                        alpha = T(1) / hWRes[b][jj];
                        cpu_symv_hemv(uplo, n, alpha, A[b], lda, hB[b] + jj * ldb, 1, beta,
                                      hA[b] + j * lda, 1);
                    }
                    // move hZRes
                    for(rocblas_int i = 0; i < n; i++)
                    {
                        for(rocblas_int j = 0; j < numMatchingEigs; j++)
                        {
                            int jj = hWResIds[j]; // Id of rocSOLVER eigen-pair associated to j-th LAPACK eigen-pair
                            if(j != jj)
                                hZRes[b][i + j * ldz] = hZRes[b][i + jj * ldz];
                        }
                    }
                }

                // error is ||hA - hZRes|| / ||hA||
                // using frobenius norm
                err = norm_error('F', n, numMatchingEigs, lda, hA[b], hZRes[b], ldz);
                *max_err = err > *max_err ? err : *max_err;
            }
            else // if(!use_legacy_tests)
            {
                //
                // Prepare input
                //

                // Get computed eigenvalues
                auto eigs
                    = *HMat::Convert(rocsolverEigs.data(), rocsolverEigs.size(),
                                     1); // convert eigenvalues from type S to type T, if required

                // Get lapack (reference) eigenvalues
                auto eigsRef
                    = *HMat::Convert(lapackEigs.data(), lapackEigs.size(),
                                     1); // convert eigenvalues from type S to type T, if required

                // Create thin wrappers of input matrices A and B
                auto AWrap = HMat::Wrap(A.data() + b * lda * n, lda, n);
                auto BWrap = HMat::Wrap(B.data() + b * ldb * n, ldb, n);

                // We want the sub-blocks starting from row 0, col 0 and with size n x n of A and B
                auto A_b = (*AWrap).block(BDesc().nrows(n).ncols(n));
                auto B_b = (*BWrap).block(BDesc().nrows(n).ncols(n));

                // Get computed eigenvectors
                auto V_b
                    = (*HMat::Wrap(hZRes[b], ldz, n)).block(BDesc().nrows(n).ncols(numRocsolverEigs));

                // If rocSOLVER computed more eigen-pairs then the number of
                // reference eigenvalues, select the eigen-pairs that match the
                // reference
                if(numRocsolverEigs > numMatchingEigs)
                {
                    rocblas_int ii;
                    for(rocblas_int i = 0; i < numMatchingEigs; ++i)
                    {
                        ii = rocsolverEigsIds[i];
                        V_b.col(i, V_b.col(ii));
                    }
                    V_b = V_b.block(BDesc().nrows(n).ncols(numMatchingEigs));
                }

                //
                // Check eigenpairs' accuracy with a "Relative Weyl" error
                // bound, which (at its simplest form) states the following.
                //
                // Let X (cond(X) < Inf), and A (A^* = A) be such that A has
                // eigenvalues {a_i} and H = X^t*A*X has eigenvalues {h_i}.
                // Then:
                //
                // |a_i - h_i| <= |a_i|*||X^t*X - I||_2
                //
                // Note: for rocSOLVER's sygv, if V is the eigenvectors' matrix
                // and B = L*L^t, then either X = L^t*V (cases 1 and 2) or X =
                // inv(L)*V (case 3).
                //
                auto VE = HMat::Empty();
                if(itype == rocblas_eform_bax)
                {
                    VE = adjoint(V_b) * inv(B_b) * V_b - HMat::Eye(numMatchingEigs);
                }
                else // if ((itype == rocblas_eform_ax) || (itype == rocblas_eform_abx))
                {
                    VE = adjoint(V_b) * B_b * V_b - HMat::Eye(numMatchingEigs);
                }
                S eta = std::max(VE.norm(), std::numeric_limits<S>::epsilon());
                *max_err = eta > *max_err ? eta : *max_err;

                auto AE = HMat::Empty();
                if(itype == rocblas_eform_abx)
                {
                    auto Z = B_b * V_b;
                    AE = adjoint(Z) * A_b * Z - HMat::Zeros(numMatchingEigs).diag(eigs);
                }
                else // if ((itype == rocblas_eform_ax) || (itype == rocblas_eform_bax))
                {
                    AE = adjoint(V_b) * A_b * V_b - HMat::Zeros(numMatchingEigs).diag(eigs);
                }
                err = AE.norm() / eigsRef.norm();
                err *= std::numeric_limits<S>::epsilon() / eta;
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <bool STRIDED, typename T, typename S, typename Td, typename Ud, typename Vd, typename Th, typename Uh, typename Vh>
void sygvdx_hegvdx_getPerfData(const rocblas_handle handle,
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
                               Vd& dNev,
                               Ud& dW,
                               const rocblas_stride stW,
                               Td& dZ,
                               const rocblas_int ldz,
                               const rocblas_stride stZ,
                               Vd& dInfo,
                               const rocblas_int bc,
                               Th& hA,
                               Th& hB,
                               Vh& hNev,
                               Uh& hW,
                               Th& hZ,
                               Vh& hInfo,
                               double* gpu_time_used,
                               double* cpu_time_used,
                               const rocblas_int hot_calls,
                               const int profile,
                               const bool profile_kernels,
                               const bool perf,
                               const bool singular)
{
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = nan("");
    }

    sygvdx_hegvdx_initData<true, false, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc,
                                           hA, hB, A, B, false, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygvdx_hegvdx_initData<false, true, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB,
                                               bc, hA, hB, A, B, false, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_sygvdx_hegvdx(
            STRIDED, handle, itype, evect, erange, uplo, n, dA.data(), lda, stA, dB.data(), ldb, stB,
            vl, vu, il, iu, dNev.data(), dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc));
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
        sygvdx_hegvdx_initData<false, true, T>(handle, itype, evect, n, dA, lda, stA, dB, ldb, stB,
                                               bc, hA, hB, A, B, false, singular);

        start = get_time_us_sync(stream);
        rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n, dA.data(), lda, stA,
                                dB.data(), ldb, stB, vl, vu, il, iu, dNev.data(), dW.data(), stW,
                                dZ.data(), ldz, stZ, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sygvdx_hegvdx(Arguments& argus)
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
    rocblas_int ldz = argus.get<rocblas_int>("ldz", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * n);
    rocblas_stride stW = argus.get<rocblas_stride>("strideW", n);
    rocblas_stride stZ = argus.get<rocblas_stride>("strideZ", ldz * n);

    S vl = S(argus.get<double>("vl", 0));
    S vu = S(argus.get<double>("vu", erangeC == 'V' ? 1 : 0));
    rocblas_int il = argus.get<rocblas_int>("il", erangeC == 'I' ? 1 : 0);
    rocblas_int iu = argus.get<rocblas_int>("iu", erangeC == 'I' ? 1 : 0);

    rocblas_eform itype = char2rocblas_eform(itypeC);
    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_erange erange = char2rocblas_erange(erangeC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stWRes = (argus.unit_check || argus.norm_check || argus.hash_check) ? stW : 0;
    rocblas_stride stZRes = (argus.unit_check || argus.norm_check || argus.hash_check) ? stZ : 0;

    // check non-supported values
    if(uplo == rocblas_fill_full || evect == rocblas_evect_tridiagonal)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                        (T* const*)nullptr, lda, stA, (T* const*)nullptr, ldb, stB,
                                        vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stW,
                                        (T* const*)nullptr, ldz, stZ, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n, (T*)nullptr,
                                        lda, stA, (T*)nullptr, ldb, stB, vl, vu, il, iu,
                                        (rocblas_int*)nullptr, (S*)nullptr, stW, (T*)nullptr, ldz,
                                        stZ, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * n;
    size_t size_W = size_t(n);
    size_t size_Z = size_t(ldz) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;
    size_t hashA = 0, hashB = 0, hashW = 0, hashZ = 0;

    size_t size_WRes = (argus.unit_check || argus.norm_check || argus.hash_check) ? size_W : 0;
    size_t size_ZRes = (argus.unit_check || argus.norm_check || argus.hash_check) ? size_Z : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || ldb < n || (evect != rocblas_evect_none && ldz < n)
                         || bc < 0 || (erange == rocblas_erange_value && vl >= vu)
                         || (erange == rocblas_erange_index && (il < 1 || iu < 0))
                         || (erange == rocblas_erange_index && (iu > n || (n > 0 && il > iu))));
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n,
                                        (T* const*)nullptr, lda, stA, (T* const*)nullptr, ldb, stB,
                                        vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stW,
                                        (T* const*)nullptr, ldz, stZ, (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n, (T*)nullptr,
                                        lda, stA, (T*)nullptr, ldb, stB, vl, vu, il, iu,
                                        (rocblas_int*)nullptr, (S*)nullptr, stW, (T*)nullptr, ldz,
                                        stZ, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_sygvdx_hegvdx(
                STRIDED, handle, itype, evect, erange, uplo, n, (T* const*)nullptr, lda, stA,
                (T* const*)nullptr, ldb, stB, vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr,
                stW, (T* const*)nullptr, ldz, stZ, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sygvdx_hegvdx(
                STRIDED, handle, itype, evect, erange, uplo, n, (T*)nullptr, lda, stA, (T*)nullptr,
                ldb, stB, vl, vu, il, iu, (rocblas_int*)nullptr, (S*)nullptr, stW, (T*)nullptr, ldz,
                stZ, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
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
    device_strided_batch_vector<rocblas_int> dNev(1, 1, 1, bc);
    device_strided_batch_vector<S> dW(size_W, 1, stW, bc);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    CHECK_HIP_ERROR(dNev.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hZ(size_Z, 1, bc);
        host_batch_vector<T> hZRes(size_ZRes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_batch_vector<T> dZ(size_Z, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_Z)
            CHECK_HIP_ERROR(dZ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n, dA.data(),
                                        lda, stA, dB.data(), ldb, stB, vl, vu, il, iu, dNev.data(),
                                        dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check || argus.hash_check)
            sygvdx_hegvdx_getError<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                dNev, dW, stW, dZ, ldz, stZ, dInfo, bc, hA, hB, hNev, hNevRes, hW, hWRes, hZ, hZRes,
                hInfo, hInfoRes, &max_error, argus.singular, hashA, hashB, hashW, hashZ);

        // collect performance data
        if(argus.timing)
            sygvdx_hegvdx_getPerfData<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                dNev, dW, stW, dZ, ldz, stZ, dInfo, bc, hA, hB, hNev, hW, hZ, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hZ(size_Z, 1, stZ, bc);
        host_strided_batch_vector<T> hZRes(size_ZRes, 1, stZRes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dZ(size_Z, 1, stZ, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_Z)
            CHECK_HIP_ERROR(dZ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_sygvdx_hegvdx(STRIDED, handle, itype, evect, erange, uplo, n, dA.data(),
                                        lda, stA, dB.data(), ldb, stB, vl, vu, il, iu, dNev.data(),
                                        dW.data(), stW, dZ.data(), ldz, stZ, dInfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check || argus.hash_check)
            sygvdx_hegvdx_getError<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                dNev, dW, stW, dZ, ldz, stZ, dInfo, bc, hA, hB, hNev, hNevRes, hW, hWRes, hZ, hZRes,
                hInfo, hInfoRes, &max_error, argus.singular, hashA, hashB, hashW, hashZ);

        // collect performance data
        if(argus.timing)
            sygvdx_hegvdx_getPerfData<STRIDED, T>(
                handle, itype, evect, erange, uplo, n, dA, lda, stA, dB, ldb, stB, vl, vu, il, iu,
                dNev, dW, stW, dZ, ldz, stZ, dInfo, bc, hA, hB, hNev, hW, hZ, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    // validate results for rocsolver-test
    // using 8 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 8 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("itype", "evect", "erange", "uplo", "n", "lda", "ldb", "vl",
                                       "vu", "il", "iu", "strideW", "ldz", "batch_c");
                rocsolver_bench_output(itypeC, evectC, erangeC, uploC, n, lda, ldb, vl, vu, il, iu,
                                       stW, ldz, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype", "evect", "erange", "uplo", "n", "lda", "ldb",
                                       "strideA", "strideB", "vl", "vu", "il", "iu", "strideW",
                                       "ldz", "strideZ", "batch_c");
                rocsolver_bench_output(itypeC, evectC, erangeC, uploC, n, lda, ldb, stA, stB, vl,
                                       vu, il, iu, stW, ldz, stZ, bc);
            }
            else
            {
                rocsolver_bench_output("itype", "evect", "erange", "uplo", "n", "lda", "ldb", "vl",
                                       "vu", "il", "iu", "ldz");
                rocsolver_bench_output(itypeC, evectC, erangeC, uploC, n, lda, ldb, vl, vu, il, iu,
                                       ldz);
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
            if(argus.hash_check)
            {
                rocsolver_bench_output("hash(A)", "hash(B)", "hash(W)", "hash(Z)");
                rocsolver_bench_output(ROCSOLVER_FORMAT_HASH(hashA), ROCSOLVER_FORMAT_HASH(hashB),
                                       ROCSOLVER_FORMAT_HASH(hashW), ROCSOLVER_FORMAT_HASH(hashZ));
                rocsolver_bench_endl();
            }
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

#define EXTERN_TESTING_SYGVDX_HEGVDX(...) \
    extern template void testing_sygvdx_hegvdx<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYGVDX_HEGVDX, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
