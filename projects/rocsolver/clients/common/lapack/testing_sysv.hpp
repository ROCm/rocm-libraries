/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstring>

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <bool STRIDED, typename T, typename U>
void sysv_checkBadArgs(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       const rocblas_int nrhs,
                       T dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       U dIpiv,
                       const rocblas_stride stP,
                       T dB,
                       const rocblas_int ldb,
                       const rocblas_stride stB,
                       U dInfo,
                       const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, nullptr, uplo, n, nrhs, dA, lda, stA, dIpiv, stP,
                                         dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, rocblas_fill_full, n, nrhs, dA, lda, stA,
                                         dIpiv, stP, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dIpiv,
                                             stP, dB, ldb, stB, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, (T) nullptr, lda, stA,
                                         dIpiv, stP, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, (U) nullptr,
                                         stP, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP,
                                         (T) nullptr, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP,
                                         dB, ldb, stB, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, 0, nrhs, (T) nullptr, lda, stA,
                                         (U) nullptr, stP, (T) nullptr, ldb, stB, dInfo, bc),
                          rocblas_status_success);

    // B is not referenced when nrhs = 0 (A is still factorized)
    EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, 0, dA, lda, stA, dIpiv, stP,
                                         (T) nullptr, ldb, stB, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dIpiv,
                                             stP, dB, ldb, stB, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sysv_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int nrhs = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_stride stB = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sysv_checkBadArgs<STRIDED>(handle, uplo, n, nrhs, dA.data(), lda, stA, dIpiv.data(), stP,
                                   dB.data(), ldb, stB, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        sysv_checkBadArgs<STRIDED>(handle, uplo, n, nrhs, dA.data(), lda, stA, dIpiv.data(), stP,
                                   dB.data(), ldb, stB, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sysv_initData(const rocblas_handle handle,
                   const rocblas_fill uplo,
                   const rocblas_int n,
                   const rocblas_int nrhs,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_stride stA,
                   Td& dB,
                   const rocblas_int ldb,
                   const rocblas_stride stB,
                   const rocblas_int bc,
                   Th& hA,
                   Th& hB,
                   const bool singular,
                   const char pivots = 'D')
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        if(nrhs > 0)
            rocblas_init<T>(hB, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            if(pivots == 'R')
            {
                // a symmetric indefinite matrix with random entries (without ties), whose
                // factorization has 1-by-1 and 2-by-2 blocks and interchanges in no particular
                // pattern
                for(rocblas_int j = 0; j < n; j++)
                {
                    for(rocblas_int i = 0; i <= j; i++)
                    {
                        T const v = hA[b][i + j * lda] - T(5.5) + T(0.001 * ((i * 31 + j * 17) % 97));
                        hA[b][i + j * lda] = v;
                        hA[b][j + i * lda] = v;
                    }
                }
            }
            else
            {
                // scale A to avoid singularities
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                    {
                        if(i == j)
                            hA[b][i + j * lda] += 400;
                        else
                            hA[b][i + j * lda] -= 4;
                    }
                }

                // shuffle rows to test pivoting (this moves the dominant elements to the
                // anti-diagonal of the referenced triangle, so that 2x2 pivots are needed)
                // always the same permutation for debugging purposes
                for(rocblas_int i = 0; i < n / 2; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                        std::swap(hA[b][i + j * lda], hA[b][n - 1 - i + j * lda]);
                }
            }

            if(singular && b == bc / 2)
            {
                // make the matrix in the middle of the batch singular, keeping the others
                // non-singular; zero rows and columns give zero 1x1 diagonal blocks in D
                // always the same elements for debugging purposes
                // the algorithm must detect the first zero pivot and leave B unchanged
                rocblas_int j = n / 4 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
                j = n / 2 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
                j = n - 1 + b;
                j -= (j / n) * n;
                for(rocblas_int i = 0; i < n; i++)
                {
                    hA[b][i + j * lda] = 0;
                    hA[b][j + i * lda] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        if(nrhs > 0)
            CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sysv_getError(const rocblas_handle handle,
                   const rocblas_fill uplo,
                   const rocblas_int n,
                   const rocblas_int nrhs,
                   Td& dA,
                   const rocblas_int lda,
                   const rocblas_stride stA,
                   Ud& dIpiv,
                   const rocblas_stride stP,
                   Td& dB,
                   const rocblas_int ldb,
                   const rocblas_stride stB,
                   Ud& dInfo,
                   const rocblas_int bc,
                   Th& hA,
                   Th& hARes,
                   Uh& hIpiv,
                   Uh& hIpivRes,
                   Th& hB,
                   Th& hBRes,
                   Uh& hInfo,
                   Uh& hInfoRes,
                   double* max_err,
                   const bool singular,
                   const char pivots = 'D')
{
    rocblas_int lwork = 64 * n;
    std::vector<T> work(lwork);

    // input data initialization
    sysv_initData<true, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB,
                                 singular, pivots);

    // keep a copy of the right-hand sides
    // (B must not be modified when D is singular, nor outside of its first n rows)
    size_t size_B = size_t(ldb) * nrhs;
    std::vector<T> hBOrig(size_B * bc);
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(size_B > 0)
            std::memcpy(hBOrig.data() + b * size_B, hB[b], sizeof(T) * size_B);
    }

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA,
                                       dIpiv.data(), stP, dB.data(), ldb, stB, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(nrhs > 0)
        CHECK_HIP_ERROR(hBRes.transfer_from(dB));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cpu_sysv(uplo, n, nrhs, hA[b], lda, hIpiv[b], (nrhs > 0 ? hB[b] : nullptr), ldb,
                 work.data(), lwork, hInfo[b]);
    }

    // error is ||hA - hARes|| / ||hA|| for the factorization (as SYTRF tests),
    // and ||hB - hBRes|| / ||hB|| for the solution
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm for the factorization and vector-induced infinity norm
    // for the solution
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // factorization (A is factorized even if nrhs = 0 or D is singular)
        err = norm_error('F', n, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;

        // also check pivoting (count the number of incorrect pivots)
        err = 0;
        for(rocblas_int i = 0; i < n; ++i)
        {
            EXPECT_EQ(hIpiv[b][i], hIpivRes[b][i]) << "where b = " << b << ", i = " << i;
            if(hIpiv[b][i] != hIpivRes[b][i])
                err++;
        }
        *max_err = err > *max_err ? err : *max_err;

        if(nrhs > 0)
        {
            T* BOrig = hBOrig.data() + b * size_B;

            if(hInfo[b][0] == 0)
            {
                // solution
                err = norm_error('I', n, nrhs, ldb, hB[b], hBRes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }

            // count the elements of B that were modified but must be bitwise unchanged:
            // all of B if D is singular, and rows n to ldb-1 in any case
            rocblas_int first_row = (hInfo[b][0] == 0 ? n : 0);
            rocblas_int nmod = 0;
            for(rocblas_int j = 0; j < nrhs; ++j)
            {
                for(rocblas_int i = first_row; i < ldb; ++i)
                {
                    if(std::memcmp(&BOrig[i + j * ldb], &hBRes[b][i + j * ldb], sizeof(T)) != 0)
                        nmod++;
                }
            }
            EXPECT_EQ(nmod, 0) << "where b = " << b << ", info = " << hInfo[b][0];
            *max_err += nmod;
        }
    }

    // also check info
    err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void sysv_getPerfData(const rocblas_handle handle,
                      const rocblas_fill uplo,
                      const rocblas_int n,
                      const rocblas_int nrhs,
                      Td& dA,
                      const rocblas_int lda,
                      const rocblas_stride stA,
                      Ud& dIpiv,
                      const rocblas_stride stP,
                      Td& dB,
                      const rocblas_int ldb,
                      const rocblas_stride stB,
                      Ud& dInfo,
                      const rocblas_int bc,
                      Th& hA,
                      Uh& hIpiv,
                      Th& hB,
                      Uh& hInfo,
                      double* gpu_time_used,
                      double* cpu_time_used,
                      const rocblas_int hot_calls,
                      const int profile,
                      const bool profile_kernels,
                      const bool perf,
                      const bool singular,
                      const char pivots = 'D')
{
    rocblas_int lwork = 64 * n;
    std::vector<T> work(lwork);

    if(!perf)
    {
        sysv_initData<true, false, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB,
                                      singular, pivots);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cpu_sysv(uplo, n, nrhs, hA[b], lda, hIpiv[b], (nrhs > 0 ? hB[b] : nullptr), ldb,
                     work.data(), lwork, hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sysv_initData<true, false, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB,
                                  singular, pivots);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sysv_initData<false, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB,
                                      singular, pivots);

        CHECK_ROCBLAS_ERROR(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA,
                                           dIpiv.data(), stP, dB.data(), ldb, stB, dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

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
        sysv_initData<false, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dB, ldb, stB, bc, hA, hB,
                                      singular, pivots);

        timer.start(stream);
        rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA, dIpiv.data(), stP,
                       dB.data(), ldb, stB, dInfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_sysv(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nrhs);
    char pivots = argus.get<char>("pivots", 'D');

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;
    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, (T* const*)nullptr,
                                                 lda, stA, (rocblas_int*)nullptr, stP,
                                                 (T* const*)nullptr, ldb, stB,
                                                 (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda,
                                                 stA, (rocblas_int*)nullptr, stP, (T*)nullptr, ldb,
                                                 stB, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(n);
    size_t size_B = size_t(ldb) * nrhs;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;
    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, (T* const*)nullptr,
                                                 lda, stA, (rocblas_int*)nullptr, stP,
                                                 (T* const*)nullptr, ldb, stB,
                                                 (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda,
                                                 stA, (rocblas_int*)nullptr, stP, (T*)nullptr, ldb,
                                                 stB, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, (T* const*)nullptr, lda,
                                             stA, (rocblas_int*)nullptr, stP, (T* const*)nullptr,
                                             ldb, stB, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda, stA,
                                             (rocblas_int*)nullptr, stP, (T*)nullptr, ldb, stB,
                                             (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        // (nrhs = 0 is not a quick return, as A is still factorized)
        if(n == 0 || bc == 0)
        {
            // info must be set to zero when n = 0
            if(bc > 0)
            {
                for(rocblas_int b = 0; b < bc; ++b)
                    hInfoRes[b][0] = -1;
                CHECK_HIP_ERROR(dInfo.transfer_from(hInfoRes));
            }

            EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA.data(), lda,
                                                 stA, dIpiv.data(), stP, dB.data(), ldb, stB,
                                                 dInfo.data(), bc),
                                  rocblas_status_success);

            if(bc > 0)
            {
                CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
                for(rocblas_int b = 0; b < bc; ++b)
                    EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
            }

            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sysv_getError<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB,
                                      dInfo, bc, hA, hARes, hIpiv, hIpivRes, hB, hBRes, hInfo,
                                      hInfoRes, &max_error, argus.singular, pivots);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            sysv_getPerfData<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                         stB, dInfo, bc, hA, hIpiv, hB, hInfo, &gpu_time_used,
                                         &cpu_time_used, hot_calls, argus.profile,
                                         argus.profile_kernels, argus.perf, argus.singular, pivots);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<rocblas_int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        // (nrhs = 0 is not a quick return, as A is still factorized)
        if(n == 0 || bc == 0)
        {
            // info must be set to zero when n = 0
            if(bc > 0)
            {
                for(rocblas_int b = 0; b < bc; ++b)
                    hInfoRes[b][0] = -1;
                CHECK_HIP_ERROR(dInfo.transfer_from(hInfoRes));
            }

            EXPECT_ROCBLAS_STATUS(rocsolver_sysv(STRIDED, handle, uplo, n, nrhs, dA.data(), lda,
                                                 stA, dIpiv.data(), stP, dB.data(), ldb, stB,
                                                 dInfo.data(), bc),
                                  rocblas_status_success);

            if(bc > 0)
            {
                CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
                for(rocblas_int b = 0; b < bc; ++b)
                    EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
            }

            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sysv_getError<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB,
                                      dInfo, bc, hA, hARes, hIpiv, hIpivRes, hB, hBRes, hInfo,
                                      hInfoRes, &max_error, argus.singular, pivots);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            sysv_getPerfData<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                         stB, dInfo, bc, hA, hIpiv, hB, hInfo, &gpu_time_used,
                                         &cpu_time_used, hot_calls, argus.profile,
                                         argus.profile_kernels, argus.perf, argus.singular, pivots);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance (n^2 * machine_precision for the random
    // indefinite matrices, which are less well conditioned; the pivots are compared exactly)
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, (pivots == 'R' ? n * n : n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb", "strideP", "batch_c");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb", "strideA", "strideP",
                                       "strideB", "batch_c");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb, stA, stP, stB, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "n", "nrhs", "lda", "ldb");
                rocsolver_bench_output(uploC, n, nrhs, lda, ldb);
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

#define EXTERN_TESTING_SYSV(...) extern template void testing_sysv<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYSV, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
