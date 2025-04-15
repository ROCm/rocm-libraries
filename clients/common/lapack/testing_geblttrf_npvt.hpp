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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"

template <bool STRIDED, typename T, typename U>
void geblttrf_npvt_checkBadArgs(const rocblas_handle handle,
                                const rocblas_int nb,
                                const rocblas_int nblocks,
                                T dA,
                                const rocblas_int lda,
                                const rocblas_stride stA,
                                T dB,
                                const rocblas_int ldb,
                                const rocblas_stride stB,
                                T dC,
                                const rocblas_int ldc,
                                const rocblas_stride stC,
                                U dInfo,
                                const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, nullptr, nb, nblocks, dA, lda, stA, dB,
                                                  ldb, stB, dC, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA,
                                                      dB, ldb, stB, dC, ldc, stC, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, (T) nullptr, lda,
                                                  stA, dB, ldb, stB, dC, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA,
                                                  (T) nullptr, ldb, stB, dC, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA, dB,
                                                  ldb, stB, (T) nullptr, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA, dB,
                                                  ldb, stB, dC, ldc, stC, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, 0, nblocks, (T) nullptr, lda,
                                                  stA, (T) nullptr, ldb, stB, (T) nullptr, ldc, stC,
                                                  dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, 0, (T) nullptr, lda, stA,
                                                  (T) nullptr, ldb, stB, (T) nullptr, ldc, stC,
                                                  dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA, lda, stA,
                                                      dB, ldb, stB, dC, ldc, stC, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geblttrf_npvt_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int nb = 1;
    rocblas_int nblocks = 2;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldc = 1;
    rocblas_stride stA = 2;
    rocblas_stride stB = 2;
    rocblas_stride stC = 2;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_batch_vector<T> dC(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        geblttrf_npvt_checkBadArgs<STRIDED>(handle, nb, nblocks, dA.data(), lda, stA, dB.data(),
                                            ldb, stB, dC.data(), ldc, stC, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<T> dC(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        geblttrf_npvt_checkBadArgs<STRIDED>(handle, nb, nblocks, dA.data(), lda, stA, dB.data(),
                                            ldb, stB, dC.data(), ldc, stC, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrf_npvt_initData(const rocblas_handle handle,
                            const rocblas_int nb,
                            const rocblas_int nblocks,
                            Td& dA,
                            const rocblas_int lda,
                            Td& dB,
                            const rocblas_int ldb,
                            Td& dC,
                            const rocblas_int ldc,
                            const rocblas_int bc,
                            Th& hA,
                            Th& hB,
                            Th& hC,
                            const bool singular)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);
        rocblas_init<T>(hC, false);

        rocblas_int n = nb * nblocks;

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale to avoid singularities
            // leaving matrix as diagonal dominant so that pivoting is not required
            for(rocblas_int i = 0; i < nb; i++)
            {
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int k = 0; k < nblocks; k++)
                    {
                        if(i == j)
                            hB[b][i + j * ldb + k * ldb * nb] += 400;
                        else
                            hB[b][i + j * ldb + k * ldb * nb] -= 4;
                    }

                    for(rocblas_int k = 0; k < nblocks - 1; k++)
                    {
                        hA[b][i + j * lda + k * lda * nb] -= 4;
                        hC[b][i + j * ldc + k * ldc * nb] -= 4;
                    }
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes)

                rocblas_int jj = n / 4 + b;
                jj -= (jj / n) * n;
                rocblas_int j = jj % nb;
                rocblas_int k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    hB[b][i + j * ldb + k * ldb * nb] = 0;
                    if(k < nblocks - 1)
                        hA[b][i + j * lda + k * lda * nb] = 0;
                    if(k > 0)
                        hC[b][i + j * ldc + (k - 1) * ldc * nb] = 0;
                }

                jj = n / 2 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    hB[b][i + j * ldb + k * ldb * nb] = 0;
                    if(k < nblocks - 1)
                        hA[b][i + j * lda + k * lda * nb] = 0;
                    if(k > 0)
                        hC[b][i + j * ldc + (k - 1) * ldc * nb] = 0;
                }

                jj = n - 1 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    hB[b][i + j * ldb + k * ldb * nb] = 0;
                    if(k < nblocks - 1)
                        hA[b][i + j * lda + k * lda * nb] = 0;
                    if(k > 0)
                        hC[b][i + j * ldc + (k - 1) * ldc * nb] = 0;
                }
            }
        }
    }

    // now copy data to the GPU
    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void geblttrf_npvt_getError(const rocblas_handle handle,
                            const rocblas_int nb,
                            const rocblas_int nblocks,
                            Td& dA,
                            const rocblas_int lda,
                            const rocblas_stride stA,
                            Td& dB,
                            const rocblas_int ldb,
                            const rocblas_stride stB,
                            Td& dC,
                            const rocblas_int ldc,
                            const rocblas_stride stC,
                            Ud& dInfo,
                            const rocblas_int bc,
                            Th& hA,
                            Th& hB,
                            Th& hBRes,
                            Th& hC,
                            Th& hCRes,
                            Uh& hInfo,
                            Uh& hInfoRes,
                            double* max_err,
                            const bool singular)
{
    int n = nb * nblocks;
    std::vector<T> L(n * n);
    std::vector<T> U(n * n);
    std::vector<T> M(n * n);
    std::vector<T> MRes(n * n);

    // input data initialization
    geblttrf_npvt_initData<true, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA,
                                          hB, hC, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda, stA,
                                                dB.data(), ldb, stB, dC.data(), ldc, stC,
                                                dInfo.data(), bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // check info for singularities
    double err = 0;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
        {
            EXPECT_GT(hInfoRes[b][0], 0) << "where b = " << b;
            if(hInfoRes[b][0] <= 0)
                err++;
        }
        else
        {
            EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;
            if(hInfoRes[b][0] != 0)
                err++;
        }
    }
    *max_err += err;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] == 0)
        {
            // compute diagonal blocks and store in full matrix L
            for(rocblas_int k = 0; k < nblocks; k++)
            {
                for(rocblas_int i = 0; i < nb; i++)
                {
                    for(rocblas_int j = 0; j < nb; j++)
                    {
                        if(i <= j)
                            L[i + j * n + k * (n + 1) * nb] = hBRes[b][i + j * ldb + k * ldb * nb];
                        else
                            L[i + j * n + k * (n + 1) * nb] = 0;
                    }
                }

                cpu_trmm(rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_unit, nb, nb, T(1), hBRes[b] + k * ldb * nb, ldb,
                         L.data() + k * (n + 1) * nb, n);
            }

            // move blocks A, updated C, and I into full matrices L and U
            for(rocblas_int k = 0; k < nblocks; k++)
            {
                for(rocblas_int i = 0; i < nb; i++)
                {
                    if(k < nblocks - 1)
                    {
                        for(rocblas_int j = 0; j < nb; j++)
                        {
                            U[i + (j + nb) * n + k * (n + 1) * nb]
                                = hCRes[b][i + j * ldc + k * ldc * nb];
                            L[(i + nb) + j * n + k * (n + 1) * nb]
                                = hA[b][i + j * lda + k * lda * nb];
                        }
                    }

                    U[i + i * n + k * (n + 1) * nb] = 1;
                }
            }

            // reconstruct input matrix from factors and store it in MRes
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, n, n, n, T(1), L.data(), n,
                     U.data(), n, T(0), MRes.data(), n);

            // form original matrix from original blocks
            for(rocblas_int k = 0; k < nblocks; k++)
            {
                for(rocblas_int i = 0; i < nb; i++)
                {
                    for(rocblas_int j = 0; j < nb; j++)
                    {
                        M[i + j * n + k * (n + 1) * nb] = hB[b][i + j * ldb + k * ldb * nb];

                        if(k < nblocks - 1)
                        {
                            M[(i + nb) + j * n + k * (n + 1) * nb]
                                = hA[b][i + j * lda + k * lda * nb];
                            M[i + (j + nb) * n + k * (n + 1) * nb]
                                = hC[b][i + j * ldc + k * ldc * nb];
                        }
                    }
                }
            }

            // error is ||M - MRes|| / ||M||
            // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
            // IT MIGHT BE REVISITED IN THE FUTURE)
            // using frobenius norm
            err = norm_error('F', n, n, n, M.data(), MRes.data());
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th>
void geblttrf_npvt_getPerfData(const rocblas_handle handle,
                               const rocblas_int nb,
                               const rocblas_int nblocks,
                               Td& dA,
                               const rocblas_int lda,
                               const rocblas_stride stA,
                               Td& dB,
                               const rocblas_int ldb,
                               const rocblas_stride stB,
                               Td& dC,
                               const rocblas_int ldc,
                               const rocblas_stride stC,
                               Ud& dInfo,
                               const rocblas_int bc,
                               Th& hA,
                               Th& hB,
                               Th& hC,
                               double* gpu_time_used,
                               double* cpu_time_used,
                               const rocblas_int hot_calls,
                               const int profile,
                               const bool profile_kernels,
                               const bool perf,
                               const bool singular)
{
    if(!perf)
    {
        // there is no direct CPU/LAPACK equivalent for this function, therefore
        // we return an invalid CPU time
        *cpu_time_used = nan("");
    }

    geblttrf_npvt_initData<true, false, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA,
                                           hB, hC, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrf_npvt_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc,
                                               hA, hB, hC, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda,
                                                    stA, dB.data(), ldb, stB, dC.data(), ldc, stC,
                                                    dInfo.data(), bc));
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
        geblttrf_npvt_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc,
                                               hA, hB, hC, singular);

        start = get_time_us_sync(stream);
        rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(), lda, stA, dB.data(), ldb,
                                stB, dC.data(), ldc, stC, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_geblttrf_npvt(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int lda = argus.get<rocblas_int>("lda", nb);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", nb);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nb);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * nb * nblocks);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nb * nblocks);
    rocblas_stride stC = argus.get<rocblas_stride>("strideC", ldc * nb * nblocks);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;
    rocblas_stride stCRes = (argus.unit_check || argus.norm_check) ? stC : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * nb * nblocks;
    size_t size_B = size_t(ldb) * nb * nblocks;
    size_t size_C = size_t(ldc) * nb * nblocks;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (nb < 0 || nblocks < 0 || lda < nb || ldb < nb || ldc < nb || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(
                rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, (T* const*)nullptr, lda, stA,
                                        (T* const*)nullptr, ldb, stB, (T* const*)nullptr, ldc, stC,
                                        (rocblas_int*)nullptr, bc),
                rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, (T*)nullptr,
                                                          lda, stA, (T*)nullptr, ldb, stB, (T*)nullptr,
                                                          ldc, stC, (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_geblttrf_npvt(
                STRIDED, handle, nb, nblocks, (T* const*)nullptr, lda, stA, (T* const*)nullptr, ldb,
                stB, (T* const*)nullptr, ldc, stC, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, (T*)nullptr,
                                                      lda, stA, (T*)nullptr, ldb, stB, (T*)nullptr,
                                                      ldc, stC, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hC(size_C, 1, bc);
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_batch_vector<T> hCRes(size_CRes, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_batch_vector<T> dC(size_C, 1, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(nb == 0 || nblocks == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(),
                                                          lda, stA, dB.data(), ldb, stB, dC.data(),
                                                          ldc, stC, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            geblttrf_npvt_getError<STRIDED, T>(handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC,
                                               ldc, stC, dInfo, bc, hA, hB, hBRes, hC, hCRes, hInfo,
                                               hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            geblttrf_npvt_getPerfData<STRIDED, T>(
                handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC, ldc, stC, dInfo, bc, hA, hB,
                hC, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hC(size_C, 1, stC, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<T> hCRes(size_CRes, 1, stCRes, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dC(size_C, 1, stC, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_C)
            CHECK_HIP_ERROR(dC.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(nb == 0 || nblocks == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt(STRIDED, handle, nb, nblocks, dA.data(),
                                                          lda, stA, dB.data(), ldb, stB, dC.data(),
                                                          ldc, stC, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            geblttrf_npvt_getError<STRIDED, T>(handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC,
                                               ldc, stC, dInfo, bc, hA, hB, hBRes, hC, hCRes, hInfo,
                                               hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            geblttrf_npvt_getPerfData<STRIDED, T>(
                handle, nb, nblocks, dA, lda, stA, dB, ldb, stB, dC, ldc, stC, dInfo, bc, hA, hB,
                hC, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                argus.perf, argus.singular);
    }

    // validate results for rocsolver-test
    // using nb * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, nb);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("nb", "nblocks", "lda", "ldb", "ldc", "batch_c");
                rocsolver_bench_output(nb, nblocks, lda, ldb, ldc, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("nb", "nblocks", "lda", "strideA", "ldb", "strideB", "ldc",
                                       "strideC", "batch_c");
                rocsolver_bench_output(nb, nblocks, lda, stA, ldb, stB, ldc, stC, bc);
            }
            else
            {
                rocsolver_bench_output("nb", "nblocks", "lda", "ldb", "ldc");
                rocsolver_bench_output(nb, nblocks, lda, ldb, ldc);
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

#define EXTERN_TESTING_GEBLTTRF_NPVT(...) \
    extern template void testing_geblttrf_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRF_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
