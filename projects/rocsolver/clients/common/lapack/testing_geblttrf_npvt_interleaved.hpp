/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename T, typename U>
void geblttrf_npvt_interleaved_checkBadArgs(const rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            T dA,
                                            const rocblas_int inca,
                                            const rocblas_int lda,
                                            const rocblas_stride stA,
                                            T dB,
                                            const rocblas_int incb,
                                            const rocblas_int ldb,
                                            const rocblas_stride stB,
                                            T dC,
                                            const rocblas_int incc,
                                            const rocblas_int ldc,
                                            const rocblas_stride stC,
                                            U dInfo,
                                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(nullptr, nb, nblocks, dA, inca, lda,
                                                              stA, dB, incb, ldb, stB, dC, incc,
                                                              ldc, stC, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, inca, lda,
                                                              stA, dB, incb, ldb, stB, dC, incc,
                                                              ldc, stC, dInfo, -1),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T) nullptr,
                                                              inca, lda, stA, dB, incb, ldb, stB,
                                                              dC, incc, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, inca, lda,
                                                              stA, (T) nullptr, incb, ldb, stB, dC,
                                                              incc, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, inca, lda,
                                                              stA, dB, incb, ldb, stB, (T) nullptr,
                                                              incc, ldc, stC, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, inca, lda,
                                                              stA, dB, incb, ldb, stB, dC, incc,
                                                              ldc, stC, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, 0, nblocks, (T) nullptr, inca,
                                                              lda, stA, (T) nullptr, incb, ldb, stB,
                                                              (T) nullptr, incc, ldc, stC, dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_geblttrf_npvt_interleaved(handle, nb, 0, (T) nullptr, inca, lda, stA, (T) nullptr,
                                            incb, ldb, stB, (T) nullptr, incc, ldc, stC, dInfo, bc),
        rocblas_status_success);

    // quick return with zero batch_count if applicable
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, inca, lda,
                                                              stA, dB, incb, ldb, stB, dC, incc,
                                                              ldc, stC, (U) nullptr, 0),
                          rocblas_status_success);
}

template <typename T>
void testing_geblttrf_npvt_interleaved_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int nb = 1;
    rocblas_int nblocks = 2;
    rocblas_int inca = 1;
    rocblas_int incb = 1;
    rocblas_int incc = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldc = 1;
    rocblas_stride stA = 2;
    rocblas_stride stB = 2;
    rocblas_stride stC = 2;
    rocblas_int bc = 1;

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
    geblttrf_npvt_interleaved_checkBadArgs(handle, nb, nblocks, dA.data(), inca, lda, stA,
                                           dB.data(), incb, ldb, stB, dC.data(), incc, ldc, stC,
                                           dInfo.data(), bc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrf_npvt_interleaved_initData(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        Td& dA,
                                        const rocblas_int inca,
                                        const rocblas_int lda,
                                        const rocblas_stride stA,
                                        Td& dB,
                                        const rocblas_int incb,
                                        const rocblas_int ldb,
                                        const rocblas_stride stB,
                                        Td& dC,
                                        const rocblas_int incc,
                                        const rocblas_int ldc,
                                        const rocblas_stride stC,
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
            T* A = hA[0] + b * stA;
            T* B = hB[0] + b * stB;
            T* C = hC[0] + b * stC;

            // scale to avoid singularities
            // leaving matrix as diagonal dominant so that pivoting is not required
            for(rocblas_int i = 0; i < nb; i++)
            {
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int k = 0; k < nblocks; k++)
                    {
                        if(i == j)
                            B[i * incb + j * ldb + k * ldb * nb] += 400;
                        else
                            B[i * incb + j * ldb + k * ldb * nb] -= 4;
                    }

                    for(rocblas_int k = 0; k < nblocks - 1; k++)
                    {
                        A[i * inca + j * lda + k * lda * nb] -= 4;
                        C[i * incc + j * ldc + k * ldc * nb] -= 4;
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
                    B[i * incb + j * ldb + k * ldb * nb] = 0;
                    if(k < nblocks - 1)
                        A[i * inca + j * lda + k * lda * nb] = 0;
                    if(k > 0)
                        C[i * incc + j * ldc + (k - 1) * ldc * nb] = 0;
                }

                jj = n / 2 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    B[i * incb + j * ldb + k * ldb * nb] = 0;
                    if(k < nblocks - 1)
                        A[i * inca + j * lda + k * lda * nb] = 0;
                    if(k > 0)
                        C[i * incc + j * ldc + (k - 1) * ldc * nb] = 0;
                }

                jj = n - 1 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    B[i * incb + j * ldb + k * ldb * nb] = 0;
                    if(k < nblocks - 1)
                        A[i * inca + j * lda + k * lda * nb] = 0;
                    if(k > 0)
                        C[i * incc + j * ldc + (k - 1) * ldc * nb] = 0;
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

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void geblttrf_npvt_interleaved_getError(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        Td& dA,
                                        const rocblas_int inca,
                                        const rocblas_int lda,
                                        const rocblas_stride stA,
                                        Td& dB,
                                        const rocblas_int incb,
                                        const rocblas_int ldb,
                                        const rocblas_stride stB,
                                        Td& dC,
                                        const rocblas_int incc,
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
    std::vector<T> Btmp(nb * n);
    std::vector<T> L(n * n);
    std::vector<T> U(n * n);
    std::vector<T> M(n * n);
    std::vector<T> MRes(n * n);

    // input data initialization
    geblttrf_npvt_interleaved_initData<true, true, T>(handle, nb, nblocks, dA, inca, lda, stA, dB,
                                                      incb, ldb, stB, dC, incc, ldc, stC, bc, hA,
                                                      hB, hC, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt_interleaved(
        handle, nb, nblocks, dA.data(), inca, lda, stA, dB.data(), incb, ldb, stB, dC.data(), incc,
        ldc, stC, dInfo.data(), bc));
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
                        Btmp[i + j * nb + k * nb * nb]
                            = hBRes[0][i * incb + j * ldb + k * ldb * nb + b * stB];

                        if(i <= j)
                            L[i + j * n + k * (n + 1) * nb] = Btmp[i + j * nb + k * nb * nb];
                        else
                            L[i + j * n + k * (n + 1) * nb] = 0;
                    }
                }

                cpu_trmm(rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                         rocblas_diagonal_unit, nb, nb, T(1), Btmp.data() + k * nb * nb, nb,
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
                                = hCRes[0][i * incc + j * ldc + k * ldc * nb + b * stC];
                            L[(i + nb) + j * n + k * (n + 1) * nb]
                                = hA[0][i * inca + j * lda + k * lda * nb + b * stA];
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
                        M[i + j * n + k * (n + 1) * nb]
                            = hB[0][i * incb + j * ldb + k * ldb * nb + b * stB];

                        if(k < nblocks - 1)
                        {
                            M[(i + nb) + j * n + k * (n + 1) * nb]
                                = hA[0][i * inca + j * lda + k * lda * nb + b * stA];
                            M[i + (j + nb) * n + k * (n + 1) * nb]
                                = hC[0][i * incc + j * ldc + k * ldc * nb + b * stC];
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

template <typename T, typename Td, typename Ud, typename Th>
void geblttrf_npvt_interleaved_getPerfData(const rocblas_handle handle,
                                           const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           Td& dA,
                                           const rocblas_int inca,
                                           const rocblas_int lda,
                                           const rocblas_stride stA,
                                           Td& dB,
                                           const rocblas_int incb,
                                           const rocblas_int ldb,
                                           const rocblas_stride stB,
                                           Td& dC,
                                           const rocblas_int incc,
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

    geblttrf_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, dA, inca, lda, stA, dB,
                                                       incb, ldb, stB, dC, incc, ldc, stC, bc, hA,
                                                       hB, hC, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrf_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, dA, inca, lda, stA,
                                                           dB, incb, ldb, stB, dC, incc, ldc, stC,
                                                           bc, hA, hB, hC, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt_interleaved(
            handle, nb, nblocks, dA.data(), inca, lda, stA, dB.data(), incb, ldb, stB, dC.data(),
            incc, ldc, stC, dInfo.data(), bc));
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
        geblttrf_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, dA, inca, lda, stA,
                                                           dB, incb, ldb, stB, dC, incc, ldc, stC,
                                                           bc, hA, hB, hC, singular);

        start = get_time_us_sync(stream);
        rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA.data(), inca, lda, stA,
                                            dB.data(), incb, ldb, stB, dC.data(), incc, ldc, stC,
                                            dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_geblttrf_npvt_interleaved(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int inca = argus.get<rocblas_int>("inca", 1);
    rocblas_int incb = argus.get<rocblas_int>("incb", 1);
    rocblas_int incc = argus.get<rocblas_int>("incc", 1);
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
    rocblas_int n = nb * nblocks;
    size_t size_A = std::max(size_t(lda) * n, size_t(stA)) * bc;
    size_t size_B = std::max(size_t(ldb) * n, size_t(stB)) * bc;
    size_t size_C = std::max(size_t(ldc) * n, size_t(stC)) * bc;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_a = (inca < 1 || lda < inca * nb);
    bool invalid_b = (incc < 1 || ldc < incc * nb);
    bool invalid_c = (incc < 1 || ldc < incc * nb);
    bool invalid_size = (nb < 0 || nblocks < 0 || bc < 0 || invalid_a || invalid_b || invalid_c);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T*)nullptr,
                                                                  inca, lda, stA, (T*)nullptr, incb,
                                                                  ldb, stB, (T*)nullptr, incc, ldc,
                                                                  stC, (rocblas_int*)nullptr, bc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_geblttrf_npvt_interleaved(
            handle, nb, nblocks, (T*)nullptr, inca, lda, stA, (T*)nullptr, incb, ldb, stB,
            (T*)nullptr, incc, ldc, stC, (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hB(size_B, 1, size_B, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hBRes(size_BRes, 1, size_BRes, 1);
    host_strided_batch_vector<T> hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dB(size_B, 1, size_B, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
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
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(
                                  handle, nb, nblocks, dA.data(), inca, lda, stA, dB.data(), incb,
                                  ldb, stB, dC.data(), incc, ldc, stC, dInfo.data(), bc),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geblttrf_npvt_interleaved_getError<T>(handle, nb, nblocks, dA, inca, lda, stA, dB, incb, ldb,
                                              stB, dC, incc, ldc, stC, dInfo, bc, hA, hB, hBRes, hC,
                                              hCRes, hInfo, hInfoRes, &max_error, argus.singular);

    // collect performance data
    if(argus.timing)
        geblttrf_npvt_interleaved_getPerfData<T>(
            handle, nb, nblocks, dA, inca, lda, stA, dB, incb, ldb, stB, dC, incc, ldc, stC, dInfo,
            bc, hA, hB, hC, &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
            argus.profile_kernels, argus.perf, argus.singular);

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
            rocsolver_bench_output("nb", "nblocks", "inca", "lda", "strideA", "incb", "ldb",
                                   "strideB", "incc", "ldc", "strideC", "batch_c");
            rocsolver_bench_output(nb, nblocks, inca, lda, stA, incb, ldb, stB, incc, ldc, stC, bc);
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

#define EXTERN_TESTING_GEBLTTRF_NPVT_INTERLEAVED(...) \
    extern template void testing_geblttrf_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRF_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
