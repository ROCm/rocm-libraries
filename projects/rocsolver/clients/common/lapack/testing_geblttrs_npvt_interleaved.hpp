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

template <typename T>
void geblttrs_npvt_interleaved_checkBadArgs(const rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            const rocblas_int nrhs,
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
                                            T dX,
                                            const rocblas_int incx,
                                            const rocblas_int ldx,
                                            const rocblas_stride stX,
                                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(nullptr, nb, nblocks, nrhs, dA, inca,
                                                              lda, stA, dB, incb, ldb, stB, dC, incc,
                                                              ldc, stC, dX, incx, ldx, stX, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, inca,
                                                              lda, stA, dB, incb, ldb, stB, dC, incc,
                                                              ldc, stC, dX, incx, ldx, stX, -1),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(
                              handle, nb, nblocks, nrhs, (T) nullptr, inca, lda, stA, dB, incb, ldb,
                              stB, dC, incc, ldc, stC, dX, incx, ldx, stX, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(
                              handle, nb, nblocks, nrhs, dA, inca, lda, stA, (T) nullptr, incb, ldb,
                              stB, dC, incc, ldc, stC, dX, incx, ldx, stX, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(
                              handle, nb, nblocks, nrhs, dA, inca, lda, stA, dB, incb, ldb, stB,
                              (T) nullptr, incc, ldc, stC, dX, incx, ldx, stX, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(
                              handle, nb, nblocks, nrhs, dA, inca, lda, stA, dB, incb, ldb, stB, dC,
                              incc, ldc, stC, (T) nullptr, incx, ldx, stX, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, 0, nblocks, nrhs, (T) nullptr,
                                                              inca, lda, stA, (T) nullptr, incb,
                                                              ldb, stB, (T) nullptr, incc, ldc, stC,
                                                              (T) nullptr, incx, ldx, stX, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, 0, nrhs, (T) nullptr,
                                                              inca, lda, stA, (T) nullptr, incb,
                                                              ldb, stB, (T) nullptr, incc, ldc, stC,
                                                              (T) nullptr, incx, ldx, stX, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, 0, dA, inca, lda,
                                                              stA, dB, incb, ldb, stB, dC, incc, ldc,
                                                              stC, (T) nullptr, incx, ldx, stX, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, inca,
                                                              lda, stA, dB, incb, ldb, stB, dC,
                                                              incc, ldc, stC, dX, incx, ldx, stX, 0),
                          rocblas_status_success);
}

template <typename T>
void testing_geblttrs_npvt_interleaved_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int nb = 1;
    rocblas_int nblocks = 2;
    rocblas_int nrhs = 1;
    rocblas_int inca = 1;
    rocblas_int incb = 1;
    rocblas_int incc = 1;
    rocblas_int incx = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldc = 1;
    rocblas_int ldx = 1;
    rocblas_stride stA = 2;
    rocblas_stride stB = 2;
    rocblas_stride stC = 2;
    rocblas_stride stX = 2;
    rocblas_int bc = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dB(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<T> dX(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dX.memcheck());

    // check bad arguments
    geblttrs_npvt_interleaved_checkBadArgs(handle, nb, nblocks, nrhs, dA.data(), inca, lda, stA,
                                           dB.data(), incb, ldb, stB, dC.data(), incc, ldc, stC,
                                           dX.data(), incx, ldx, stX, bc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_initData(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
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
                                        Td& dX,
                                        const rocblas_int incx,
                                        const rocblas_int ldx,
                                        const rocblas_stride stX,
                                        const rocblas_int bc,
                                        Th& hA,
                                        Th& hB,
                                        Th& hC,
                                        Th& hX,
                                        Th& hRHS)
{
    if(CPU)
    {
        int info;
        int n = nb * nblocks;
        std::vector<T> M(n * n);
        std::vector<T> XX(n * nrhs);
        std::vector<T> XB(n * nrhs);
        std::vector<rocblas_int> ipiv(nb);

        // initialize blocks of the original matrix
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);
        rocblas_init<T>(hC, false);

        // initialize solution vectors
        rocblas_init<T>(hX, false);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            T* A = hA[0] + b * stA;
            T* B = hB[0] + b * stB;
            T* C = hC[0] + b * stC;
            T* X = hX[0] + b * stX;
            T* RHS = hRHS[0] + b * stX;

            // form original matrix M and scale to avoid singularities
            for(rocblas_int k = 0; k < nblocks; k++)
            {
                for(rocblas_int i = 0; i < nb; i++)
                {
                    for(rocblas_int j = 0; j < nb; j++)
                    {
                        if(i == j)
                            M[i + j * n + k * (n + 1) * nb]
                                = B[i * incb + j * ldb + k * ldb * nb] + 400;
                        else
                            M[i + j * n + k * (n + 1) * nb]
                                = B[i * incb + j * ldb + k * ldb * nb] - 4;

                        if(k < nblocks - 1)
                        {
                            M[(i + nb) + j * n + k * (n + 1) * nb]
                                = A[i * inca + j * lda + k * lda * nb] - 4;
                            M[i + (j + nb) * n + k * (n + 1) * nb]
                                = C[i * incc + j * ldc + k * ldc * nb] - 4;
                        }
                    }
                }
            }

            // move blocks of X to full matrix XX
            for(rocblas_int k = 0; k < nblocks; k++)
                for(rocblas_int i = 0; i < nb; i++)
                    for(rocblas_int j = 0; j < nrhs; j++)
                        XX[i + j * n + k * nb] = X[i * incx + j * ldx + k * ldx * nrhs];

            // generate the full matrix of right-hand-side vectors XB by computing M * XX
            cpu_gemm(rocblas_operation_none, rocblas_operation_none, n, nrhs, n, T(1), M.data(), n,
                     XX.data(), n, T(0), XB.data(), n);

            // move XB to block format in hRHS
            for(rocblas_int k = 0; k < nblocks; k++)
                for(rocblas_int i = 0; i < nb; i++)
                    for(rocblas_int j = 0; j < nrhs; j++)
                        RHS[i * incx + j * ldx + k * ldx * nrhs] = XB[i + j * n + k * nb];

            // factorize M
            cpu_getrf(nb, nb, M.data(), n, ipiv.data(), &info);
            for(rocblas_int k = 0; k < nblocks - 1; k++)
            {
                cpu_getrs(rocblas_operation_none, nb, nb, M.data() + k * (n + 1) * nb, n,
                          ipiv.data(), M.data() + nb * n + k * (n + 1) * nb, n);

                cpu_gemm(rocblas_operation_none, rocblas_operation_none, nb, nb, nb, T(-1),
                         M.data() + nb + k * (n + 1) * nb, n, M.data() + nb * n + k * (n + 1) * nb,
                         n, T(1), M.data() + (k + 1) * (n + 1) * nb, n);

                cpu_getrf(nb, nb, M.data() + (k + 1) * (n + 1) * nb, n, ipiv.data(), &info);
            }

            // move factorized blocks from M into hA, hB, and hC
            for(rocblas_int k = 0; k < nblocks; k++)
            {
                for(rocblas_int i = 0; i < nb; i++)
                {
                    for(rocblas_int j = 0; j < nb; j++)
                    {
                        B[i * incb + j * ldb + k * ldb * nb] = M[i + j * n + k * (n + 1) * nb];

                        if(k < nblocks - 1)
                        {
                            A[i * inca + j * lda + k * lda * nb]
                                = M[(i + nb) + j * n + k * (n + 1) * nb];
                            C[i * incc + j * ldc + k * ldc * nb]
                                = M[i + (j + nb) * n + k * (n + 1) * nb];
                        }
                    }
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
        CHECK_HIP_ERROR(dX.transfer_from(hRHS));
    }
}

template <typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_getError(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
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
                                        Td& dX,
                                        const rocblas_int incx,
                                        const rocblas_int ldx,
                                        const rocblas_stride stX,
                                        const rocblas_int bc,
                                        Th& hA,
                                        Th& hB,
                                        Th& hC,
                                        Th& hX,
                                        Th& hXRes,
                                        double* max_err)
{
    std::vector<T> Xtmp(nb * nrhs * nblocks);
    std::vector<T> XtmpRes(nb * nrhs * nblocks);

    // input data initialization
    geblttrs_npvt_interleaved_initData<true, true, T>(handle, nb, nblocks, nrhs, dA, inca, lda, stA,
                                                      dB, incb, ldb, stB, dC, incc, ldc, stC, dX,
                                                      incx, ldx, stX, bc, hA, hB, hC, hX, hXRes);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt_interleaved(
        handle, nb, nblocks, nrhs, dA.data(), inca, lda, stA, dB.data(), incb, ldb, stB, dC.data(),
        incc, ldc, stC, dX.data(), incx, ldx, stX, bc));
    CHECK_HIP_ERROR(hXRes.transfer_from(dX));

    double err = 0;
    *max_err = 0;

    // error is ||hX - hXRes|| / ||hX||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // put X and XRes into Xtmp and XtmpRes in column-major format
        for(rocblas_int k = 0; k < nblocks; k++)
        {
            for(rocblas_int i = 0; i < nb; i++)
            {
                for(rocblas_int j = 0; j < nrhs; j++)
                {
                    Xtmp[i + j * nb + k * nb * nrhs]
                        = hX[0][i * incx + j * ldx + k * ldx * nrhs + b * stX];
                    XtmpRes[i + j * nb + k * nb * nrhs]
                        = hXRes[0][i * incx + j * ldx + k * ldx * nrhs + b * stX];
                }
            }
        }

        err = norm_error('F', nb, nrhs * nblocks, nb, Xtmp.data(), XtmpRes.data());
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_getPerfData(const rocblas_handle handle,
                                           const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           const rocblas_int nrhs,
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
                                           Td& dX,
                                           const rocblas_int incx,
                                           const rocblas_int ldx,
                                           const rocblas_stride stX,
                                           const rocblas_int bc,
                                           Th& hA,
                                           Th& hB,
                                           Th& hC,
                                           Th& hX,
                                           Th& hXRes,
                                           double* gpu_time_used,
                                           double* cpu_time_used,
                                           const rocblas_int hot_calls,
                                           const int profile,
                                           const bool profile_kernels,
                                           const bool perf)
{
    if(!perf)
    {
        // there is no direct CPU/LAPACK equivalent for this function, therefore
        // we return an invalid CPU time
        *cpu_time_used = nan("");
    }

    geblttrs_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, nrhs, dA, inca, lda, stA,
                                                       dB, incb, ldb, stB, dC, incc, ldc, stC, dX,
                                                       incx, ldx, stX, bc, hA, hB, hC, hX, hXRes);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrs_npvt_interleaved_initData<false, true, T>(
            handle, nb, nblocks, nrhs, dA, inca, lda, stA, dB, incb, ldb, stB, dC, incc, ldc, stC,
            dX, incx, ldx, stX, bc, hA, hB, hC, hX, hXRes);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt_interleaved(
            handle, nb, nblocks, nrhs, dA.data(), inca, lda, stA, dB.data(), incb, ldb, stB,
            dC.data(), incc, ldc, stC, dX.data(), incx, ldx, stX, bc));
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
        geblttrs_npvt_interleaved_initData<false, true, T>(
            handle, nb, nblocks, nrhs, dA, inca, lda, stA, dB, incb, ldb, stB, dC, incc, ldc, stC,
            dX, incx, ldx, stX, bc, hA, hB, hC, hX, hXRes);

        start = get_time_us_sync(stream);
        rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(), inca, lda, stA,
                                            dB.data(), incb, ldb, stB, dC.data(), incc, ldc, stC,
                                            dX.data(), incx, ldx, stX, bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_geblttrs_npvt_interleaved(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs");
    rocblas_int inca = argus.get<rocblas_int>("inca", 1);
    rocblas_int incb = argus.get<rocblas_int>("incb", 1);
    rocblas_int incc = argus.get<rocblas_int>("incc", 1);
    rocblas_int incx = argus.get<rocblas_int>("incx", 1);
    rocblas_int lda = argus.get<rocblas_int>("lda", nb);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", nb);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nb);
    rocblas_int ldx = argus.get<rocblas_int>("ldx", nb);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * nb * nblocks);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nb * nblocks);
    rocblas_stride stC = argus.get<rocblas_stride>("strideC", ldc * nb * nblocks);
    rocblas_stride stX = argus.get<rocblas_stride>("strideX", ldx * nrhs * nblocks);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stXRes = stX;

    // check non-supported values
    // N/A

    // determine sizes
    rocblas_int n = nb * nblocks;
    size_t size_A = std::max(size_t(lda) * n, size_t(stA)) * bc;
    size_t size_B = std::max(size_t(ldb) * n, size_t(stB)) * bc;
    size_t size_C = std::max(size_t(ldc) * n, size_t(stC)) * bc;
    size_t size_X = std::max(size_t(ldx) * nrhs * nblocks, size_t(stX)) * bc;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_XRes = size_X;

    // check invalid sizes
    bool invalid_a = (inca < 1 || lda < inca * nb);
    bool invalid_b = (incc < 1 || ldc < incc * nb);
    bool invalid_c = (incc < 1 || ldc < incc * nb);
    bool invalid_x = (incx < 1 || ldx < incx * nb);
    bool invalid_size = (nb < 0 || nblocks < 0 || nrhs < 0 || bc < 0 || invalid_a || invalid_b
                         || invalid_c || invalid_x);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, (T*)nullptr, inca, lda,
                                                stA, (T*)nullptr, incb, ldb, stB, (T*)nullptr, incc,
                                                ldc, stC, (T*)nullptr, incx, ldx, stX, bc),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_geblttrs_npvt_interleaved(
            handle, nb, nblocks, nrhs, (T*)nullptr, inca, lda, stA, (T*)nullptr, incb, ldb, stB,
            (T*)nullptr, incc, ldc, stC, (T*)nullptr, incx, ldx, stX, bc));

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
    host_strided_batch_vector<T> hX(size_X, 1, size_X, 1);
    host_strided_batch_vector<T> hXRes(size_XRes, 1, size_XRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dB(size_B, 1, size_B, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T> dX(size_X, 1, size_X, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_B)
        CHECK_HIP_ERROR(dB.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    if(size_X)
        CHECK_HIP_ERROR(dX.memcheck());

    // check quick return
    if(nb == 0 || nblocks == 0 || nrhs == 0 || bc == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(), inca, lda,
                                                stA, dB.data(), incb, ldb, stB, dC.data(), incc,
                                                ldc, stC, dX.data(), incx, ldx, stX, bc),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geblttrs_npvt_interleaved_getError<T>(handle, nb, nblocks, nrhs, dA, inca, lda, stA, dB,
                                              incb, ldb, stB, dC, incc, ldc, stC, dX, incx, ldx,
                                              stX, bc, hA, hB, hC, hX, hXRes, &max_error);

    // collect performance data
    if(argus.timing)
        geblttrs_npvt_interleaved_getPerfData<T>(
            handle, nb, nblocks, nrhs, dA, inca, lda, stA, dB, incb, ldb, stB, dC, incc, ldc, stC,
            dX, incx, ldx, stX, bc, hA, hB, hC, hX, hXRes, &gpu_time_used, &cpu_time_used,
            hot_calls, argus.profile, argus.profile_kernels, argus.perf);

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
            rocsolver_bench_output("nb", "nblocks", "nrhs", "inca", "lda", "strideA", "incb", "ldb",
                                   "strideB", "incc", "ldc", "strideC", "incx", "ldx", "strideX",
                                   "batch_c");
            rocsolver_bench_output(nb, nblocks, nrhs, inca, lda, stA, incb, ldb, stB, incc, ldc,
                                   stC, incx, ldx, stX, bc);
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

#define EXTERN_TESTING_GEBLTTRS_NPVT_INTERLEAVED(...) \
    extern template void testing_geblttrs_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRS_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
