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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <bool STRIDED, typename I, typename Td, typename Id>
void sytrs_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const I n,
                        const I nrhs,
                        Td dA,
                        const I lda,
                        const rocblas_stride stA,
                        Id dIpiv,
                        const rocblas_stride stP,
                        Td dB,
                        const I ldb,
                        const rocblas_stride stB,
                        const I bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, nullptr, uplo, n, nrhs, dA, lda, stA, dIpiv, stP,
                                          dB, ldb, stB, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, rocblas_fill(0), n, nrhs, dA, lda, stA,
                                          dIpiv, stP, dB, ldb, stB, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dIpiv,
                                              stP, dB, ldb, stB, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, (Td) nullptr, lda, stA,
                                          dIpiv, stP, dB, ldb, stB, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA, lda, stA,
                                          (Id) nullptr, stP, dB, ldb, stB, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP,
                                          (Td) nullptr, ldb, stB, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, 0, nrhs, (Td) nullptr, lda, stA,
                                          (Id) nullptr, stP, (Td) nullptr, ldb, stB, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, 0, dA, lda, stA, dIpiv, stP,
                                          (Td) nullptr, ldb, stB, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA, lda, stA, dIpiv,
                                              stP, dB, ldb, stB, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_sytrs_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    I n = 1;
    I nrhs = 1;
    I lda = 1;
    I ldb = 1;
    rocblas_stride stA = 1;
    rocblas_stride stP = 1;
    rocblas_stride stB = 1;
    I bc = 1;
    rocblas_fill uplo = rocblas_fill_lower;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_strided_batch_vector<I> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        sytrs_checkBadArgs<STRIDED>(handle, uplo, n, nrhs, dA.data(), lda, stA, dIpiv.data(), stP,
                                    dB.data(), ldb, stB, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<I> dIpiv(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());

        // check bad arguments
        sytrs_checkBadArgs<STRIDED>(handle, uplo, n, nrhs, dA.data(), lda, stA, dIpiv.data(), stP,
                                    dB.data(), ldb, stB, bc);
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Id, typename Th, typename Ih, typename Uh>
void sytrs_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const I n,
                    const I nrhs,
                    Td& dA,
                    const I lda,
                    const rocblas_stride stA,
                    Id& dIpiv,
                    const rocblas_stride stP,
                    Td& dB,
                    const I ldb,
                    const rocblas_stride stB,
                    const I bc,
                    Th& hA,
                    Ih& hIpiv,
                    Uh& hIpiv_cpu,
                    Th& hB)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);

        for(I b = 0; b < bc; ++b)
        {
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

            // shuffle rows to test pivoting
            // always the same permutation for debugging purposes
            for(rocblas_int i = 0; i < n / 2; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    std::swap(hA[b][i + j * lda], hA[b][n - 1 - i + j * lda]);
                }
            }

        } // end for b

        // do the symmetric decomposition of matrix A w/ the reference LAPACK routine
        for(I b = 0; b < bc; ++b)
        {
            int info = 0;
            int lwork = lda * n;
            std::vector<T> work(lwork);

            cpu_sytrf(uplo, n, hA[b], lda, hIpiv_cpu[b], work.data(), lwork, &info);
            assert(info == 0);

            for(I i = 0; i < n; i++)
            {
                hIpiv[b][i] = hIpiv_cpu[b][i];
            }
        }
    }

    if(GPU)
    {
        // now copy pivoting indices and matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <bool STRIDED, typename T, typename I, typename Td, typename Id, typename Th, typename Ih, typename Uh>
void sytrs_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const I n,
                    const I nrhs,
                    Td& dA,
                    const I lda,
                    const rocblas_stride stA,
                    Id& dIpiv,
                    const rocblas_stride stP,
                    Td& dB,
                    const I ldb,
                    const rocblas_stride stB,
                    const I bc,
                    Th& hA,
                    Ih& hIpiv,
                    Uh& hIpiv_cpu,
                    Th& hB,
                    Th& hBRes,
                    double* max_err)
{
    // input data initialization
    sytrs_initData<true, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc,
                                  hA, hIpiv, hIpiv_cpu, hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA,
                                        dIpiv.data(), stP, dB.data(), ldb, stB, bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));

    // CPU lapack
    for(I b = 0; b < bc; ++b)
    {
        cpu_sytrs(uplo, n, nrhs, hA[b], lda, hIpiv_cpu[b], hB[b], ldb);
    }

    // error is ||hB - hBRes|| / ||hB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(I b = 0; b < bc; ++b)
    {
        err = norm_error('I', n, nrhs, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, typename T, typename I, typename Td, typename Id, typename Th, typename Ih, typename Uh>
void sytrs_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const I n,
                       const I nrhs,
                       Td& dA,
                       const I lda,
                       const rocblas_stride stA,
                       Id& dIpiv,
                       const rocblas_stride stP,
                       Td& dB,
                       const I ldb,
                       const rocblas_stride stB,
                       const I bc,
                       Th& hA,
                       Ih& hIpiv,
                       Uh& hIpiv_cpu,
                       Th& hB,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        sytrs_initData<true, false, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                       stB, bc, hA, hIpiv, hIpiv_cpu, hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(I b = 0; b < bc; ++b)
        {
            cpu_sytrs(uplo, n, nrhs, hA[b], lda, hIpiv_cpu[b], hB[b], ldb);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sytrs_initData<true, false, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB,
                                   bc, hA, hIpiv, hIpiv_cpu, hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sytrs_initData<false, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                       stB, bc, hA, hIpiv, hIpiv_cpu, hB);

        CHECK_ROCBLAS_ERROR(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA,
                                            dIpiv.data(), stP, dB.data(), ldb, stB, bc));
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

    for(int iter = 0; iter < hot_calls; iter++)
    {
        sytrs_initData<false, true, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                       stB, bc, hA, hIpiv, hIpiv_cpu, hB);

        timer.start(stream);
        rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda, stA, dIpiv.data(), stP,
                        dB.data(), ldb, stB, bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_sytrs(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    I n = argus.get<rocblas_int>("n");
    I nrhs = argus.get<rocblas_int>("nrhs", n);
    I lda = argus.get<rocblas_int>("lda", n);
    I ldb = argus.get<rocblas_int>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nrhs);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    I bc = argus.batch_count;
    int hot_calls = argus.iters;

    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;

    // check non-supported values
    if((uplo != rocblas_fill_upper) && (uplo != rocblas_fill_lower))
    {
        if(BATCHED)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs,
                                                  (T* const*)nullptr, lda, stA, (I*)nullptr, stP,
                                                  (T* const*)nullptr, ldb, stB, bc),
                                  rocblas_status_invalid_value);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda,
                                                  stA, (I*)nullptr, stP, (T*)nullptr, ldb, stB, bc),
                                  rocblas_status_invalid_value);
        }
        if(argus.timing)
        {
            rocsolver_bench_inform(inform_invalid_args);
        }

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * nrhs;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs,
                                                  (T* const*)nullptr, lda, stA, (I*)nullptr, stP,
                                                  (T* const*)nullptr, ldb, stB, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda,
                                                  stA, (I*)nullptr, stP, (T*)nullptr, ldb, stB, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, (T* const*)nullptr,
                                              lda, stA, (I*)nullptr, stP, (T* const*)nullptr, ldb,
                                              stB, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, (T*)nullptr, lda, stA,
                                              (I*)nullptr, stP, (T*)nullptr, ldb, stB, bc));

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
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        host_strided_batch_vector<I> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpiv_cpu(size_P, 1, stP, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        device_strided_batch_vector<I> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda,
                                                  stA, dIpiv.data(), stP, dB.data(), ldb, stB, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytrs_getError<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                       stB, bc, hA, hIpiv, hIpiv_cpu, hB, hBRes, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            sytrs_getPerfData<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                          stB, bc, hA, hIpiv, hIpiv_cpu, hB, &gpu_time_used,
                                          &cpu_time_used, hot_calls, argus.profile,
                                          argus.profile_kernels, argus.perf);
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<I> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hIpiv_cpu(size_P, 1, stP, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        device_strided_batch_vector<I> dIpiv(size_P, 1, stP, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        // check quick return
        if(n == 0 || nrhs == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_sytrs(STRIDED, handle, uplo, n, nrhs, dA.data(), lda,
                                                  stA, dIpiv.data(), stP, dB.data(), ldb, stB, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            sytrs_getError<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                       stB, bc, hA, hIpiv, hIpiv_cpu, hB, hBRes, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            sytrs_getPerfData<STRIDED, T>(handle, uplo, n, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb,
                                          stB, bc, hA, hIpiv, hIpiv_cpu, hB, &gpu_time_used,
                                          &cpu_time_used, hot_calls, argus.profile,
                                          argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

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

#define EXTERN_TESTING_SYTRS(...) extern template void testing_sytrs<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SYTRS,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
