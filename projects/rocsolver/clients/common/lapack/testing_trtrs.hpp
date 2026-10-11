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

template <bool STRIDED, typename T, typename I, typename U>
void trtrs_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_operation trans,
                        const rocblas_diagonal diag,
                        const I n,
                        const I nrhs,
                        T dA,
                        const I lda,
                        const rocblas_stride stA,
                        T dB,
                        const I ldb,
                        const rocblas_stride stB,
                        U dInfo,
                        const I bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, nullptr, uplo, trans, diag, n, nrhs, dA, lda,
                                          stA, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, rocblas_fill_full, trans, diag, n, nrhs,
                                          dA, lda, stA, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, rocblas_operation(0), diag, n,
                                          nrhs, dA, lda, stA, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, rocblas_diagonal(0), n,
                                          nrhs, dA, lda, stA, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, dA, lda,
                                              stA, dB, ldb, stB, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, (T) nullptr,
                                          lda, stA, dB, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, dA, lda, stA,
                                          (T) nullptr, ldb, stB, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, dA, lda, stA,
                                          dB, ldb, stB, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, 0, nrhs, (T) nullptr,
                                          lda, stA, (T) nullptr, ldb, stB, dInfo, bc),
                          rocblas_status_success);
    // (B is not needed when nrhs = 0, but the diagonal of A is still checked)
    EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, 0, dA, lda, stA,
                                          (T) nullptr, ldb, stB, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, dA, lda,
                                              stA, dB, ldb, stB, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_trtrs_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    I n = 1;
    I nrhs = 1;
    I lda = 1;
    I ldb = 1;
    rocblas_stride stA = 1;
    rocblas_stride stB = 1;
    I bc = 1;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_diagonal diag = rocblas_diagonal_non_unit;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_batch_vector<T> dB(1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        trtrs_checkBadArgs<STRIDED>(handle, uplo, trans, diag, n, nrhs, dA.data(), lda, stA,
                                    dB.data(), ldb, stB, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dB(1, 1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        trtrs_checkBadArgs<STRIDED>(handle, uplo, trans, diag, n, nrhs, dA.data(), lda, stA,
                                    dB.data(), ldb, stB, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void trtrs_initData(const rocblas_handle handle,
                    const I n,
                    const I nrhs,
                    Td& dA,
                    const I lda,
                    Td& dB,
                    const I ldb,
                    const I bc,
                    Th& hA,
                    Th& hB,
                    const bool singular)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);

        for(I b = 0; b < bc; ++b)
        {
            // make A diagonally dominant (and well conditioned): the diagonal entries are at
            // least 1.1 in absolute value and the off-diagonal entries of each row and column
            // add up to less than 0.75 in absolute value
            for(I j = 0; j < n; j++)
            {
                for(I i = 0; i < n; i++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = hA[b][i + j * lda] / 10.0 + 1;
                    else
                    {
                        hA[b][i + j * lda] = hA[b][i + j * lda] / (20.0 * n);
                        if((i + 2 * j) % 3 == 1)
                            hA[b][i + j * lda] = -hA[b][i + j * lda];
                    }
                }
            }

            if(singular && b == bc / 2)
            {
                // add some singularities in one instance of the batch
                // (always the same elements for debugging purposes);
                // if the diagonal is referenced, the first zero must be reported and the
                // right-hand sides of this instance must not be modified
                I k = n / 2;
                hA[b][k + k * lda] = 0;
                k = n - 1;
                hA[b][k + k * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        if(nrhs > 0)
            CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool STRIDED, typename T, typename I, typename Td, typename Ud, typename Th, typename Uh>
void trtrs_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_operation trans,
                    const rocblas_diagonal diag,
                    const I n,
                    const I nrhs,
                    Td& dA,
                    const I lda,
                    const rocblas_stride stA,
                    Td& dB,
                    const I ldb,
                    const rocblas_stride stB,
                    Ud& dInfo,
                    const I bc,
                    Th& hA,
                    Th& hB,
                    Th& hBRes,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err,
                    const bool singular)
{
    // input data initialization
    trtrs_initData<true, true, T>(handle, n, nrhs, dA, lda, dB, ldb, bc, hA, hB, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, dA.data(), lda,
                                        stA, dB.data(), ldb, stB, dInfo.data(), bc));
    if(nrhs > 0)
        CHECK_HIP_ERROR(hBRes.transfer_from(dB));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    // (B is not modified if A is singular)
    for(I b = 0; b < bc; ++b)
    {
        rocblas_int info;
        cpu_trtrs(uplo, trans, diag, n, nrhs, hA[b], lda, hB[b], ldb, &info);
        hInfo[b][0] = info;
    }

    // check info for singularities
    double err = 0;
    *max_err = 0;
    for(I b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;

    if(nrhs == 0)
        return;

    // error is ||hB - hBRes|| / ||hB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(I b = 0; b < bc; ++b)
    {
        if(hInfo[b][0] == 0)
        {
            err = norm_error('F', n, nrhs, ldb, hB[b], hBRes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
        else
        {
            // B must be bitwise equal to the input
            I modified = 0;
            for(I j = 0; j < nrhs; ++j)
            {
                if(std::memcmp(hB[b] + j * ldb, hBRes[b] + j * ldb, sizeof(T) * n) != 0)
                    modified++;
            }
            EXPECT_EQ(modified, 0) << "where b = " << b;
            if(modified != 0)
                *max_err += 1;
        }
    }
}

template <bool STRIDED, typename T, typename I, typename Td, typename Ud, typename Th>
void trtrs_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_operation trans,
                       const rocblas_diagonal diag,
                       const I n,
                       const I nrhs,
                       Td& dA,
                       const I lda,
                       const rocblas_stride stA,
                       Td& dB,
                       const I ldb,
                       const rocblas_stride stB,
                       Ud& dInfo,
                       const I bc,
                       Th& hA,
                       Th& hB,
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
        trtrs_initData<true, false, T>(handle, n, nrhs, dA, lda, dB, ldb, bc, hA, hB, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(I b = 0; b < bc; ++b)
        {
            rocblas_int info;
            cpu_trtrs(uplo, trans, diag, n, nrhs, hA[b], lda, hB[b], ldb, &info);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    trtrs_initData<true, false, T>(handle, n, nrhs, dA, lda, dB, ldb, bc, hA, hB, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trtrs_initData<false, true, T>(handle, n, nrhs, dA, lda, dB, ldb, bc, hA, hB, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, dA.data(),
                                            lda, stA, dB.data(), ldb, stB, dInfo.data(), bc));
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
        trtrs_initData<false, true, T>(handle, n, nrhs, dA, lda, dB, ldb, bc, hA, hB, singular);

        timer.start(stream);
        rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, dA.data(), lda, stA, dB.data(),
                        ldb, stB, dInfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_trtrs(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    char transC = argus.get<char>("trans");
    char diagC = argus.get<char>("diag");
    I n = argus.get<I>("n");
    I nrhs = argus.get<I>("nrhs", n);
    I lda = argus.get<I>("lda", n);
    I ldb = argus.get<I>("ldb", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stB = argus.get<rocblas_stride>("strideB", ldb * nrhs);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_operation trans = char2rocblas_operation(transC);
    rocblas_diagonal diag = char2rocblas_diagonal(diagC);
    I bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;

    // check non-supported values
    if((uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
       || (trans != rocblas_operation_none && trans != rocblas_operation_transpose
           && trans != rocblas_operation_conjugate_transpose)
       || (diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit))
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs,
                                                  (T* const*)nullptr, lda, stA, (T* const*)nullptr,
                                                  ldb, stB, (I*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs,
                                                  (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB,
                                                  (I*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * nrhs;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nrhs < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs,
                                                  (T* const*)nullptr, lda, stA, (T* const*)nullptr,
                                                  ldb, stB, (I*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs,
                                                  (T*)nullptr, lda, stA, (T*)nullptr, ldb, stB,
                                                  (I*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs,
                                              (T* const*)nullptr, lda, stA, (T* const*)nullptr, ldb,
                                              stB, (I*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs, (T*)nullptr,
                                              lda, stA, (T*)nullptr, ldb, stB, (I*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // info arrays (allocated for every layout)
    host_strided_batch_vector<I> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<I> hInfoRes(1, 1, 1, bc);
    device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
    if(bc)
        CHECK_HIP_ERROR(dInfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hB(size_B, 1, bc);
        host_batch_vector<T> hBRes(size_BRes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_batch_vector<T> dB(size_B, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            // info must be set to zero
            for(I b = 0; b < bc; ++b)
                hInfoRes[b][0] = -1;
            if(bc)
                CHECK_HIP_ERROR(dInfo.transfer_from(hInfoRes));

            EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs,
                                                  dA.data(), lda, stA, dB.data(), ldb, stB,
                                                  dInfo.data(), bc),
                                  rocblas_status_success);

            if(bc)
                CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
            for(I b = 0; b < bc; ++b)
                EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;

            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            trtrs_getError<STRIDED, T>(handle, uplo, trans, diag, n, nrhs, dA, lda, stA, dB, ldb,
                                       stB, dInfo, bc, hA, hB, hBRes, hInfo, hInfoRes, &max_error,
                                       argus.singular);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            trtrs_getPerfData<STRIDED, T>(handle, uplo, trans, diag, n, nrhs, dA, lda, stA, dB, ldb,
                                          stB, dInfo, bc, hA, hB, &gpu_time_used, &cpu_time_used,
                                          hot_calls, argus.profile, argus.profile_kernels,
                                          argus.perf, argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T> hBRes(size_BRes, 1, stBRes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            // info must be set to zero
            for(I b = 0; b < bc; ++b)
                hInfoRes[b][0] = -1;
            if(bc)
                CHECK_HIP_ERROR(dInfo.transfer_from(hInfoRes));

            EXPECT_ROCBLAS_STATUS(rocsolver_trtrs(STRIDED, handle, uplo, trans, diag, n, nrhs,
                                                  dA.data(), lda, stA, dB.data(), ldb, stB,
                                                  dInfo.data(), bc),
                                  rocblas_status_success);

            if(bc)
                CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
            for(I b = 0; b < bc; ++b)
                EXPECT_EQ(hInfoRes[b][0], 0) << "where b = " << b;

            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            trtrs_getError<STRIDED, T>(handle, uplo, trans, diag, n, nrhs, dA, lda, stA, dB, ldb,
                                       stB, dInfo, bc, hA, hB, hBRes, hInfo, hInfoRes, &max_error,
                                       argus.singular);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            trtrs_getPerfData<STRIDED, T>(handle, uplo, trans, diag, n, nrhs, dA, lda, stA, dB, ldb,
                                          stB, dInfo, bc, hA, hB, &gpu_time_used, &cpu_time_used,
                                          hot_calls, argus.profile, argus.profile_kernels,
                                          argus.perf, argus.singular);
    }

    // validate results for rocsolver-test
    // using max(n, 4) * machine_precision as tolerance (a complex division alone can differ from
    // the reference by more than one rounding error)
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, std::max(n, I(4)));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("uplo", "trans", "diag", "n", "nrhs", "lda", "ldb", "batch_c");
                rocsolver_bench_output(uploC, transC, diagC, n, nrhs, lda, ldb, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("uplo", "trans", "diag", "n", "nrhs", "lda", "ldb",
                                       "strideA", "strideB", "batch_c");
                rocsolver_bench_output(uploC, transC, diagC, n, nrhs, lda, ldb, stA, stB, bc);
            }
            else
            {
                rocsolver_bench_output("uplo", "trans", "diag", "n", "nrhs", "lda", "ldb");
                rocsolver_bench_output(uploC, transC, diagC, n, nrhs, lda, ldb);
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

#define EXTERN_TESTING_TRTRS(...) extern template void testing_trtrs<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_TRTRS,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
