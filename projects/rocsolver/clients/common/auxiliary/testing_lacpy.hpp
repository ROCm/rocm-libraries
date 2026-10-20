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
#include <limits>

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <typename T, typename I>
void lacpy_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const I m,
                        const I n,
                        T dA,
                        const I lda,
                        T dB,
                        const I ldb)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(nullptr, uplo, m, n, dA, lda, dB, ldb),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lacpy(handle, static_cast<rocblas_fill>(0), m, n, dA, lda, dB, ldb),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(handle, uplo, m, n, (T) nullptr, lda, dB, ldb),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(handle, uplo, m, n, dA, lda, (T) nullptr, ldb),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(handle, uplo, (I)0, n, (T) nullptr, lda, (T) nullptr, ldb),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(handle, uplo, m, (I)0, (T) nullptr, lda, (T) nullptr, ldb),
                          rocblas_status_success);
}

template <typename T, typename I>
void testing_lacpy_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_full;
    I m = 1;
    I n = 1;
    I lda = 1;
    I ldb = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dB(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());

    // check bad arguments
    lacpy_checkBadArgs(handle, uplo, m, n, dA.data(), lda, dB.data(), ldb);
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void lacpy_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Td& dB,
                    const I ldb,
                    Th& hA,
                    Th& hB)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));
        const S nan = std::numeric_limits<S>::quiet_NaN();

        rocblas_init<T>(hA, true);

        // fill all of B (including rows m to ldb-1) with a sentinel pattern of
        // negative values and NaNs, which cannot be in A;
        // only the part of B given by uplo can be overwritten
        for(I j = 0; j < n; j++)
        {
            for(I i = 0; i < ldb; i++)
                hB[0][i + j * ldb] = ((i + j) % 3 == 0) ? T(nan) : T(-S(1 + (i + j * ldb) % 1000));
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <typename T>
size_t lacpy_count_mismatches(const size_t size, const T* gold, const T* comp, size_t* first)
{
    size_t count = 0;
    for(size_t k = 0; k < size; k++)
    {
        if(std::memcmp(gold + k, comp + k, sizeof(T)) != 0)
        {
            if(count == 0)
                *first = k;
            count++;
        }
    }
    return count;
}

template <typename T, typename I, typename Td, typename Th>
void lacpy_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Td& dB,
                    const I ldb,
                    Th& hA,
                    Th& hARes,
                    Th& hB,
                    Th& hBRes,
                    double* max_err)
{
    // initialize data
    lacpy_initData<true, true, T>(handle, uplo, m, n, dA, lda, dB, ldb, hA, hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_lacpy(handle, uplo, m, n, dA.data(), lda, dB.data(), ldb));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));

    // CPU lapack
    cpu_lacpy(uplo, m, n, hA[0], lda, hB[0], ldb);

    // error is the number of elements that differ (elements must be bitwise identical):
    // the copied part of B must be equal to A, the rest of B (including the
    // rows m to ldb-1) and all of A must be unchanged
    size_t first = 0;
    size_t nbad = lacpy_count_mismatches(size_t(ldb) * n, hB[0], hBRes[0], &first);
    EXPECT_EQ(nbad, 0) << "first difference in B at i = " << first % ldb << ", j = " << first / ldb;
    *max_err = nbad;

    nbad = lacpy_count_mismatches(size_t(lda) * n, hA[0], hARes[0], &first);
    EXPECT_EQ(nbad, 0) << "first difference in A at i = " << first % lda << ", j = " << first / lda;
    *max_err += nbad;
}

template <typename T, typename I, typename Td, typename Th>
void lacpy_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const I m,
                       const I n,
                       Td& dA,
                       const I lda,
                       Td& dB,
                       const I ldb,
                       Th& hA,
                       Th& hB,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        lacpy_initData<true, false, T>(handle, uplo, m, n, dA, lda, dB, ldb, hA, hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_lacpy(uplo, m, n, hA[0], lda, hB[0], ldb);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    lacpy_initData<true, false, T>(handle, uplo, m, n, dA, lda, dB, ldb, hA, hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lacpy_initData<false, true, T>(handle, uplo, m, n, dA, lda, dB, ldb, hA, hB);

        CHECK_ROCBLAS_ERROR(rocsolver_lacpy(handle, uplo, m, n, dA.data(), lda, dB.data(), ldb));
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
        lacpy_initData<false, true, T>(handle, uplo, m, n, dA, lda, dB, ldb, hA, hB);

        timer.start(stream);
        rocsolver_lacpy(handle, uplo, m, n, dA.data(), lda, dB.data(), ldb);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T, typename I>
void testing_lacpy(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I lda = argus.get<I>("lda", m);
    I ldb = argus.get<I>("ldb", m);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower && uplo != rocblas_fill_full)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(handle, uplo, m, n, (T*)nullptr, lda, (T*)nullptr, ldb),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_B = size_t(ldb) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || ldb < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(handle, uplo, m, n, (T*)nullptr, lda, (T*)nullptr, ldb),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_lacpy(handle, uplo, m, n, (T*)nullptr, lda, (T*)nullptr, ldb));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hB(size_B, 1, size_B, 1);
    host_strided_batch_vector<T> hBRes(size_BRes, 1, size_BRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dB(size_B, 1, size_B, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_B)
        CHECK_HIP_ERROR(dB.memcheck());

    // check quick return
    if(m == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lacpy(handle, uplo, m, n, dA.data(), lda, dB.data(), ldb),
                              rocblas_status_success);

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lacpy_getError<T>(handle, uplo, m, n, dA, lda, dB, ldb, hA, hARes, hB, hBRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        lacpy_getPerfData<T>(handle, uplo, m, n, dA, lda, dB, ldb, hA, hB, &gpu_time_used,
                             &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels,
                             argus.perf);

    // validate results for rocsolver-test
    // no tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 0);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "m", "n", "lda", "ldb");
            rocsolver_bench_output(uploC, m, n, lda, ldb);

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

#define EXTERN_TESTING_LACPY(...) extern template void testing_lacpy<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LACPY, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
