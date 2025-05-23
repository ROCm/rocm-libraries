/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

template <bool STRIDED, bool GETRF, typename I, typename Td, typename Id>
void getf2_getrf_npvt_checkBadArgs(const rocblas_handle handle,
                                   const I m,
                                   const I n,
                                   Td dA,
                                   const I lda,
                                   const rocblas_stride stA,
                                   Id dInfo,
                                   const I bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_getf2_getrf_npvt(STRIDED, GETRF, nullptr, m, n, dA, lda, stA, dInfo, bc),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, dA, lda, stA, dInfo, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, (Td) nullptr, lda, stA, dInfo, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, dA, lda, stA, (Id) nullptr, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, 0, n, (Td) nullptr, lda, stA, dInfo, bc),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, 0, (Td) nullptr, lda, stA, dInfo, bc),
        rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, dA, lda, stA, dInfo, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, bool GETRF, typename T, typename I>
void testing_getf2_getrf_npvt_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    I m = 1;
    I n = 1;
    I lda = 1;
    rocblas_stride stA = 1;
    I bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getf2_getrf_npvt_checkBadArgs<STRIDED, GETRF>(handle, m, n, dA.data(), lda, stA,
                                                      dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getf2_getrf_npvt_checkBadArgs<STRIDED, GETRF>(handle, m, n, dA.data(), lda, stA,
                                                      dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Id, typename Th>
void getf2_getrf_npvt_initData(const rocblas_handle handle,
                               const I m,
                               const I n,
                               Td& dA,
                               const I lda,
                               const rocblas_stride stA,
                               Id& dInfo,
                               const I bc,
                               Th& hA,
                               const bool singular)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        // leaving matrix as diagonal dominant so that pivoting is not required
        for(I b = 0; b < bc; ++b)
        {
            for(I i = 0; i < m; i++)
            {
                for(I j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes).
                // The algorithm must detect the first zero element in the
                // diagonal of those matrices in the batch that are singular
                I j = n / 4 + b;
                j -= (j / n) * n;
                for(I i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
                j = n / 2 + b;
                j -= (j / n) * n;
                for(I i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
                j = n - 1 + b;
                j -= (j / n) * n;
                for(I i = 0; i < m; i++)
                    hA[b][i + j * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, bool GETRF, typename T, typename I, typename Td, typename Id, typename Th, typename Ih, typename Uh>
void getf2_getrf_npvt_getError(const rocblas_handle handle,
                               const I m,
                               const I n,
                               Td& dA,
                               const I lda,
                               const rocblas_stride stA,
                               Id& dInfo,
                               const I bc,
                               Th& hA,
                               Th& hARes,
                               Uh& hIpiv,
                               Uh& hInfo,
                               Ih& hInfoRes,
                               double* max_err,
                               const bool singular)
{
    // input data initialization
    getf2_getrf_npvt_initData<true, true, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, dA.data(), lda,
                                                   stA, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(I b = 0; b < bc; ++b)
    {
        GETRF ? cpu_getrf(m, n, hA[b], lda, hIpiv[b], hInfo[b])
              : cpu_getf2(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
    }

    // expecting original matrix to be non-singular
    // error is ||hA - hARes|| / ||hA|| (ideally ||LU - Lres Ures|| / ||LU||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for(I b = 0; b < bc; ++b)
    {
        err = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }

    // also check info for singularities
    err = 0;
    for(I b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <bool STRIDED, bool GETRF, typename T, typename I, typename Td, typename Id, typename Th, typename Uh>
void getf2_getrf_npvt_getPerfData(const rocblas_handle handle,
                                  const I m,
                                  const I n,
                                  Td& dA,
                                  const I lda,
                                  const rocblas_stride stA,
                                  Id& dInfo,
                                  const I bc,
                                  Th& hA,
                                  Uh& hIpiv,
                                  Uh& hInfo,
                                  double* gpu_time_used,
                                  double* cpu_time_used,
                                  const int hot_calls,
                                  const int profile,
                                  const bool profile_kernels,
                                  const bool perf,
                                  const bool singular)
{
    if(!perf)
    {
        getf2_getrf_npvt_initData<true, false, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA,
                                                  singular);

        // cpu-lapack performance (only if no perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(I b = 0; b < bc; ++b)
        {
            GETRF ? cpu_getrf(m, n, hA[b], lda, hIpiv[b], hInfo[b])
                  : cpu_getf2(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getf2_getrf_npvt_initData<true, false, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getf2_getrf_npvt_initData<false, true, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA,
                                                  singular);

        CHECK_ROCBLAS_ERROR(rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, dA.data(), lda,
                                                       stA, dInfo.data(), bc));
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
    for(int iter = 0; iter < hot_calls; iter++)
    {
        getf2_getrf_npvt_initData<false, true, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA,
                                                  singular);

        start = get_time_us_sync(stream);
        rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, dA.data(), lda, stA, dInfo.data(),
                                   bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, bool GETRF, typename T, typename I>
void testing_getf2_getrf_npvt(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    I m = argus.get<rocblas_int>("m");
    I n = argus.get<rocblas_int>("n", m);
    I lda = argus.get<rocblas_int>("lda", m);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", min(m, n));

    I bc = argus.batch_count;
    int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(min(m, n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n,
                                                             (T* const*)nullptr, lda, stA,
                                                             (I*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n,
                                                             (T*)nullptr, lda, stA, (I*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf_npvt(
                STRIDED, GETRF, handle, m, n, (T* const*)nullptr, lda, stA, (I*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n, (T*)nullptr,
                                                         lda, stA, (I*)nullptr, bc));

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
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<I> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n,
                                                             dA.data(), lda, stA, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getf2_getrf_npvt_getError<STRIDED, GETRF, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA,
                                                         hARes, hIpiv, hInfo, hInfoRes, &max_error,
                                                         argus.singular);

        // collect performance data
        if(argus.timing)
            getf2_getrf_npvt_getPerfData<STRIDED, GETRF, T>(
                handle, m, n, dA, lda, stA, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<I> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_getf2_getrf_npvt(STRIDED, GETRF, handle, m, n,
                                                             dA.data(), lda, stA, dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getf2_getrf_npvt_getError<STRIDED, GETRF, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA,
                                                         hARes, hIpiv, hInfo, hInfoRes, &max_error,
                                                         argus.singular);

        // collect performance data
        if(argus.timing)
            getf2_getrf_npvt_getPerfData<STRIDED, GETRF, T>(
                handle, m, n, dA, lda, stA, dInfo, bc, hA, hIpiv, hInfo, &gpu_time_used,
                &cpu_time_used, hot_calls, argus.profile, argus.profile_kernels, argus.perf,
                argus.singular);
    }

    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, min(m, n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "batch_c");
                rocsolver_bench_output(m, n, lda, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
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

#define EXTERN_TESTING_GETF2_GETRF_NPVT(...) \
    extern template void testing_getf2_getrf_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GETF2_GETRF_NPVT,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
