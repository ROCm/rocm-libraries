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

template <bool STRIDED, typename T, typename U>
void getri_npvt_checkBadArgs(const rocblas_handle handle,
                             const rocblas_int n,
                             T dA,
                             const rocblas_int lda,
                             const rocblas_stride stA,
                             U dInfo,
                             const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, nullptr, n, dA, lda, stA, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, handle, n, dA, lda, stA, dInfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, handle, n, (T) nullptr, lda, stA, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, handle, n, dA, lda, stA, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, handle, 0, (T) nullptr, lda, stA, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, handle, n, dA, lda, stA, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_getri_npvt_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getri_npvt_checkBadArgs<STRIDED>(handle, n, dA.data(), lda, stA, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getri_npvt_checkBadArgs<STRIDED>(handle, n, dA.data(), lda, stA, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th, typename Uh>
void getri_npvt_initData(const rocblas_handle handle,
                         const rocblas_int n,
                         Td& dA,
                         const rocblas_int lda,
                         const rocblas_int bc,
                         Th& hA,
                         Uh& hIpiv,
                         Uh& hInfo,
                         const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
            // leaving matrix as diagonal dominant so that pivoting is not required
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

            // do the LU decomposition of matrix A w/ the reference LAPACK routine
            cpu_getrf(n, n, hA[b], lda, hIpiv[b], hInfo[b]);

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // add some singularities
                // always the same elements for debugging purposes
                // the algorithm must detect the first zero pivot in those
                // matrices in the batch that are singular
                rocblas_int i = n / 4 + b;
                i -= (i / n) * n;
                hA[b][i + i * lda] = 0;
                i = n / 2 + b;
                i -= (i / n) * n;
                hA[b][i + i * lda] = 0;
                i = n - 1 + b;
                i -= (i / n) * n;
                hA[b][i + i * lda] = 0;
            }
        }
    }

    // now copy data to the GPU
    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getri_npvt_getError(const rocblas_handle handle,
                         const rocblas_int n,
                         Td& dA,
                         const rocblas_int lda,
                         const rocblas_stride stA,
                         Ud& dInfo,
                         const rocblas_int bc,
                         Th& hA,
                         Th& hARes,
                         Uh& hIpiv,
                         Uh& hInfo,
                         Uh& hInfoRes,
                         double* max_err,
                         const bool singular)
{
    rocblas_int sizeW = n;
    std::vector<T> hW(sizeW);

    // input data initialization
    getri_npvt_initData<true, true, T>(handle, n, dA, lda, bc, hA, hIpiv, hInfo, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_getri_npvt(STRIDED, handle, n, dA.data(), lda, stA, dInfo.data(), bc));

    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
    {
        cpu_getri(n, hA[b], lda, hIpiv[b], hW.data(), sizeW, hInfo[b]);
    }

    // check info for singularities
    double err = 0;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] == 0)
        {
            err = norm_error('F', n, n, lda, hA[b], hARes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getri_npvt_getPerfData(const rocblas_handle handle,
                            const rocblas_int n,
                            Td& dA,
                            const rocblas_int lda,
                            const rocblas_stride stA,
                            Ud& dInfo,
                            const rocblas_int bc,
                            Th& hA,
                            Uh& hIpiv,
                            Uh& hInfo,
                            double* gpu_time_used,
                            double* cpu_time_used,
                            const rocblas_int hot_calls,
                            const int profile,
                            const bool profile_kernels,
                            const bool perf,
                            const bool singular)
{
    rocblas_int sizeW = n;
    std::vector<T> hW(sizeW);

    if(!perf)
    {
        getri_npvt_initData<true, false, T>(handle, n, dA, lda, bc, hA, hIpiv, hInfo, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
        {
            cpu_getri(n, hA[b], lda, hIpiv[b], hW.data(), sizeW, hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getri_npvt_initData<true, false, T>(handle, n, dA, lda, bc, hA, hIpiv, hInfo, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getri_npvt_initData<false, true, T>(handle, n, dA, lda, bc, hA, hIpiv, hInfo, singular);

        CHECK_ROCBLAS_ERROR(
            rocsolver_getri_npvt(STRIDED, handle, n, dA.data(), lda, stA, dInfo.data(), bc));
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
        getri_npvt_initData<false, true, T>(handle, n, dA, lda, bc, hA, hIpiv, hInfo, singular);

        start = get_time_us_sync(stream);
        rocsolver_getri_npvt(STRIDED, handle, n, dA.data(), lda, stA, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_getri_npvt(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stP = argus.get<rocblas_stride>("strideP", n);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_P = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, handle, n, (T* const*)nullptr, lda,
                                                       stA, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_getri_npvt(STRIDED, handle, n, (T*)nullptr, lda, stA,
                                                       (rocblas_int*)nullptr, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_getri_npvt(STRIDED, handle, n, (T* const*)nullptr, lda, stA,
                                                   (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_getri_npvt(STRIDED, handle, n, (T*)nullptr, lda, stA,
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
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_getri_npvt(STRIDED, handle, n, dA.data(), lda, stA, dInfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getri_npvt_getError<STRIDED, T>(handle, n, dA, lda, stA, dInfo, bc, hA, hARes, hIpiv,
                                            hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            getri_npvt_getPerfData<STRIDED, T>(handle, n, dA, lda, stA, dInfo, bc, hA, hIpiv, hInfo,
                                               &gpu_time_used, &cpu_time_used, hot_calls,
                                               argus.profile, argus.profile_kernels, argus.perf,
                                               argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<rocblas_int> hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
        host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(
                rocsolver_getri_npvt(STRIDED, handle, n, dA.data(), lda, stA, dInfo.data(), bc),
                rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            getri_npvt_getError<STRIDED, T>(handle, n, dA, lda, stA, dInfo, bc, hA, hARes, hIpiv,
                                            hInfo, hInfoRes, &max_error, argus.singular);

        // collect performance data
        if(argus.timing)
            getri_npvt_getPerfData<STRIDED, T>(handle, n, dA, lda, stA, dInfo, bc, hA, hIpiv, hInfo,
                                               &gpu_time_used, &cpu_time_used, hot_calls,
                                               argus.profile, argus.profile_kernels, argus.perf,
                                               argus.singular);
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
                rocsolver_bench_output("n", "lda", "batch_c");
                rocsolver_bench_output(n, lda, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("n", "lda", "strideA", "batch_c");
                rocsolver_bench_output(n, lda, stA, bc);
            }
            else
            {
                rocsolver_bench_output("n", "lda");
                rocsolver_bench_output(n, lda);
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

#define EXTERN_TESTING_GETRI_NPVT(...) \
    extern template void testing_getri_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GETRI_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
