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

template <typename T>
void lasyf_checkBadArgs(const rocblas_handle handle,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        const rocblas_int nb,
                        rocblas_int* kb,
                        T dA,
                        const rocblas_int lda,
                        rocblas_int* ipiv,
                        rocblas_int* info)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lasyf(nullptr, uplo, n, nb, kb, dA, lda, ipiv, info),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_lasyf(handle, rocblas_fill_full, n, nb, kb, dA, lda, ipiv, info),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasyf(handle, uplo, n, nb, (rocblas_int*)nullptr, dA, lda, ipiv, info),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lasyf(handle, uplo, n, nb, kb, (T) nullptr, lda, ipiv, info),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasyf(handle, uplo, n, nb, kb, dA, lda, (rocblas_int*)nullptr, info),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasyf(handle, uplo, n, nb, kb, dA, lda, ipiv, (rocblas_int*)nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lasyf(handle, uplo, 0, 0, kb, (T) nullptr, lda, (rocblas_int*)nullptr, info),
        rocblas_status_success);
}

template <typename T>
void testing_lasyf_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_int n = 1;
    rocblas_int nb = 1;
    rocblas_int lda = 1;

    // memory allocations
    device_strided_batch_vector<rocblas_int> dKB(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dKB.memcheck());
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    lasyf_checkBadArgs(handle, uplo, n, nb, dKB.data(), dA.data(), lda, dIpiv.data(), dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void lasyf_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    Th& hA,
                    const bool singular)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // shuffle rows to test pivoting
        // always the same permuation for debugging purposes
        for(rocblas_int i = 0; i < n / 2; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                tmp = hA[0][i + j * lda];
                hA[0][i + j * lda] = hA[0][n - 1 - i + j * lda];
                hA[0][n - 1 - i + j * lda] = tmp;
            }
        }

        if(singular)
        {
            // add some singularities
            // always the same elements for debugging purposes
            // the algorithm must detect the first zero pivot in those
            // matrices in the batch that are singular
            rocblas_int j = n / 4;
            j -= (j / n) * n;
            for(rocblas_int i = 0; i < n; i++)
            {
                hA[0][i + j * lda] = 0;
                hA[0][j + i * lda] = 0;
            }
            j = n / 2;
            j -= (j / n) * n;
            for(rocblas_int i = 0; i < n; i++)
            {
                hA[0][i + j * lda] = 0;
                hA[0][j + i * lda] = 0;
            }
            j = n - 1;
            j -= (j / n) * n;
            for(rocblas_int i = 0; i < n; i++)
            {
                hA[0][i + j * lda] = 0;
                hA[0][j + i * lda] = 0;
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void lasyf_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int nb,
                    Ud& dKB,
                    Td& dA,
                    const rocblas_int lda,
                    Ud& dIpiv,
                    Ud& dInfo,
                    Uh& hKB,
                    Uh& hKBRes,
                    Th& hA,
                    Th& hARes,
                    Uh& hIpiv,
                    Uh& hIpivRes,
                    Uh& hInfo,
                    Uh& hInfoRes,
                    double* max_err,
                    const bool singular)
{
    int ldw = n;
    int lwork = ldw * nb;
    std::vector<T> work(lwork);

    // input data initialization
    lasyf_initData<true, true, T>(handle, n, dA, lda, hA, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_lasyf(handle, uplo, n, nb, dKB.data(), dA.data(), lda,
                                        dIpiv.data(), dInfo.data()));
    CHECK_HIP_ERROR(hKBRes.transfer_from(dKB));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    cpu_lasyf(uplo, n, nb, hKB[0], hA[0], lda, hIpiv[0], work.data(), ldw, hInfo[0]);

    // error is ||hA - hARes|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    *max_err = 0;
    err = norm_error('F', n, n, lda, hA[0], hARes[0]);
    *max_err = err > *max_err ? err : *max_err;

    // also check pivoting (count the number of incorrect pivots)
    err = 0;
    if(uplo == rocblas_fill_upper)
    {
        for(rocblas_int i = n - hKBRes[0][0]; i < n; ++i)
        {
            EXPECT_EQ(hIpiv[0][i], hIpivRes[0][i]) << "where i = " << i;
            if(hIpiv[0][i] != hIpivRes[0][i])
                err++;
        }
    }
    else
    {
        for(rocblas_int i = 0; i < hKBRes[0][0]; ++i)
        {
            EXPECT_EQ(hIpiv[0][i], hIpivRes[0][i]) << "where i = " << i;
            if(hIpiv[0][i] != hIpivRes[0][i])
                err++;
        }
    }
    *max_err = err > *max_err ? err : *max_err;

    // also check kb
    err = 0;
    EXPECT_EQ(hKB[0][0], hKBRes[0][0]);
    if(hKB[0][0] != hKBRes[0][0])
        err++;
    *max_err += err;

    // also check info
    err = 0;
    EXPECT_EQ(hInfo[0][0], hInfoRes[0][0]);
    if(hInfo[0][0] != hInfoRes[0][0])
        err++;
    *max_err += err;
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void lasyf_getPerfData(const rocblas_handle handle,
                       const rocblas_fill uplo,
                       const rocblas_int n,
                       const rocblas_int nb,
                       Ud& dKB,
                       Td& dA,
                       const rocblas_int lda,
                       Ud& dIpiv,
                       Ud& dInfo,
                       Uh& hKB,
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
    int ldw = n;
    int lwork = ldw * nb;
    std::vector<T> work(lwork);

    if(!perf)
    {
        lasyf_initData<true, false, T>(handle, n, dA, lda, hA, singular);

        // cpu-lapack performance
        *cpu_time_used = get_time_us_no_sync();
        cpu_lasyf(uplo, n, nb, hKB[0], hA[0], lda, hIpiv[0], work.data(), ldw, hInfo[0]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    lasyf_initData<true, false, T>(handle, n, dA, lda, hA, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lasyf_initData<false, true, T>(handle, n, dA, lda, hA, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_lasyf(handle, uplo, n, nb, dKB.data(), dA.data(), lda,
                                            dIpiv.data(), dInfo.data()));
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
        lasyf_initData<false, true, T>(handle, n, dA, lda, hA, singular);

        start = get_time_us_sync(stream);
        rocsolver_lasyf(handle, uplo, n, nb, dKB.data(), dA.data(), lda, dIpiv.data(), dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_lasyf(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nb = argus.get<rocblas_int>("nb", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lasyf(handle, uplo, n, nb, (rocblas_int*)nullptr, (T*)nullptr,
                                              lda, (rocblas_int*)nullptr, (rocblas_int*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = lda * n;
    size_t size_Ipiv = n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_IpivRes = (argus.unit_check || argus.norm_check) ? size_Ipiv : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nb < 0 || nb > n || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lasyf(handle, uplo, n, nb, (rocblas_int*)nullptr, (T*)nullptr,
                                              lda, (rocblas_int*)nullptr, (rocblas_int*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_lasyf(handle, uplo, n, nb, (rocblas_int*)nullptr, (T*)nullptr,
                                          lda, (rocblas_int*)nullptr, (rocblas_int*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<rocblas_int> hKB(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hKBRes(1, 1, 1, 1);
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<rocblas_int> hIpiv(size_Ipiv, 1, size_Ipiv, 1);
    host_strided_batch_vector<rocblas_int> hIpivRes(size_IpivRes, 1, size_IpivRes, 1);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, 1);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dKB(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<rocblas_int> dIpiv(size_Ipiv, 1, size_Ipiv, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_Ipiv)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dKB.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(nb == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_lasyf(handle, uplo, n, nb, dKB.data(), dA.data(), lda,
                                              dIpiv.data(), dInfo.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lasyf_getError<T>(handle, uplo, n, nb, dKB, dA, lda, dIpiv, dInfo, hKB, hKBRes, hA, hARes,
                          hIpiv, hIpivRes, hInfo, hInfoRes, &max_error, argus.singular);

    // collect performance data
    if(argus.timing)
        lasyf_getPerfData<T>(handle, uplo, n, nb, dKB, dA, lda, dIpiv, dInfo, hKB, hA, hIpiv, hInfo,
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                             argus.profile_kernels, argus.perf, argus.singular);

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
            rocsolver_bench_output("uplo", "n", "nb", "lda");
            rocsolver_bench_output(uploC, n, nb, lda);
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

#define EXTERN_TESTING_LASYF(...) extern template void testing_lasyf<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LASYF, FOREACH_SCALAR_TYPE, APPLY_STAMP)
