/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Th,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nb,
                          Td& dA,
                          const rocblas_int lda,
                          Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale band of size nb of A to avoid singularities
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i <= j + nb && i >= j - nb)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = i+1; j < n; j++)
                hA[0][i + j * lda] = hA[0][j + i * lda];
        }           
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Th,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nb,
                          Td& dA,
                          const rocblas_int lda,
                          Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale band of size nb of A to avoid singularities
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j)
                    hA[0][i + j * lda] = hA[0][i + j * lda].real() + 400;
                if(i <= j + k && i >= j - k)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = i+1; j < n; j++)
                hA[0][i + j * lda] = conj(hA[0][j + i * lda]);
        }           
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename Td, typename Th>
void sy2sb_he2hb_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int nb,
                    const rocblas_int k,
                    Td& dA,
                    const rocblas_int lda,
                    Td& dV,
                    Td& dW,
                    Th& hA,
                    Th& hARes,
                    Th& hTau,
                    Th& hAB,
                    const rocblas_int ldab,
                    double* max_err)
{
    size_t lwork = n * k + n * std::max(k, 128) + 2 * k * k;
    std::vector<T> hwork(lwork);

    // input data initialization
    sy2sb_he2hb_initData<true, true, T>(handle, n, nb, dA, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sy2sb_he2hb(handle, n, nb, k, dA.data(), lda, dV.data(), dW.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));

    // CPU lapack
    cpu_sy2sb_he2hb(uplo,
        n,
        k,
        hA[0],
        lda,
        hAB[0],
        ldab,
        hTau[0],
        hwork.data(),
        lwork);

    // error is ||hARes - hAB|| / ||hAB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    rocblas_int offset = (uplo == rocblas_fill_lower) ? k : 0;
    *max_err = 0;
    err = norm_error('F', k+1, n, ldab, hAB[0], hARes[0]);
    *max_err = err > *max_err ? err : *max_err;

    // TODO: Check V and W
}

template <typename T, typename Td, typename Th>
void sy2sb_he2hb_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       const rocblas_int nb,
                       const rocblas_int k,
                       Td& dA,
                       const rocblas_int lda,
                       Td& dV,
                       Td& dW,
                       Th& hA,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = nan("");
    }

    sy2sb_he2hb_initData<true, false, T>(handle, n, nb, dA, lda, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sy2sb_he2hb_initData<false, true, T>(handle, n, nb, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_sy2sb_he2hb(handle, n, nb, k, dA.data(), lda, dV.data(), dW.data()));
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
        sy2sb_he2hb_initData<false, true, T>(handle, n, nb, dA, lda, hA);

        start = get_time_us_sync(stream);
        rocsolver_sy2sb_he2hb(handle, n, nb, k, dA.data(), lda, dV.data(), dW.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}


template <typename T>
void testing_sy2sb_he2hb(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nb = argus.get<rocblas_int>("nb", 1);
    rocblas_int k = argus.get<rocblas_int>("k", 1);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldab = nb + 1;

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_V = (n - nb) * (n - nb);
    size_t size_W = (n - nb) * (n - nb);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_tau = (argus.unit_check || argus.norm_check) ? std::max(n - nb, 0) : 0;
    size_t size_AB = (argus.unit_check || argus.norm_check) ? ldab * n : 0;
    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || k < 0 || lda < n || nb < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, n, nb, k, (T*)nullptr, lda, (T*)nullptr, (T*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_sy2sb_he2hb(handle, n, nb, k, (T*)nullptr, lda, (T*)nullptr, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hAB(size_AB, 1, size_AB, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
    device_strided_batch_vector<T> dW(size_W, 1, size_W, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dW.memcheck());

    // check quick return
    if(k == 0 || n == 0 || nb == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, n, nb, k, dA.data(), lda, dV.data(), dW.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        sy2sb_he2hb_getError<T>(handle, n, nb, k, dA, lda, dV, dW, hA, hARes, hTau, hAB, ldab, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        sy2sb_he2hb_getPerfData<T>(handle, n, nb, k, dA, lda, dV, dW, hA, 
                             &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
                             argus.profile_kernels, argus.perf);

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
            rocsolver_bench_output("n", "nb", "k", "lda");
            rocsolver_bench_output(n, nb, k, lda);
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

#define EXTERN_TESTING_SY2SB_HE2HB(...) extern template void testing_sy2sb_he2hb<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SY2SB_HE2HB, FOREACH_SCALAR_TYPE, APPLY_STAMP)
