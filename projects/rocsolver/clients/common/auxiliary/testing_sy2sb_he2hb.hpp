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
          typename Ud,
          typename Th,
          typename Uh,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int k,
                          Ud& dA,
                          const rocblas_int lda,
                          Td& dAB,
                          const rocblas_int ldab,
                          Uh& hA,
                          Th& hAB,
                          const rocblas_int bc)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i <= j + k && i >= j - k)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            for(rocblas_int i = 0; i < k + 1; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    hAB[b][i + j * ldab] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dAB.transfer_from(hAB));
    }
}

template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Ud,
          typename Th,
          typename Uh,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int k,
                          Ud& dA,
                          const rocblas_int lda,
                          Td& dAB,
                          const rocblas_int ldab,
                          Uh& hA,
                          Th& hAB,
                          const rocblas_int bc)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] = hA[b][i + j * lda].real() + 400;
                    if(i <= j + k && i >= j - k)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            for(rocblas_int i = 0; i < k + 1; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    hAB[b][i + j * ldab] = 0;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dAB.transfer_from(hAB));
    }
}

template <typename T, typename Ud, typename Td, typename Uh, typename Th>
void sy2sb_he2hb_getError(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_int n,
                    const rocblas_int k,
                    Ud& dA,
                    const rocblas_int lda,
                    Td& dAB,
                    const rocblas_int ldab,
                    Td& dTau,
                    Uh& hA,
                    Th& hARes,
                    Th& hTau,
                    Th& hTauRes,
                    Th& hAB,
                    Th& hABRes,
                    double* max_err)
{
    size_t lwork = n * k + n * std::max(k, 128) + 2 * k * k;
    std::vector<T> hwork(lwork);

    // input data initialization
    sy2sb_he2hb_initData<true, true, T>(handle, n, k, dA, lda, dAB, ldab, hA, hAB, 1);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sy2sb_he2hb(handle, uplo, n, k, dA.data(), lda, dAB.data(), ldab, dTau.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hABRes.transfer_from(dAB));
    CHECK_HIP_ERROR(hTauRes.transfer_from(dTau));

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

    // error is max(||hA - hARes|| / ||hA||, ||hW - hWRes|| / ||hW||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE) using frobenius norm
    double err;
    rocblas_int offset = (uplo == rocblas_fill_lower) ? k : 0;
    *max_err = 0;
    err = norm_error('F', k+1, n, ldab, hAB[0], hABRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    if(n > k + 1)
    {
        err = norm_error('F', n - k, 1, 1, hTau[0], hTauRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }

    // TODO: Check HH reflectors in A
}

template <typename T>
void testing_sy2sb_he2hb(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int k = argus.get<rocblas_int>("k", n);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_int ldab = argus.get<rocblas_int>("ldb", k+1);

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, k, (T*)nullptr, lda, (T*)nullptr, ldab, (T*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = lda * n;
    size_t size_AB = ldab * n;
    size_t size_tau = std::max(n - k, 0);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_ABRes = (argus.unit_check || argus.norm_check) ? size_AB : 0;
    size_t size_tauRes = (argus.unit_check || argus.norm_check) ? size_tau : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || k < 0 || lda < n || ldab < k+1);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, k, (T*)nullptr, lda, (T*)nullptr, ldab, (T*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_sy2sb_he2hb(handle, uplo, n, k, (T*)nullptr, lda, (T*)nullptr, ldab, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hAB(size_AB, 1, size_AB, 1);
    host_strided_batch_vector<T> hABRes(size_ABRes, 1, size_ABRes, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);
    host_strided_batch_vector<T> hTauRes(size_tauRes, 1, size_tauRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dAB(size_AB, 1, size_AB, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_AB)
        CHECK_HIP_ERROR(dAB.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());

    // check quick return
    if(k == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, uplo, n, k, dA.data(), lda, dAB.data(), ldab, dTau.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        sy2sb_he2hb_getError<T>(handle, uplo, n, k, dA, lda, dAB, ldab, dTau, hA, hARes, hTau, hTauRes, hAB,
                          hABRes, &max_error);

    // // collect performance data
    // if(argus.timing && hot_calls > 0)
    //     sy2sb_he2hb_getPerfData<T>(handle, uplo, n, k, dA, lda, dE, dTau, dW, ldw, hA, hE, hTau, hW,
    //                          &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
    //                          argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using k*n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, k * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "k", "lda", "ldab");
            rocsolver_bench_output(uploC, n, k, lda, ldab);
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
