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

//------------------------------------------------------------------------------
// todo: Can dA and hA have different lda? Or does lda apply only to hA? Weird argument order.
template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Th,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int kd,
                          Td& dA,
                          const rocblas_int lda,
                          Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale band of size kd of A to avoid singularities
        // todo: remove. no need to scale.
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i <= j + kd && i >= j - kd)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // Symmetrize matrix.
        // todo: should this be in he2hb, where it (may) be needed,
        // and should be timed?
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

//------------------------------------------------------------------------------
// todo: why is a complex version needed?
// - Use different conj, not std::conj.
// - Making diagonal real. Use real(x) instead of x.real().
// Anything else?
template <bool CPU,
          bool GPU,
          typename T,
          typename Td,
          typename Th,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int kd,
                          Td& dA,
                          const rocblas_int lda,
                          Th& hA)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale band of size kd of A to avoid singularities
        // todo: remove. no need to scale.
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                if(i == j)
                    hA[0][i + j * lda] = hA[0][i + j * lda].real() + 400;
                if(i <= j + kd && i >= j - kd)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // Symmetrize matrix.
        // todo: should this be in he2hb, where it (may) be needed,
        // and should be timed?
        for(rocblas_int i = 0; i < n; i++)
        {
            for(rocblas_int j = i+1; j < n; j++)
                hA[0][i + j * lda] = std::conj(hA[0][j + i * lda]);
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

//------------------------------------------------------------------------------
// n        -- matrix dimension
// kd       -- desired bandwidth
// nb       -- outer blocksize to use
// dA       -- matrix on GPU, lda-by-n, lda >= n
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 3 kd (size needed for 2nd stage)
// hA       -- matrix on CPU, lda-by-n, lda >= n
// hAres    -- output matrix on CPU to copy GPU result, lda-by-n, lda >= n
//              hmm... can I reuse this for copying dAband?
// hTau     -- output vector on CPU, length n.
// hAband   -- output band matrix on CPU, ldab-by-n, ldab >= kd+1
//
template <typename T, typename Td, typename Th>
void sy2sb_he2hb_getError(const rocblas_handle handle,
                    const rocblas_int n,
                    const rocblas_int kd,
                    const rocblas_int nb,
                    Td& dA,
                    const rocblas_int lda,
                    Td& dTau,
                    Td& dAband,
                    const rocblas_int ldab,
                    Th& hARes,
                    Th& hTauRes,
                    Th& hAbandRes,
                    Th& hA,
                    Th& hTau,
                    Th& hAband,
                    double* max_err)
{
    // lwork for LAPACK hetrd_he2hb
    size_t lwork = n * kd + n * std::max(kd, 128) + 2 * kd * kd;
    std::vector<T> hwork(lwork);

    // input data initialization
    sy2sb_he2hb_initData<true, true, T>(handle, n, kd, dA, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_sy2sb_he2hb(
            handle, n, kd, nb, dA.data(), lda, dTau.data(), dAband.data(), ldab));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hAband_Res.transfer_from(dAband));

    // CPU lapack
    cpu_sy2sb_he2hb(rocblas_fill_lower,
        n,
        kd,
        hA[0],
        lda,
        hAband[0],
        ldab,
        hTau[0],
        hwork.data(),
        lwork);

    // error is ||hARes - hAband|| / ||hAband||
    // using frobenius norm
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE)

    // todo: report all errors (A, V).
    double err = 0;
    *max_err = 0;
    err = norm_error('F', kd+1, n, ldab, hAband[0], hAbandRes[0]);
    *max_err = std::max( err, *max_err );

    // TODO: Check V and tau. Check orthogonality of Q using unmtr/ungtr.
}

//------------------------------------------------------------------------------
// n        -- matrix dimension
// kd       -- desired bandwidth
// nb       -- outer blocksize to use
// dA       -- matrix on GPU, lda-by-n, lda >= n
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 3 kd (size needed for 2nd stage)
// hA       -- matrix on CPU, lda-by-n, lda >= n
template <typename T, typename Td, typename Th>
void sy2sb_he2hb_getPerfData(const rocblas_handle handle,
                       const rocblas_int n,
                       const rocblas_int kd,
                       const rocblas_int nb,
                       Td& dA,
                       const rocblas_int lda,
                       Td& dAband,
                       const rocblas_int ldab,
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

    sy2sb_he2hb_initData<true, false, T>(handle, n, kd, dA, lda, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sy2sb_he2hb_initData<false, true, T>(handle, n, kd, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(
            rocsolver_sy2sb_he2hb(
                handle, n, kd, nb,
                dA.data(), lda,
                dTau.data(),
                dAband.data(), ldab));
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
        sy2sb_he2hb_initData<false, true, T>(handle, n, kd, dA, lda, hA);

        start = get_time_us_sync(stream);
        rocsolver_sy2sb_he2hb(
            handle, n, kd, nb,
            dA.data(), lda,
            dTau.data(),
            dAband.data(), ldab);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

//------------------------------------------------------------------------------
template <typename T>
void testing_sy2sb_he2hb(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int kd = argus.get<rocblas_int>("kd", 1);
    rocblas_int nb = argus.get<rocblas_int>("nb", kd);
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    // rocSolver 2nd stage needs 3*kd. LAPACK needs kd+1.
    // todo: get from argus?
    rocblas_int ldab = 3*kd;  //kd + 1;

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_Aband = ldab * n;
    size_t size_tau = std::max( n - kd, 0 );
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes     = (argus.unit_check || argus.norm_check) ? size_A     : 0;
    size_t size_AbandRes = (argus.unit_check || argus.norm_check) ? size_Aband : 0;
    size_t size_tauRes   = (argus.unit_check || argus.norm_check) ? size_tau   : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || nb < kd || lda < n || ldab < 3*kd || (n > 0 && kd < 1));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_sy2sb_he2hb(
                handle, n, kd, nb,
                (T*)nullptr, lda,
                (T*)nullptr,
                (T*)nullptr, ldab),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(
            rocsolver_sy2sb_he2hb(
                handle, n, kd, nb,
                (T*)nullptr, lda,
                (T*)nullptr,
                (T*)nullptr, ldab));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAband(size_Aband, 1, size_Aband, 1);
    host_strided_batch_vector<T> hTau(size_tau, 1, size_tau, 1);

    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hAbandRes(size_AbandRes, 1, size_AbandRes, 1);
    host_strided_batch_vector<T> hTauRes(size_tauRes, 1, size_tauRes, 1);

    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dAband(size_Aband, 1, size_Aband, 1);
    device_strided_batch_vector<T> dTau(size_tau, 1, size_tau, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_Aband)
        CHECK_HIP_ERROR(dAband.memcheck());
    if(size_tau)
        CHECK_HIP_ERROR(dTau.memcheck());

    // check quick return
    if(nb == 0 || n == 0 || kd == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_sy2sb_he2hb(
                handle, n, kd, nb,
                dA.data(), lda,
                dTau.data(),
                dAband.data(), ldab),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check) {
        sy2sb_he2hb_getError<T>(
            handle, n, kd, nb,
            dA, lda, dTau, dAband, ldab,
            hARes, hAbandRes, hTauRes,
            hA, hTau, hAband,
            &max_error);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0) {
        sy2sb_he2hb_getPerfData<T>(
            handle, n, kd, nb,
            dA, lda, dTau, dAband, ldab, hA,
            &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
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
            rocsolver_bench_output("n", "kd", "nb", "lda");
            rocsolver_bench_output(n, kd, nb, lda);
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
