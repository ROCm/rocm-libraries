/* **************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "common/misc/generate.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <typename T, typename I>
void sy2sb_he2hb_checkBadArgs(const rocblas_handle handle,
                              const I n,
                              const I kd,
                              const I nb,
                              T dA,
                              const I lda,
                              T dAband,
                              const I ldab,
                              T dTau)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(nullptr, n, kd, nb, dA, lda, dAband, ldab, dTau),
                          rocblas_status_invalid_handle);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_sy2sb_he2hb(handle, n, kd, nb, (T) nullptr, lda, dAband, ldab, dTau),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, n, kd, nb, dA, lda, (T) nullptr, ldab, dTau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, n, kd, nb, dA, lda, dAband, ldab, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, I(0), kd, nb, (T) nullptr, lda, (T) nullptr,
                                                ldab, (T) nullptr),
                          rocblas_status_success);
}

template <typename T, typename I>
void testing_sy2sb_he2hb_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    I n = 1;
    I kd = 1;
    I nb = 1;
    I lda = 1;
    I ldab = 3;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dAband(ldab, 1, ldab, 1);
    device_strided_batch_vector<T> dTau(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dAband.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());

    // check bad arguments
    sy2sb_he2hb_checkBadArgs(handle, n, kd, nb, dA.data(), lda, dAband.data(), ldab, dTau.data());
}

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void sy2sb_he2hb_initData(const rocblas_handle handle,
                          const I n,
                          const I kd, // unused
                          Td& dA,
                          const I lda,
                          Th& hA)
{
    if(CPU)
    {
        herand(rocblas_fill_full, n, hA[0], lda, true);
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
//
// dA       -- matrix on GPU, lda-by-n, lda >= n
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 3 kd - 1 (size needed for 2nd stage)
// dTau     -- output vector on GPU, length n-kd
//
// hARes    -- output matrix on CPU to copy GPU result, lda-by-n, lda >= n
// hAbandRes-- output band matrix on CPU to copy GPU result, ldab-by-n
// hTauRes  -- output vector on CPU to copy GPU result, length n-kd
//
// hA       -- matrix on CPU, lda-by-n, lda >= n
// hAband   -- output band matrix on CPU, ldab-by-n, ldab >= kd+1
// hTau     -- output vector on CPU, length n-kd
//
template <typename T, typename I, typename Td, typename Th>
void sy2sb_he2hb_getError(const rocblas_handle handle,
                          const I n,
                          const I kd,
                          const I nb,

                          Td& dA,
                          const I lda,
                          Td& dAband,
                          const I ldab,
                          Td& dTau,

                          Th& hARes,
                          Th& hAbandRes,
                          Th& hTauRes,

                          Th& hA,
                          Th& hAband,
                          Th& hTau,
                          double* max_err)
{
    using S = decltype(std::real(T{}));

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    // lwork for LAPACK hetrd_he2hb
    size_t lwork = n * kd + n * std::max<I>(kd, 128) + 2 * kd * kd;
    std::vector<T> hwork(lwork);

    // input data initialization
    sy2sb_he2hb_initData<true, true, T, I>(handle, n, kd, dA, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sy2sb_he2hb(handle, n, kd, nb, // opts
                                              dA.data(), lda, // A
                                              dAband.data(), ldab, // Aband
                                              dTau.data())); // tau
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hAbandRes.transfer_from(dAband));
    CHECK_HIP_ERROR(hTauRes.transfer_from(dTau));

    // CPU lapack
    cpu_sy2sb_he2hb(rocblas_fill_lower, n, kd, hA[0], lda, hAband[0], ldab, hTau[0], hwork.data(),
                    lwork);

    // error is ||hAband - hAbandRes|| / (n * ||hAband||)
    // and ||hTau - hTauRes|| / (n * ||hTau||)
    // using frobenius norm
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE)

    S dummy_work;
    S A_norm = cpu_lange('F', kd + 1, n, hAband[0], ldab, &dummy_work);
    S tau_norm = cpu_lange('F', n - kd, 1, hTau[0], n - kd, &dummy_work);

    // hAband -= hAbandRes (restricted to lower band structure).
    // hAbandRes has "don't care" values outside the band structure.
    I idiag = kd - 1;
    for(int j = 0; j < n; ++j)
    {
        for(int k = 0; k < kd + 1 && k + j < n; ++k)
        {
            hAband[0][k + j * ldab] -= hAbandRes[0][k + idiag + j * ldab];
        }
    }
    // hTau -= hTauRes
    cpu_axpy(n - kd, T(-1.0), hTauRes[0], 1, hTau[0], 1);

    // todo: check orthogonality of Q and backward error.
    double err;
    *max_err = 0;
    err = cpu_lange('F', kd + 1, n, hAband[0], ldab, &dummy_work);
    err /= n;
    if(A_norm != 0)
        err /= A_norm;
    *max_err = rocblas_max_nan(err, *max_err);

    err = cpu_lange('F', n - kd, 1, hTau[0], n - kd, &dummy_work);
    err /= n;
    if(tau_norm != 0)
        err /= tau_norm;
    *max_err = rocblas_max_nan(err, *max_err);
}

//------------------------------------------------------------------------------
// n        -- matrix dimension
// kd       -- desired bandwidth
// nb       -- outer blocksize to use
// dA       -- matrix on GPU, lda-by-n, lda >= n
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 3 kd - 1 (size needed for 2nd stage)
// hA       -- matrix on CPU, lda-by-n, lda >= n
template <typename T, typename I, typename Td, typename Th>
void sy2sb_he2hb_getPerfData(const rocblas_handle handle,
                             const I n,
                             const I kd,
                             const I nb,
                             Td& dA,
                             const I lda,
                             Td& dAband,
                             const I ldab,
                             Td& dTau,
                             Th& hA,
                             Th& hAband,
                             Th& hTau,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    rocsolver_timer timer;
    double start, time;
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    if(!perf)
    {
        sy2sb_he2hb_initData<true, false, T, I>(handle, n, kd, dA, lda, hA);

        // lwork for LAPACK hetrd_he2hb
        size_t lwork = n * kd + n * std::max<I>(kd, 128) + 2 * kd * kd;
        std::vector<T> hwork(lwork);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_sy2sb_he2hb(rocblas_fill_lower, n, kd, hA[0], lda, hAband[0], ldab, hTau[0],
                        hwork.data(), lwork);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sy2sb_he2hb_initData<true, false, T, I>(handle, n, kd, dA, lda, hA);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sy2sb_he2hb_initData<false, true, T, I>(handle, n, kd, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_sy2sb_he2hb(handle, n, kd, nb, // opts
                                                  dA.data(), lda, // A
                                                  dAband.data(), ldab, // Aband
                                                  dTau.data())); // tau
    }

    // gpu-lapack performance
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
        sy2sb_he2hb_initData<false, true, T, I>(handle, n, kd, dA, lda, hA);

        timer.start(stream);
        rocsolver_sy2sb_he2hb(handle, n, kd, nb, // opts
                              dA.data(), lda, // A
                              dAband.data(), ldab, // Aband
                              dTau.data()); // tau
        time = timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void testing_sy2sb_he2hb(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    I n = argus.get<I>("n");
    I kd = argus.get<I>("kd", 1);
    I nb = argus.get<I>("nb", kd);
    I lda = argus.get<I>("lda", n);
    // rocSolver 2nd stage hb2st needs 3*kd - 1. LAPACK needs kd+1.
    I ldab = 3 * kd - 1;

    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = lda * n;
    size_t size_Aband = ldab * n;
    size_t size_tau = std::max<I>(n - kd, 0);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_AbandRes = (argus.unit_check || argus.norm_check) ? size_Aband : 0;
    size_t size_tauRes = (argus.unit_check || argus.norm_check) ? size_tau : 0;

    // check invalid sizes
    bool invalid_size
        = (n < 0 || kd < 1 || nb < 1 || nb < kd || nb % kd != 0 || lda < n || ldab < 3 * kd - 1);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, n, kd, nb, // opts
                                                    (T*)nullptr, lda, // A
                                                    (T*)nullptr, ldab, // Aband
                                                    (T*)nullptr), // tau
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_sy2sb_he2hb(handle, n, kd, nb, // opts
                                                (T*)nullptr, lda, // A
                                                (T*)nullptr, ldab, // Aband
                                                (T*)nullptr)); // tau

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
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sy2sb_he2hb(handle, n, kd, nb, // opts
                                                    dA.data(), lda, // A
                                                    dAband.data(), ldab, // Aband
                                                    dTau.data()), // tau
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        sy2sb_he2hb_getError<T, I>(handle, n, kd, nb, dA, lda, dAband, ldab, dTau, hARes, hAbandRes,
                                   hTauRes, hA, hAband, hTau, &max_error);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        sy2sb_he2hb_getPerfData<T, I>(handle, n, kd, nb, dA, lda, dAband, ldab, dTau, hA, hAband,
                                      hTau, &gpu_time_used, &cpu_time_used, hot_calls,
                                      argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 10*machine_precision as tolerance
    // max_errors is already normalized, e.g., by n.
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 10);

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

#define EXTERN_TESTING_SY2SB_HE2HB(...) \
    extern template void testing_sy2sb_he2hb<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SY2SB_HE2HB, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
