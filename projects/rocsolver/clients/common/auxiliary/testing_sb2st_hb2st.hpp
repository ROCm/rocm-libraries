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

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename I, typename Ud, typename Uh>
void sb2st_hb2st_initData(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const I n,
                          const I kd,
                          Ud& dAband,
                          const I ldab,
                          Uh& hAband)
{
    if(CPU)
    {
        // Upper band doesn't need kd-th diagonal.
        hbrand(n, kd, kd - 1, hAband[0], ldab);
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dAband.transfer_from(hAband));
    }
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void sb2st_hb2st_checkBadArgs(const rocblas_handle handle,
                              const I n,
                              const I kd,
                              T* dAband,
                              const I ldab,
                              decltype(std::real(T{}))* dD,
                              decltype(std::real(T{}))* dE,
                              T* dV,
                              const I ldv,
                              T* dTau)
{
    using S = decltype(std::real(T{}));

    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(nullptr, rocblas_fill_lower, n, kd, dAband, ldab,
                                                dD, dE, dV, ldv, dTau),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_upper, n, kd, dAband, ldab, dD,
                                                dE, dV, ldv, dTau),
                          rocblas_status_not_implemented);

    // sizes
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, -1, kd, dAband, ldab,
                                                dD, dE, dV, ldv, dTau),
                          rocblas_status_invalid_size);
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, -1, dAband, ldab, dD,
                                                dE, dV, ldv, dTau),
                          rocblas_status_invalid_size);
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, dAband,
                                                3 * kd - 2, dD, dE, dV, ldv, dTau),
                          rocblas_status_invalid_size);
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, dAband, ldab, dD,
                                                dE, dV, 2 * kd - 2, dTau),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, (T*)nullptr,
                                                ldab, dD, dE, dV, ldv, dTau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, dAband, ldab,
                                                (S*)nullptr, dE, dV, ldv, dTau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, dAband, ldab, dD,
                                                (S*)nullptr, dV, ldv, dTau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, dAband, ldab, dD,
                                                dE, (T*)nullptr, ldv, dTau),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, dAband, ldab, dD,
                                                dE, dV, ldv, (T*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, 0, kd, (T*)nullptr,
                                                ldab, (S*)nullptr, (S*)nullptr, (T*)nullptr, ldv,
                                                (T*)nullptr),
                          rocblas_status_success);
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void testing_sb2st_hb2st_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    I n = 2;
    I kd = 1;
    I ldab = 3 * kd - 1; // == 2
    I ldv = 2 * kd - 1; // == 1

    // nv = nv_blocks * kd = 1 * 1 = 1
    I nv = 1;

    // memory allocations
    device_strided_batch_vector<T> dAband(ldab * n, 1, ldab * n, 1);
    device_strided_batch_vector<S> dD(n, 1, n, 1);
    device_strided_batch_vector<S> dE(n - 1, 1, n - 1, 1);
    device_strided_batch_vector<T> dV(ldv * nv, 1, ldv * nv, 1);
    device_strided_batch_vector<T> dTau(nv, 1, nv, 1);
    CHECK_HIP_ERROR(dAband.memcheck());
    CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dE.memcheck());
    CHECK_HIP_ERROR(dV.memcheck());
    CHECK_HIP_ERROR(dTau.memcheck());

#ifndef ROCSOLVER_ENABLE_EIG_2STAGE
    // hb2st is gated behind ROCSOLVER_ENABLE_EIG_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, rocblas_fill_lower, n, kd, dAband.data(), ldab,
                                                dD.data(), dE.data(), dV.data(), ldv, dTau.data()),
                          rocblas_status_not_implemented);
    return;
#endif

    // check bad arguments
    sb2st_hb2st_checkBadArgs(handle, n, kd, dAband.data(), ldab, dD.data(), dE.data(), dV.data(),
                             ldv, dTau.data());
}

//------------------------------------------------------------------------------
template <typename T, typename I, typename Ud, typename Td, typename Uh, typename Th>
void sb2st_hb2st_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const I n,
                          const I kd,
                          Ud& dAband,
                          const I ldab,
                          Td& dD,
                          Td& dE,
                          Ud& dV,
                          const I ldv,
                          Ud& dTau,
                          Uh& hAband,
                          Uh& hAbandRes,
                          Th& hDRes,
                          Th& hERes,
                          Th& hW,
                          double* max_err)
{
    using S = decltype(std::real(T{}));
    using std::abs, std::imag, std::real;

    I idiag = kd - 1;

    // input data initialization
    sb2st_hb2st_initData<true, true, T, I>(handle, uplo, n, kd, dAband, ldab, hAband);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_sb2st_hb2st(handle, uplo, n, kd, dAband.data(), ldab, dD.data(),
                                              dE.data(), dV.data(), ldv, dTau.data()));
    CHECK_HIP_ERROR(hAbandRes.transfer_from(dAband));
    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hERes.transfer_from(dE));

    S err = 0;
    *max_err = 0;
    if constexpr(rocblas_is_complex<T>)
    {
        // Check that diag is real to working precision.
        for(I j = 0; j < n; ++j)
        {
            err = rocblas_max_nan(err, abs(imag(hAbandRes[0][idiag + j * ldab])));
        }
        *max_err = rocblas_max_nan(err, *max_err);

        // Check that subdiag is real to working precision.
        err = 0;
        for(I j = 0; j < n - 1; ++j)
        {
            err = rocblas_max_nan(err, abs(imag(hAbandRes[0][idiag + 1 + j * ldab])));
        }
        *max_err = rocblas_max_nan(err, *max_err);
    }

    // Check that diag( A ) == D.
    err = 0;
    for(I j = 0; j < n; ++j)
    {
        err += hDRes[0][j] != real(hAbandRes[0][idiag + j * ldab]);
    }
    *max_err = rocblas_max_nan(err, *max_err);

    // Check that diag( A, -1 ) == E.
    err = 0;
    for(I j = 0; j < n - 1; ++j)
    {
        err += hERes[0][j] != real(hAbandRes[0][idiag + 1 + j * ldab]);
    }
    *max_err = rocblas_max_nan(err, *max_err);

    // Compute eigenvalues of tridiagonal matrix
    cpu_sterf(n, hDRes.data(), hERes.data());

    // CPU lapack
    // Compute eigenvalues of banded matrix
    int info;
    int worksize = rocblas_is_complex<T> ? int(n) : std::max(1, 3 * int(n) - 2);
    std::vector<T> work(worksize, T(0));
    int worksize_real = rocblas_is_complex<T> ? std::max(1, 3 * int(n) - 2) : 0;
    std::vector<S> work_real(worksize_real, S(0));
    T dummy; // for eigenvectors, Z
    // Assuming lower.
    cpu_sbev_hbev(rocblas_evect_none, uplo, int(n), int(kd), &hAband[0][idiag], int(ldab),
                  hW.data(), &dummy, 1, work.data(), work_real.data(), &info);

    // Compare CPU and GPU eigval results in hW and hDRes, respectively.
    err = norm_error('F', 1, n, 1, hW.data(), hDRes.data());
    *max_err = rocblas_max_nan(*max_err, err);

    // Orthogonality of V and backward error checked in unmtr_hb2st tester.
}

//------------------------------------------------------------------------------
template <typename T, typename I, typename Td, typename Ud, typename Uh>
void sb2st_hb2st_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const I n,
                             const I kd,
                             Ud& dAband,
                             const I ldab,
                             Td& dD,
                             Td& dE,
                             Ud& dV,
                             const I ldv,
                             Ud& dTau,
                             Uh& hAband,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf)
{
    rocsolver_timer timer;
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        using S = decltype(std::real(T{}));
        std::vector<T> hAband_cpu(hAband[0], hAband[0] + ldab * n);
        std::vector<S> hD_cpu(n);
        std::vector<S> hE_cpu(n - 1);
        std::vector<T> work_cpu(n);

        *cpu_time_used = get_time_us_no_sync();
        cpu_sb2st_hb2st<T>(uplo, n, kd, hAband_cpu.data(), ldab, hD_cpu.data(), hE_cpu.data(),
                           work_cpu.data());
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sb2st_hb2st_initData<true, false, T, I>(handle, uplo, n, kd, dAband, ldab, hAband);

    // cold calls
    double start, time;
    for(int iter = 0; iter < 2; iter++)
    {
        sb2st_hb2st_initData<false, true, T, I>(handle, uplo, n, kd, dAband, ldab, hAband);

        CHECK_ROCBLAS_ERROR(rocsolver_sb2st_hb2st(handle, uplo, n, kd, dAband.data(), ldab,
                                                  dD.data(), dE.data(), dV.data(), ldv, dTau.data()));
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
        sb2st_hb2st_initData<false, true, T, I>(handle, uplo, n, kd, dAband, ldab, hAband);

        timer.start(stream);
        CHECK_ROCBLAS_ERROR(rocsolver_sb2st_hb2st(handle, uplo, n, kd, dAband.data(), ldab,
                                                  dD.data(), dE.data(), dV.data(), ldv, dTau.data()));
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void testing_sb2st_hb2st(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    I n = argus.get<I>("n");
    I kd = argus.get<I>("kd");
    I ldab = argus.get<I>("ldab", 3 * kd - 1);
    I ldv = argus.get<I>("ldv", 2 * kd - 1);

    // V is ldv x nv
    I nt = 0;
    if(kd > 0)
        nt = ceildiv(n - 1, kd);
    I nv_blocks = nt * (nt + 1) / 2;
    I nv = nv_blocks * kd;

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

#ifndef ROCSOLVER_ENABLE_EIG_2STAGE
    // hb2st is gated behind ROCSOLVER_ENABLE_EIG_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, uplo, n, kd, // opts
                                                (T*)nullptr, ldab, // Aband
                                                (S*)nullptr, (S*)nullptr, // D, E
                                                (T*)nullptr, ldv, // V
                                                (T*)nullptr), // tau
                          rocblas_status_not_implemented);
    return;
#endif

    // determine sizes
    size_t size_Aband = ldab * n;
    size_t size_V = ldv * nv;
    size_t size_Tau = nv;
    size_t size_D = n;
    size_t size_E = n - 1;
    size_t size_W = size_D;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ares = (argus.unit_check || argus.norm_check) ? size_Aband : 0;
    size_t size_Dres = (argus.unit_check || argus.norm_check) ? size_D : 0;
    size_t size_Eres = (argus.unit_check || argus.norm_check) ? size_E : 0;
    size_t size_Vres = 0; // todo: not yet checked

    // check unimplemented uplo
    bool not_implemented = (uplo == rocblas_fill_upper);
    if(not_implemented)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, uplo, n, kd, (T*)nullptr, ldab,
                                                    (S*)nullptr, (S*)nullptr, (T*)nullptr, ldv,
                                                    (T*)nullptr),
                              rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // check invalid sizes
    bool invalid_size = (n < 0 || kd < 1 || ldab < 3 * kd - 1 || ldv < 2 * kd - 1);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, uplo, n, kd, (T*)nullptr, ldab,
                                                    (S*)nullptr, (S*)nullptr, (T*)nullptr, ldv,
                                                    (T*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));

        CHECK_ALLOC_QUERY(rocsolver_sb2st_hb2st(handle, uplo, n, kd, (T*)nullptr, ldab, (S*)nullptr,
                                                (S*)nullptr, (T*)nullptr, ldv, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hAband(size_Aband, 1, size_Aband, 1);
    host_strided_batch_vector<S> hW(size_W, 1, size_W, 1);
    host_strided_batch_vector<T> hAbandRes(size_Ares, 1, size_Ares, 1);
    host_strided_batch_vector<S> hDRes(size_Dres, 1, size_Dres, 1);
    host_strided_batch_vector<S> hERes(size_Eres, 1, size_Eres, 1);

    device_strided_batch_vector<T> dAband(size_Aband, 1, size_Aband, 1);
    device_strided_batch_vector<T> dV(size_V, 1, size_V, 1);
    device_strided_batch_vector<T> dTau(size_Tau, 1, size_Tau, 1);
    device_strided_batch_vector<S> dD(size_D, 1, size_D, 1);
    device_strided_batch_vector<S> dE(size_E, 1, size_E, 1);

    if(size_Aband)
        CHECK_HIP_ERROR(dAband.memcheck());
    if(size_V)
        CHECK_HIP_ERROR(dV.memcheck());
    if(size_Tau)
        CHECK_HIP_ERROR(dTau.memcheck());
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_sb2st_hb2st(handle, uplo, n, kd, dAband.data(), ldab,
                                                    dD.data(), dE.data(), dV.data(), ldv, dTau.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);
        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        sb2st_hb2st_getError<T, I>(handle, uplo, n, kd, dAband, ldab, dD, dE, dV, ldv, dTau, hAband,
                                   hAbandRes, hDRes, hERes, hW, &max_error);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        sb2st_hb2st_getPerfData<T, I>(handle, uplo, n, kd, dAband, ldab, dD, dE, dV, ldv, dTau,
                                      hAband, &gpu_time_used, &cpu_time_used, hot_calls,
                                      argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "kd", "ldab", "ldv");
            rocsolver_bench_output(uploC, n, kd, ldab, ldv);
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

#define EXTERN_TESTING_SB2ST_HB2ST(...) \
    extern template void testing_sb2st_hb2st<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_SB2ST_HB2ST, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
