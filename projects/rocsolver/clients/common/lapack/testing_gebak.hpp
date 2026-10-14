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

#include "common/lapack/testing_gebal.hpp"
#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <bool STRIDED, typename T, typename S, typename U>
void gebak_checkBadArgs(const rocblas_handle handle,
                        const rocsolver_balance job,
                        const rocblas_side side,
                        const rocblas_int n,
                        U dIlo,
                        U dIhi,
                        S dScale,
                        const rocblas_stride stS,
                        const rocblas_int m,
                        T dV,
                        const rocblas_int ldv,
                        const rocblas_stride stV,
                        const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, nullptr, job, side, n, dIlo, dIhi, dScale, stS,
                                          m, dV, ldv, stV, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, rocsolver_balance(0), side, n, dIlo,
                                          dIhi, dScale, stS, m, dV, ldv, stV, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, rocblas_side_both, n, dIlo, dIhi,
                                          dScale, stS, m, dV, ldv, stV, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo, dIhi, dScale,
                                              stS, m, dV, ldv, stV, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, (U) nullptr, dIhi, dScale,
                                          stS, m, dV, ldv, stV, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo, (U) nullptr, dScale,
                                          stS, m, dV, ldv, stV, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo, dIhi, (S) nullptr,
                                          stS, m, dV, ldv, stV, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo, dIhi, dScale, stS, m,
                                          (T) nullptr, ldv, stV, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, 0, (U) nullptr, (U) nullptr,
                                          (S) nullptr, stS, m, (T) nullptr, ldv, stV, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo, dIhi, dScale, stS, 0,
                                          (T) nullptr, ldv, stV, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, (U) nullptr,
                                              (U) nullptr, dScale, stS, m, dV, ldv, stV, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gebak_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocsolver_balance job = rocsolver_balance_both;
    rocblas_side side = rocblas_side_right;
    rocblas_int n = 1;
    rocblas_int m = 1;
    rocblas_int ldv = 1;
    rocblas_stride stS = 1;
    rocblas_stride stV = 1;
    rocblas_int bc = 1;

#ifdef ROCSOLVER_ENABLE_BALANCE
    // memory allocations
    device_strided_batch_vector<rocblas_int> dIlo(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dIhi(1, 1, 1, 1);
    device_strided_batch_vector<S> dScale(1, 1, 1, 1);
    CHECK_HIP_ERROR(dIlo.memcheck());
    CHECK_HIP_ERROR(dIhi.memcheck());
    CHECK_HIP_ERROR(dScale.memcheck());

    if(BATCHED)
    {
        device_batch_vector<T> dV(1, 1, 1);
        CHECK_HIP_ERROR(dV.memcheck());

        // check bad arguments
        gebak_checkBadArgs<STRIDED>(handle, job, side, n, dIlo.data(), dIhi.data(), dScale.data(),
                                    stS, m, dV.data(), ldv, stV, bc);
    }
    else
    {
        device_strided_batch_vector<T> dV(1, 1, 1, 1);
        CHECK_HIP_ERROR(dV.memcheck());

        // check bad arguments
        gebak_checkBadArgs<STRIDED>(handle, job, side, n, dIlo.data(), dIhi.data(), dScale.data(),
                                    stS, m, dV.data(), ldv, stV, bc);
    }
#endif
}

/** GEBAK_INITDATA generates the balancing information (ilo, ihi and scale) by calling
    the host LAPACK GEBAL on a matrix built by gebal_genMatrix, and a random matrix V
    of eigenvectors. **/
template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void gebak_initData(const rocblas_handle handle,
                    const rocsolver_balance job,
                    const rocblas_int n,
                    Ud& dIlo,
                    Ud& dIhi,
                    Sd& dScale,
                    const rocblas_int m,
                    Td& dV,
                    const rocblas_int ldv,
                    const rocblas_int bc,
                    Uh& hIlo,
                    Uh& hIhi,
                    Sh& hScale,
                    Th& hV,
                    const rocblas_int mtype)
{
    if(CPU)
    {
        rocblas_init<T>(hV, true);

        std::vector<T> A(size_t(n) * n);
        rocblas_int info;
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(size_t k = 0; k < A.size(); k++)
                A[k] = random_generator<T>();
            gebal_genMatrix(n, A.data(), n, mtype, b);
            cpu_gebal(job, n, A.data(), n, hIlo[b], hIhi[b], hScale[b], &info);
        }

        // new random eigenvectors
        rocblas_init<T>(hV, false);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dIlo.transfer_from(hIlo));
        CHECK_HIP_ERROR(dIhi.transfer_from(hIhi));
        CHECK_HIP_ERROR(dScale.transfer_from(hScale));
        CHECK_HIP_ERROR(dV.transfer_from(hV));
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void gebak_getError(const rocblas_handle handle,
                    const rocsolver_balance job,
                    const rocblas_side side,
                    const rocblas_int n,
                    Ud& dIlo,
                    Ud& dIhi,
                    Sd& dScale,
                    const rocblas_stride stS,
                    const rocblas_int m,
                    Td& dV,
                    const rocblas_int ldv,
                    const rocblas_stride stV,
                    const rocblas_int bc,
                    Uh& hIlo,
                    Uh& hIhi,
                    Sh& hScale,
                    Th& hV,
                    Th& hVRes,
                    const rocblas_int mtype,
                    double* max_err)
{
    // input data initialization
    gebak_initData<true, true, T>(handle, job, n, dIlo, dIhi, dScale, m, dV, ldv, bc, hIlo, hIhi,
                                  hScale, hV, mtype);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo.data(), dIhi.data(),
                                        dScale.data(), stS, m, dV.data(), ldv, stV, bc));
    CHECK_HIP_ERROR(hVRes.transfer_from(dV));

    // CPU lapack
    rocblas_int info;
    for(rocblas_int b = 0; b < bc; ++b)
        cpu_gebak(job, side, n, hIlo[b][0], hIhi[b][0], hScale[b], m, hV[b], ldv, &info);

    // error is ||hV - hVRes|| / ||hV||
    // (the transformation only swaps rows and scales them by powers of two, so
    // the results should coincide)
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', n, m, ldv, hV[b], hVRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void gebak_getPerfData(const rocblas_handle handle,
                       const rocsolver_balance job,
                       const rocblas_side side,
                       const rocblas_int n,
                       Ud& dIlo,
                       Ud& dIhi,
                       Sd& dScale,
                       const rocblas_stride stS,
                       const rocblas_int m,
                       Td& dV,
                       const rocblas_int ldv,
                       const rocblas_stride stV,
                       const rocblas_int bc,
                       Uh& hIlo,
                       Uh& hIhi,
                       Sh& hScale,
                       Th& hV,
                       const rocblas_int mtype,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    rocblas_int info;

    if(!perf)
    {
        gebak_initData<true, false, T>(handle, job, n, dIlo, dIhi, dScale, m, dV, ldv, bc, hIlo,
                                       hIhi, hScale, hV, mtype);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_gebak(job, side, n, hIlo[b][0], hIhi[b][0], hScale[b], m, hV[b], ldv, &info);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gebak_initData<true, false, T>(handle, job, n, dIlo, dIhi, dScale, m, dV, ldv, bc, hIlo, hIhi,
                                   hScale, hV, mtype);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gebak_initData<false, true, T>(handle, job, n, dIlo, dIhi, dScale, m, dV, ldv, bc, hIlo,
                                       hIhi, hScale, hV, mtype);

        CHECK_ROCBLAS_ERROR(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo.data(), dIhi.data(),
                                            dScale.data(), stS, m, dV.data(), ldv, stV, bc));
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
        gebak_initData<false, true, T>(handle, job, n, dIlo, dIhi, dScale, m, dV, ldv, bc, hIlo,
                                       hIhi, hScale, hV, mtype);

        timer.start(stream);
        rocsolver_gebak(STRIDED, handle, job, side, n, dIlo.data(), dIhi.data(), dScale.data(), stS,
                        m, dV.data(), ldv, stV, bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gebak(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char jobC = argus.get<char>("job");
    char sideC = argus.get<char>("side");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int m = argus.get<rocblas_int>("m", n);
    rocblas_int ldv = argus.get<rocblas_int>("ldv", n);
    rocblas_stride stS = argus.get<rocblas_stride>("strideS", n);
    rocblas_stride stV = argus.get<rocblas_stride>("strideV", rocblas_stride(ldv) * m);
    rocblas_int mtype = argus.get<rocblas_int>("mtype", 2);

    rocsolver_balance job = char2rocsolver_balance(jobC);
    rocblas_side side = char2rocblas_side(sideC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stVRes = (argus.unit_check || argus.norm_check) ? stV : 0;

    // check non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, (rocblas_int*)nullptr,
                                                  (rocblas_int*)nullptr, (S*)nullptr, stS, m,
                                                  (T* const*)nullptr, ldv, stV, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n,
                                                  (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (S*)nullptr, stS, m, (T*)nullptr, ldv, stV, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_V = size_t(ldv) * m;
    size_t size_S = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_VRes = (argus.unit_check || argus.norm_check) ? size_V : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_BALANCE
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, (rocblas_int*)nullptr,
                                                  (rocblas_int*)nullptr, (S*)nullptr, stS, m,
                                                  (T* const*)nullptr, ldv, stV, bc),
                                  rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n,
                                                  (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (S*)nullptr, stS, m, (T*)nullptr, ldv, stV, bc),
                                  rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || m < 0 || ldv < n || ldv < 1 || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, (rocblas_int*)nullptr,
                                                  (rocblas_int*)nullptr, (S*)nullptr, stS, m,
                                                  (T* const*)nullptr, ldv, stV, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n,
                                                  (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (S*)nullptr, stS, m, (T*)nullptr, ldv, stV, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_gebak(STRIDED, handle, job, side, n, (rocblas_int*)nullptr,
                                              (rocblas_int*)nullptr, (S*)nullptr, stS, m,
                                              (T* const*)nullptr, ldv, stV, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_gebak(STRIDED, handle, job, side, n, (rocblas_int*)nullptr,
                                              (rocblas_int*)nullptr, (S*)nullptr, stS, m,
                                              (T*)nullptr, ldv, stV, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    host_strided_batch_vector<rocblas_int> hIlo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hIhi(1, 1, 1, bc);
    host_strided_batch_vector<S> hScale(size_S, 1, stS, bc);
    device_strided_batch_vector<rocblas_int> dIlo(1, 1, 1, bc);
    device_strided_batch_vector<rocblas_int> dIhi(1, 1, 1, bc);
    device_strided_batch_vector<S> dScale(size_S, 1, stS, bc);
    if(bc)
    {
        CHECK_HIP_ERROR(dIlo.memcheck());
        CHECK_HIP_ERROR(dIhi.memcheck());
    }
    if(size_S)
        CHECK_HIP_ERROR(dScale.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hV(size_V, 1, bc);
        host_batch_vector<T> hVRes(size_VRes, 1, bc);
        device_batch_vector<T> dV(size_V, 1, bc);
        if(size_V)
            CHECK_HIP_ERROR(dV.memcheck());

        // check quick return
        if(n == 0 || m == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo.data(),
                                                  dIhi.data(), dScale.data(), stS, m, dV.data(),
                                                  ldv, stV, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebak_getError<STRIDED, T>(handle, job, side, n, dIlo, dIhi, dScale, stS, m, dV, ldv,
                                       stV, bc, hIlo, hIhi, hScale, hV, hVRes, mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gebak_getPerfData<STRIDED, T>(handle, job, side, n, dIlo, dIhi, dScale, stS, m, dV, ldv,
                                          stV, bc, hIlo, hIhi, hScale, hV, mtype, &gpu_time_used,
                                          &cpu_time_used, hot_calls, argus.profile,
                                          argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hV(size_V, 1, stV, bc);
        host_strided_batch_vector<T> hVRes(size_VRes, 1, stVRes, bc);
        device_strided_batch_vector<T> dV(size_V, 1, stV, bc);
        if(size_V)
            CHECK_HIP_ERROR(dV.memcheck());

        // check quick return
        if(n == 0 || m == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebak(STRIDED, handle, job, side, n, dIlo.data(),
                                                  dIhi.data(), dScale.data(), stS, m, dV.data(),
                                                  ldv, stV, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebak_getError<STRIDED, T>(handle, job, side, n, dIlo, dIhi, dScale, stS, m, dV, ldv,
                                       stV, bc, hIlo, hIhi, hScale, hV, hVRes, mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gebak_getPerfData<STRIDED, T>(handle, job, side, n, dIlo, dIhi, dScale, stS, m, dV, ldv,
                                          stV, bc, hIlo, hIhi, hScale, hV, mtype, &gpu_time_used,
                                          &cpu_time_used, hot_calls, argus.profile,
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
            if(BATCHED)
            {
                rocsolver_bench_output("job", "side", "n", "m", "strideS", "ldv", "batch_c");
                rocsolver_bench_output(jobC, sideC, n, m, stS, ldv, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("job", "side", "n", "m", "strideS", "ldv", "strideV",
                                       "batch_c");
                rocsolver_bench_output(jobC, sideC, n, m, stS, ldv, stV, bc);
            }
            else
            {
                rocsolver_bench_output("job", "side", "n", "m", "ldv");
                rocsolver_bench_output(jobC, sideC, n, m, ldv);
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

#define EXTERN_TESTING_GEBAK(...) extern template void testing_gebak<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBAK, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
