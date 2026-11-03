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

#include "common/auxiliary/testing_equ_helpers.hpp"
#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

/* Tests for GEEQU (POW2 = false) and GEEQUB (POW2 = true).

   Besides the usual arguments, the tests accept:
   - scale: 'N' (default), 'L' or 'S'; magnitude of the entries of A (see equ_shift),
   - zero_row, zero_col: 1-based index of a row and a column of A set to zero (0 = none).

   All outputs are pre-filled with a sentinel (-1) both on the device and in the host reference, so
   that outputs that LAPACK leaves unset when info > 0 must keep the sentinel. */

template <bool POW2, typename T, typename I, typename S>
void geequ_checkBadArgs(const rocblas_handle handle,
                        const I m,
                        const I n,
                        T dA,
                        const I lda,
                        S dR,
                        S dC,
                        S drowcnd,
                        S dcolcnd,
                        S damax,
                        I* dinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, nullptr, m, n, dA, lda, dR, dC, drowcnd,
                                                 dcolcnd, damax, dinfo),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, (T) nullptr, lda, dR, dC,
                                                 drowcnd, dcolcnd, damax, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, dA, lda, (S) nullptr, dC,
                                                 drowcnd, dcolcnd, damax, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, dA, lda, dR, (S) nullptr,
                                                 drowcnd, dcolcnd, damax, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, dA, lda, dR, dC, (S) nullptr,
                                                 dcolcnd, damax, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, dA, lda, dR, dC, drowcnd,
                                                 (S) nullptr, damax, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, dA, lda, dR, dC, drowcnd,
                                                 dcolcnd, (S) nullptr, dinfo),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, dA, lda, dR, dC, drowcnd,
                                                 dcolcnd, damax, (I*)nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers (info is still required)
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, (I)0, n, (T) nullptr, lda,
                                                 (S) nullptr, (S) nullptr, (S) nullptr, (S) nullptr,
                                                 (S) nullptr, dinfo),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, (I)0, (T) nullptr, lda,
                                                 (S) nullptr, (S) nullptr, (S) nullptr, (S) nullptr,
                                                 (S) nullptr, dinfo),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, (I)0, n, (T) nullptr, lda,
                                                 (S) nullptr, (S) nullptr, (S) nullptr, (S) nullptr,
                                                 (S) nullptr, (I*)nullptr),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, (I)0, (T) nullptr, lda,
                                                 (S) nullptr, (S) nullptr, (S) nullptr, (S) nullptr,
                                                 (S) nullptr, (I*)nullptr),
                          rocblas_status_invalid_pointer);
}

template <bool POW2, typename T, typename I>
void testing_geequ_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    I m = 1;
    I n = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dR(1, 1, 1, 1);
    device_strided_batch_vector<S> dC(1, 1, 1, 1);
    device_strided_batch_vector<S> dscal(3, 1, 3, 1);
    device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dR.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check bad arguments
    geequ_checkBadArgs<POW2>(handle, m, n, dA.data(), lda, dR.data(), dC.data(), dscal.data(),
                             dscal.data() + 1, dscal.data() + 2, dinfo.data());
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void geequ_initData(const rocblas_handle handle,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    const char scale,
                    const I zero_row,
                    const I zero_col,
                    Th& hA)
{
    if(CPU)
    {
        // the padding rows hold large values; they must not be read
        equ_init_general(hA[0], m, n, lda, scale, zero_row, zero_col, false);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool POW2, typename T, typename I, typename S, typename Td, typename Sd, typename Id, typename Th, typename Sh, typename Ih>
void geequ_getError(const rocblas_handle handle,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& dR,
                    Sd& dC,
                    Sd& dscal,
                    Id& dinfo,
                    const char scale,
                    const I zero_row,
                    const I zero_col,
                    Th& hA,
                    Sh& hR,
                    Sh& hC,
                    Sh& hscal,
                    Sh& hRRes,
                    Sh& hCRes,
                    Sh& hscalRes,
                    Ih& hinfoRes,
                    double* max_err)
{
    // initialize data
    geequ_initData<true, true, T>(handle, m, n, dA, lda, scale, zero_row, zero_col, hA);

    // pre-fill all outputs with the same sentinel on the host and on the device
    const S sentinel = S(-1);
    for(I i = 0; i < m; i++)
        hR[0][i] = sentinel;
    for(I j = 0; j < n; j++)
        hC[0][j] = sentinel;
    for(int k = 0; k < 3; k++)
        hscal[0][k] = sentinel;
    hinfoRes[0][0] = -1;
    CHECK_HIP_ERROR(dR.transfer_from(hR));
    CHECK_HIP_ERROR(dC.transfer_from(hC));
    CHECK_HIP_ERROR(dscal.transfer_from(hscal));
    CHECK_HIP_ERROR(dinfo.transfer_from(hinfoRes));

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_geequ_geequb(POW2, handle, m, n, dA.data(), lda, dR.data(),
                                               dC.data(), dscal.data(), dscal.data() + 1,
                                               dscal.data() + 2, dinfo.data()));
    CHECK_HIP_ERROR(hRRes.transfer_from(dR));
    CHECK_HIP_ERROR(hCRes.transfer_from(dC));
    CHECK_HIP_ERROR(hscalRes.transfer_from(dscal));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    // CPU lapack
    rocblas_int hinfo = -1;
    if(POW2)
        cpu_geequb(m, n, hA[0], lda, hR[0], hC[0], hscal[0], hscal[0] + 1, hscal[0] + 2, &hinfo);
    else
        cpu_geequ(m, n, hA[0], lda, hR[0], hC[0], hscal[0], hscal[0] + 1, hscal[0] + 2, &hinfo);

    // error is the largest relative error of any output (including the sentinels of the outputs
    // that are not set); GEEQUB computes powers of 2 and its results must be exact
    const double eps = get_epsilon<T>();
    const double tol_RC = POW2 ? 0 : 2 * eps;
    const double tol_cnd = POW2 ? 0 : 4 * eps;
    double err_R = 0, err_C = 0;
    for(I i = 0; i < m; i++)
        err_R = std::max(err_R, equ_relerr(hRRes[0][i], hR[0][i]));
    for(I j = 0; j < n; j++)
        err_C = std::max(err_C, equ_relerr(hCRes[0][j], hC[0][j]));
    double err_rowcnd = equ_relerr(hscalRes[0][0], hscal[0][0]);
    double err_colcnd = equ_relerr(hscalRes[0][1], hscal[0][1]);
    double err_amax = equ_relerr(hscalRes[0][2], hscal[0][2]);

    EXPECT_EQ(hinfoRes[0][0], I(hinfo)) << "info differs";
    EXPECT_LE(err_R, tol_RC) << "R differs (info = " << hinfo << ")";
    EXPECT_LE(err_C, tol_RC) << "C differs (info = " << hinfo << ")";
    EXPECT_LE(err_rowcnd, tol_cnd) << "rowcnd differs (info = " << hinfo << ")";
    EXPECT_LE(err_colcnd, tol_cnd) << "colcnd differs (info = " << hinfo << ")";
    EXPECT_EQ(err_amax, 0) << "amax differs (info = " << hinfo << ")";

    *max_err = std::max({err_R, err_C, err_rowcnd, err_colcnd, err_amax});
    if(hinfoRes[0][0] != I(hinfo))
        *max_err = std::numeric_limits<double>::infinity();
}

template <bool POW2, typename T, typename I, typename S, typename Td, typename Sd, typename Id, typename Th, typename Sh>
void geequ_getPerfData(const rocblas_handle handle,
                       const I m,
                       const I n,
                       Td& dA,
                       const I lda,
                       Sd& dR,
                       Sd& dC,
                       Sd& dscal,
                       Id& dinfo,
                       const char scale,
                       const I zero_row,
                       const I zero_col,
                       Th& hA,
                       Sh& hR,
                       Sh& hC,
                       Sh& hscal,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    rocblas_int hinfo;

    // A is not modified, so the data is initialized only once
    geequ_initData<true, true, T>(handle, m, n, dA, lda, scale, zero_row, zero_col, hA);

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        if(POW2)
            cpu_geequb(m, n, hA[0], lda, hR[0], hC[0], hscal[0], hscal[0] + 1, hscal[0] + 2, &hinfo);
        else
            cpu_geequ(m, n, hA[0], lda, hR[0], hC[0], hscal[0], hscal[0] + 1, hscal[0] + 2, &hinfo);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        CHECK_ROCBLAS_ERROR(rocsolver_geequ_geequb(POW2, handle, m, n, dA.data(), lda, dR.data(),
                                                   dC.data(), dscal.data(), dscal.data() + 1,
                                                   dscal.data() + 2, dinfo.data()));
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
        timer.start(stream);
        rocsolver_geequ_geequb(POW2, handle, m, n, dA.data(), lda, dR.data(), dC.data(),
                               dscal.data(), dscal.data() + 1, dscal.data() + 2, dinfo.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool POW2, typename T, typename I>
void testing_geequ(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I lda = argus.get<I>("lda", m);
    char scale = argus.get<char>("scale", 'N');
    I zero_row = argus.get<I>("zero_row", 0);
    I zero_col = argus.get<I>("zero_col", 0);

    rocblas_int hot_calls = argus.iters;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, (T*)nullptr, lda,
                                                     (S*)nullptr, (S*)nullptr, (S*)nullptr,
                                                     (S*)nullptr, (S*)nullptr, (I*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_geequ_geequb(POW2, handle, m, n, (T*)nullptr, lda, (S*)nullptr,
                                                 (S*)nullptr, (S*)nullptr, (S*)nullptr, (S*)nullptr,
                                                 (I*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_R = m;
    size_t size_C = n;
    size_t size_scal = 3; // rowcnd, colcnd, amax
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_RRes = (argus.unit_check || argus.norm_check) ? size_R : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<S> hR(size_R, 1, size_R, 1);
    host_strided_batch_vector<S> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<S> hscal(size_scal, 1, size_scal, 1);
    host_strided_batch_vector<S> hRRes(size_RRes, 1, size_RRes, 1);
    host_strided_batch_vector<S> hCRes(size_CRes, 1, size_CRes, 1);
    host_strided_batch_vector<S> hscalRes(size_scal, 1, size_scal, 1);
    host_strided_batch_vector<I> hinfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dR(size_R, 1, size_R, 1);
    device_strided_batch_vector<S> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<S> dscal(size_scal, 1, size_scal, 1);
    device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_R)
        CHECK_HIP_ERROR(dR.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        // pre-fill the outputs with a sentinel; rowcnd = colcnd = 1, amax = 0 and info = 0 must be
        // set, and the other outputs must not be modified
        for(size_t i = 0; i < size_R; i++)
            hR[0][i] = S(-1);
        for(size_t j = 0; j < size_C; j++)
            hC[0][j] = S(-1);
        for(size_t k = 0; k < size_scal; k++)
            hscalRes[0][k] = S(-1);
        hinfoRes[0][0] = -1;
        CHECK_HIP_ERROR(dR.transfer_from(hR));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(dscal.transfer_from(hscalRes));
        CHECK_HIP_ERROR(dinfo.transfer_from(hinfoRes));

        EXPECT_ROCBLAS_STATUS(rocsolver_geequ_geequb(POW2, handle, m, n, dA.data(), lda, dR.data(),
                                                     dC.data(), dscal.data(), dscal.data() + 1,
                                                     dscal.data() + 2, dinfo.data()),
                              rocblas_status_success);

        CHECK_HIP_ERROR(hR.transfer_from(dR));
        CHECK_HIP_ERROR(hC.transfer_from(dC));
        CHECK_HIP_ERROR(hscalRes.transfer_from(dscal));
        CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
        EXPECT_EQ(hscalRes[0][0], S(1)) << "rowcnd";
        EXPECT_EQ(hscalRes[0][1], S(1)) << "colcnd";
        EXPECT_EQ(hscalRes[0][2], S(0)) << "amax";
        EXPECT_EQ(hinfoRes[0][0], 0) << "info";
        for(size_t i = 0; i < size_R; i++)
            EXPECT_EQ(hR[0][i], S(-1)) << "R[" << i << "] was modified";
        for(size_t j = 0; j < size_C; j++)
            EXPECT_EQ(hC[0][j], S(-1)) << "C[" << j << "] was modified";

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geequ_getError<POW2, T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, dinfo, scale, zero_row,
                                      zero_col, hA, hR, hC, hscal, hRRes, hCRes, hscalRes, hinfoRes,
                                      &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        geequ_getPerfData<POW2, T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, dinfo, scale,
                                         zero_row, zero_col, hA, hR, hC, hscal, &gpu_time_used,
                                         &cpu_time_used, hot_calls, argus.profile,
                                         argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // GEEQU: R and C within 2 eps, rowcnd and colcnd within 4 eps, amax exact
    // GEEQUB: all exact
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, POW2 ? 0 : 4);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("m", "n", "lda");
            rocsolver_bench_output(m, n, lda);

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

#define EXTERN_TESTING_GEEQU(...) extern template void testing_geequ<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEEQU,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
