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

/* Tests for POEQU (POW2 = false) and POEQUB (POW2 = true).

   Besides the usual arguments, the tests accept:
   - scale: 'N' (default), 'L' or 'S'; magnitude of the diagonal of A (see equ_shift),
   - zero_diag, neg_diag: 1-based index of a diagonal element of A set to zero or to a negative
     value (0 = none).

   Only the (real part of the) diagonal of A may be read: the rest of A holds NaN or large values.
   All outputs are pre-filled with a sentinel (-1) both on the device and in the host reference, so
   that outputs that LAPACK leaves unset when info > 0 must keep the sentinel. */

template <bool POW2, typename T, typename I, typename S>
void poequ_checkBadArgs(const rocblas_handle handle,
                        const I n,
                        T dA,
                        const I lda,
                        S dS,
                        S dscond,
                        S damax,
                        I* dinfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_poequ_poequb(POW2, nullptr, n, dA, lda, dS, dscond, damax, dinfo),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_poequ_poequb(POW2, handle, n, (T) nullptr, lda, dS, dscond, damax, dinfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_poequ_poequb(POW2, handle, n, dA, lda, (S) nullptr, dscond, damax, dinfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_poequ_poequb(POW2, handle, n, dA, lda, dS, (S) nullptr, damax, dinfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_poequ_poequb(POW2, handle, n, dA, lda, dS, dscond, (S) nullptr, dinfo),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_poequ_poequb(POW2, handle, n, dA, lda, dS, dscond, damax, (I*)nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers (info is still required)
    EXPECT_ROCBLAS_STATUS(rocsolver_poequ_poequb(POW2, handle, (I)0, (T) nullptr, lda, (S) nullptr,
                                                 (S) nullptr, (S) nullptr, dinfo),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_poequ_poequb(POW2, handle, (I)0, (T) nullptr, lda, (S) nullptr,
                                                 (S) nullptr, (S) nullptr, (I*)nullptr),
                          rocblas_status_invalid_pointer);
}

template <bool POW2, typename T, typename I>
void testing_poequ_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    I n = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dS(1, 1, 1, 1);
    device_strided_batch_vector<S> dscal(2, 1, 2, 1);
    device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check bad arguments
    poequ_checkBadArgs<POW2>(handle, n, dA.data(), lda, dS.data(), dscal.data(), dscal.data() + 1,
                             dinfo.data());
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void poequ_initData(const rocblas_handle handle,
                    const I n,
                    Td& dA,
                    const I lda,
                    const char scale,
                    const I zero_diag,
                    const I neg_diag,
                    Th& hA)
{
    if(CPU)
    {
        equ_init_diagonal(hA[0], n, lda, scale, zero_diag, neg_diag);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool POW2, typename T, typename I, typename S, typename Td, typename Sd, typename Id, typename Th, typename Sh, typename Ih>
void poequ_getError(const rocblas_handle handle,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& dS,
                    Sd& dscal,
                    Id& dinfo,
                    const char scale,
                    const I zero_diag,
                    const I neg_diag,
                    Th& hA,
                    Sh& hS,
                    Sh& hscal,
                    Sh& hSRes,
                    Sh& hscalRes,
                    Ih& hinfoRes,
                    double* max_err)
{
    // initialize data
    poequ_initData<true, true, T>(handle, n, dA, lda, scale, zero_diag, neg_diag, hA);

    // pre-fill all outputs with the same sentinel on the host and on the device
    const S sentinel = S(-1);
    for(I i = 0; i < n; i++)
        hS[0][i] = sentinel;
    for(int k = 0; k < 2; k++)
        hscal[0][k] = sentinel;
    hinfoRes[0][0] = -1;
    CHECK_HIP_ERROR(dS.transfer_from(hS));
    CHECK_HIP_ERROR(dscal.transfer_from(hscal));
    CHECK_HIP_ERROR(dinfo.transfer_from(hinfoRes));

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_poequ_poequb(POW2, handle, n, dA.data(), lda, dS.data(),
                                               dscal.data(), dscal.data() + 1, dinfo.data()));
    CHECK_HIP_ERROR(hSRes.transfer_from(dS));
    CHECK_HIP_ERROR(hscalRes.transfer_from(dscal));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));

    // CPU lapack
    rocblas_int hinfo = -1;
    if(POW2)
        cpu_poequb(n, hA[0], lda, hS[0], hscal[0], hscal[0] + 1, &hinfo);
    else
        cpu_poequ(n, hA[0], lda, hS[0], hscal[0], hscal[0] + 1, &hinfo);

    // error is the largest relative error of any output (including the sentinels of the outputs
    // that are not set); POEQUB computes powers of 2 and its scale factors must be exact
    const double eps = get_epsilon<T>();
    double err_S = 0;
    for(I i = 0; i < n; i++)
        err_S = std::max(err_S, equ_relerr(hSRes[0][i], hS[0][i]));
    double err_scond = equ_relerr(hscalRes[0][0], hscal[0][0]);
    double err_amax = equ_relerr(hscalRes[0][1], hscal[0][1]);

    EXPECT_EQ(hinfoRes[0][0], I(hinfo)) << "info differs";
    EXPECT_LE(err_S, POW2 ? 0 : 2 * eps) << "S differs (info = " << hinfo << ")";
    EXPECT_LE(err_scond, 4 * eps) << "scond differs (info = " << hinfo << ")";
    EXPECT_EQ(err_amax, 0) << "amax differs (info = " << hinfo << ")";

    *max_err = std::max({err_S, err_scond, err_amax});
    if(hinfoRes[0][0] != I(hinfo))
        *max_err = std::numeric_limits<double>::infinity();
}

template <bool POW2, typename T, typename I, typename S, typename Td, typename Sd, typename Id, typename Th, typename Sh>
void poequ_getPerfData(const rocblas_handle handle,
                       const I n,
                       Td& dA,
                       const I lda,
                       Sd& dS,
                       Sd& dscal,
                       Id& dinfo,
                       const char scale,
                       const I zero_diag,
                       const I neg_diag,
                       Th& hA,
                       Sh& hS,
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
    poequ_initData<true, true, T>(handle, n, dA, lda, scale, zero_diag, neg_diag, hA);

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        if(POW2)
            cpu_poequb(n, hA[0], lda, hS[0], hscal[0], hscal[0] + 1, &hinfo);
        else
            cpu_poequ(n, hA[0], lda, hS[0], hscal[0], hscal[0] + 1, &hinfo);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        CHECK_ROCBLAS_ERROR(rocsolver_poequ_poequb(POW2, handle, n, dA.data(), lda, dS.data(),
                                                   dscal.data(), dscal.data() + 1, dinfo.data()));
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
        rocsolver_poequ_poequb(POW2, handle, n, dA.data(), lda, dS.data(), dscal.data(),
                               dscal.data() + 1, dinfo.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool POW2, typename T, typename I>
void testing_poequ(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    I n = argus.get<I>("n");
    I lda = argus.get<I>("lda", n);
    char scale = argus.get<char>("scale", 'N');
    I zero_diag = argus.get<I>("zero_diag", 0);
    I neg_diag = argus.get<I>("neg_diag", 0);

    rocblas_int hot_calls = argus.iters;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_poequ_poequb(POW2, handle, n, (T*)nullptr, lda, (S*)nullptr,
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
        CHECK_ALLOC_QUERY(rocsolver_poequ_poequb(POW2, handle, n, (T*)nullptr, lda, (S*)nullptr,
                                                 (S*)nullptr, (S*)nullptr, (I*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_S = n;
    size_t size_scal = 2; // scond, amax
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_SRes = (argus.unit_check || argus.norm_check) ? size_S : 0;

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<S> hS(size_S, 1, size_S, 1);
    host_strided_batch_vector<S> hscal(size_scal, 1, size_scal, 1);
    host_strided_batch_vector<S> hSRes(size_SRes, 1, size_SRes, 1);
    host_strided_batch_vector<S> hscalRes(size_scal, 1, size_scal, 1);
    host_strided_batch_vector<I> hinfoRes(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dS(size_S, 1, size_S, 1);
    device_strided_batch_vector<S> dscal(size_scal, 1, size_scal, 1);
    device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_S)
        CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    // check quick return
    if(n == 0)
    {
        // pre-fill the outputs with a sentinel; scond = 1, amax = 0 and info = 0 must be set
        for(size_t k = 0; k < size_scal; k++)
            hscalRes[0][k] = S(-1);
        hinfoRes[0][0] = -1;
        CHECK_HIP_ERROR(dscal.transfer_from(hscalRes));
        CHECK_HIP_ERROR(dinfo.transfer_from(hinfoRes));

        EXPECT_ROCBLAS_STATUS(rocsolver_poequ_poequb(POW2, handle, n, dA.data(), lda, dS.data(),
                                                     dscal.data(), dscal.data() + 1, dinfo.data()),
                              rocblas_status_success);

        CHECK_HIP_ERROR(hscalRes.transfer_from(dscal));
        CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
        EXPECT_EQ(hscalRes[0][0], S(1)) << "scond";
        EXPECT_EQ(hscalRes[0][1], S(0)) << "amax";
        EXPECT_EQ(hinfoRes[0][0], 0) << "info";

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        poequ_getError<POW2, T, I, S>(handle, n, dA, lda, dS, dscal, dinfo, scale, zero_diag,
                                      neg_diag, hA, hS, hscal, hSRes, hscalRes, hinfoRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        poequ_getPerfData<POW2, T, I, S>(handle, n, dA, lda, dS, dscal, dinfo, scale, zero_diag,
                                         neg_diag, hA, hS, hscal, &gpu_time_used, &cpu_time_used,
                                         hot_calls, argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // S within 2 eps (exact for POEQUB), scond within 4 eps, amax exact
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 4);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "lda");
            rocsolver_bench_output(n, lda);

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

#define EXTERN_TESTING_POEQU(...) extern template void testing_poequ<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_POEQU,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
