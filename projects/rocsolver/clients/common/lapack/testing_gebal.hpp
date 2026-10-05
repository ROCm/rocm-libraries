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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

/** GEBAL_GENMATRIX modifies the random n-by-n matrix A according to mtype:
    - mtype = 0: A is left as is.
    - mtype = 1: A is replaced by D^(-1) A D, where D is diagonal with entries
                 ranging from 1e-6 to 1e6, so that A is badly scaled.
    - mtype = 2: as mtype = 1, but n/6 leading columns and n/6 trailing rows are
                 first made triangular so that eigenvalues can be isolated, and A is
                 then permuted symmetrically to hide this structure.
    - mtype = 3: A is upper triangular and then permuted symmetrically, so that
                 all the eigenvalues can be isolated.
    - mtype = 4: as mtype = 1, but with a zero row and a zero column.
    - mtype = 5: as mtype = 1, but the entries of D range from 1e-150 to 1e150
                 (1e-18 to 1e18 in single precision), to reach the limits of the
                 scaling factors.
    - mtype = 6: A is lower triangular.
    The permutation depends on seed so that the matrices of a batch differ. **/
template <typename T>
void gebal_genMatrix(const rocblas_int n,
                     T* A,
                     const rocblas_int lda,
                     const rocblas_int mtype,
                     const rocblas_int seed)
{
    using S = decltype(std::real(T{}));

    if(mtype == 2)
    {
        rocblas_int nlo = n / 6;
        rocblas_int nhi = n / 6;
        for(rocblas_int j = 0; j < nlo; j++)
            for(rocblas_int i = j + 1; i < n; i++)
                A[i + rocblas_stride(j) * lda] = 0;
        for(rocblas_int i = n - nhi; i < n; i++)
            for(rocblas_int j = 0; j < i; j++)
                A[i + rocblas_stride(j) * lda] = 0;
    }

    if(mtype == 3 || mtype == 6)
    {
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
                if((mtype == 3 && i > j) || (mtype == 6 && i < j))
                    A[i + rocblas_stride(j) * lda] = 0;
    }

    if(mtype == 4 && n > 7)
    {
        for(rocblas_int i = 0; i < n; i++)
            A[i + 3 * rocblas_stride(lda)] = 0;
        for(rocblas_int j = 0; j < n; j++)
            A[7 + rocblas_stride(j) * lda] = 0;
    }

    if((mtype == 1 || mtype == 2 || mtype == 4 || mtype == 5) && n > 1)
    {
        double range = 6.0;
        if(mtype == 5)
            range = std::is_same<S, float>::value ? 18.0 : 150.0;
        std::vector<double> d(n);
        for(rocblas_int i = 0; i < n; i++)
            d[i] = std::pow(10.0, -range + 2.0 * range * i / (n - 1));
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
                A[i + rocblas_stride(j) * lda] = A[i + rocblas_stride(j) * lda] * T(d[j] / d[i]);
    }

    if(mtype == 2 || mtype == 3)
    {
        // deterministic pseudo-random permutation
        std::vector<rocblas_int> p(n);
        for(rocblas_int i = 0; i < n; i++)
            p[i] = i;
        uint32_t state = 12345u + 7919u * seed;
        for(rocblas_int i = n - 1; i > 0; i--)
        {
            state = state * 1664525u + 1013904223u;
            std::swap(p[i], p[state % (i + 1)]);
        }

        std::vector<T> B(size_t(n) * n);
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
                B[i + size_t(j) * n] = A[p[i] + p[j] * lda];
        for(rocblas_int j = 0; j < n; j++)
            for(rocblas_int i = 0; i < n; i++)
                A[i + rocblas_stride(j) * lda] = B[i + size_t(j) * n];
    }
}

template <bool STRIDED, typename T, typename S, typename U>
void gebal_checkBadArgs(const rocblas_handle handle,
                        const rocsolver_balance job,
                        const rocblas_int n,
                        T dA,
                        const rocblas_int lda,
                        const rocblas_stride stA,
                        U dIlo,
                        U dIhi,
                        S dScale,
                        const rocblas_stride stS,
                        const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gebal(STRIDED, nullptr, job, n, dA, lda, stA, dIlo, dIhi, dScale, stS, bc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, rocsolver_balance(0), n, dA, lda, stA,
                                          dIlo, dIhi, dScale, stS, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_gebal(STRIDED, handle, job, n, dA, lda, stA, dIlo, dIhi, dScale, stS, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, (T) nullptr, lda, stA, dIlo,
                                          dIhi, dScale, stS, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gebal(STRIDED, handle, job, n, dA, lda, stA, (U) nullptr, dIhi, dScale, stS, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gebal(STRIDED, handle, job, n, dA, lda, stA, dIlo, (U) nullptr, dScale, stS, bc),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_gebal(STRIDED, handle, job, n, dA, lda, stA, dIlo, dIhi, (S) nullptr, stS, bc),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, 0, (T) nullptr, lda, stA, dIlo,
                                          dIhi, (S) nullptr, stS, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, dA, lda, stA, (U) nullptr,
                                              (U) nullptr, dScale, stS, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gebal_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocsolver_balance job = rocsolver_balance_both;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stS = 1;
    rocblas_int bc = 1;

#ifdef ROCSOLVER_ENABLE_BALANCE
    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIlo(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIhi(1, 1, 1, 1);
        device_strided_batch_vector<S> dScale(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIlo.memcheck());
        CHECK_HIP_ERROR(dIhi.memcheck());
        CHECK_HIP_ERROR(dScale.memcheck());

        // check bad arguments
        gebal_checkBadArgs<STRIDED>(handle, job, n, dA.data(), lda, stA, dIlo.data(), dIhi.data(),
                                    dScale.data(), stS, bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIlo(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dIhi(1, 1, 1, 1);
        device_strided_batch_vector<S> dScale(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIlo.memcheck());
        CHECK_HIP_ERROR(dIhi.memcheck());
        CHECK_HIP_ERROR(dScale.memcheck());

        // check bad arguments
        gebal_checkBadArgs<STRIDED>(handle, job, n, dA.data(), lda, stA, dIlo.data(), dIhi.data(),
                                    dScale.data(), stS, bc);
    }
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void gebal_initData(const rocblas_handle handle,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_int bc,
                    Th& hA,
                    const rocblas_int mtype)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        for(rocblas_int b = 0; b < bc; ++b)
            gebal_genMatrix(n, hA[b], lda, mtype, b);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void gebal_getError(const rocblas_handle handle,
                    const rocsolver_balance job,
                    const rocblas_int n,
                    Td& dA,
                    const rocblas_int lda,
                    const rocblas_stride stA,
                    Ud& dIlo,
                    Ud& dIhi,
                    Sd& dScale,
                    const rocblas_stride stS,
                    const rocblas_int bc,
                    Th& hA,
                    Th& hARes,
                    Uh& hIlo,
                    Uh& hIhi,
                    Uh& hIloRes,
                    Uh& hIhiRes,
                    Sh& hScale,
                    Sh& hScaleRes,
                    const rocblas_int mtype,
                    double* max_err)
{
    // input data initialization
    gebal_initData<true, true, T>(handle, n, dA, lda, bc, hA, mtype);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_gebal(STRIDED, handle, job, n, dA.data(), lda, stA, dIlo.data(),
                                        dIhi.data(), dScale.data(), stS, bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIloRes.transfer_from(dIlo));
    CHECK_HIP_ERROR(hIhiRes.transfer_from(dIhi));
    CHECK_HIP_ERROR(hScaleRes.transfer_from(dScale));

    // CPU lapack
    rocblas_int info;
    for(rocblas_int b = 0; b < bc; ++b)
        cpu_gebal(job, n, hA[b], lda, hIlo[b], hIhi[b], hScale[b], &info);

    // the GPU and CPU results must coincide:
    // - error is ||hA - hARes|| / ||hA|| and max_i |hScale(i) - hScaleRes(i)| / |hScale(i)|
    //   (as the scaling factors are powers of two, these errors are zero unless the tiny
    //   differences in the computed norms change a balancing decision; a different
    //   decision on any factor, however small, gives an error of at least 1/2)
    // - ilo and ihi must be equal
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', n, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;

        for(rocblas_int i = 0; i < n; i++)
        {
            const double s = hScale[b][i];
            err = std::abs(double(hScaleRes[b][i]) - s) / (s != 0 ? std::abs(s) : 1.0);
            *max_err = err > *max_err ? err : *max_err;
        }

        EXPECT_EQ(hIlo[b][0], hIloRes[b][0]) << "where b = " << b;
        EXPECT_EQ(hIhi[b][0], hIhiRes[b][0]) << "where b = " << b;
        if(hIlo[b][0] != hIloRes[b][0] || hIhi[b][0] != hIhiRes[b][0])
            *max_err += 1;
    }
}

template <bool STRIDED, typename T, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void gebal_getPerfData(const rocblas_handle handle,
                       const rocsolver_balance job,
                       const rocblas_int n,
                       Td& dA,
                       const rocblas_int lda,
                       const rocblas_stride stA,
                       Ud& dIlo,
                       Ud& dIhi,
                       Sd& dScale,
                       const rocblas_stride stS,
                       const rocblas_int bc,
                       Th& hA,
                       Uh& hIlo,
                       Uh& hIhi,
                       Sh& hScale,
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
        gebal_initData<true, false, T>(handle, n, dA, lda, bc, hA, mtype);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_gebal(job, n, hA[b], lda, hIlo[b], hIhi[b], hScale[b], &info);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    gebal_initData<true, false, T>(handle, n, dA, lda, bc, hA, mtype);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        gebal_initData<false, true, T>(handle, n, dA, lda, bc, hA, mtype);

        CHECK_ROCBLAS_ERROR(rocsolver_gebal(STRIDED, handle, job, n, dA.data(), lda, stA,
                                            dIlo.data(), dIhi.data(), dScale.data(), stS, bc));
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
        gebal_initData<false, true, T>(handle, n, dA, lda, bc, hA, mtype);

        timer.start(stream);
        rocsolver_gebal(STRIDED, handle, job, n, dA.data(), lda, stA, dIlo.data(), dIhi.data(),
                        dScale.data(), stS, bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_gebal(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    char jobC = argus.get<char>("job");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int lda = argus.get<rocblas_int>("lda", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", rocblas_stride(lda) * n);
    rocblas_stride stS = argus.get<rocblas_stride>("strideS", n);
    rocblas_int mtype = argus.get<rocblas_int>("mtype", 2);

    rocsolver_balance job = char2rocsolver_balance(jobC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stSRes = (argus.unit_check || argus.norm_check) ? stS : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_S = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_SRes = (argus.unit_check || argus.norm_check) ? size_S : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_BALANCE
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, (T* const*)nullptr, lda,
                                                  stA, (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (S*)nullptr, stS, bc),
                                  rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, (T*)nullptr, lda, stA,
                                                  (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (S*)nullptr, stS, bc),
                                  rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || lda < 1 || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, (T* const*)nullptr, lda,
                                                  stA, (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (S*)nullptr, stS, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, (T*)nullptr, lda, stA,
                                                  (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                  (S*)nullptr, stS, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_gebal(STRIDED, handle, job, n, (T* const*)nullptr, lda, stA,
                                              (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                              (S*)nullptr, stS, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_gebal(STRIDED, handle, job, n, (T*)nullptr, lda, stA,
                                              (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                              (S*)nullptr, stS, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations (all cases)
    host_strided_batch_vector<rocblas_int> hIlo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hIhi(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hIloRes(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hIhiRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hScale(size_S, 1, stS, bc);
    host_strided_batch_vector<S> hScaleRes(size_SRes, 1, stSRes, bc);
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
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, dA.data(), lda, stA,
                                                  dIlo.data(), dIhi.data(), dScale.data(), stS, bc),
                                  rocblas_status_success);
            if(bc > 0)
            {
                // LAPACK returns ilo = 1 and ihi = 0
                CHECK_HIP_ERROR(hIloRes.transfer_from(dIlo));
                CHECK_HIP_ERROR(hIhiRes.transfer_from(dIhi));
                for(rocblas_int b = 0; b < bc; ++b)
                {
                    EXPECT_EQ(hIloRes[b][0], 1) << "where b = " << b;
                    EXPECT_EQ(hIhiRes[b][0], 0) << "where b = " << b;
                }
            }
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebal_getError<STRIDED, T>(handle, job, n, dA, lda, stA, dIlo, dIhi, dScale, stS, bc,
                                       hA, hARes, hIlo, hIhi, hIloRes, hIhiRes, hScale, hScaleRes,
                                       mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gebal_getPerfData<STRIDED, T>(handle, job, n, dA, lda, stA, dIlo, dIhi, dScale, stS, bc,
                                          hA, hIlo, hIhi, hScale, mtype, &gpu_time_used,
                                          &cpu_time_used, hot_calls, argus.profile,
                                          argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_gebal(STRIDED, handle, job, n, dA.data(), lda, stA,
                                                  dIlo.data(), dIhi.data(), dScale.data(), stS, bc),
                                  rocblas_status_success);
            if(bc > 0)
            {
                // LAPACK returns ilo = 1 and ihi = 0
                CHECK_HIP_ERROR(hIloRes.transfer_from(dIlo));
                CHECK_HIP_ERROR(hIhiRes.transfer_from(dIhi));
                for(rocblas_int b = 0; b < bc; ++b)
                {
                    EXPECT_EQ(hIloRes[b][0], 1) << "where b = " << b;
                    EXPECT_EQ(hIhiRes[b][0], 0) << "where b = " << b;
                }
            }
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            gebal_getError<STRIDED, T>(handle, job, n, dA, lda, stA, dIlo, dIhi, dScale, stS, bc,
                                       hA, hARes, hIlo, hIhi, hIloRes, hIhiRes, hScale, hScaleRes,
                                       mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            gebal_getPerfData<STRIDED, T>(handle, job, n, dA, lda, stA, dIlo, dIhi, dScale, stS, bc,
                                          hA, hIlo, hIhi, hScale, mtype, &gpu_time_used,
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
                rocsolver_bench_output("job", "n", "lda", "strideS", "batch_c");
                rocsolver_bench_output(jobC, n, lda, stS, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("job", "n", "lda", "strideA", "strideS", "batch_c");
                rocsolver_bench_output(jobC, n, lda, stA, stS, bc);
            }
            else
            {
                rocsolver_bench_output("job", "n", "lda");
                rocsolver_bench_output(jobC, n, lda);
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

#define EXTERN_TESTING_GEBAL(...) extern template void testing_gebal<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBAL, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
