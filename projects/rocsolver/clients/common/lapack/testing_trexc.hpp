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

#include <random>

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <bool STRIDED, typename U>
void trexc_checkBadArgs(const rocblas_handle handle,
                        const rocsolver_schur_vectors compq,
                        const rocblas_int n,
                        U dT,
                        const rocblas_int ldt,
                        const rocblas_stride stT,
                        U dQ,
                        const rocblas_int ldq,
                        const rocblas_stride stQ,
                        const rocblas_int ifst,
                        const rocblas_int ilst,
                        const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trexc(STRIDED, nullptr, compq, n, dT, ldt, stT, dQ, ldq, stQ, ifst, ilst, bc),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, rocsolver_schur_vectors(0), n, dT, ldt,
                                          stT, dQ, ldq, stQ, ifst, ilst, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, rocsolver_schur_vectors_initialize, n,
                                          dT, ldt, stT, dQ, ldq, stQ, ifst, ilst, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trexc(STRIDED, handle, compq, n, dT, ldt, stT, dQ, ldq, stQ, ifst, ilst, -1),
            rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, (U) nullptr, ldt, stT, dQ, ldq,
                                          stQ, ifst, ilst, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, dT, ldt, stT, (U) nullptr, ldq,
                                          stQ, ifst, ilst, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, 0, (U) nullptr, ldt, stT,
                                          (U) nullptr, ldq, stQ, ifst, ilst, bc),
                          rocblas_status_success);
    // Q is not referenced when compq = none
    EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, rocsolver_schur_vectors_none, n, dT, ldt,
                                          stT, (U) nullptr, ldq, stQ, ifst, ilst, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trexc(STRIDED, handle, compq, n, dT, ldt, stT, dQ, ldq, stQ, ifst, ilst, 0),
            rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_trexc_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocsolver_schur_vectors compq = rocsolver_schur_vectors_update;
    rocblas_int n = 2;
    rocblas_int ldt = 2;
    rocblas_int ldq = 2;
    rocblas_int ifst = 1;
    rocblas_int ilst = 2;
    rocblas_stride stT = 1;
    rocblas_stride stQ = 1;
    rocblas_int bc = 1;

#ifdef ROCSOLVER_ENABLE_HSEQR
    if(BATCHED)
    {
        device_batch_vector<T> dT(1, 1, 1);
        device_batch_vector<T> dQ(1, 1, 1);
        CHECK_HIP_ERROR(dT.memcheck());
        CHECK_HIP_ERROR(dQ.memcheck());

        // check bad arguments
        trexc_checkBadArgs<STRIDED>(handle, compq, n, dT.data(), ldt, stT, dQ.data(), ldq, stQ,
                                    ifst, ilst, bc);
    }
    else
    {
        device_strided_batch_vector<T> dT(1, 1, 1, 1);
        device_strided_batch_vector<T> dQ(1, 1, 1, 1);
        CHECK_HIP_ERROR(dT.memcheck());
        CHECK_HIP_ERROR(dQ.memcheck());

        // check bad arguments
        trexc_checkBadArgs<STRIDED>(handle, compq, n, dT.data(), ldt, stT, dQ.data(), ldq, stQ,
                                    ifst, ilst, bc);
    }
#endif
}

/** TREXC_INITDATA generates an n-by-n upper triangular matrix T and a matrix Q
    (with normally distributed entries) according to mtype:
    - mtype = 0: T as is.
    - mtype = 1: T scaled by a tiny factor (1e-300, or 1e-30 in single precision).
    - mtype = 2: T scaled by a huge factor (1e300, or 1e30 in single precision).
    - mtype = 3: T with repeated diagonal entries (T(k,k) = T(1,1) for odd k). **/
template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void trexc_initData(const rocblas_handle handle,
                    const rocsolver_schur_vectors compq,
                    const rocblas_int n,
                    Td& dT,
                    const rocblas_int ldt,
                    Td& dQ,
                    const rocblas_int ldq,
                    const rocblas_int bc,
                    Th& hT,
                    Th& hQ,
                    const rocblas_int mtype)
{
    using S = decltype(std::real(T{}));

    if(CPU)
    {
        const S factor = (mtype == 1) ? (std::is_same<S, float>::value ? S(1e-30) : S(1e-300))
            : (mtype == 2)            ? (std::is_same<S, float>::value ? S(1e30) : S(1e300))
                                      : S(1);
        for(rocblas_int b = 0; b < bc; ++b)
        {
            std::mt19937_64 rng(7919ull * n + 104729ull * b + 31ull * mtype);
            std::normal_distribution<double> gauss(0.0, 1.0);
            auto rnd = [&]() { return T(S(gauss(rng)), S(gauss(rng))); };
            for(rocblas_int j = 0; j < n; j++)
                for(rocblas_int i = 0; i < n; i++)
                    hT[b][i + size_t(j) * ldt] = (i <= j) ? factor * rnd() : T(0);
            if(mtype == 3)
                for(rocblas_int k = 2; k < n; k += 2)
                    hT[b][k + size_t(k) * ldt] = hT[b][0];
            if(compq == rocsolver_schur_vectors_update)
                for(rocblas_int j = 0; j < n; j++)
                    for(rocblas_int i = 0; i < n; i++)
                        hQ[b][i + size_t(j) * ldq] = rnd();
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dT.transfer_from(hT));
        if(compq == rocsolver_schur_vectors_update)
            CHECK_HIP_ERROR(dQ.transfer_from(hQ));
    }
}

template <bool STRIDED, typename T, typename Td, typename Th>
void trexc_getError(const rocblas_handle handle,
                    const rocsolver_schur_vectors compq,
                    const rocblas_int n,
                    Td& dT,
                    const rocblas_int ldt,
                    const rocblas_stride stT,
                    Td& dQ,
                    const rocblas_int ldq,
                    const rocblas_stride stQ,
                    const rocblas_int ifst,
                    const rocblas_int ilst,
                    const rocblas_int bc,
                    Th& hT,
                    Th& hQ,
                    Th& hTRes,
                    Th& hQRes,
                    const rocblas_int mtype,
                    double* max_err)
{
    const bool wantq = (compq == rocsolver_schur_vectors_update);

    // input data initialization
    trexc_initData<true, true, T>(handle, compq, n, dT, ldt, dQ, ldq, bc, hT, hQ, mtype);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_trexc(STRIDED, handle, compq, n, dT.data(), ldt, stT, dQ.data(),
                                        ldq, stQ, ifst, ilst, bc));
    CHECK_HIP_ERROR(hTRes.transfer_from(dT));
    if(wantq)
        CHECK_HIP_ERROR(hQRes.transfer_from(dQ));

    // CPU lapack
    rocblas_int info;
    for(rocblas_int b = 0; b < bc; ++b)
        cpu_trexc(compq, n, hT[b], ldt, hQ[b], ldq, ifst, ilst, &info);

    // error is ||hT - hTRes|| / ||hT|| and ||hQ - hQRes|| / ||hQ||
    // (both apply the same sequence of plane rotations, so the results
    // coincide up to rounding)
    double err;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        err = norm_error('F', n, n, ldt, hT[b], hTRes[b]);
        *max_err = err > *max_err ? err : *max_err;
        if(wantq)
        {
            err = norm_error('F', n, n, ldq, hQ[b], hQRes[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <bool STRIDED, typename T, typename Td, typename Th>
void trexc_getPerfData(const rocblas_handle handle,
                       const rocsolver_schur_vectors compq,
                       const rocblas_int n,
                       Td& dT,
                       const rocblas_int ldt,
                       const rocblas_stride stT,
                       Td& dQ,
                       const rocblas_int ldq,
                       const rocblas_stride stQ,
                       const rocblas_int ifst,
                       const rocblas_int ilst,
                       const rocblas_int bc,
                       Th& hT,
                       Th& hQ,
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
        trexc_initData<true, false, T>(handle, compq, n, dT, ldt, dQ, ldq, bc, hT, hQ, mtype);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cpu_trexc(compq, n, hT[b], ldt, hQ[b], ldq, ifst, ilst, &info);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    trexc_initData<true, false, T>(handle, compq, n, dT, ldt, dQ, ldq, bc, hT, hQ, mtype);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trexc_initData<false, true, T>(handle, compq, n, dT, ldt, dQ, ldq, bc, hT, hQ, mtype);

        CHECK_ROCBLAS_ERROR(rocsolver_trexc(STRIDED, handle, compq, n, dT.data(), ldt, stT,
                                            dQ.data(), ldq, stQ, ifst, ilst, bc));
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
        trexc_initData<false, true, T>(handle, compq, n, dT, ldt, dQ, ldq, bc, hT, hQ, mtype);

        timer.start(stream);
        rocsolver_trexc(STRIDED, handle, compq, n, dT.data(), ldt, stT, dQ.data(), ldq, stQ, ifst,
                        ilst, bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_trexc(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char compqC = argus.get<char>("compq");
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldt = argus.get<rocblas_int>("ldt", n);
    rocblas_int ldq = argus.get<rocblas_int>("ldq", n);
    rocblas_int ifst = argus.get<rocblas_int>("ifst", 1);
    rocblas_int ilst = argus.get<rocblas_int>("ilst", n);
    rocblas_stride stT = argus.get<rocblas_stride>("strideT", rocblas_stride(ldt) * n);
    rocblas_stride stQ = argus.get<rocblas_stride>("strideQ", rocblas_stride(ldq) * n);
    rocblas_int mtype = argus.get<rocblas_int>("mtype", 0);

    rocsolver_schur_vectors compq = char2rocsolver_schur_vectors(compqC);
    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;
    const bool wantq = (compq == rocsolver_schur_vectors_update);

    // check non-supported values
    if(compq != rocsolver_schur_vectors_none && compq != rocsolver_schur_vectors_update)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, (T* const*)nullptr, ldt,
                                                  stT, (T* const*)nullptr, ldq, stQ, ifst, ilst, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, (T*)nullptr, ldt, stT,
                                                  (T*)nullptr, ldq, stQ, ifst, ilst, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_T = size_t(ldt) * n;
    size_t size_Q = wantq ? size_t(ldq) * n : 1;
    stQ = wantq ? stQ : 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    bool check = (argus.unit_check || argus.norm_check);
    size_t size_TRes = check ? size_T : 0;
    size_t size_QRes = check ? size_Q : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_HSEQR
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, (T* const*)nullptr, ldt,
                                                  stT, (T* const*)nullptr, ldq, stQ, ifst, ilst, bc),
                                  rocblas_status_not_implemented);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, (T*)nullptr, ldt, stT,
                                                  (T*)nullptr, ldq, stQ, ifst, ilst, bc),
                                  rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || ldt < n || ldt < 1 || ldq < 1 || (wantq && ldq < n) || bc < 0
                         || (n > 0 && (ifst < 1 || ifst > n || ilst < 1 || ilst > n)));
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, (T* const*)nullptr, ldt,
                                                  stT, (T* const*)nullptr, ldq, stQ, ifst, ilst, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, (T*)nullptr, ldt, stT,
                                                  (T*)nullptr, ldq, stQ, ifst, ilst, bc),
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
            CHECK_ALLOC_QUERY(rocsolver_trexc(STRIDED, handle, compq, n, (T* const*)nullptr, ldt,
                                              stT, (T* const*)nullptr, ldq, stQ, ifst, ilst, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_trexc(STRIDED, handle, compq, n, (T*)nullptr, ldt, stT,
                                              (T*)nullptr, ldq, stQ, ifst, ilst, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hT(size_T, 1, bc);
        host_batch_vector<T> hTRes(size_TRes, 1, bc);
        host_batch_vector<T> hQ(size_Q, 1, bc);
        host_batch_vector<T> hQRes(size_QRes, 1, bc);
        device_batch_vector<T> dT(size_T, 1, bc);
        device_batch_vector<T> dQ(size_Q, 1, bc);
        if(size_T)
            CHECK_HIP_ERROR(dT.memcheck());
        CHECK_HIP_ERROR(dQ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, dT.data(), ldt, stT,
                                                  dQ.data(), ldq, stQ, ifst, ilst, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(check)
            trexc_getError<STRIDED, T>(handle, compq, n, dT, ldt, stT, dQ, ldq, stQ, ifst, ilst, bc,
                                       hT, hQ, hTRes, hQRes, mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            trexc_getPerfData<STRIDED, T>(handle, compq, n, dT, ldt, stT, dQ, ldq, stQ, ifst, ilst,
                                          bc, hT, hQ, mtype, &gpu_time_used, &cpu_time_used,
                                          hot_calls, argus.profile, argus.profile_kernels,
                                          argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hT(size_T, 1, stT, bc);
        host_strided_batch_vector<T> hTRes(size_TRes, 1, stT, bc);
        host_strided_batch_vector<T> hQ(size_Q, 1, stQ, bc);
        host_strided_batch_vector<T> hQRes(size_QRes, 1, stQ, bc);
        device_strided_batch_vector<T> dT(size_T, 1, stT, bc);
        device_strided_batch_vector<T> dQ(size_Q, 1, stQ, bc);
        if(size_T)
            CHECK_HIP_ERROR(dT.memcheck());
        CHECK_HIP_ERROR(dQ.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_trexc(STRIDED, handle, compq, n, dT.data(), ldt, stT,
                                                  dQ.data(), ldq, stQ, ifst, ilst, bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(check)
            trexc_getError<STRIDED, T>(handle, compq, n, dT, ldt, stT, dQ, ldq, stQ, ifst, ilst, bc,
                                       hT, hQ, hTRes, hQRes, mtype, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            trexc_getPerfData<STRIDED, T>(handle, compq, n, dT, ldt, stT, dQ, ldq, stQ, ifst, ilst,
                                          bc, hT, hQ, mtype, &gpu_time_used, &cpu_time_used,
                                          hot_calls, argus.profile, argus.profile_kernels,
                                          argus.perf);
    }

    // validate results for rocsolver-test
    // using max(n, 32) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, std::max(n, 32));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("compq", "n", "ldt", "ldq", "ifst", "ilst", "batch_c");
                rocsolver_bench_output(compqC, n, ldt, ldq, ifst, ilst, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("compq", "n", "ldt", "strideT", "ldq", "strideQ", "ifst",
                                       "ilst", "batch_c");
                rocsolver_bench_output(compqC, n, ldt, stT, ldq, stQ, ifst, ilst, bc);
            }
            else
            {
                rocsolver_bench_output("compq", "n", "ldt", "ldq", "ifst", "ilst");
                rocsolver_bench_output(compqC, n, ldt, ldq, ifst, ilst);
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

#define EXTERN_TESTING_TREXC(...) extern template void testing_trexc<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_TREXC, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_COMPLEX_TYPE, APPLY_STAMP)
