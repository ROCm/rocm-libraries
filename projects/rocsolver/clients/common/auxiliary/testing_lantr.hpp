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

#include <cmath>
#include <limits>

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

/* The test-only argument "magnitude" scales the random test matrix close to the overflow ('B')
   or underflow ('S') threshold, so that the squares of the entries overflow or underflow while
   the norm itself is representable, or sets one referenced entry to NaN ('Q'; the norm must be
   NaN, as in LAPACK). It is 'N' (no scaling) by default. */
template <typename S, typename I>
S lantr_magnitude_scale(const char magnitude, const I m, const I n)
{
    if(magnitude == 'B')
    {
        // entries are at most 16 in absolute value, so that any norm is at most
        // 16 * max(m, n) times the scale, which is less than 2^(max_exponent - 2)
        int log2mn = 0;
        while((I(1) << log2mn) < std::max(m, n))
            log2mn++;
        return std::ldexp(S(1), std::numeric_limits<S>::max_exponent - 6 - log2mn);
    }
    else if(magnitude == 'S')
        return std::ldexp(S(1), std::numeric_limits<S>::min_exponent + 4);
    else
        return S(1);
}

// relative error of the computed norm; it is zero if both values are equal (including when both
// are infinity or both are NaN)
template <typename S>
double lantr_relerror(const S gold, const S comp)
{
    if(gold == comp || (std::isnan(gold) && std::isnan(comp)))
        return 0;
    if(!std::isfinite(gold) || !std::isfinite(comp))
        return std::numeric_limits<double>::infinity();

    double err = std::abs(double(gold) - double(comp));
    return (gold != 0) ? err / std::abs(double(gold)) : std::numeric_limits<double>::infinity();
}

template <typename T, typename I, typename S>
void lantr_checkBadArgs(const rocblas_handle handle,
                        const rocsolver_norm_type norm_type,
                        const rocblas_fill uplo,
                        const rocblas_diagonal diag,
                        const I m,
                        const I n,
                        T dA,
                        const I lda,
                        S dnorm)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_lantr(nullptr, norm_type, uplo, diag, m, n, dA, lda, dnorm),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_lantr(handle, static_cast<rocsolver_norm_type>(0), uplo, diag,
                                          m, n, dA, lda, dnorm),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lantr(handle, norm_type, rocblas_fill_full, diag, m, n, dA, lda, dnorm),
        rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lantr(handle, norm_type, uplo, rocblas_diagonal(0), m, n, dA, lda, dnorm),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lantr(handle, norm_type, uplo, diag, m, n, (T) nullptr, lda, dnorm),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_lantr(handle, norm_type, uplo, diag, m, n, dA, lda, (S) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lantr(handle, norm_type, uplo, diag, (I)0, n, (T) nullptr, lda, (S) nullptr),
        rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_lantr(handle, norm_type, uplo, diag, m, (I)0, (T) nullptr, lda, (S) nullptr),
        rocblas_status_success);
}

template <typename T, typename I>
void testing_lantr_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocsolver_norm_type norm_type = rocsolver_norm_type_one;
    rocblas_fill uplo = rocblas_fill_upper;
    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    I m = 1;
    I n = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dnorm(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dnorm.memcheck());

    // check bad arguments
    lantr_checkBadArgs(handle, norm_type, uplo, diag, m, n, dA.data(), lda, dnorm.data());
}

template <bool CPU, bool GPU, typename T, typename I, typename S, typename Td, typename Th>
void lantr_initData(const rocblas_handle handle,
                    const rocblas_fill uplo,
                    const rocblas_diagonal diag,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Th& hA,
                    const char magnitude)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        const S scale = lantr_magnitude_scale<S>(magnitude, m, n);
        const T nan = T(std::numeric_limits<S>::quiet_NaN());
        const bool unit = (diag == rocblas_diagonal_unit);

        for(I j = 0; j < n; j++)
        {
            for(I i = 0; i < lda; i++)
            {
                bool referenced = (i < m)
                    && (uplo == rocblas_fill_upper ? (unit ? i < j : i <= j)
                                                   : (unit ? i > j : i >= j));
                if(referenced)
                {
                    // mix the signs and scale to the requested magnitude
                    T factor = ((i + 2 * j) % 3 == 1) ? T(-scale) : T(scale);
                    hA[0][i + j * lda] = hA[0][i + j * lda] * factor;
                }
                else
                {
                    // the other part, the unit diagonal (if applicable) and the padding rows
                    // must not be referenced
                    hA[0][i + j * lda] = nan;
                }
            }
        }

        // a NaN in the referenced part: the last column of the first row (upper), or the first
        // column of the last row (lower), if referenced
        if(magnitude == 'Q' && m > 0 && n > 0)
        {
            const I i = (uplo == rocblas_fill_upper) ? 0 : m - 1;
            const I j = (uplo == rocblas_fill_upper) ? n - 1 : 0;
            if(!unit || i != j)
                hA[0][i + j * lda] = nan;
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void lantr_getError(const rocblas_handle handle,
                    const rocsolver_norm_type norm_type,
                    const rocblas_fill uplo,
                    const rocblas_diagonal diag,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& dnorm,
                    Th& hA,
                    Sh& hnorm,
                    Sh& hnorm_res,
                    const char magnitude,
                    double* max_err)
{
    // workspace for CPU lantr (only needed for the infinity-norm)
    std::vector<S> work(m);

    // initialize data
    lantr_initData<true, true, T, I, S>(handle, uplo, diag, m, n, dA, lda, hA, magnitude);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(
        rocsolver_lantr(handle, norm_type, uplo, diag, m, n, dA.data(), lda, dnorm.data()));
    CHECK_HIP_ERROR(hnorm_res.transfer_from(dnorm));

    // CPU lapack
    char norm = rocsolver2char_norm_type(norm_type);
    char uploC = rocblas2char_fill(uplo);
    char diagC = rocblas2char_diagonal(diag);
    hnorm[0][0] = cpu_lantr<T, S>(norm, uploC, diagC, m, n, hA[0], lda, work.data());

    // error is |hnorm - hnorm_res| / |hnorm|
    *max_err = lantr_relerror(hnorm[0][0], hnorm_res[0][0]);
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void lantr_getPerfData(const rocblas_handle handle,
                       const rocsolver_norm_type norm_type,
                       const rocblas_fill uplo,
                       const rocblas_diagonal diag,
                       const I m,
                       const I n,
                       Td& dA,
                       const I lda,
                       Sd& dnorm,
                       Th& hA,
                       Sh& hnorm,
                       const char magnitude,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    // workspace for CPU lantr
    std::vector<S> work(m);

    // only init CPU data once as it is not overwritten
    lantr_initData<true, false, T, I, S>(handle, uplo, diag, m, n, dA, lda, hA, magnitude);

    if(!perf)
    {
        // cpu-lapack performance (only if not in perf mode)
        char norm = rocsolver2char_norm_type(norm_type);
        char uploC = rocblas2char_fill(uplo);
        char diagC = rocblas2char_diagonal(diag);
        *cpu_time_used = get_time_us_no_sync();
        hnorm[0][0] = cpu_lantr<T, S>(norm, uploC, diagC, m, n, hA[0], lda, work.data());
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        lantr_initData<false, true, T, I, S>(handle, uplo, diag, m, n, dA, lda, hA, magnitude);

        CHECK_ROCBLAS_ERROR(
            rocsolver_lantr(handle, norm_type, uplo, diag, m, n, dA.data(), lda, dnorm.data()));
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
        lantr_initData<false, true, T, I, S>(handle, uplo, diag, m, n, dA, lda, hA, magnitude);

        timer.start(stream);
        rocsolver_lantr(handle, norm_type, uplo, diag, m, n, dA.data(), lda, dnorm.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T, typename I>
void testing_lantr(Arguments& argus)
{
    using S = real_t<T>;

    // get arguments
    rocblas_local_handle handle;
    char norm_typeC = argus.get<char>("norm_type");
    char uploC = argus.get<char>("uplo");
    char diagC = argus.get<char>("diag");
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I lda = argus.get<I>("lda", m);
    char magnitude = argus.get<char>("magnitude", 'N');

    rocsolver_norm_type norm_type = char2rocsolver_norm_type(norm_typeC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_diagonal diag = char2rocblas_diagonal(diagC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if((norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_frobenius
        && norm_type != rocsolver_norm_type_infinity && norm_type != rocsolver_norm_type_max)
       || (uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
       || (diag != rocblas_diagonal_unit && diag != rocblas_diagonal_non_unit))
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_lantr(handle, norm_type, uplo, diag, m, n, (T*)nullptr, lda, (S*)nullptr),
            rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_norm = 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_norm_res = (argus.unit_check || argus.norm_check) ? size_norm : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_lantr(handle, norm_type, uplo, diag, m, n, (T*)nullptr, lda, (S*)nullptr),
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
            rocsolver_lantr(handle, norm_type, uplo, diag, m, n, (T*)nullptr, lda, (S*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<S> hnorm(size_norm, 1, size_norm, 1);
    host_strided_batch_vector<S> hnorm_res(size_norm_res, 1, size_norm_res, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dnorm(size_norm, 1, size_norm, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dnorm.memcheck());

    // check quick return
    if(m == 0 || n == 0)
    {
        // the norm must be set to zero
        hnorm[0][0] = S(-1);
        CHECK_HIP_ERROR(dnorm.transfer_from(hnorm));

        EXPECT_ROCBLAS_STATUS(
            rocsolver_lantr(handle, norm_type, uplo, diag, m, n, dA.data(), lda, dnorm.data()),
            rocblas_status_success);

        CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
        EXPECT_EQ(hnorm[0][0], S(0));

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        lantr_getError<T, I, S>(handle, norm_type, uplo, diag, m, n, dA, lda, dnorm, hA, hnorm,
                                hnorm_res, magnitude, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        lantr_getPerfData<T, I, S>(handle, norm_type, uplo, diag, m, n, dA, lda, dnorm, hA, hnorm,
                                   magnitude, &gpu_time_used, &cpu_time_used, hot_calls,
                                   argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    if(argus.unit_check)
    {
        if(norm_type == rocsolver_norm_type_one)
        {
            ROCSOLVER_TEST_CHECK(T, max_error, m); // column sums of at most m elements
        }
        else if(norm_type == rocsolver_norm_type_infinity)
        {
            ROCSOLVER_TEST_CHECK(T, max_error, n); // row sums of at most n elements
        }
        else if(norm_type == rocsolver_norm_type_max)
        {
            // no arithmetic for real types; complex absolute values may differ in the last bit
            ROCSOLVER_TEST_CHECK(T, max_error, (rocblas_is_complex<T> ? 1 : 0));
        }
        else
        {
            ROCSOLVER_TEST_CHECK(T, max_error, m + n); // sum of squares of n columns of m elements
        }
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("norm_type", "uplo", "diag", "m", "n", "lda");
            rocsolver_bench_output(norm_typeC, uploC, diagC, m, n, lda);

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

#define EXTERN_TESTING_LANTR(...) extern template void testing_lantr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LANTR, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
