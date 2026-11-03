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

/* Tests for LAQSY (HERM = false) and LAQHE (HERM = true).

   The argument equed ('N', or 'Y' or 'B'; default 'Y') selects whether the scaling must be applied,
   and the test-only argument equ_mode selects how it is forced:
   - 'F' (default): through scond (0.5 or 0.05), with amax = 0.75,
   - 'T': amax below the small threshold (scond = 0.5); the scaling is applied and equed is ignored,
   - 'H': same with amax above the large threshold,
   - 'E': scond = 0.1 and amax = small exactly; no scaling must be applied and equed is ignored,
   - 'G' and 'P': S, scond and amax are computed from A by POEQU ('G') or POEQUB ('P'), on the device
     for the device run and on the host for the reference; equed is ignored.
   The triangle of A that is not referenced and the padding rows hold NaN and must not be modified.
   The diagonal of A has nonzero imaginary parts (complex types), which LAQHE sets to zero when it
   applies the scaling. */

template <bool HERM, typename T, typename I, typename S>
void laqsy_laqhe_checkBadArgs(const rocblas_handle handle,
                              const rocblas_fill uplo,
                              const I n,
                              T dA,
                              const I lda,
                              S dS,
                              S dscond,
                              S damax,
                              rocsolver_equilibration* dequed)
{
    using E = rocsolver_equilibration*;

    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqsy_laqhe(HERM, nullptr, uplo, n, dA, lda, dS, dscond, damax, dequed),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_laqsy_laqhe(HERM, handle, rocblas_fill_full, n, dA, lda, dS,
                                                dscond, damax, dequed),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqsy_laqhe(HERM, handle, rocblas_fill(0), n, dA, lda, dS, dscond, damax, dequed),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqsy_laqhe(HERM, handle, uplo, n, (T) nullptr, lda, dS, dscond, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA, lda, (S) nullptr, dscond, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA, lda, dS, (S) nullptr, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA, lda, dS, dscond, (S) nullptr, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA, lda, dS, dscond, damax, (E) nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers (equed is still required)
    EXPECT_ROCBLAS_STATUS(rocsolver_laqsy_laqhe(HERM, handle, uplo, (I)0, (T) nullptr, lda,
                                                (S) nullptr, (S) nullptr, (S) nullptr, dequed),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_laqsy_laqhe(HERM, handle, uplo, (I)0, (T) nullptr, lda,
                                                (S) nullptr, (S) nullptr, (S) nullptr, (E) nullptr),
                          rocblas_status_invalid_pointer);
}

template <bool HERM, typename T, typename I>
void testing_laqsy_laqhe_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_fill uplo = rocblas_fill_upper;
    I n = 1;
    I lda = 1;

    // memory allocation
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<S> dS(1, 1, 1, 1);
    device_strided_batch_vector<S> dscal(2, 1, 2, 1);
    device_strided_batch_vector<rocsolver_equilibration> dequed(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dequed.memcheck());

    // check bad arguments
    laqsy_laqhe_checkBadArgs<HERM>(handle, uplo, n, dA.data(), lda, (const S*)dS.data(),
                                   (const S*)dscal.data(), (const S*)dscal.data() + 1, dequed.data());
}

/* Equilibration that LAQSY/LAQHE must apply for the given equed and equ_mode (see above); returns 0
   for the modes in which it is not known beforehand. */
inline rocsolver_equilibration laqsy_laqhe_expected(const char equedC, const char mode)
{
    switch(mode)
    {
    case 'F':
        return char2rocsolver_equilibration(equedC) == rocsolver_equilibration_none
            ? rocsolver_equilibration_none
            : rocsolver_equilibration_both;
    case 'T':
    case 'H': return rocsolver_equilibration_both;
    case 'E': return rocsolver_equilibration_none;
    default: return static_cast<rocsolver_equilibration>(0);
    }
}

template <bool CPU, bool GPU, bool HERM, typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void laqsy_laqhe_initData(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const I n,
                          Td& dA,
                          const I lda,
                          Sd& dS,
                          Sd& dscal,
                          const char equedC,
                          const char mode,
                          Th& hA,
                          Sh& hS,
                          Sh& hscal)
{
    const bool chain = (mode == 'G' || mode == 'P');

    if(CPU)
    {
        // the diagonal has a positive real part spanning many orders of magnitude
        equ_init_triangle(hA[0], uplo, n, lda);

        if(chain)
        {
            // scalings computed on the host
            rocblas_int hinfo;
            if(mode == 'P')
                cpu_poequb(n, hA[0], lda, hS[0], hscal[0], hscal[0] + 1, &hinfo);
            else
                cpu_poequ(n, hA[0], lda, hS[0], hscal[0], hscal[0] + 1, &hinfo);
        }
        else
        {
            // random S; scond and amax select the equilibration
            for(I i = 0; i < n; i++)
                hS[0][i] = equ_positive<S>(equ_rand_int(-equ_spread / 2, equ_spread / 2));

            S& scond = hscal[0][0];
            S& amax = hscal[0][1];
            switch(mode)
            {
            case 'T':
            case 'H':
                scond = S(0.5);
                amax = (mode == 'T') ? equ_small<S>() / 2 : equ_large<S>() * 2;
                break;
            case 'E':
                scond = S(0.1);
                amax = equ_small<S>();
                break;
            default:
            {
                bool none = char2rocsolver_equilibration(equedC) == rocsolver_equilibration_none;
                scond = none ? S(0.5) : S(0.05);
                amax = S(0.75);
                break;
            }
            }
        }
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        if(chain)
        {
            // compute the scalings on the device
            device_strided_batch_vector<I> dinfo(1, 1, 1, 1);
            CHECK_HIP_ERROR(dinfo.memcheck());
            CHECK_ROCBLAS_ERROR(rocsolver_poequ_poequb(mode == 'P', handle, n, dA.data(), lda,
                                                       dS.data(), dscal.data(), dscal.data() + 1,
                                                       dinfo.data()));
        }
        else
        {
            CHECK_HIP_ERROR(dS.transfer_from(hS));
            CHECK_HIP_ERROR(dscal.transfer_from(hscal));
        }
    }
}

template <bool HERM, typename T, typename I, typename S>
void laqsy_laqhe_cpu(const rocblas_fill uplo,
                     const I n,
                     T* A,
                     const I lda,
                     S* scal,
                     const S scond,
                     const S amax,
                     char* equed)
{
    char uploC = rocblas2char_fill(uplo);
    if constexpr(HERM)
        cpu_laqhe(uploC, n, A, lda, scal, scond, amax, equed);
    else
        cpu_laqsy(uploC, n, A, lda, scal, scond, amax, equed);
}

template <bool HERM, typename T, typename I, typename S, typename Td, typename Sd, typename Ed, typename Th, typename Sh, typename Eh>
void laqsy_laqhe_getError(const rocblas_handle handle,
                          const rocblas_fill uplo,
                          const I n,
                          Td& dA,
                          const I lda,
                          Sd& dS,
                          Sd& dscal,
                          Ed& dequed,
                          const char equedC,
                          const char mode,
                          Th& hA,
                          Th& hARes,
                          Sh& hS,
                          Sh& hscal,
                          Eh& hequedRes,
                          double* max_err)
{
    // initialize data
    laqsy_laqhe_initData<true, true, HERM, T, I, S>(handle, uplo, n, dA, lda, dS, dscal, equedC,
                                                    mode, hA, hS, hscal);

    // equed is pre-filled with an invalid value
    hequedRes[0][0] = static_cast<rocsolver_equilibration>(0);
    CHECK_HIP_ERROR(dequed.transfer_from(hequedRes));

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA.data(), lda, dS.data(),
                                              dscal.data(), dscal.data() + 1, dequed.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hequedRes.transfer_from(dequed));

    // CPU lapack
    char hequedC = '?';
    laqsy_laqhe_cpu<HERM>(uplo, n, hA[0], lda, hS[0], hscal[0][0], hscal[0][1], &hequedC);
    rocsolver_equilibration hequed = char2rocsolver_equilibration(hequedC);

    // check the reference against the equilibration selected by the test
    rocsolver_equilibration expected = laqsy_laqhe_expected(equedC, mode);
    if(expected != static_cast<rocsolver_equilibration>(0))
        EXPECT_EQ(hequed, expected) << "unexpected equilibration in the host reference";

    // error is the largest relative error of any element of the referenced triangle (the other
    // triangle and the padding rows must be bitwise unchanged)
    *max_err = equ_matrix_error(hARes[0], hA[0], n, lda, [n, uplo](I i, I j) {
        return i < n && (uplo == rocblas_fill_upper ? i <= j : i >= j);
    });

    EXPECT_EQ(hequedRes[0][0], hequed) << "equed differs";
    if(hequedRes[0][0] != hequed)
        *max_err = std::numeric_limits<double>::infinity();

    // LAQHE sets the imaginary part of the diagonal to zero when it scales A
    if(HERM && hequedRes[0][0] == rocsolver_equilibration_both)
    {
        for(I j = 0; j < n; j++)
        {
            EXPECT_EQ(std::imag(hARes[0][j + j * lda]), 0) << "where j = " << j;
            if(std::imag(hARes[0][j + j * lda]) != 0)
                *max_err = std::numeric_limits<double>::infinity();
        }
    }
}

template <bool HERM, typename T, typename I, typename S, typename Td, typename Sd, typename Ed, typename Th, typename Sh>
void laqsy_laqhe_getPerfData(const rocblas_handle handle,
                             const rocblas_fill uplo,
                             const I n,
                             Td& dA,
                             const I lda,
                             Sd& dS,
                             Sd& dscal,
                             Ed& dequed,
                             const char equedC,
                             const char mode,
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
    char hequedC;

    if(!perf)
    {
        laqsy_laqhe_initData<true, false, HERM, T, I, S>(handle, uplo, n, dA, lda, dS, dscal,
                                                         equedC, mode, hA, hS, hscal);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        laqsy_laqhe_cpu<HERM>(uplo, n, hA[0], lda, hS[0], hscal[0][0], hscal[0][1], &hequedC);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    laqsy_laqhe_initData<true, false, HERM, T, I, S>(handle, uplo, n, dA, lda, dS, dscal, equedC,
                                                     mode, hA, hS, hscal);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        laqsy_laqhe_initData<false, true, HERM, T, I, S>(handle, uplo, n, dA, lda, dS, dscal,
                                                         equedC, mode, hA, hS, hscal);

        CHECK_ROCBLAS_ERROR(rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA.data(), lda, dS.data(),
                                                  dscal.data(), dscal.data() + 1, dequed.data()));
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
        laqsy_laqhe_initData<false, true, HERM, T, I, S>(handle, uplo, n, dA, lda, dS, dscal,
                                                         equedC, mode, hA, hS, hscal);

        timer.start(stream);
        rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA.data(), lda, dS.data(), dscal.data(),
                              dscal.data() + 1, dequed.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool HERM, typename T, typename I>
void testing_laqsy_laqhe(Arguments& argus)
{
    using S = decltype(std::real(T{}));
    using E = rocsolver_equilibration;

    // get arguments
    rocblas_local_handle handle;
    char uploC = argus.get<char>("uplo");
    I n = argus.get<I>("n");
    I lda = argus.get<I>("lda", n);
    char equedC = argus.get<char>("equed", 'Y');
    char mode = argus.get<char>("equ_mode", 'F');

    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_laqsy_laqhe(HERM, handle, uplo, n, (T*)nullptr, lda,
                                                    (S*)nullptr, (S*)nullptr, (S*)nullptr,
                                                    (E*)nullptr),
                              rocblas_status_invalid_value);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_args);

        return;
    }

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_laqsy_laqhe(HERM, handle, uplo, n, (T*)nullptr, lda,
                                                    (S*)nullptr, (S*)nullptr, (S*)nullptr,
                                                    (E*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_laqsy_laqhe(HERM, handle, uplo, n, (T*)nullptr, lda,
                                                (S*)nullptr, (S*)nullptr, (S*)nullptr, (E*)nullptr));

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

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<S> hS(size_S, 1, size_S, 1);
    host_strided_batch_vector<S> hscal(size_scal, 1, size_scal, 1);
    host_strided_batch_vector<E> hequedRes(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dS(size_S, 1, size_S, 1);
    device_strided_batch_vector<S> dscal(size_scal, 1, size_scal, 1);
    device_strided_batch_vector<E> dequed(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_S)
        CHECK_HIP_ERROR(dS.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dequed.memcheck());

    // check quick return
    if(n == 0)
    {
        // equed is pre-filled with an invalid value and must be set to none
        hequedRes[0][0] = static_cast<E>(0);
        CHECK_HIP_ERROR(dequed.transfer_from(hequedRes));

        EXPECT_ROCBLAS_STATUS(rocsolver_laqsy_laqhe(HERM, handle, uplo, n, dA.data(), lda, dS.data(),
                                                    dscal.data(), dscal.data() + 1, dequed.data()),
                              rocblas_status_success);

        CHECK_HIP_ERROR(hequedRes.transfer_from(dequed));
        EXPECT_EQ(hequedRes[0][0], rocsolver_equilibration_none) << "equed";

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        laqsy_laqhe_getError<HERM, T, I, S>(handle, uplo, n, dA, lda, dS, dscal, dequed, equedC,
                                            mode, hA, hARes, hS, hscal, hequedRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        laqsy_laqhe_getPerfData<HERM, T, I, S>(handle, uplo, n, dA, lda, dS, dscal, dequed, equedC,
                                               mode, hA, hS, hscal, &gpu_time_used, &cpu_time_used,
                                               hot_calls, argus.profile, argus.profile_kernels,
                                               argus.perf);

    // validate results for rocsolver-test
    // LAPACK scales a(i,j) as (S(j) * S(i)) * a(i,j), with at most two roundings; when the scalings
    // come from POEQU, their own rounding errors add up
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, mode == 'G' ? 8 : 2);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("uplo", "n", "lda", "equed");
            rocsolver_bench_output(uploC, n, lda, equedC);

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

#define EXTERN_TESTING_LAQSY(...) \
    extern template void testing_laqsy_laqhe<false, __VA_ARGS__>(Arguments&);
#define EXTERN_TESTING_LAQHE(...) \
    extern template void testing_laqsy_laqhe<true, __VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LAQSY, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
INSTANTIATE(EXTERN_TESTING_LAQHE, FOREACH_COMPLEX_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
