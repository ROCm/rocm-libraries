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

/* Tests for LAQGE.

   The argument equed ('N', 'R', 'C' or 'B'; default 'B') selects the equilibration to be applied, and
   the test-only argument equ_mode selects how it is forced:
   - 'F' (default): through rowcnd and colcnd (0.5 or 0.05), with amax = 0.75,
   - 'T': amax below the small threshold (rowcnd = 0.5): row scaling is forced, and colcnd selects
     between equed = 'R' and 'B',
   - 'H': same with amax above the large threshold,
   - 'E': rowcnd = colcnd = 0.1 and amax = small exactly; no scaling must be applied,
   - 'G' and 'P': R, C, rowcnd, colcnd and amax are computed from A by GEEQU ('G') or GEEQUB ('P'),
     on the device for the device run and on the host for the reference; equed is ignored.
   The padding rows of A hold NaN and must not be modified. */

template <typename T, typename I, typename S>
void laqge_checkBadArgs(const rocblas_handle handle,
                        const I m,
                        const I n,
                        T dA,
                        const I lda,
                        S dR,
                        S dC,
                        S drowcnd,
                        S dcolcnd,
                        S damax,
                        rocsolver_equilibration* dequed)
{
    using E = rocsolver_equilibration*;

    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(nullptr, m, n, dA, lda, dR, dC, drowcnd, dcolcnd, damax, dequed),
        rocblas_status_invalid_handle);

    // values
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(handle, m, n, (T) nullptr, lda, dR, dC, drowcnd, dcolcnd, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(handle, m, n, dA, lda, (S) nullptr, dC, drowcnd, dcolcnd, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(handle, m, n, dA, lda, dR, (S) nullptr, drowcnd, dcolcnd, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(handle, m, n, dA, lda, dR, dC, (S) nullptr, dcolcnd, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(handle, m, n, dA, lda, dR, dC, drowcnd, (S) nullptr, damax, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(handle, m, n, dA, lda, dR, dC, drowcnd, dcolcnd, (S) nullptr, dequed),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_laqge(handle, m, n, dA, lda, dR, dC, drowcnd, dcolcnd, damax, (E) nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers (equed is still required)
    EXPECT_ROCBLAS_STATUS(rocsolver_laqge(handle, (I)0, n, (T) nullptr, lda, (S) nullptr,
                                          (S) nullptr, (S) nullptr, (S) nullptr, (S) nullptr, dequed),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_laqge(handle, m, (I)0, (T) nullptr, lda, (S) nullptr,
                                          (S) nullptr, (S) nullptr, (S) nullptr, (S) nullptr, dequed),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_laqge(handle, (I)0, n, (T) nullptr, lda, (S) nullptr, (S) nullptr,
                                          (S) nullptr, (S) nullptr, (S) nullptr, (E) nullptr),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_laqge(handle, m, (I)0, (T) nullptr, lda, (S) nullptr, (S) nullptr,
                                          (S) nullptr, (S) nullptr, (S) nullptr, (E) nullptr),
                          rocblas_status_invalid_pointer);
}

template <typename T, typename I>
void testing_laqge_bad_arg()
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
    device_strided_batch_vector<rocsolver_equilibration> dequed(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dR.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dequed.memcheck());

    // check bad arguments
    laqge_checkBadArgs(handle, m, n, dA.data(), lda, (const S*)dR.data(), (const S*)dC.data(),
                       (const S*)dscal.data(), (const S*)dscal.data() + 1,
                       (const S*)dscal.data() + 2, dequed.data());
}

/* Equilibration that LAQGE must apply for the given equed and equ_mode (see above); returns 0 for the
   modes in which it is not known beforehand. */
inline rocsolver_equilibration laqge_expected(const char equedC, const char mode)
{
    rocsolver_equilibration equed = char2rocsolver_equilibration(equedC);
    bool col = (equed == rocsolver_equilibration_column || equed == rocsolver_equilibration_both);
    switch(mode)
    {
    case 'F': return equed;
    case 'T':
    case 'H': return col ? rocsolver_equilibration_both : rocsolver_equilibration_row;
    case 'E': return rocsolver_equilibration_none;
    default: return static_cast<rocsolver_equilibration>(0);
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename S, typename Td, typename Sd, typename Th, typename Sh>
void laqge_initData(const rocblas_handle handle,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& dR,
                    Sd& dC,
                    Sd& dscal,
                    const char equedC,
                    const char mode,
                    Th& hA,
                    Sh& hR,
                    Sh& hC,
                    Sh& hscal)
{
    const bool chain = (mode == 'G' || mode == 'P');

    if(CPU)
    {
        if(chain)
        {
            // A with widely varying row and column magnitudes, and its scalings computed on the host
            rocblas_int hinfo;
            equ_init_general(hA[0], m, n, lda, 'N', I(0), I(0), true);
            if(mode == 'P')
                cpu_geequb(m, n, hA[0], lda, hR[0], hC[0], hscal[0], hscal[0] + 1, hscal[0] + 2,
                           &hinfo);
            else
                cpu_geequ(m, n, hA[0], lda, hR[0], hC[0], hscal[0], hscal[0] + 1, hscal[0] + 2,
                          &hinfo);
        }
        else
        {
            // random A, R and C; rowcnd, colcnd and amax select the equilibration
            for(I j = 0; j < n; j++)
            {
                for(I i = 0; i < m; i++)
                    hA[0][i + j * lda]
                        = equ_element<T>(equ_rand_int(-equ_spread / 2, equ_spread / 2));
                for(I i = m; i < lda; i++)
                    hA[0][i + j * lda] = equ_nan<T>();
            }
            for(I i = 0; i < m; i++)
                hR[0][i] = equ_positive<S>(equ_rand_int(-equ_spread / 2, equ_spread / 2));
            for(I j = 0; j < n; j++)
                hC[0][j] = equ_positive<S>(equ_rand_int(-equ_spread / 2, equ_spread / 2));

            rocsolver_equilibration equed = char2rocsolver_equilibration(equedC);
            bool row
                = (equed == rocsolver_equilibration_row || equed == rocsolver_equilibration_both);
            bool col
                = (equed == rocsolver_equilibration_column || equed == rocsolver_equilibration_both);
            S& rowcnd = hscal[0][0];
            S& colcnd = hscal[0][1];
            S& amax = hscal[0][2];
            switch(mode)
            {
            case 'T':
            case 'H':
                rowcnd = S(0.5);
                colcnd = col ? S(0.05) : S(0.5);
                amax = (mode == 'T') ? equ_small<S>() / 2 : equ_large<S>() * 2;
                break;
            case 'E':
                rowcnd = S(0.1);
                colcnd = S(0.1);
                amax = equ_small<S>();
                break;
            default:
                rowcnd = row ? S(0.05) : S(0.5);
                colcnd = col ? S(0.05) : S(0.5);
                amax = S(0.75);
                break;
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
            CHECK_ROCBLAS_ERROR(rocsolver_geequ_geequb(
                mode == 'P', handle, m, n, dA.data(), lda, dR.data(), dC.data(), dscal.data(),
                dscal.data() + 1, dscal.data() + 2, dinfo.data()));
        }
        else
        {
            CHECK_HIP_ERROR(dR.transfer_from(hR));
            CHECK_HIP_ERROR(dC.transfer_from(hC));
            CHECK_HIP_ERROR(dscal.transfer_from(hscal));
        }
    }
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Ed, typename Th, typename Sh, typename Eh>
void laqge_getError(const rocblas_handle handle,
                    const I m,
                    const I n,
                    Td& dA,
                    const I lda,
                    Sd& dR,
                    Sd& dC,
                    Sd& dscal,
                    Ed& dequed,
                    const char equedC,
                    const char mode,
                    Th& hA,
                    Th& hARes,
                    Sh& hR,
                    Sh& hC,
                    Sh& hscal,
                    Eh& hequedRes,
                    double* max_err)
{
    // initialize data
    laqge_initData<true, true, T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, equedC, mode, hA, hR,
                                        hC, hscal);

    // equed is pre-filled with an invalid value
    hequedRes[0][0] = static_cast<rocsolver_equilibration>(0);
    CHECK_HIP_ERROR(dequed.transfer_from(hequedRes));

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_laqge(handle, m, n, dA.data(), lda, dR.data(), dC.data(),
                                        dscal.data(), dscal.data() + 1, dscal.data() + 2,
                                        dequed.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hequedRes.transfer_from(dequed));

    // CPU lapack
    char hequedC = '?';
    cpu_laqge(m, n, hA[0], lda, hR[0], hC[0], hscal[0][0], hscal[0][1], hscal[0][2], &hequedC);
    rocsolver_equilibration hequed = char2rocsolver_equilibration(hequedC);

    // check the reference against the equilibration selected by the test
    rocsolver_equilibration expected = laqge_expected(equedC, mode);
    if(expected != static_cast<rocsolver_equilibration>(0))
        EXPECT_EQ(hequed, expected) << "unexpected equilibration in the host reference";

    // error is the largest relative error of any element of A (the padding rows must be bitwise
    // unchanged)
    *max_err = equ_matrix_error(hARes[0], hA[0], n, lda, [m](I i, I) { return i < m; });

    EXPECT_EQ(hequedRes[0][0], hequed) << "equed differs";
    if(hequedRes[0][0] != hequed)
        *max_err = std::numeric_limits<double>::infinity();
}

template <typename T, typename I, typename S, typename Td, typename Sd, typename Ed, typename Th, typename Sh>
void laqge_getPerfData(const rocblas_handle handle,
                       const I m,
                       const I n,
                       Td& dA,
                       const I lda,
                       Sd& dR,
                       Sd& dC,
                       Sd& dscal,
                       Ed& dequed,
                       const char equedC,
                       const char mode,
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
    char hequedC;

    if(!perf)
    {
        laqge_initData<true, false, T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, equedC, mode, hA,
                                             hR, hC, hscal);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cpu_laqge(m, n, hA[0], lda, hR[0], hC[0], hscal[0][0], hscal[0][1], hscal[0][2], &hequedC);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    laqge_initData<true, false, T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, equedC, mode, hA, hR,
                                         hC, hscal);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        laqge_initData<false, true, T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, equedC, mode, hA,
                                             hR, hC, hscal);

        CHECK_ROCBLAS_ERROR(rocsolver_laqge(handle, m, n, dA.data(), lda, dR.data(), dC.data(),
                                            dscal.data(), dscal.data() + 1, dscal.data() + 2,
                                            dequed.data()));
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
        laqge_initData<false, true, T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, equedC, mode, hA,
                                             hR, hC, hscal);

        timer.start(stream);
        rocsolver_laqge(handle, m, n, dA.data(), lda, dR.data(), dC.data(), dscal.data(),
                        dscal.data() + 1, dscal.data() + 2, dequed.data());
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T, typename I>
void testing_laqge(Arguments& argus)
{
    using S = decltype(std::real(T{}));
    using E = rocsolver_equilibration;

    // get arguments
    rocblas_local_handle handle;
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I lda = argus.get<I>("lda", m);
    char equedC = argus.get<char>("equed", 'B');
    char mode = argus.get<char>("equ_mode", 'F');

    rocblas_int hot_calls = argus.iters;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_laqge(handle, m, n, (T*)nullptr, lda, (S*)nullptr, (S*)nullptr,
                                              (S*)nullptr, (S*)nullptr, (S*)nullptr, (E*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_laqge(handle, m, n, (T*)nullptr, lda, (S*)nullptr, (S*)nullptr,
                                          (S*)nullptr, (S*)nullptr, (S*)nullptr, (E*)nullptr));

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

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<S> hR(size_R, 1, size_R, 1);
    host_strided_batch_vector<S> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<S> hscal(size_scal, 1, size_scal, 1);
    host_strided_batch_vector<E> hequedRes(1, 1, 1, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<S> dR(size_R, 1, size_R, 1);
    device_strided_batch_vector<S> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<S> dscal(size_scal, 1, size_scal, 1);
    device_strided_batch_vector<E> dequed(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_R)
        CHECK_HIP_ERROR(dR.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dscal.memcheck());
    CHECK_HIP_ERROR(dequed.memcheck());

    // check quick return
    if(n == 0 || m == 0)
    {
        // equed is pre-filled with an invalid value and must be set to none
        hequedRes[0][0] = static_cast<E>(0);
        CHECK_HIP_ERROR(dequed.transfer_from(hequedRes));

        EXPECT_ROCBLAS_STATUS(rocsolver_laqge(handle, m, n, dA.data(), lda, dR.data(), dC.data(),
                                              dscal.data(), dscal.data() + 1, dscal.data() + 2,
                                              dequed.data()),
                              rocblas_status_success);

        CHECK_HIP_ERROR(hequedRes.transfer_from(dequed));
        EXPECT_EQ(hequedRes[0][0], rocsolver_equilibration_none) << "equed";

        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        laqge_getError<T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, dequed, equedC, mode, hA,
                                hARes, hR, hC, hscal, hequedRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        laqge_getPerfData<T, I, S>(handle, m, n, dA, lda, dR, dC, dscal, dequed, equedC, mode, hA,
                                   hR, hC, hscal, &gpu_time_used, &cpu_time_used, hot_calls,
                                   argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // LAPACK scales a(i,j) as (C(j) * R(i)) * a(i,j), with at most two roundings; when the scalings
    // come from GEEQU, their own rounding errors add up
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, mode == 'G' ? 8 : 2);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("m", "n", "lda", "equed");
            rocsolver_bench_output(m, n, lda, equedC);

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

#define EXTERN_TESTING_LAQGE(...) extern template void testing_laqge<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_LAQGE, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
