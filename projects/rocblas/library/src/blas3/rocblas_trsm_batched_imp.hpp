/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_trmm.hpp"
#include "rocblas_trsm.hpp"
#include "trtri_trsm.hpp"
#include "utility.hpp"

// Shared memory usuage is (128/2)^2 * sizeof(float) = 32K. LDS is 64K per CU. Theoretically
// you can use all 64K, but in practice no.
// constexpr rocblas_int STRSM_BLOCK = 128;
// constexpr rocblas_int DTRSM_BLOCK = 128;

namespace
{
    template <typename>
    constexpr char rocblas_trsm_name[] = "unknown";
    template <>
    constexpr char rocblas_trsm_name<float>[] = ROCBLAS_API_STR(rocblas_batched_strsm);
    template <>
    constexpr char rocblas_trsm_name<double>[] = ROCBLAS_API_STR(rocblas_batched_dtrsm);
    template <>
    constexpr char rocblas_trsm_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_batched_ctrsm);
    template <>
    constexpr char rocblas_trsm_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_batched_ztrsm);

    /* ============================================================================================ */

    template <typename API_INT, typename T>
    rocblas_status rocblas_trsm_batched_ex_impl(rocblas_handle    handle,
                                                rocblas_side      side,
                                                rocblas_fill      uplo,
                                                rocblas_operation transA,
                                                rocblas_diagonal  diag,
                                                API_INT           m,
                                                API_INT           n,
                                                const T*          alpha,
                                                const T* const    A[],
                                                API_INT           lda,
                                                T* const          B[],
                                                API_INT           ldb,
                                                API_INT           batch_count,
                                                const T* const    supplied_invA[]    = nullptr,
                                                API_INT           supplied_invA_size = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        /////////////
        // LOGGING //
        /////////////
        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto side_letter   = rocblas_side_letter(side);
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transA_letter = rocblas_transpose_letter(transA);
                auto diag_letter   = rocblas_diag_letter(diag);

                if(layer_mode & rocblas_layer_mode_log_trace)
                    logger.log_trace(handle,
                                     rocblas_trsm_name<T>,
                                     side,
                                     uplo,
                                     transA,
                                     diag,
                                     m,
                                     n,
                                     LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                     A,
                                     lda,
                                     B,
                                     ldb,
                                     batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f trsm_batched -r",
                                     rocblas_precision_string<T>,
                                     "--side",
                                     side_letter,
                                     "--uplo",
                                     uplo_letter,
                                     "--transposeA",
                                     transA_letter,
                                     "--diag",
                                     diag_letter,
                                     "-m",
                                     m,
                                     "-n",
                                     n,
                                     LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                     "--lda",
                                     lda,
                                     "--ldb",
                                     ldb,
                                     "--batch_count",
                                     batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    logger.log_profile(handle,
                                       rocblas_trsm_name<T>,
                                       "side",
                                       side_letter,
                                       "uplo",
                                       uplo_letter,
                                       "transA",
                                       transA_letter,
                                       "diag",
                                       diag_letter,
                                       "m",
                                       m,
                                       "n",
                                       n,
                                       "lda",
                                       lda,
                                       "ldb",
                                       ldb,
                                       "batch_count",
                                       batch_count);
                }
            }
        }

        rocblas_status arg_status = rocblas_trsm_arg_check(
            handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count);

        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(rocblas_pointer_mode_host == handle->pointer_mode && 0 == *alpha)
        {
            set_block_unit<T>(handle, m, n, B, ldb, 0, batch_count, 0);
            return rocblas_status_success;
        }

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trsm_check_numerics_status
                = rocblas_trmm_check_numerics(rocblas_trsm_name<T>,
                                              handle,
                                              side,
                                              uplo,
                                              transA,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              0,
                                              B,
                                              ldb,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(trsm_check_numerics_status != rocblas_status_success)
                return trsm_check_numerics_status;
        }

        rocblas_status status = rocblas_status_success;
        //////////////////////
        // MEMORY MANAGEMENT//
        //////////////////////
        //kernel function is enclosed inside the brackets so that the handle device memory used by the kernel is released after the computation.
        {
            // Proxy object holds the allocation. It must stay alive as long as mem_* pointers below are alive.
            auto  w_mem = handle->device_malloc(0);
            void* w_mem_x_temp;
            void* w_mem_x_temp_arr;
            void* w_mem_invA;
            void* w_mem_invA_arr;

            rocblas_status perf_status
                = ROCBLAS_API(rocblas_internal_trsm_template_mem)<true, T>(handle,
                                                                           side,
                                                                           transA,
                                                                           m,
                                                                           n,
                                                                           lda,
                                                                           ldb,
                                                                           batch_count,
                                                                           w_mem,
                                                                           w_mem_x_temp,
                                                                           w_mem_x_temp_arr,
                                                                           w_mem_invA,
                                                                           w_mem_invA_arr,
                                                                           supplied_invA,
                                                                           supplied_invA_size);

            if(perf_status != rocblas_status_success && perf_status != rocblas_status_perf_degraded)
                return perf_status;

            bool optimal_mem = perf_status == rocblas_status_success;

            status = ROCBLAS_API(rocblas_internal_trsm_batched_template)(handle,
                                                                         side,
                                                                         uplo,
                                                                         transA,
                                                                         diag,
                                                                         m,
                                                                         n,
                                                                         alpha,
                                                                         A,
                                                                         0,
                                                                         lda,
                                                                         0,
                                                                         B,
                                                                         0,
                                                                         ldb,
                                                                         0,
                                                                         batch_count,
                                                                         optimal_mem,
                                                                         w_mem_x_temp,
                                                                         w_mem_x_temp_arr,
                                                                         w_mem_invA,
                                                                         w_mem_invA_arr,
                                                                         supplied_invA,
                                                                         supplied_invA_size,
                                                                         0,
                                                                         0);
            status = (status != rocblas_status_success) ? status : perf_status;
            if(status != rocblas_status_success && status != rocblas_status_perf_degraded)
                return status;
        }

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trsm_check_numerics_status
                = rocblas_trmm_check_numerics(rocblas_trsm_name<T>,
                                              handle,
                                              side,
                                              uplo,
                                              transA,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              0,
                                              B,
                                              ldb,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(trsm_check_numerics_status != rocblas_status_success)
                return trsm_check_numerics_status;
        }
        return status;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, TI_, T_)                                                     \
    rocblas_status routine_name_(rocblas_handle    handle,                               \
                                 rocblas_side      side,                                 \
                                 rocblas_fill      uplo,                                 \
                                 rocblas_operation transA,                               \
                                 rocblas_diagonal  diag,                                 \
                                 TI_               m,                                    \
                                 TI_               n,                                    \
                                 const T_*         alpha,                                \
                                 const T_* const   A[],                                  \
                                 TI_               lda,                                  \
                                 T_* const         B[],                                  \
                                 TI_               ldb,                                  \
                                 TI_               batch_count)                          \
    try                                                                                  \
    {                                                                                    \
        return rocblas_trsm_batched_ex_impl<TI_>(                                        \
            handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, batch_count); \
    }                                                                                    \
    catch(...)                                                                           \
    {                                                                                    \
        return exception_to_rocblas_status();                                            \
    }

#define INST_TRSM_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                           \
    IMPL(ROCBLAS_API(rocblas_strsm_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dtrsm_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_ctrsm_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_ztrsm_batched), TI_, rocblas_double_complex); \
    } // extern "C"
