/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_syr2k_her2k.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_syr2k_name[] = "unknown";
    template <>
    constexpr char rocblas_syr2k_name<float>[] = ROCBLAS_API_STR(rocblas_ssyr2k_batched);
    template <>
    constexpr char rocblas_syr2k_name<double>[] = ROCBLAS_API_STR(rocblas_dsyr2k_batched);
    template <>
    constexpr char rocblas_syr2k_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_csyr2k_batched);
    template <>
    constexpr char rocblas_syr2k_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zsyr2k_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_syr2k_batched_impl(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              API_INT           n,
                                              API_INT           k,
                                              const T*          alpha,
                                              const T* const    A[],
                                              API_INT           lda,
                                              const T* const    B[],
                                              API_INT           ldb,
                                              const T*          beta,
                                              T* const          C[],
                                              API_INT           ldc,
                                              API_INT           batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Copy alpha and beta to host if on device. This is because gemm is called and it
        // requires alpha and beta to be on host
        T alpha_h;
        T beta_h;
        RETURN_IF_ROCBLAS_ERROR(
            rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);

            if(layer_mode & rocblas_layer_mode_log_trace)
                logger.log_trace(handle,
                                 rocblas_syr2k_name<T>,
                                 uplo,
                                 transA,
                                 n,
                                 k,
                                 LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                 A,
                                 lda,
                                 B,
                                 ldb,
                                 LOG_TRACE_SCALAR_VALUE(handle, beta),
                                 C,
                                 ldc,
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f syr2k_batched -r",
                                 rocblas_precision_string<T>,
                                 "--uplo",
                                 uplo_letter,
                                 "--transposeA",
                                 transA_letter,
                                 "-n",
                                 n,
                                 "-k",
                                 k,
                                 LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                 "--lda",
                                 lda,
                                 "--ldb",
                                 ldb,
                                 LOG_BENCH_SCALAR_VALUE(handle, beta),
                                 "--ldc",
                                 ldc,
                                 "--batch_count",
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_syr2k_name<T>,
                                   "uplo",
                                   uplo_letter,
                                   "transA",
                                   transA_letter,
                                   "N",
                                   n,
                                   "K",
                                   k,
                                   "lda",
                                   lda,
                                   "ldb",
                                   ldb,
                                   "ldc",
                                   ldc,
                                   "batch_count",
                                   batch_count);
        }

        static constexpr rocblas_stride offset_C = 0, offset_A = 0, offset_B = 0;
        static constexpr rocblas_stride stride_C = 0, stride_A = 0, stride_B = 0;

        rocblas_status arg_status = rocblas_syr2k_arg_check<API_INT>(handle,
                                                                     uplo,
                                                                     transA,
                                                                     n,
                                                                     k,
                                                                     alpha,
                                                                     A,
                                                                     offset_A,
                                                                     lda,
                                                                     stride_A,
                                                                     B,
                                                                     offset_B,
                                                                     ldb,
                                                                     stride_B,
                                                                     beta,
                                                                     C,
                                                                     offset_C,
                                                                     ldc,
                                                                     stride_C,
                                                                     batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        static constexpr bool Hermetian = false;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status syr2k_check_numerics_status
                = rocblas_her2k_syr2k_check_numerics<Hermetian>(rocblas_syr2k_name<T>,
                                                                handle,
                                                                uplo,
                                                                transA,
                                                                n,
                                                                k,
                                                                A,
                                                                lda,
                                                                stride_A,
                                                                B,
                                                                ldb,
                                                                stride_B,
                                                                C,
                                                                ldc,
                                                                stride_C,
                                                                batch_count,
                                                                check_numerics,
                                                                is_input);

            if(syr2k_check_numerics_status != rocblas_status_success)
                return syr2k_check_numerics_status;
        }

        rocblas_status status
            = ROCBLAS_API(rocblas_internal_syr2k_batched_template)<T>(handle,
                                                                      uplo,
                                                                      transA,
                                                                      n,
                                                                      k,
                                                                      alpha,
                                                                      A,
                                                                      offset_A,
                                                                      lda,
                                                                      stride_A,
                                                                      B,
                                                                      offset_B,
                                                                      ldb,
                                                                      stride_B,
                                                                      beta,
                                                                      C,
                                                                      offset_C,
                                                                      ldc,
                                                                      stride_C,
                                                                      batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status syr2k_check_numerics_status
                = rocblas_her2k_syr2k_check_numerics<Hermetian>(rocblas_syr2k_name<T>,
                                                                handle,
                                                                uplo,
                                                                transA,
                                                                n,
                                                                k,
                                                                A,
                                                                lda,
                                                                stride_A,
                                                                B,
                                                                ldb,
                                                                stride_B,
                                                                C,
                                                                ldc,
                                                                stride_C,
                                                                batch_count,
                                                                check_numerics,
                                                                is_input);

            if(syr2k_check_numerics_status != rocblas_status_success)
                return syr2k_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                                       \
    rocblas_status routine_name_(rocblas_handle    handle,                                 \
                                 rocblas_fill      uplo,                                   \
                                 rocblas_operation transA,                                 \
                                 TI_               n,                                      \
                                 TI_               k,                                      \
                                 const T_*         alpha,                                  \
                                 const T_* const   A[],                                    \
                                 TI_               lda,                                    \
                                 const T_* const   B[],                                    \
                                 TI_               ldb,                                    \
                                 const T_*         beta,                                   \
                                 T_* const         C[],                                    \
                                 TI_               ldc,                                    \
                                 TI_               batch_count)                            \
    try                                                                                    \
    {                                                                                      \
        return rocblas_syr2k_batched_impl<TI_>(                                            \
            handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count); \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        return exception_to_rocblas_status();                                              \
    }

#define INST_SYR2K_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                            \
    IMPL(ROCBLAS_API(rocblas_ssyr2k_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dsyr2k_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_csyr2k_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zsyr2k_batched), TI_, rocblas_double_complex); \
    } // extern "C"
