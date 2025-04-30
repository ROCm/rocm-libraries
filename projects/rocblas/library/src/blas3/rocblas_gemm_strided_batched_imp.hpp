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
#include "rocblas_gemm.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_gemm_strided_batched_name[] = "unknown";

    template <>
    constexpr char rocblas_gemm_strided_batched_name<rocblas_half>[]
        = ROCBLAS_API_STR(rocblas_hgemm_strided_batched);

    template <>
    constexpr char rocblas_gemm_strided_batched_name<float>[]
        = ROCBLAS_API_STR(rocblas_sgemm_strided_batched);

    template <>
    constexpr char rocblas_gemm_strided_batched_name<double>[]
        = ROCBLAS_API_STR(rocblas_dgemm_strided_batched);

    template <>
    constexpr char rocblas_gemm_strided_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cgemm_strided_batched);

    template <>
    constexpr char rocblas_gemm_strided_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zgemm_strided_batched);

    /*******************************************************************************
    * Strided / Batched GEMM implementation
    ******************************************************************************/
    template <typename API_INT, typename T>
    auto rocblas_gemm_strided_batched_impl(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           API_INT           m,
                                           API_INT           n,
                                           API_INT           k,
                                           const T*          alpha,
                                           const T*          A,
                                           API_INT           lda,
                                           rocblas_stride    stride_a,
                                           const T*          B,
                                           API_INT           ldb,
                                           rocblas_stride    stride_b,
                                           const T*          beta,
                                           T*                C,
                                           API_INT           ldc,
                                           rocblas_stride    stride_c,
                                           API_INT           batch_count)

    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Copy alpha and beta to host if on device
        T alpha_h, beta_h;
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
            auto trans_a_letter = rocblas_transpose_letter(trans_a);
            auto trans_b_letter = rocblas_transpose_letter(trans_b);

            if(layer_mode & rocblas_layer_mode_log_trace)
                logger.log_trace(handle,
                                 rocblas_gemm_strided_batched_name<T>,
                                 trans_a,
                                 trans_b,
                                 m,
                                 n,
                                 k,
                                 LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                 A,
                                 lda,
                                 stride_a,
                                 B,
                                 ldb,
                                 stride_b,
                                 LOG_TRACE_SCALAR_VALUE(handle, beta),
                                 C,
                                 ldc,
                                 stride_c,
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f gemm_strided_batched -r",
                                 rocblas_precision_string<T>,
                                 "--transposeA",
                                 trans_a_letter,
                                 "--transposeB",
                                 trans_b_letter,
                                 "-m",
                                 m,
                                 "-n",
                                 n,
                                 "-k",
                                 k,
                                 LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                 "--lda",
                                 lda,
                                 "--stride_a",
                                 stride_a,
                                 "--ldb",
                                 ldb,
                                 "--stride_b",
                                 stride_b,
                                 LOG_BENCH_SCALAR_VALUE(handle, beta),
                                 "--ldc",
                                 ldc,
                                 "--stride_c",
                                 stride_c,
                                 "--batch_count",
                                 batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                logger.log_profile(handle,
                                   rocblas_gemm_strided_batched_name<T>,
                                   "transA",
                                   trans_a_letter,
                                   "transB",
                                   trans_b_letter,
                                   "M",
                                   m,
                                   "N",
                                   n,
                                   "K",
                                   k,
                                   "alpha",
                                   rocblas_internal_value_category(*alpha),
                                   "lda",
                                   lda,
                                   "stride_a",
                                   stride_a,
                                   "ldb",
                                   ldb,
                                   "stride_b",
                                   stride_b,
                                   "beta",
                                   rocblas_internal_value_category(*beta),
                                   "ldc",
                                   ldc,
                                   "stride_c",
                                   stride_c,
                                   "batch_count",
                                   batch_count);
            }
        }

        auto validArgs = rocblas_gemm_arg_check(
            handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);

        if(validArgs != rocblas_status_continue)
            return validArgs;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status gemm_check_numerics_status
                = rocblas_gemm_check_numerics(rocblas_gemm_strided_batched_name<T>,
                                              handle,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              A,
                                              0,
                                              lda,
                                              stride_a,
                                              B,
                                              0,
                                              ldb,
                                              stride_b,
                                              C,
                                              0,
                                              ldc,
                                              stride_c,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(gemm_check_numerics_status != rocblas_status_success)
                return gemm_check_numerics_status;
        }
        rocblas_status status = rocblas_status_success;

        API_INT a_n2 = rocblas_operation_none == trans_a ? k : m;
        API_INT b_n2 = rocblas_operation_none == trans_b ? n : k;

        status = ROCBLAS_API(rocblas_internal_gemm_template)(handle,
                                                             trans_a,
                                                             trans_b,
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             A,
                                                             rocblas_stride(0),
                                                             lda,
                                                             stride_a,
                                                             B,
                                                             rocblas_stride(0),
                                                             ldb,
                                                             stride_b,
                                                             beta,
                                                             C,
                                                             rocblas_stride(0),
                                                             ldc,
                                                             stride_c,
                                                             batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status gemm_check_numerics_status
                = rocblas_gemm_check_numerics(rocblas_gemm_strided_batched_name<T>,
                                              handle,
                                              trans_a,
                                              trans_b,
                                              m,
                                              n,
                                              k,
                                              A,
                                              0,
                                              lda,
                                              stride_a,
                                              B,
                                              0,
                                              ldb,
                                              stride_b,
                                              C,
                                              0,
                                              ldc,
                                              stride_c,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(gemm_check_numerics_status != rocblas_status_success)
                return gemm_check_numerics_status;
        }
        return status;
    }
}

/*******************************************************************************
 * Strided_Batched GEMM APIs
 ******************************************************************************/
#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, TI_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle    handle,              \
                                 rocblas_operation trans_a,             \
                                 rocblas_operation trans_b,             \
                                 TI_               m,                   \
                                 TI_               n,                   \
                                 TI_               k,                   \
                                 const T_*         alpha,               \
                                 const T_*         A,                   \
                                 TI_               lda,                 \
                                 rocblas_stride    stride_a,            \
                                 const T_*         B,                   \
                                 TI_               ldb,                 \
                                 rocblas_stride    stride_b,            \
                                 const T_*         beta,                \
                                 T_*               C,                   \
                                 TI_               ldc,                 \
                                 rocblas_stride    stride_c,            \
                                 TI_               batch_count)         \
    try                                                                 \
    {                                                                   \
        return rocblas_gemm_strided_batched_impl<TI_, T_>(handle,       \
                                                          trans_a,      \
                                                          trans_b,      \
                                                          m,            \
                                                          n,            \
                                                          k,            \
                                                          alpha,        \
                                                          A,            \
                                                          lda,          \
                                                          stride_a,     \
                                                          B,            \
                                                          ldb,          \
                                                          stride_b,     \
                                                          beta,         \
                                                          C,            \
                                                          ldc,          \
                                                          stride_c,     \
                                                          batch_count); \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

#define INST_GEMM_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_hgemm_strided_batched), TI_, rocblas_half);           \
    IMPL(ROCBLAS_API(rocblas_sgemm_strided_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dgemm_strided_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_cgemm_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zgemm_strided_batched), TI_, rocblas_double_complex); \
    } // extern "C"
