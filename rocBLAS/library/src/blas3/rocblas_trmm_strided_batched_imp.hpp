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
#include "rocblas_trmm.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_trmm_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_trmm_strided_batched_name<float>[]
        = ROCBLAS_API_STR(rocblas_strmm_strided_batched);
    template <>
    constexpr char rocblas_trmm_strided_batched_name<double>[]
        = ROCBLAS_API_STR(rocblas_dtrmm_strided_batched);
    template <>
    constexpr char rocblas_trmm_strided_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_ctrmm_strided_batched);
    template <>
    constexpr char rocblas_trmm_strided_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_ztrmm_strided_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_trmm_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_side      side,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transa,
                                                     rocblas_diagonal  diag,
                                                     API_INT           m,
                                                     API_INT           n,
                                                     const T*          alpha,
                                                     const T*          a,
                                                     API_INT           lda,
                                                     rocblas_stride    stride_a,
                                                     const T*          b,
                                                     API_INT           ldb,
                                                     rocblas_stride    stride_b,
                                                     T*                c,
                                                     API_INT           ldc,
                                                     rocblas_stride    stride_c,
                                                     API_INT           batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        T        alpha_h, beta_h;
        const T* beta = nullptr;
        RETURN_IF_ROCBLAS_ERROR(rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h, beta_h, m && n));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile)
           && (!handle->is_device_memory_size_query()))
        {
            auto side_letter   = rocblas_side_letter(side);
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transa_letter = rocblas_transpose_letter(transa);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_trace)
                logger.log_trace(handle,
                                 rocblas_trmm_strided_batched_name<T>,
                                 side,
                                 uplo,
                                 transa,
                                 diag,
                                 m,
                                 n,
                                 LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                 a,
                                 lda,
                                 stride_a,
                                 b,
                                 ldb,
                                 stride_b,
                                 c,
                                 ldc,
                                 stride_c,
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f trmm_strided_batched -r",
                                 rocblas_precision_string<T>,
                                 "--side",
                                 side_letter,
                                 "--uplo",
                                 uplo_letter,
                                 "--transposeA",
                                 transa_letter,
                                 "--diag",
                                 diag_letter,
                                 "-m",
                                 m,
                                 "-n",
                                 n,
                                 LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                 "--lda",
                                 lda,
                                 "--stride_a",
                                 stride_a,
                                 "--ldb",
                                 ldb,
                                 "--stride_b",
                                 stride_b,
                                 "--ldc",
                                 ldc,
                                 "--stride_c",
                                 stride_c,
                                 "--batch_count",
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_trmm_strided_batched_name<T>,
                                   "side",
                                   side_letter,
                                   "uplo",
                                   uplo_letter,
                                   "transa",
                                   transa_letter,
                                   "diag",
                                   diag_letter,
                                   "m",
                                   m,
                                   "n",
                                   n,
                                   "lda",
                                   lda,
                                   "stride_a",
                                   stride_a,
                                   "ldb",
                                   ldb,
                                   "stride_b",
                                   stride_b,
                                   "ldc",
                                   ldc,
                                   "stride_c",
                                   stride_c,
                                   "batch_count",
                                   batch_count);
        }

        static constexpr rocblas_stride offset_a     = 0;
        static constexpr rocblas_stride offset_b     = 0;
        static constexpr rocblas_stride offset_c     = 0;
        static constexpr rocblas_stride stride_alpha = 0;

        rocblas_status arg_status = rocblas_trmm_arg_check(
            handle, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, c, ldc, batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(rocblas_pointer_mode_host == handle->pointer_mode && 0 == *alpha)
        {
            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(rocblas_set_matrix_zero_if_alpha_zero_template(
                handle, m, n, alpha, 0, c, ldc, stride_c, batch_count));
            return rocblas_status_success;
        }
        else if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // set matrix to zero and continue calculation. This will give
            // the same functionality as Legacy BLAS. alpha is on device and
            // it should not be copied from device to host because this is
            // an asynchronous function and the copy would make it synchronous.
            PRINT_AND_RETURN_IF_ROCBLAS_ERROR(rocblas_set_matrix_zero_if_alpha_zero_template(
                handle, m, n, alpha, 0, c, ldc, stride_c, batch_count));
        }

        if(rocblas_pointer_mode_host == handle->pointer_mode && !a)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trmm_check_numerics_status
                = rocblas_trmm_check_numerics(rocblas_trmm_strided_batched_name<T>,
                                              handle,
                                              side,
                                              uplo,
                                              transa,
                                              m,
                                              n,
                                              a,
                                              lda,
                                              stride_a,
                                              b,
                                              ldb,
                                              stride_b,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(trmm_check_numerics_status != rocblas_status_success)
                return trmm_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_trmm_template)(handle,
                                                                            side,
                                                                            uplo,
                                                                            transa,
                                                                            diag,
                                                                            m,
                                                                            n,
                                                                            alpha,
                                                                            stride_alpha,
                                                                            a,
                                                                            offset_a,
                                                                            lda,
                                                                            stride_a,
                                                                            b,
                                                                            offset_b,
                                                                            ldb,
                                                                            stride_b,
                                                                            c,
                                                                            offset_c,
                                                                            ldc,
                                                                            stride_c,
                                                                            batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trmm_check_numerics_status
                = rocblas_trmm_check_numerics(rocblas_trmm_strided_batched_name<T>,
                                              handle,
                                              side,
                                              uplo,
                                              transa,
                                              m,
                                              n,
                                              a,
                                              lda,
                                              stride_a,
                                              c,
                                              ldc,
                                              stride_c,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(trmm_check_numerics_status != rocblas_status_success)
                return trmm_check_numerics_status;
        }
        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, TI_, T_)                            \
    rocblas_status routine_name_(rocblas_handle    handle,      \
                                 rocblas_side      side,        \
                                 rocblas_fill      uplo,        \
                                 rocblas_operation transa,      \
                                 rocblas_diagonal  diag,        \
                                 TI_               m,           \
                                 TI_               n,           \
                                 const T_*         alpha,       \
                                 const T_*         a,           \
                                 TI_               lda,         \
                                 rocblas_stride    stride_a,    \
                                 const T_*         b,           \
                                 TI_               ldb,         \
                                 rocblas_stride    stride_b,    \
                                 T_*               c,           \
                                 TI_               ldc,         \
                                 rocblas_stride    stride_c,    \
                                 TI_               batch_count) \
    try                                                         \
    {                                                           \
        return rocblas_trmm_strided_batched_impl(handle,        \
                                                 side,          \
                                                 uplo,          \
                                                 transa,        \
                                                 diag,          \
                                                 m,             \
                                                 n,             \
                                                 alpha,         \
                                                 a,             \
                                                 lda,           \
                                                 stride_a,      \
                                                 b,             \
                                                 ldb,           \
                                                 stride_b,      \
                                                 c,             \
                                                 ldc,           \
                                                 stride_c,      \
                                                 batch_count);  \
    }                                                           \
    catch(...)                                                  \
    {                                                           \
        return exception_to_rocblas_status();                   \
    }

#define INST_TRMM_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_strmm_strided_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dtrmm_strided_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_ctrmm_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_ztrmm_strided_batched), TI_, rocblas_double_complex); \
    }
