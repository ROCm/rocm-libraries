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
 * FOR AP PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_tpsv.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_tpsv_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_tpsv_strided_batched_name<float>[]
        = ROCBLAS_API_STR(rocblas_stpsv_strided_batched);
    template <>
    constexpr char rocblas_tpsv_strided_batched_name<double>[]
        = ROCBLAS_API_STR(rocblas_dtpsv_strided_batched);
    template <>
    constexpr char rocblas_tpsv_strided_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_ctpsv_strided_batched);
    template <>
    constexpr char rocblas_tpsv_strided_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_ztpsv_strided_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_tpsv_strided_batched_impl(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation transA,
                                                     rocblas_diagonal  diag,
                                                     API_INT           n,
                                                     const T*          AP,
                                                     rocblas_stride    stride_A,
                                                     T*                x,
                                                     API_INT           incx,
                                                     rocblas_stride    stride_x,
                                                     API_INT           batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        auto                    layer_mode = handle->layer_mode;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle,
                             rocblas_tpsv_strided_batched_name<T>,
                             uplo,
                             transA,
                             diag,
                             n,
                             AP,
                             stride_A,
                             x,
                             incx,
                             stride_x,
                             batch_count);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if(handle->pointer_mode == rocblas_pointer_mode_host)
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f tpsv_strided_batched -r",
                                     rocblas_precision_string<T>,
                                     "--uplo",
                                     uplo_letter,
                                     "--transposeA",
                                     transA_letter,
                                     "--diag",
                                     diag_letter,
                                     "-n",
                                     n,
                                     "--stride_a",
                                     stride_A,
                                     "--incx",
                                     incx,
                                     "--stride_x",
                                     stride_x,
                                     "--batch_count",
                                     batch_count);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_tpsv_strided_batched_name<T>,
                                   "uplo",
                                   uplo_letter,
                                   "transA",
                                   transA_letter,
                                   "diag",
                                   diag_letter,
                                   "N",
                                   n,
                                   "stride_a",
                                   stride_A,
                                   "incx",
                                   incx,
                                   "stride_x",
                                   stride_x,
                                   "batch_count",
                                   batch_count);
        }

        rocblas_status arg_status = rocblas_tpsv_arg_check<API_INT>(
            handle, uplo, transA, diag, n, AP, x, incx, batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        auto check_numerics = handle->check_numerics;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status tpsv_check_numerics_status
                = rocblas_tpsv_check_numerics(rocblas_tpsv_strided_batched_name<T>,
                                              handle,
                                              n,
                                              AP,
                                              0,
                                              stride_A,
                                              x,
                                              0,
                                              incx,
                                              stride_x,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tpsv_check_numerics_status != rocblas_status_success)
                return tpsv_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_tpsv_launcher)(
            handle, uplo, transA, diag, n, AP, 0, stride_A, x, 0, incx, stride_x, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status tpsv_check_numerics_status
                = rocblas_tpsv_check_numerics(rocblas_tpsv_strided_batched_name<T>,
                                              handle,
                                              n,
                                              AP,
                                              0,
                                              stride_A,
                                              x,
                                              0,
                                              incx,
                                              stride_x,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(tpsv_check_numerics_status != rocblas_status_success)
                return tpsv_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                                      \
    rocblas_status routine_name_(rocblas_handle    handle,                                \
                                 rocblas_fill      uplo,                                  \
                                 rocblas_operation transA,                                \
                                 rocblas_diagonal  diag,                                  \
                                 TI_               n,                                     \
                                 const T_*         AP,                                    \
                                 rocblas_stride    stride_A,                              \
                                 T_*               x,                                     \
                                 TI_               incx,                                  \
                                 rocblas_stride    stride_x,                              \
                                 TI_               batch_count)                           \
    try                                                                                   \
    {                                                                                     \
        return rocblas_tpsv_strided_batched_impl<TI_>(                                    \
            handle, uplo, transA, diag, n, AP, stride_A, x, incx, stride_x, batch_count); \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        return exception_to_rocblas_status();                                             \
    }

#define INST_TPSV_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_stpsv_strided_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dtpsv_strided_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_ctpsv_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_ztpsv_strided_batched), TI_, rocblas_double_complex); \
    } // extern "C"
