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
#include "rocblas_trmv.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_trmv_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_trmv_batched_name<float>[] = ROCBLAS_API_STR(rocblas_strmv_batched);
    template <>
    constexpr char rocblas_trmv_batched_name<double>[] = ROCBLAS_API_STR(rocblas_dtrmv_batched);
    template <>
    constexpr char rocblas_trmv_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_ctrmv_batched);
    template <>
    constexpr char rocblas_trmv_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_ztrmv_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_trmv_batched_impl(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transa,
                                             rocblas_diagonal  diag,
                                             API_INT           n,
                                             const T* const*   a,
                                             API_INT           lda,
                                             T* const*         x,
                                             API_INT           incx,
                                             API_INT           batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        rocblas_internal_logger logger;

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transa_letter = rocblas_transpose_letter(transa);
                auto diag_letter   = rocblas_diag_letter(diag);
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    logger.log_trace(handle,
                                     rocblas_trmv_batched_name<T>,
                                     uplo,
                                     transa,
                                     diag,
                                     n,
                                     a,
                                     lda,
                                     x,
                                     incx,
                                     batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f trmv_batched -r",
                                     rocblas_precision_string<T>,
                                     "--uplo",
                                     uplo_letter,
                                     "--transposeA",
                                     transa_letter,
                                     "--diag",
                                     diag_letter,
                                     "-n",
                                     n,
                                     "--lda",
                                     lda,
                                     "--incx",
                                     incx,
                                     "--batch_count",
                                     batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    logger.log_profile(handle,
                                       rocblas_trmv_batched_name<T>,
                                       "uplo",
                                       uplo_letter,
                                       "transA",
                                       transa_letter,
                                       "diag",
                                       diag_letter,
                                       "N",
                                       n,
                                       "lda",
                                       lda,
                                       "incx",
                                       incx,
                                       "batch_count",
                                       batch_count);
                }
            }
        }

        size_t         dev_bytes;
        rocblas_status arg_status = rocblas_trmv_arg_check<API_INT, T>(
            handle, uplo, transa, diag, n, a, lda, x, incx, batch_count, dev_bytes);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        auto workspace = handle->device_malloc(dev_bytes);
        if(!workspace)
            return rocblas_status_memory_error;

        rocblas_stride stride_w = n;

        auto check_numerics = handle->check_numerics;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trmv_check_numerics_status
                = rocblas_trmv_check_numerics(rocblas_trmv_batched_name<T>,
                                              handle,
                                              uplo,
                                              n,
                                              a,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(trmv_check_numerics_status != rocblas_status_success)
                return trmv_check_numerics_status;
        }

        constexpr rocblas_stride offset_a = 0, offset_x = 0;
        constexpr rocblas_stride stride_a = 0, stride_x = 0;
        rocblas_status status = ROCBLAS_API(rocblas_internal_trmv_batched_template)(handle,
                                                                                    uplo,
                                                                                    transa,
                                                                                    diag,
                                                                                    n,
                                                                                    a,
                                                                                    offset_a,
                                                                                    lda,
                                                                                    stride_a,
                                                                                    x,
                                                                                    offset_x,
                                                                                    incx,
                                                                                    stride_x,
                                                                                    (T*)workspace,
                                                                                    stride_w,
                                                                                    batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trmv_check_numerics_status
                = rocblas_trmv_check_numerics(rocblas_trmv_batched_name<T>,
                                              handle,
                                              uplo,
                                              n,
                                              a,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(trmv_check_numerics_status != rocblas_status_success)
                return trmv_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                      \
    rocblas_status routine_name_(rocblas_handle    handle,                \
                                 rocblas_fill      uplo,                  \
                                 rocblas_operation transa,                \
                                 rocblas_diagonal  diag,                  \
                                 TI_               n,                     \
                                 const T_* const*  a,                     \
                                 TI_               lda,                   \
                                 T_* const*        x,                     \
                                 TI_               incx,                  \
                                 TI_               batch_count)           \
    try                                                                   \
    {                                                                     \
        return rocblas_trmv_batched_impl<TI_>(                            \
            handle, uplo, transa, diag, n, a, lda, x, incx, batch_count); \
    }                                                                     \
    catch(...)                                                            \
    {                                                                     \
        return exception_to_rocblas_status();                             \
    }

#define INST_TRMV_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                           \
    IMPL(ROCBLAS_API(rocblas_strmv_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dtrmv_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_ctrmv_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_ztrmv_batched), TI_, rocblas_double_complex); \
    } // extern "C"
