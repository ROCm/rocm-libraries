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

#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_spr.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_spr_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_spr_batched_name<float>[] = ROCBLAS_API_STR(rocblas_sspr_batched);
    template <>
    constexpr char rocblas_spr_batched_name<double>[] = ROCBLAS_API_STR(rocblas_dspr_batched);
    template <>
    constexpr char rocblas_spr_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cspr_batched);
    template <>
    constexpr char rocblas_spr_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zspr_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_spr_batched_impl(rocblas_handle handle,
                                            rocblas_fill   uplo,
                                            API_INT        n,
                                            const T*       alpha,
                                            const T* const x[],
                                            API_INT        incx,
                                            T* const       AP[],
                                            API_INT        batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter = rocblas_fill_letter(uplo);

            if(layer_mode & rocblas_layer_mode_log_trace)
                logger.log_trace(handle,
                                 rocblas_spr_batched_name<T>,
                                 uplo,
                                 n,
                                 LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                 0,
                                 x,
                                 incx,
                                 AP);

            if(layer_mode & rocblas_layer_mode_log_bench)
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f spr_batched -r",
                                 rocblas_precision_string<T>,
                                 "--uplo",
                                 uplo_letter,
                                 "-n",
                                 n,
                                 LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                 "--incx",
                                 incx,
                                 "--batch_count",
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_spr_batched_name<T>,
                                   "uplo",
                                   uplo_letter,
                                   "N",
                                   n,
                                   "incx",
                                   incx,
                                   "batch_count",
                                   batch_count);
        }

        static constexpr rocblas_stride offset_x = 0, offset_A = 0;
        static constexpr rocblas_stride stride_x = 0, stride_A = 0;

        rocblas_status arg_status = rocblas_spr_arg_check(handle,
                                                          uplo,
                                                          n,
                                                          alpha,
                                                          x,
                                                          offset_x,
                                                          incx,
                                                          stride_x,
                                                          AP,
                                                          offset_A,
                                                          stride_A,
                                                          batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status spr_check_numerics_status
                = rocblas_spr_check_numerics(rocblas_spr_batched_name<T>,
                                             handle,
                                             n,
                                             AP,
                                             offset_A,
                                             stride_A,
                                             x,
                                             offset_x,
                                             incx,
                                             stride_x,
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(spr_check_numerics_status != rocblas_status_success)
                return spr_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_spr_launcher)(handle,
                                                                           uplo,
                                                                           n,
                                                                           alpha,
                                                                           x,
                                                                           offset_x,
                                                                           incx,
                                                                           stride_x,
                                                                           AP,
                                                                           offset_A,
                                                                           stride_A,
                                                                           batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status spr_check_numerics_status
                = rocblas_spr_check_numerics(rocblas_spr_batched_name<T>,
                                             handle,
                                             n,
                                             AP,
                                             offset_A,
                                             stride_A,
                                             x,
                                             offset_x,
                                             incx,
                                             stride_x,
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(spr_check_numerics_status != rocblas_status_success)
                return spr_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                                            \
    rocblas_status routine_name_(rocblas_handle  handle,                                        \
                                 rocblas_fill    uplo,                                          \
                                 TI_             n,                                             \
                                 const T_*       alpha,                                         \
                                 const T_* const x[],                                           \
                                 TI_             incx,                                          \
                                 T_* const       AP[],                                          \
                                 TI_             batch_count)                                   \
    try                                                                                         \
    {                                                                                           \
        return rocblas_spr_batched_impl<TI_>(handle, uplo, n, alpha, x, incx, AP, batch_count); \
    }                                                                                           \
    catch(...)                                                                                  \
    {                                                                                           \
        return exception_to_rocblas_status();                                                   \
    }

#define INST_SPR_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                          \
    IMPL(ROCBLAS_API(rocblas_sspr_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dspr_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_cspr_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zspr_batched), TI_, rocblas_double_complex); \
    } // extern "C"
