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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_hpmv.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_hpmv_name[] = "unknown";
    template <>
    constexpr char rocblas_hpmv_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_chpmv_strided_batched);
    template <>
    constexpr char rocblas_hpmv_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zhpmv_strided_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_hpmv_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     API_INT        n,
                                                     const T*       alpha,
                                                     const T*       AP,
                                                     rocblas_stride stride_A,
                                                     const T*       x,
                                                     API_INT        incx,
                                                     rocblas_stride stride_x,
                                                     const T*       beta,
                                                     T*             y,
                                                     API_INT        incy,
                                                     rocblas_stride stride_y,
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
                                 rocblas_hpmv_name<T>,
                                 uplo,
                                 n,
                                 LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                 AP,
                                 stride_A,
                                 x,
                                 incx,
                                 stride_x,
                                 LOG_TRACE_SCALAR_VALUE(handle, beta),
                                 y,
                                 incy,
                                 stride_y,
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f hpmv_strided_batched -r",
                                 rocblas_precision_string<T>,
                                 "--uplo",
                                 uplo_letter,
                                 "-n",
                                 n,
                                 LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                 "--stride_a",
                                 stride_A,
                                 "--incx",
                                 incx,
                                 "--stride_x",
                                 stride_x,
                                 LOG_BENCH_SCALAR_VALUE(handle, beta),
                                 "--incy",
                                 incy,
                                 "--stride_y",
                                 stride_y,
                                 "--batch_count",
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_hpmv_name<T>,
                                   "uplo",
                                   uplo_letter,
                                   "N",
                                   n,
                                   "stride_a",
                                   stride_A,
                                   "incx",
                                   incx,
                                   "stride_x",
                                   stride_x,
                                   "incy",
                                   incy,
                                   "stride_y",
                                   stride_y,
                                   "batch_count",
                                   batch_count);
        }

        constexpr rocblas_stride offset_A = 0, offset_x = 0, offset_y = 0;

        rocblas_status arg_status = rocblas_hpmv_arg_check(handle,
                                                           uplo,
                                                           n,
                                                           alpha,
                                                           AP,
                                                           offset_A,
                                                           stride_A,
                                                           x,
                                                           offset_x,
                                                           incx,
                                                           stride_x,
                                                           beta,
                                                           y,
                                                           offset_y,
                                                           incy,
                                                           stride_y,
                                                           batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status hpmv_check_numerics_status
                = rocblas_hpmv_check_numerics(rocblas_hpmv_name<T>,
                                              handle,
                                              n,
                                              AP,
                                              offset_A,
                                              stride_A,
                                              x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(hpmv_check_numerics_status != rocblas_status_success)
                return hpmv_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_hpmv_launcher)(handle,
                                                                   uplo,
                                                                   n,
                                                                   alpha,
                                                                   AP,
                                                                   offset_A,
                                                                   stride_A,
                                                                   x,
                                                                   offset_x,
                                                                   incx,
                                                                   stride_x,
                                                                   beta,
                                                                   y,
                                                                   offset_y,
                                                                   incy,
                                                                   stride_y,
                                                                   batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status hpmv_check_numerics_status
                = rocblas_hpmv_check_numerics(rocblas_hpmv_name<T>,
                                              handle,
                                              n,
                                              AP,
                                              offset_A,
                                              stride_A,
                                              x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(hpmv_check_numerics_status != rocblas_status_success)
                return hpmv_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle handle,                 \
                                 rocblas_fill   uplo,                   \
                                 TI_            n,                      \
                                 const T_*      alpha,                  \
                                 const T_*      AP,                     \
                                 rocblas_stride stride_A,               \
                                 const T_*      x,                      \
                                 TI_            incx,                   \
                                 rocblas_stride stride_x,               \
                                 const T_*      beta,                   \
                                 T_*            y,                      \
                                 TI_            incy,                   \
                                 rocblas_stride stride_y,               \
                                 TI_            batch_count)            \
    try                                                                 \
    {                                                                   \
        return rocblas_hpmv_strided_batched_impl<TI_, T_>(handle,       \
                                                          uplo,         \
                                                          n,            \
                                                          alpha,        \
                                                          AP,           \
                                                          stride_A,     \
                                                          x,            \
                                                          incx,         \
                                                          stride_x,     \
                                                          beta,         \
                                                          y,            \
                                                          incy,         \
                                                          stride_y,     \
                                                          batch_count); \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

#define INST_HPMV_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_chpmv_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zhpmv_strided_batched), TI_, rocblas_double_complex); \
    } // extern "C"
