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
#include "rocblas_hpr2.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_hpr2_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_hpr2_strided_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_chpr2_strided_batched);
    template <>
    constexpr char rocblas_hpr2_strided_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zhpr2_strided_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_hpr2_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     API_INT        n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     API_INT        incx,
                                                     rocblas_stride stridex,
                                                     const T*       y,
                                                     API_INT        incy,
                                                     rocblas_stride stridey,
                                                     T*             AP,
                                                     rocblas_stride strideA,
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
                                 rocblas_hpr2_strided_batched_name<T>,
                                 uplo,
                                 n,
                                 LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                 x,
                                 incx,
                                 stridex,
                                 y,
                                 incy,
                                 stridey,
                                 AP,
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f hpr2_strided_batched -r",
                                 rocblas_precision_string<T>,
                                 "--uplo",
                                 uplo_letter,
                                 "-n",
                                 n,
                                 LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                 "--incx",
                                 incx,
                                 "--stride_x",
                                 stridex,
                                 "--incy",
                                 incy,
                                 "--stride_y",
                                 stridey,
                                 "--stride_a",
                                 strideA,
                                 "--batch_count",
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_hpr2_strided_batched_name<T>,
                                   "uplo",
                                   uplo_letter,
                                   "N",
                                   n,
                                   "incx",
                                   incx,
                                   "stride_x",
                                   stridex,
                                   "incy",
                                   incy,
                                   "stride_y",
                                   stridey,
                                   "stride_a",
                                   strideA,
                                   "batch_count",
                                   batch_count);
        }

        static constexpr rocblas_stride offset_x = 0, offset_y = 0, offset_A = 0;

        rocblas_status arg_status = rocblas_hpr2_arg_check(handle,
                                                           uplo,
                                                           n,
                                                           alpha,
                                                           x,
                                                           offset_x,
                                                           incx,
                                                           stridex,
                                                           y,
                                                           offset_y,
                                                           incy,
                                                           stridey,
                                                           AP,
                                                           offset_A,
                                                           strideA,
                                                           batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status hpr2_check_numerics_status
                = rocblas_hpr2_check_numerics(rocblas_hpr2_strided_batched_name<T>,
                                              handle,
                                              n,
                                              AP,
                                              offset_A,
                                              strideA,
                                              x,
                                              offset_x,
                                              incx,
                                              stridex,
                                              y,
                                              offset_y,
                                              incy,
                                              stridey,
                                              1,
                                              check_numerics,
                                              is_input);
            if(hpr2_check_numerics_status != rocblas_status_success)
                return hpr2_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_hpr2_launcher)(handle,
                                                                            uplo,
                                                                            n,
                                                                            alpha,
                                                                            x,
                                                                            offset_x,
                                                                            incx,
                                                                            stridex,
                                                                            y,
                                                                            offset_y,
                                                                            incy,
                                                                            stridey,
                                                                            AP,
                                                                            offset_A,
                                                                            strideA,
                                                                            batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status hpr2_check_numerics_status
                = rocblas_hpr2_check_numerics(rocblas_hpr2_strided_batched_name<T>,
                                              handle,
                                              n,
                                              AP,
                                              offset_A,
                                              strideA,
                                              x,
                                              offset_x,
                                              incx,
                                              stridex,
                                              y,
                                              offset_y,
                                              incy,
                                              stridey,
                                              1,
                                              check_numerics,
                                              is_input);
            if(hpr2_check_numerics_status != rocblas_status_success)
                return hpr2_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                                               \
    rocblas_status routine_name_(rocblas_handle handle,                                            \
                                 rocblas_fill   uplo,                                              \
                                 TI_            n,                                                 \
                                 const T_*      alpha,                                             \
                                 const T_*      x,                                                 \
                                 TI_            incx,                                              \
                                 rocblas_stride stridex,                                           \
                                 const T_*      y,                                                 \
                                 TI_            incy,                                              \
                                 rocblas_stride stridey,                                           \
                                 T_*            AP,                                                \
                                 rocblas_stride strideA,                                           \
                                 TI_            batch_count)                                       \
    try                                                                                            \
    {                                                                                              \
        return rocblas_hpr2_strided_batched_impl<TI_>(                                             \
            handle, uplo, n, alpha, x, incx, stridex, y, incy, stridey, AP, strideA, batch_count); \
    }                                                                                              \
    catch(...)                                                                                     \
    {                                                                                              \
        return exception_to_rocblas_status();                                                      \
    }

#define INST_HPR2_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_chpr2_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zhpr2_strided_batched), TI_, rocblas_double_complex); \
    } // extern "C"
