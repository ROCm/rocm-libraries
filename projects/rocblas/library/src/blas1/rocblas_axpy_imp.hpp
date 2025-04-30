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
#include "rocblas.h"
#include "rocblas_axpy.hpp"
#include "rocblas_block_sizes.h"
#include "utility.hpp"

namespace
{

    template <typename>
    constexpr char rocblas_axpy_name[] = "unknown";
    template <>
    constexpr char rocblas_axpy_name<float>[] = ROCBLAS_API_STR(rocblas_saxpy);
    template <>
    constexpr char rocblas_axpy_name<double>[] = ROCBLAS_API_STR(rocblas_daxpy);
    template <>
    constexpr char rocblas_axpy_name<rocblas_half>[] = ROCBLAS_API_STR(rocblas_haxpy);
    template <>
    constexpr char rocblas_axpy_name<rocblas_float_complex>[] = ROCBLAS_API_STR(rocblas_caxpy);
    template <>
    constexpr char rocblas_axpy_name<rocblas_double_complex>[] = ROCBLAS_API_STR(rocblas_zaxpy);

    template <typename API_INT, rocblas_int NB, typename T>
    rocblas_status rocblas_axpy_impl(rocblas_handle handle,
                                     API_INT        n,
                                     const T*       alpha,
                                     const T*       x,
                                     API_INT        incx,
                                     T*             y,
                                     API_INT        incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        rocblas_internal_logger logger;

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle,
                             rocblas_axpy_name<T>,
                             n,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             x,
                             incx,
                             y,
                             incy);

        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f axpy -r",
                             rocblas_precision_string<T>,
                             "-n",
                             n,
                             LOG_BENCH_SCALAR_VALUE(handle, alpha),
                             "--incx",
                             incx,
                             "--incy",
                             incy);

        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle, rocblas_axpy_name<T>, "N", n, "incx", incx, "incy", incy);

        static constexpr API_INT        batch_count_1 = 1;
        static constexpr rocblas_stride stride_0      = 0;
        static constexpr rocblas_stride offset_0      = 0;

        rocblas_status arg_status = rocblas_axpy_arg_check(handle,
                                                           n,
                                                           alpha,
                                                           x,
                                                           offset_0,
                                                           incx,
                                                           stride_0,
                                                           y,
                                                           offset_0,
                                                           incy,
                                                           stride_0,
                                                           batch_count_1);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status axpy_check_numerics_status
                = rocblas_axpy_check_numerics(rocblas_axpy_name<T>,
                                              handle,
                                              n,
                                              x,
                                              offset_0,
                                              incx,
                                              stride_0,
                                              y,
                                              offset_0,
                                              incy,
                                              stride_0,
                                              batch_count_1,
                                              check_numerics,
                                              is_input);
            if(axpy_check_numerics_status != rocblas_status_success)
                return axpy_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_axpy_template)(handle,
                                                                            n,
                                                                            alpha,
                                                                            stride_0,
                                                                            x,
                                                                            offset_0,
                                                                            incx,
                                                                            stride_0,
                                                                            y,
                                                                            offset_0,
                                                                            incy,
                                                                            stride_0,
                                                                            batch_count_1);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status axpy_check_numerics_status
                = rocblas_axpy_check_numerics(rocblas_axpy_name<T>,
                                              handle,
                                              n,
                                              x,
                                              offset_0,
                                              incx,
                                              stride_0,
                                              y,
                                              offset_0,
                                              incy,
                                              stride_0,
                                              batch_count_1,
                                              check_numerics,
                                              is_input);
            if(axpy_check_numerics_status != rocblas_status_success)
                return axpy_check_numerics_status;
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

#define IMPL(name_, TI_, T_)                                                                   \
    rocblas_status name_(                                                                      \
        rocblas_handle handle, TI_ n, const T_* alpha, const T_* x, TI_ incx, T_* y, TI_ incy) \
    try                                                                                        \
    {                                                                                          \
        return rocblas_axpy_impl<TI_, ROCBLAS_AXPY_NB>(handle, n, alpha, x, incx, y, incy);    \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        return exception_to_rocblas_status();                                                  \
    }

#define INST_AXPY_C_API(TI_)                                       \
    extern "C" {                                                   \
    IMPL(ROCBLAS_API(rocblas_haxpy), TI_, rocblas_half);           \
    IMPL(ROCBLAS_API(rocblas_saxpy), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_daxpy), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_caxpy), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zaxpy), TI_, rocblas_double_complex); \
    } // extern "C"
