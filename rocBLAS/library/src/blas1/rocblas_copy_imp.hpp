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
#include "rocblas_block_sizes.h"
#include "rocblas_copy.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_copy_name[] = "unknown";
    template <>
    constexpr char rocblas_copy_name<float>[] = ROCBLAS_API_STR(rocblas_scopy);
    template <>
    constexpr char rocblas_copy_name<double>[] = ROCBLAS_API_STR(rocblas_dcopy);
    template <>
    constexpr char rocblas_copy_name<rocblas_half>[] = ROCBLAS_API_STR(rocblas_hcopy);
    template <>
    constexpr char rocblas_copy_name<rocblas_float_complex>[] = ROCBLAS_API_STR(rocblas_ccopy);
    template <>
    constexpr char rocblas_copy_name<rocblas_double_complex>[] = ROCBLAS_API_STR(rocblas_zcopy);

    template <typename API_INT, rocblas_int NB, typename T>
    rocblas_status rocblas_copy_impl(
        rocblas_handle handle, API_INT n, const T* x, API_INT incx, T* y, API_INT incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle, rocblas_copy_name<T>, n, x, incx, y, incy);

        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f copy -r",
                             rocblas_precision_string<T>,
                             "-n",
                             n,
                             "--incx",
                             incx,
                             "--incy",
                             incy);

        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle, rocblas_copy_name<T>, "N", n, "incx", incx, "incy", incy);

        rocblas_status arg_status
            = rocblas_copy_arg_check(handle, n, x, 0, incx, 0, y, 0, incy, 0, API_INT(1));
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status copy_check_numerics_status
                = rocblas_copy_check_numerics(rocblas_copy_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(copy_check_numerics_status != rocblas_status_success)
                return copy_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_copy_launcher)<API_INT, NB>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, 1);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status copy_check_numerics_status
                = rocblas_copy_check_numerics(rocblas_copy_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(copy_check_numerics_status != rocblas_status_success)
                return copy_check_numerics_status;
        }
        return status;
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(name_, TI_, T_)                                                                   \
    rocblas_status name_(rocblas_handle handle, TI_ n, const T_* x, TI_ incx, T_* y, TI_ incy) \
    try                                                                                        \
    {                                                                                          \
        return rocblas_copy_impl<TI_, ROCBLAS_COPY_NB>(handle, n, x, incx, y, incy);           \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        return exception_to_rocblas_status();                                                  \
    }

#define INST_COPY_C_API(TI_)                                       \
    extern "C" {                                                   \
    IMPL(ROCBLAS_API(rocblas_scopy), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dcopy), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_hcopy), TI_, rocblas_half);           \
    IMPL(ROCBLAS_API(rocblas_ccopy), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zcopy), TI_, rocblas_double_complex); \
    } // extern "C"
