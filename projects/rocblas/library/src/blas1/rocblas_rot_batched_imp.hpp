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
#include "rocblas_block_sizes.h"
#include "rocblas_rot.hpp"

namespace
{
    constexpr int NB = ROCBLAS_ROT_NB;

    template <typename T, typename = T>
    constexpr char rocblas_rot_name[] = "unknown";
    template <>
    constexpr char rocblas_rot_name<float>[] = ROCBLAS_API_STR(rocblas_srot_batched);
    template <>
    constexpr char rocblas_rot_name<double>[] = ROCBLAS_API_STR(rocblas_drot_batched);
    template <>
    constexpr char rocblas_rot_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_crot_batched);
    template <>
    constexpr char rocblas_rot_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zrot_batched);
    template <>
    constexpr char rocblas_rot_name<rocblas_float_complex, float>[]
        = ROCBLAS_API_STR(rocblas_csrot_batched);
    template <>
    constexpr char rocblas_rot_name<rocblas_double_complex, double>[]
        = ROCBLAS_API_STR(rocblas_zdrot_batched);

    template <typename API_INT, typename T, typename U, typename V>
    rocblas_status rocblas_rot_batched_impl(rocblas_handle handle,
                                            API_INT        n,
                                            T* const       x[],
                                            API_INT        incx,
                                            T* const       y[],
                                            API_INT        incy,
                                            const U*       c,
                                            const V*       s,
                                            API_INT        batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(
                handle, rocblas_rot_name<T, V>, n, x, incx, y, incy, c, s, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH "-f rot_batched --a_type",
                             rocblas_precision_string<T>,
                             "--b_type",
                             rocblas_precision_string<U>,
                             "--c_type",
                             rocblas_precision_string<V>,
                             "-n",
                             n,
                             "--incx",
                             incx,
                             "--incy",
                             incy,
                             "--batch_count",
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_rot_name<T, V>,
                               "N",
                               n,
                               "incx",
                               incx,
                               "incy",
                               incy,
                               "batch_count",
                               batch_count);

        if(n <= 0 || batch_count <= 0)
            return rocblas_status_success;
        if(!x || !y || !c || !s)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rot_check_numerics_status
                = rocblas_rot_check_numerics(rocblas_rot_name<T>,
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
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(rot_check_numerics_status != rocblas_status_success)
                return rot_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_rot_launcher)<API_INT, NB, T>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, c, 0, s, 0, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rot_check_numerics_status
                = rocblas_rot_check_numerics(rocblas_rot_name<T>,
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
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(rot_check_numerics_status != rocblas_status_success)
                return rot_check_numerics_status;
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

#define IMPL(name_, TI_, T_, U_, V_)                         \
    rocblas_status name_(rocblas_handle handle,              \
                         TI_            n,                   \
                         T_* const      x[],                 \
                         TI_            incx,                \
                         T_* const      y[],                 \
                         TI_            incy,                \
                         const U_*      c,                   \
                         const V_*      s,                   \
                         TI_            batch_count)         \
    try                                                      \
    {                                                        \
        return rocblas_rot_batched_impl<TI_, T_, U_, V_>(    \
            handle, n, x, incx, y, incy, c, s, batch_count); \
    }                                                        \
    catch(...)                                               \
    {                                                        \
        return exception_to_rocblas_status();                \
    }

#define INST_ROT_BATCHED_C_API(TI_)                                                        \
    extern "C" {                                                                           \
    IMPL(ROCBLAS_API(rocblas_srot_batched), TI_, float, float, float);                     \
    IMPL(ROCBLAS_API(rocblas_drot_batched), TI_, double, double, double);                  \
    IMPL(ROCBLAS_API(rocblas_crot_batched),                                                \
         TI_,                                                                              \
         rocblas_float_complex,                                                            \
         float,                                                                            \
         rocblas_float_complex);                                                           \
    IMPL(ROCBLAS_API(rocblas_csrot_batched), TI_, rocblas_float_complex, float, float);    \
    IMPL(ROCBLAS_API(rocblas_zrot_batched),                                                \
         TI_,                                                                              \
         rocblas_double_complex,                                                           \
         double,                                                                           \
         rocblas_double_complex);                                                          \
    IMPL(ROCBLAS_API(rocblas_zdrot_batched), TI_, rocblas_double_complex, double, double); \
    } // extern "C"
