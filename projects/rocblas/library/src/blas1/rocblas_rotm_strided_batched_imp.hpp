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
#include "rocblas_rotm.hpp"
#include "utility.hpp"

namespace
{

    template <typename>
    constexpr char rocblas_rotm_name[] = "unknown";
    template <>
    constexpr char rocblas_rotm_name<float>[] = ROCBLAS_API_STR(rocblas_srotm_strided_batched);
    template <>
    constexpr char rocblas_rotm_name<double>[] = ROCBLAS_API_STR(rocblas_drotm_strided_batched);

    template <typename API_INT, class T>
    rocblas_status rocblas_rotm_strided_batched_impl(rocblas_handle handle,
                                                     API_INT        n,
                                                     T*             x,
                                                     API_INT        incx,
                                                     rocblas_stride stride_x,
                                                     T*             y,
                                                     API_INT        incy,
                                                     rocblas_stride stride_y,
                                                     const T*       param,
                                                     rocblas_stride stride_param,
                                                     API_INT        batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle,
                             rocblas_rotm_name<T>,
                             n,
                             x,
                             incx,
                             stride_x,
                             y,
                             incy,
                             stride_y,
                             param,
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f rotm_strided_batched -r",
                             rocblas_precision_string<T>,
                             "-n",
                             n,
                             "--incx",
                             incx,
                             "--stride_x",
                             stride_x,
                             "--incy",
                             incy,
                             "--stride_y",
                             stride_y,
                             "--batch_count",
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_rotm_name<T>,
                               "N",
                               n,
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

        if(n <= 0 || batch_count <= 0)
            return rocblas_status_success;

        if(!param)
            return rocblas_status_invalid_pointer;

        if(rocblas_rotm_quick_return_param(handle, param, stride_param))
            return rocblas_status_success;

        if(!x || !y)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rotm_check_numerics_status
                = rocblas_rotm_check_numerics(rocblas_rotm_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              stride_x,
                                              y,
                                              0,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(rotm_check_numerics_status != rocblas_status_success)
                return rotm_check_numerics_status;
        }
        rocblas_status status
            = rocblas_internal_rotm_launcher<API_INT, ROCBLAS_ROTM_NB, true>(handle,
                                                                             n,
                                                                             x,
                                                                             0,
                                                                             incx,
                                                                             stride_x,
                                                                             y,
                                                                             0,
                                                                             incy,
                                                                             stride_y,
                                                                             param,
                                                                             0,
                                                                             stride_param,
                                                                             batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rotm_check_numerics_status
                = rocblas_rotm_check_numerics(rocblas_rotm_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              stride_x,
                                              y,
                                              0,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(rotm_check_numerics_status != rocblas_status_success)
                return rotm_check_numerics_status;
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

#define IMPL(name_, TI_, T_)                                                                    \
    rocblas_status name_(rocblas_handle handle,                                                 \
                         TI_            n,                                                      \
                         T_*            x,                                                      \
                         TI_            incx,                                                   \
                         rocblas_stride stride_x,                                               \
                         T_*            y,                                                      \
                         TI_            incy,                                                   \
                         rocblas_stride stride_y,                                               \
                         const T_*      param,                                                  \
                         rocblas_stride stride_param,                                           \
                         TI_            batch_count)                                            \
    try                                                                                         \
    {                                                                                           \
        return rocblas_rotm_strided_batched_impl(                                               \
            handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count); \
    }                                                                                           \
    catch(...)                                                                                  \
    {                                                                                           \
        return exception_to_rocblas_status();                                                   \
    }

#define INST_ROTM_STRIDED_BATCHED_C_API(TI_)                       \
    extern "C" {                                                   \
    IMPL(ROCBLAS_API(rocblas_srotm_strided_batched), TI_, float);  \
    IMPL(ROCBLAS_API(rocblas_drotm_strided_batched), TI_, double); \
    } // extern "C"
