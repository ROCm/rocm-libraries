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
#include "rocblas_rotmg.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_rotmg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotmg_name<float>[] = ROCBLAS_API_STR(rocblas_srotmg_strided_batched);
    template <>
    constexpr char rocblas_rotmg_name<double>[] = ROCBLAS_API_STR(rocblas_drotmg_strided_batched);

    template <typename API_INT, class T>
    rocblas_status rocblas_rotmg_strided_batched_impl(rocblas_handle handle,
                                                      T*             d1,
                                                      rocblas_stride stride_d1,
                                                      T*             d2,
                                                      rocblas_stride stride_d2,
                                                      T*             x1,
                                                      rocblas_stride stride_x1,
                                                      const T*       y1,
                                                      rocblas_stride stride_y1,
                                                      T*             param,
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
                             rocblas_rotmg_name<T>,
                             d1,
                             stride_d1,
                             d2,
                             stride_d2,
                             x1,
                             stride_x1,
                             y1,
                             stride_y1,
                             param,
                             stride_param,
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f rotmg_strided_batched -r",
                             rocblas_precision_string<T>,
                             "--batch_count",
                             batch_count,
                             "--stride_a",
                             stride_d1,
                             "--stride_b",
                             stride_d2,
                             "--stride_c",
                             stride_param,
                             "--stride_x",
                             stride_x1,
                             "--stride_y",
                             stride_y1);
        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_rotmg_name<T>,
                               "stride_a",
                               stride_d1,
                               "stride_b",
                               stride_d2,
                               "stride_c",
                               stride_param,
                               "stride_x",
                               stride_x1,
                               "stride_y",
                               stride_y1,
                               "batch_count",
                               batch_count);

        if(batch_count <= 0)
            return rocblas_status_success;
        if(!d1 || !d2 || !x1 || !y1 || !param)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rotmg_check_numerics_status
                = rocblas_rotmg_check_numerics_template(rocblas_rotmg_name<T>,
                                                        handle,
                                                        d1,
                                                        0,
                                                        stride_d1,
                                                        d2,
                                                        0,
                                                        stride_d2,
                                                        x1,
                                                        0,
                                                        stride_x1,
                                                        y1,
                                                        0,
                                                        stride_y1,
                                                        batch_count,
                                                        check_numerics,
                                                        is_input);
            if(rotmg_check_numerics_status != rocblas_status_success)
                return rotmg_check_numerics_status;
        }
        rocblas_status status = ROCBLAS_API(rocblas_internal_rotmg_launcher)<API_INT>(handle,
                                                                                      d1,
                                                                                      0,
                                                                                      stride_d1,
                                                                                      d2,
                                                                                      0,
                                                                                      stride_d2,
                                                                                      x1,
                                                                                      0,
                                                                                      stride_x1,
                                                                                      y1,
                                                                                      0,
                                                                                      stride_y1,
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
                = rocblas_rotmg_check_numerics_template(rocblas_rotmg_name<T>,
                                                        handle,
                                                        d1,
                                                        0,
                                                        stride_d1,
                                                        d2,
                                                        0,
                                                        stride_d2,
                                                        x1,
                                                        0,
                                                        stride_x1,
                                                        y1,
                                                        0,
                                                        stride_y1,
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

#define IMPL(name_, TI_, T_)                                    \
    rocblas_status name_(rocblas_handle handle,                 \
                         T_*            d1,                     \
                         rocblas_stride stride_d1,              \
                         T_*            d2,                     \
                         rocblas_stride stride_d2,              \
                         T_*            x1,                     \
                         rocblas_stride stride_x1,              \
                         const T_*      y1,                     \
                         rocblas_stride stride_y1,              \
                         T_*            param,                  \
                         rocblas_stride stride_param,           \
                         TI_            batch_count)            \
    try                                                         \
    {                                                           \
        return rocblas_rotmg_strided_batched_impl(handle,       \
                                                  d1,           \
                                                  stride_d1,    \
                                                  d2,           \
                                                  stride_d2,    \
                                                  x1,           \
                                                  stride_x1,    \
                                                  y1,           \
                                                  stride_y1,    \
                                                  param,        \
                                                  stride_param, \
                                                  batch_count); \
    }                                                           \
    catch(...)                                                  \
    {                                                           \
        return exception_to_rocblas_status();                   \
    }

#define INST_ROTMG_STRIDED_BATCHED_C_API(TI_)                       \
    extern "C" {                                                    \
    IMPL(ROCBLAS_API(rocblas_srotmg_strided_batched), TI_, float);  \
    IMPL(ROCBLAS_API(rocblas_drotmg_strided_batched), TI_, double); \
    } // extern "C"
