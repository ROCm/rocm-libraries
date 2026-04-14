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
#include "rocblas_rotg.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_rotg_name[] = "unknown";
    template <>
    constexpr char rocblas_rotg_name<float>[] = ROCBLAS_API_STR(rocblas_srotg_strided_batched);
    template <>
    constexpr char rocblas_rotg_name<double>[] = ROCBLAS_API_STR(rocblas_drotg_strided_batched);
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_crotg_strided_batched);
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zrotg_strided_batched);

    template <typename API_INT, class T, class U>
    rocblas_status rocblas_rotg_strided_batched_impl(rocblas_handle handle,
                                                     T*             a,
                                                     rocblas_stride stride_a,
                                                     T*             b,
                                                     rocblas_stride stride_b,
                                                     U*             c,
                                                     rocblas_stride stride_c,
                                                     T*             s,
                                                     rocblas_stride stride_s,
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
                             rocblas_rotg_name<T>,
                             a,
                             stride_a,
                             b,
                             stride_b,
                             c,
                             stride_c,
                             s,
                             stride_s,
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f rotg_strided_batched --a_type",
                             rocblas_precision_string<T>,
                             "--b_type",
                             rocblas_precision_string<U>,
                             "--stride_a",
                             stride_a,
                             "--stride_b",
                             stride_b,
                             "--stride_c",
                             stride_c,
                             "--stride_d",
                             stride_s,
                             "--batch_count",
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_rotg_name<T>,
                               "stride_a",
                               stride_a,
                               "stride_b",
                               stride_b,
                               "stride_c",
                               stride_c,
                               "stride_d",
                               stride_s,
                               "batch_count",
                               batch_count);

        if(batch_count <= 0)
            return rocblas_status_success;

        if(!a || !b || !c || !s)
            return rocblas_status_invalid_pointer;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rotg_check_numerics_status
                = rocblas_rotg_check_numerics_template(rocblas_rotg_name<T>,
                                                       handle,
                                                       a,
                                                       0,
                                                       stride_a,
                                                       b,
                                                       0,
                                                       stride_b,
                                                       c,
                                                       0,
                                                       stride_c,
                                                       s,
                                                       0,
                                                       stride_s,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_rotg_launcher)<API_INT>(
            handle, a, 0, stride_a, b, 0, stride_b, c, 0, stride_c, s, 0, stride_s, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rotg_check_numerics_status
                = rocblas_rotg_check_numerics_template(rocblas_rotg_name<T>,
                                                       handle,
                                                       a,
                                                       0,
                                                       stride_a,
                                                       b,
                                                       0,
                                                       stride_b,
                                                       c,
                                                       0,
                                                       stride_c,
                                                       s,
                                                       0,
                                                       stride_s,
                                                       batch_count,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
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

#define IMPL(name_, TI_, T_, Tc_)                                                     \
    rocblas_status name_(rocblas_handle handle,                                       \
                         T_*            a,                                            \
                         rocblas_stride stride_a,                                     \
                         T_*            b,                                            \
                         rocblas_stride stride_b,                                     \
                         Tc_*           c,                                            \
                         rocblas_stride stride_c,                                     \
                         T_*            s,                                            \
                         rocblas_stride stride_s,                                     \
                         TI_            batch_count)                                  \
    try                                                                               \
    {                                                                                 \
        return rocblas_rotg_strided_batched_impl(                                     \
            handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count); \
    }                                                                                 \
    catch(...)                                                                        \
    {                                                                                 \
        return exception_to_rocblas_status();                                         \
    }

#define INST_ROTG_STRIDED_BATCHED_C_API(TI_)                                               \
    extern "C" {                                                                           \
    IMPL(ROCBLAS_API(rocblas_srotg_strided_batched), TI_, float, float);                   \
    IMPL(ROCBLAS_API(rocblas_drotg_strided_batched), TI_, double, double);                 \
    IMPL(ROCBLAS_API(rocblas_crotg_strided_batched), TI_, rocblas_float_complex, float);   \
    IMPL(ROCBLAS_API(rocblas_zrotg_strided_batched), TI_, rocblas_double_complex, double); \
    } // extern "C"
