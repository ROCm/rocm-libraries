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
    constexpr char rocblas_rotg_name<float>[] = ROCBLAS_API_STR(rocblas_srotg);
    template <>
    constexpr char rocblas_rotg_name<double>[] = ROCBLAS_API_STR(rocblas_drotg);
    template <>
    constexpr char rocblas_rotg_name<rocblas_float_complex>[] = ROCBLAS_API_STR(rocblas_crotg);
    template <>
    constexpr char rocblas_rotg_name<rocblas_double_complex>[] = ROCBLAS_API_STR(rocblas_zrotg);

    template <typename API_INT, class T, class U>
    rocblas_status rocblas_rotg_impl(rocblas_handle handle, T* a, T* b, U* c, T* s)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle, rocblas_rotg_name<T>, a, b, c, s);
        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f rotg --a_type",
                             rocblas_precision_string<T>,
                             "--b_type",
                             rocblas_precision_string<U>);
        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle, rocblas_rotg_name<T>);

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
                                                       0,
                                                       b,
                                                       0,
                                                       0,
                                                       c,
                                                       0,
                                                       0,
                                                       s,
                                                       0,
                                                       0,
                                                       1,
                                                       check_numerics,
                                                       is_input);
            if(rotg_check_numerics_status != rocblas_status_success)
                return rotg_check_numerics_status;
        }

        // note the _64 API here just calls the 32bit API as only batched and strided have 64bit args
        rocblas_status status = ROCBLAS_API(rocblas_internal_rotg_launcher)<API_INT>(
            handle, a, 0, 0, b, 0, 0, c, 0, 0, s, 0, 0, 1);
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
                                                       0,
                                                       b,
                                                       0,
                                                       0,
                                                       c,
                                                       0,
                                                       0,
                                                       s,
                                                       0,
                                                       0,
                                                       1,
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

#define IMPL(name_, TI_, T_, Tc_)                                            \
    rocblas_status name_(rocblas_handle handle, T_* a, T_* b, Tc_* c, T_* s) \
    try                                                                      \
    {                                                                        \
        return rocblas_rotg_impl<TI_>(handle, a, b, c, s);                   \
    }                                                                        \
    catch(...)                                                               \
    {                                                                        \
        return exception_to_rocblas_status();                                \
    }

#define INST_ROTG_C_API(TI_)                                               \
    extern "C" {                                                           \
    IMPL(ROCBLAS_API(rocblas_srotg), TI_, float, float);                   \
    IMPL(ROCBLAS_API(rocblas_drotg), TI_, double, double);                 \
    IMPL(ROCBLAS_API(rocblas_crotg), TI_, rocblas_float_complex, float);   \
    IMPL(ROCBLAS_API(rocblas_zrotg), TI_, rocblas_double_complex, double); \
    } // extern "C"
