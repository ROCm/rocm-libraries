/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblas_block_sizes.h"
#include "rocblas_iamax_iamin.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_iamax_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_iamax_batched_name<float>[] = ROCBLAS_API_STR(rocblas_isamax_batched);
    template <>
    constexpr char rocblas_iamax_batched_name<double>[] = ROCBLAS_API_STR(rocblas_idamax_batched);
    template <>
    constexpr char rocblas_iamax_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_icamax_batched);
    template <>
    constexpr char rocblas_iamax_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_izamax_batched);

    // allocate workspace inside this API
    template <typename API_INT, typename S, typename T>
    rocblas_status rocblas_iamax_batched_impl(rocblas_handle  handle,
                                              API_INT         n,
                                              const T* const* x,
                                              API_INT         incx,
                                              API_INT         batch_count,
                                              API_INT*        result)
    {
        using index_val_t = std::conditional_t<std::is_same_v<API_INT, rocblas_int>,
                                               rocblas_index_value_t<S>,
                                               rocblas_index_64_value_t<S>>;

        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes
            = rocblas_single_pass_reduction_workspace_size<API_INT, ROCBLAS_IAMAX_NB, index_val_t>(
                n, batch_count);

        if(handle->is_device_memory_size_query())
        {
            if(n <= 0 || incx <= 0 || batch_count <= 0)
                return rocblas_status_size_unchanged;
            else
                return handle->set_optimal_device_memory_size(dev_bytes);
        }

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle, rocblas_iamax_batched_name<T>, n, x, incx, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f iamax_batched",
                             "-r",
                             rocblas_precision_string<T>,
                             "-n",
                             n,
                             "--incx",
                             incx,
                             "--batch_count",
                             batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_iamax_batched_name<T>,
                               "N",
                               n,
                               "incx",
                               incx,
                               "batch_count",
                               batch_count);

        static constexpr rocblas_stride shiftx_0  = 0;
        static constexpr rocblas_stride stridex_0 = 0;

        rocblas_status arg_status
            = rocblas_iamax_iamin_arg_check(handle, n, x, incx, stridex_0, batch_count, result);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
        {
            return rocblas_status_memory_error;
        }

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status check_numerics_status
                = rocblas_internal_check_numerics_vector_template(rocblas_iamax_batched_name<T>,
                                                                  handle,
                                                                  n,
                                                                  x,
                                                                  shiftx_0,
                                                                  incx,
                                                                  stridex_0,
                                                                  batch_count,
                                                                  check_numerics,
                                                                  is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }

        return ROCBLAS_API(rocblas_internal_iamax_batched_template)(
            handle, n, x, shiftx_0, incx, stridex_0, batch_count, result, (index_val_t*)w_mem);
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_routine_, TI_, T_, S_)                                                          \
    rocblas_status name_routine_(                                                                 \
        rocblas_handle handle, TI_ n, const T_* const* x, TI_ incx, TI_ batch_count, TI_* result) \
    try                                                                                           \
    {                                                                                             \
        return rocblas_iamax_batched_impl<TI_, S_>(handle, n, x, incx, batch_count, result);      \
    }                                                                                             \
    catch(...)                                                                                    \
    {                                                                                             \
        return exception_to_rocblas_status();                                                     \
    }

#define INST_IAMAX_BATCHED_C_API(TI_)                                               \
    extern "C" {                                                                    \
    IMPL(ROCBLAS_API(rocblas_isamax_batched), TI_, float, float);                   \
    IMPL(ROCBLAS_API(rocblas_idamax_batched), TI_, double, double);                 \
    IMPL(ROCBLAS_API(rocblas_icamax_batched), TI_, rocblas_float_complex, float);   \
    IMPL(ROCBLAS_API(rocblas_izamax_batched), TI_, rocblas_double_complex, double); \
    } // extern "C"
