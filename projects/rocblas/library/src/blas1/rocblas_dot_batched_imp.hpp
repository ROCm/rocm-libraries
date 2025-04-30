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
#include "rocblas_dot.hpp"
#include "utility.hpp"

namespace
{

    template <bool, typename>
    constexpr char rocblas_dot_batched_name[] = "unknown";
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, float>[] = ROCBLAS_API_STR(rocblas_sdot_batched);
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, double>[] = ROCBLAS_API_STR(rocblas_ddot_batched);
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, rocblas_half>[]
        = ROCBLAS_API_STR(rocblas_hdot_batched);
    template <bool CONJ>
    constexpr char rocblas_dot_batched_name<CONJ, rocblas_bfloat16>[]
        = ROCBLAS_API_STR(rocblas_bfdot_batched);
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cdotc_batched);
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cdotu_batched);
    template <>
    constexpr char rocblas_dot_batched_name<true, rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zdotc_batched);
    template <>
    constexpr char rocblas_dot_batched_name<false, rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zdotu_batched);

    // allocate workspace inside this API
    template <typename API_INT, bool CONJ, typename T, typename T2 = T>
    inline rocblas_status rocblas_dot_batched_impl(rocblas_handle handle,
                                                   API_INT        n,
                                                   const T* const x[],
                                                   API_INT        incx,
                                                   const T* const y[],
                                                   API_INT        incy,
                                                   API_INT        batch_count,
                                                   T*             results)
    {
        static constexpr int WIN = rocblas_dot_WIN<T>();

        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes = rocblas_reduction_workspace_size<API_INT, ROCBLAS_DOT_NB * WIN, T2>(
            n, incx, incy, batch_count);
        if(handle->is_device_memory_size_query())
        {
            if(n <= 0 || batch_count <= 0)
                return rocblas_status_size_unchanged;
            else
                return handle->set_optimal_device_memory_size(dev_bytes);
        }

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(
                handle, rocblas_dot_batched_name<CONJ, T>, n, x, incx, y, incy, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f dot_batched -r",
                             rocblas_precision_string<T>,
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
                               rocblas_dot_batched_name<CONJ, T>,
                               "N",
                               n,
                               "incx",
                               incx,
                               "incy",
                               incy,
                               "batch_count",
                               batch_count);

        // Quick return if possible.
        if(batch_count <= 0)
        {
            return rocblas_status_success;
        }

        if(n <= 0)
        {
            if(!results)
                return rocblas_status_invalid_pointer;
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    results, 0, sizeof(*results) * batch_count, handle->get_stream()));
            else
                memset(results, 0, sizeof(*results) * batch_count);
            return rocblas_status_success;
        }

        if(!x || !y || !results)
            return rocblas_status_invalid_pointer;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status dot_check_numerics_status
                = rocblas_dot_check_numerics(rocblas_dot_batched_name<CONJ, T>,
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
            if(dot_check_numerics_status != rocblas_status_success)
                return dot_check_numerics_status;
        }

        rocblas_status status;
        if constexpr(rocblas_is_complex<T> && CONJ)
            status = ROCBLAS_API(rocblas_internal_dotc_batched_template)(
                handle, n, x, 0, incx, 0, y, 0, incy, 0, batch_count, results, (T2*)w_mem);
        else
            status = ROCBLAS_API(rocblas_internal_dot_batched_template)(
                handle, n, x, 0, incx, 0, y, 0, incy, 0, batch_count, results, (T2*)w_mem);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status dot_check_numerics_status
                = rocblas_dot_check_numerics(rocblas_dot_batched_name<CONJ, T>,
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
            if(dot_check_numerics_status != rocblas_status_success)
                return dot_check_numerics_status;
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

#define IMPL(name_, TI_, conj_, T_, Tex_)                       \
    rocblas_status name_(rocblas_handle  handle,                \
                         TI_             n,                     \
                         const T_* const x[],                   \
                         TI_             incx,                  \
                         const T_* const y[],                   \
                         TI_             incy,                  \
                         TI_             batch_count,           \
                         T_*             results)               \
    try                                                         \
    {                                                           \
        return rocblas_dot_batched_impl<TI_, conj_, T_, Tex_>(  \
            handle, n, x, incx, y, incy, batch_count, results); \
    }                                                           \
    catch(...)                                                  \
    {                                                           \
        return exception_to_rocblas_status();                   \
    }

#define INST_DOT_BATCHED_C_API(TI_)                                                  \
    extern "C" {                                                                     \
    IMPL(ROCBLAS_API(rocblas_sdot_batched), TI_, false, float, float);               \
    IMPL(ROCBLAS_API(rocblas_ddot_batched), TI_, false, double, double);             \
    IMPL(ROCBLAS_API(rocblas_hdot_batched), TI_, false, rocblas_half, rocblas_half); \
    IMPL(ROCBLAS_API(rocblas_bfdot_batched), TI_, false, rocblas_bfloat16, float);   \
    IMPL(ROCBLAS_API(rocblas_cdotu_batched),                                         \
         TI_,                                                                        \
         false,                                                                      \
         rocblas_float_complex,                                                      \
         rocblas_float_complex);                                                     \
    IMPL(ROCBLAS_API(rocblas_zdotu_batched),                                         \
         TI_,                                                                        \
         false,                                                                      \
         rocblas_double_complex,                                                     \
         rocblas_double_complex);                                                    \
    IMPL(ROCBLAS_API(rocblas_cdotc_batched),                                         \
         TI_,                                                                        \
         true,                                                                       \
         rocblas_float_complex,                                                      \
         rocblas_float_complex);                                                     \
    IMPL(ROCBLAS_API(rocblas_zdotc_batched),                                         \
         TI_,                                                                        \
         true,                                                                       \
         rocblas_double_complex,                                                     \
         rocblas_double_complex);                                                    \
    } // extern "C"
