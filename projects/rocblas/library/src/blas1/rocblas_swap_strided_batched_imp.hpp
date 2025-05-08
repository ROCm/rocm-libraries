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
#include "rocblas_block_sizes.h"
#include "rocblas_swap.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_swap_strided_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_swap_strided_batched_name<float>[]
        = ROCBLAS_API_STR(rocblas_sswap_strided_batched);
    template <>
    constexpr char rocblas_swap_strided_batched_name<double>[]
        = ROCBLAS_API_STR(rocblas_dswap_strided_batched);
    template <>
    constexpr char rocblas_swap_strided_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cswap_strided_batched);
    template <>
    constexpr char rocblas_swap_strided_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zswap_strided_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_swap_strided_batched_impl(rocblas_handle handle,
                                                     API_INT        n,
                                                     T*             x,
                                                     API_INT        incx,
                                                     rocblas_stride stridex,
                                                     T*             y,
                                                     API_INT        incy,
                                                     rocblas_stride stridey,
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
                             rocblas_swap_strided_batched_name<T>,
                             n,
                             x,
                             incx,
                             stridex,
                             y,
                             incy,
                             stridey,
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f swap_strided_batched -r",
                             rocblas_precision_string<T>,
                             "-n",
                             n,
                             "--incx",
                             incx,
                             "--incy",
                             incy,
                             "--stride_x",
                             stridex,
                             "--stride_y",
                             stridey,
                             "--batch_count",
                             batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_swap_strided_batched_name<T>,
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
                               "batch_count",
                               batch_count);

        if(batch_count <= 0 || n <= 0)
            return rocblas_status_success;

        if(!x || !y)
            return rocblas_status_invalid_pointer;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status swap_check_numerics_status
                = rocblas_swap_check_numerics(rocblas_swap_strided_batched_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              stridex,
                                              y,
                                              0,
                                              incy,
                                              stridey,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(swap_check_numerics_status != rocblas_status_success)
                return swap_check_numerics_status;
        }

        static constexpr rocblas_int NB = ROCBLAS_SWAP_NB;
        rocblas_status status           = ROCBLAS_API(rocblas_internal_swap_launcher)<API_INT, NB>(
            handle, n, x, 0, incx, stridex, y, 0, incy, stridey, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status swap_check_numerics_status
                = rocblas_swap_check_numerics(rocblas_swap_strided_batched_name<T>,
                                              handle,
                                              n,
                                              x,
                                              0,
                                              incx,
                                              stridex,
                                              y,
                                              0,
                                              incy,
                                              stridey,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(swap_check_numerics_status != rocblas_status_success)
                return swap_check_numerics_status;
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

#define IMPL(name_, Ti_, T_)                                             \
    rocblas_status name_(rocblas_handle handle,                          \
                         Ti_            n,                               \
                         T_*            x,                               \
                         Ti_            incx,                            \
                         rocblas_stride stridex,                         \
                         T_*            y,                               \
                         Ti_            incy,                            \
                         rocblas_stride stridey,                         \
                         Ti_            batch_count)                     \
    try                                                                  \
    {                                                                    \
        return rocblas_swap_strided_batched_impl<Ti_>(                   \
            handle, n, x, incx, stridex, y, incy, stridey, batch_count); \
    }                                                                    \
    catch(...)                                                           \
    {                                                                    \
        return exception_to_rocblas_status();                            \
    }

#define INST_SWAP_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_sswap_strided_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dswap_strided_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_cswap_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zswap_strided_batched), TI_, rocblas_double_complex); \
    } // extern "C"
