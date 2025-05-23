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
#include "rocblas_asum_nrm2.hpp"
#include "rocblas_block_sizes.h"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_asum_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_asum_batched_name<float>[] = ROCBLAS_API_STR(rocblas_sasum_batched);
    template <>
    constexpr char rocblas_asum_batched_name<double>[] = ROCBLAS_API_STR(rocblas_dasum_batched);
    template <>
    constexpr char rocblas_asum_batched_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_scasum_batched);
    template <>
    constexpr char rocblas_asum_batched_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_dzasum_batched);

    // allocate workspace inside this API
    template <typename API_INT, rocblas_int NB, typename Ti>
    rocblas_status rocblas_asum_batched_impl(rocblas_handle  handle,
                                             API_INT         n,
                                             const Ti* const x[],
                                             API_INT         incx,
                                             API_INT         batch_count,
                                             real_t<Ti>*     result)
    {
        using To = real_t<Ti>;
        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes
            = rocblas_reduction_workspace_size<API_INT, NB, To>(n, incx, incx, batch_count);

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
            logger.log_trace(handle, rocblas_asum_batched_name<Ti>, n, x, incx, batch_count);

        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH " -f asum_batched -r",
                             rocblas_precision_string<Ti>,
                             "-n",
                             n,
                             "--incx",
                             incx,
                             "--batch_count",
                             batch_count);

        if(layer_mode & rocblas_layer_mode_log_profile)
            logger.log_profile(handle,
                               rocblas_asum_batched_name<Ti>,
                               "N",
                               n,
                               "incx",
                               incx,
                               "batch_count",
                               batch_count);

        static constexpr rocblas_stride stridex_0 = 0;
        static constexpr rocblas_stride shiftx_0  = 0;

        rocblas_status arg_status
            = rocblas_asum_nrm2_arg_check(handle, n, x, incx, stridex_0, batch_count, result);
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
                = rocblas_internal_check_numerics_vector_template(rocblas_asum_batched_name<Ti>,
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

        return ROCBLAS_API(rocblas_internal_asum_nrm2_launcher)<API_INT,
                                                                NB,
                                                                rocblas_fetch_asum<To>,
                                                                rocblas_finalize_identity>(
            handle, n, x, shiftx_0, incx, stridex_0, batch_count, (To*)w_mem, result);
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

#define IMPL(name_, TI_, T_)                                    \
    rocblas_status name_(rocblas_handle  handle,                \
                         TI_             n,                     \
                         const T_* const x[],                   \
                         TI_             incx,                  \
                         TI_             batch_count,           \
                         real_t<T_>*     result)                \
    try                                                         \
    {                                                           \
        return rocblas_asum_batched_impl<TI_, ROCBLAS_ASUM_NB>( \
            handle, n, x, incx, batch_count, result);           \
    }                                                           \
    catch(...)                                                  \
    {                                                           \
        return exception_to_rocblas_status();                   \
    }

#define INST_ASUM_BATCHED_C_API(TI_)                                        \
    extern "C" {                                                            \
    IMPL(ROCBLAS_API(rocblas_sasum_batched), TI_, float);                   \
    IMPL(ROCBLAS_API(rocblas_dasum_batched), TI_, double);                  \
    IMPL(ROCBLAS_API(rocblas_scasum_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_dzasum_batched), TI_, rocblas_double_complex); \
    }
