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
#include "rocblas_rot_ex.hpp"

namespace
{
    constexpr int NB = ROCBLAS_ROT_NB;

    template <typename API_INT>
    rocblas_status rocblas_rot_batched_ex_impl(rocblas_handle   handle,
                                               API_INT          n,
                                               void*            x,
                                               rocblas_datatype x_type,
                                               API_INT          incx,
                                               void*            y,
                                               rocblas_datatype y_type,
                                               API_INT          incy,
                                               const void*      c,
                                               const void*      s,
                                               rocblas_datatype cs_type,
                                               API_INT          batch_count,
                                               rocblas_datatype execution_type)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto                    layer_mode  = handle->layer_mode;
        auto                    x_type_str  = rocblas_datatype_string(x_type);
        auto                    y_type_str  = rocblas_datatype_string(y_type);
        auto                    cs_type_str = rocblas_datatype_string(cs_type);
        auto                    ex_type_str = rocblas_datatype_string(execution_type);
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(handle,
                             ROCBLAS_API_STR(rocblas_rot_batched_ex),
                             n,
                             x,
                             x_type_str,
                             incx,
                             y,
                             y_type_str,
                             incy,
                             c,
                             s,
                             cs_type_str,
                             batch_count,
                             ex_type_str);
        if(layer_mode & rocblas_layer_mode_log_bench)
            logger.log_bench(handle,
                             ROCBLAS_API_BENCH "-f rot_batched_ex --a_type",
                             x_type_str,
                             "--b_type",
                             y_type_str,
                             "--c_type",
                             cs_type_str,
                             "--compute_type",
                             ex_type_str,
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
                               ROCBLAS_API_STR(rocblas_rot_batched_ex),
                               "N",
                               n,
                               "a_type",
                               x_type_str,
                               "incx",
                               incx,
                               "b_type",
                               y_type_str,
                               "incy",
                               incy,
                               "c_type",
                               cs_type_str,
                               "batch_count",
                               batch_count,
                               "compute_type",
                               ex_type_str);

        if(n <= 0 || batch_count <= 0)
            return rocblas_status_success;

        if(!x || !y || !c || !s)
            return rocblas_status_invalid_pointer;

        static constexpr rocblas_stride stride_0 = 0;
        return rocblas_rot_ex_template<API_INT, NB, true>(handle,
                                                          n,
                                                          x,
                                                          x_type,
                                                          incx,
                                                          stride_0,
                                                          y,
                                                          y_type,
                                                          incy,
                                                          stride_0,
                                                          c,
                                                          s,
                                                          cs_type,
                                                          batch_count,
                                                          execution_type);
    }

} // namespace

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(name_, TI_)                                             \
    rocblas_status name_(rocblas_handle   handle,                    \
                         TI_              n,                         \
                         void*            x,                         \
                         rocblas_datatype x_type,                    \
                         TI_              incx,                      \
                         void*            y,                         \
                         rocblas_datatype y_type,                    \
                         TI_              incy,                      \
                         const void*      c,                         \
                         const void*      s,                         \
                         rocblas_datatype cs_type,                   \
                         TI_              batch_count,               \
                         rocblas_datatype execution_type)            \
    {                                                                \
        try                                                          \
        {                                                            \
            return rocblas_rot_batched_ex_impl<TI_>(handle,          \
                                                    n,               \
                                                    x,               \
                                                    x_type,          \
                                                    incx,            \
                                                    y,               \
                                                    y_type,          \
                                                    incy,            \
                                                    c,               \
                                                    s,               \
                                                    cs_type,         \
                                                    batch_count,     \
                                                    execution_type); \
        }                                                            \
        catch(...)                                                   \
        {                                                            \
            return exception_to_rocblas_status();                    \
        }                                                            \
    }

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define INST_ROT_BATCHED_EX_C_API(TI_)              \
    extern "C" {                                    \
    IMPL(ROCBLAS_API(rocblas_rot_batched_ex), TI_); \
    } // extern "C"
