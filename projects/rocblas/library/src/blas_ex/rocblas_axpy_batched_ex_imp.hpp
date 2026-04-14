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
#include "rocblas_axpy_ex.hpp"
#include "rocblas_block_sizes.h"
#include "utility.hpp"

namespace
{
    template <typename API_INT>
    rocblas_status rocblas_axpy_batched_ex_impl(rocblas_handle   handle,
                                                API_INT          n,
                                                const void*      alpha,
                                                rocblas_datatype alpha_type,
                                                const void*      x,
                                                rocblas_datatype x_type,
                                                API_INT          incx,
                                                void*            y,
                                                rocblas_datatype y_type,
                                                API_INT          incy,
                                                API_INT          batch_count,
                                                rocblas_datatype execution_type)
    {
        if(!handle)
        {
            return rocblas_status_invalid_handle;
        }

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto                    layer_mode = handle->layer_mode;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto alpha_type_str = rocblas_datatype_string(alpha_type);
            auto x_type_str     = rocblas_datatype_string(x_type);
            auto y_type_str     = rocblas_datatype_string(y_type);
            auto ex_type_str    = rocblas_datatype_string(execution_type);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    rocblas_internal_ostream alphass, betass;
                    (void)rocblas_internal_log_trace_alpha_beta_ex(
                        alpha_type, alpha, nullptr, alphass, betass);
                    logger.log_trace(handle,
                                     ROCBLAS_API_STR(rocblas_axpy_batched_ex),
                                     n,
                                     alphass.str(),
                                     alpha_type_str,
                                     x,
                                     x_type_str,
                                     incx,
                                     y,
                                     y_type_str,
                                     incy,
                                     batch_count,
                                     ex_type_str);
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    (void)rocblas_internal_log_bench_alpha_beta_ex(
                        alpha_type, alpha, nullptr, alphas, betas);
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f axpy_batched_ex",
                                     "-n",
                                     n,
                                     alphas,
                                     "--a_type",
                                     alpha_type_str,
                                     "--b_type",
                                     x_type_str,
                                     "--incx",
                                     incx,
                                     "--c_type",
                                     y_type_str,
                                     "--incy",
                                     incy,
                                     "--batch_count",
                                     batch_count,
                                     "--compute_type",
                                     ex_type_str);
                }
            }
            else if(layer_mode & rocblas_layer_mode_log_trace)
            {
                logger.log_trace(handle,
                                 ROCBLAS_API_STR(rocblas_axpy_batched_ex),
                                 n,
                                 alpha_type_str,
                                 x,
                                 x_type_str,
                                 incx,
                                 y,
                                 y_type_str,
                                 incy,
                                 batch_count,
                                 ex_type_str);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                logger.log_profile(handle,
                                   ROCBLAS_API_STR(rocblas_axpy_batched_ex),
                                   "N",
                                   n,
                                   "a_type",
                                   alpha_type_str,
                                   "b_type",
                                   x_type_str,
                                   "incx",
                                   incx,
                                   "c_type",
                                   y_type_str,
                                   "incy",
                                   incy,
                                   "batch_count",
                                   batch_count,
                                   "compute_type",
                                   ex_type_str);
            }
        }

        static constexpr rocblas_stride stride_0 = 0;
        static constexpr rocblas_stride offset_0 = 0;
        return rocblas_axpy_ex_template<API_INT, ROCBLAS_AXPY_NB, true>(handle,
                                                                        n,
                                                                        alpha,
                                                                        alpha_type,
                                                                        stride_0,
                                                                        x,
                                                                        x_type,
                                                                        offset_0,
                                                                        incx,
                                                                        stride_0,
                                                                        y,
                                                                        y_type,
                                                                        offset_0,
                                                                        incy,
                                                                        stride_0,
                                                                        batch_count,
                                                                        execution_type);
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define INST_AXPY_BATCHED_EX_C_API(TI_)                                                  \
    extern "C" {                                                                         \
    rocblas_status ROCBLAS_API(rocblas_axpy_batched_ex)(rocblas_handle   handle,         \
                                                        TI_              n,              \
                                                        const void*      alpha,          \
                                                        rocblas_datatype alpha_type,     \
                                                        const void*      x,              \
                                                        rocblas_datatype x_type,         \
                                                        TI_              incx,           \
                                                        void*            y,              \
                                                        rocblas_datatype y_type,         \
                                                        TI_              incy,           \
                                                        TI_              batch_count,    \
                                                        rocblas_datatype execution_type) \
    try                                                                                  \
    {                                                                                    \
        return rocblas_axpy_batched_ex_impl<TI_>(handle,                                 \
                                                 n,                                      \
                                                 alpha,                                  \
                                                 alpha_type,                             \
                                                 x,                                      \
                                                 x_type,                                 \
                                                 incx,                                   \
                                                 y,                                      \
                                                 y_type,                                 \
                                                 incy,                                   \
                                                 batch_count,                            \
                                                 execution_type);                        \
    }                                                                                    \
    catch(...)                                                                           \
    {                                                                                    \
        return exception_to_rocblas_status();                                            \
    }                                                                                    \
    } // extern "C"
