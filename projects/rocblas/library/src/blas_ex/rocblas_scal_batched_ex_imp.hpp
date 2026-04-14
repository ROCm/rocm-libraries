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
#include "utility.hpp"

namespace
{
    template <typename API_INT>
    rocblas_status rocblas_scal_batched_ex_impl(rocblas_handle   handle,
                                                API_INT          n,
                                                const void*      alpha,
                                                rocblas_datatype alpha_type,
                                                void*            x,
                                                rocblas_datatype x_type,
                                                API_INT          incx,
                                                API_INT          batch_count,
                                                rocblas_datatype execution_type)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode = handle->layer_mode;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto                    alpha_type_str = rocblas_datatype_string(alpha_type);
            auto                    x_type_str     = rocblas_datatype_string(x_type);
            auto                    ex_type_str    = rocblas_datatype_string(execution_type);
            rocblas_internal_logger logger;

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    rocblas_internal_ostream alphass, betass;
                    (void)rocblas_internal_log_trace_alpha_beta_ex(
                        alpha_type, alpha, nullptr, alphass, betass);
                    logger.log_trace(handle,
                                     ROCBLAS_API_STR(rocblas_scal_batched_ex),
                                     n,
                                     alphass.str(),
                                     alpha_type_str,
                                     x,
                                     x_type_str,
                                     incx,
                                     batch_count,
                                     ex_type_str);
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    (void)rocblas_internal_log_bench_alpha_beta_ex(
                        alpha_type, alpha, nullptr, alphas, betas);
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f scal_batched_ex",
                                     "-n",
                                     n,
                                     alphas,
                                     "--incx",
                                     incx,
                                     "--batch_count",
                                     batch_count,
                                     rocblas_internal_log_bench_ex_precisions(
                                         alpha_type, x_type, execution_type));
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    logger.log_trace(handle,
                                     ROCBLAS_API_STR(rocblas_scal_batched_ex),
                                     n,
                                     alpha_type_str,
                                     x,
                                     x_type_str,
                                     incx,
                                     batch_count,
                                     ex_type_str);
            }
            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   ROCBLAS_API_STR(rocblas_scal_batched_ex),
                                   "N",
                                   n,
                                   "a_type",
                                   alpha_type_str,
                                   "b_type",
                                   x_type_str,
                                   "incx",
                                   incx,
                                   "batch_count",
                                   batch_count,
                                   "compute_type",
                                   ex_type_str);
        }

        static constexpr rocblas_stride stride_0 = 0;
        return rocblas_scal_ex_template<API_INT, ROCBLAS_SCAL_NB, true>(
            handle, n, alpha, alpha_type, x, x_type, incx, stride_0, batch_count, execution_type);
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

#define IMPL(name_, TI_)                                                                 \
    rocblas_status name_(rocblas_handle   handle,                                        \
                         TI_              n,                                             \
                         const void*      alpha,                                         \
                         rocblas_datatype alpha_type,                                    \
                         void*            x,                                             \
                         rocblas_datatype x_type,                                        \
                         TI_              incx,                                          \
                         TI_              batch_count,                                   \
                         rocblas_datatype execution_type)                                \
    try                                                                                  \
    {                                                                                    \
        return rocblas_scal_batched_ex_impl<TI_>(                                        \
            handle, n, alpha, alpha_type, x, x_type, incx, batch_count, execution_type); \
    }                                                                                    \
    catch(...)                                                                           \
    {                                                                                    \
        return exception_to_rocblas_status();                                            \
    }

#define INST_SCAL_BATCHED_EX_C_API(TI_)              \
    extern "C" {                                     \
    IMPL(ROCBLAS_API(rocblas_scal_batched_ex), TI_); \
    } // extern "C"
