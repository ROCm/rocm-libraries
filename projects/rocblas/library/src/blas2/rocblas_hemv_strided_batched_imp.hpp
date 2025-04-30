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
#include "rocblas_hemv_symv.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_hemv_name[] = "unknown";
    template <>
    constexpr char rocblas_hemv_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_chemv_strided_batched);
    template <>
    constexpr char rocblas_hemv_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zhemv_strided_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_hemv_strided_batched_impl(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     API_INT        n,
                                                     const T*       alpha,
                                                     const T*       A,
                                                     API_INT        lda,
                                                     rocblas_stride stride_A,
                                                     const T*       x,
                                                     API_INT        incx,
                                                     rocblas_stride stride_x,
                                                     const T*       beta,
                                                     T*             y,
                                                     API_INT        incy,
                                                     rocblas_stride stride_y,
                                                     API_INT        batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter = rocblas_fill_letter(uplo);

                if(layer_mode & rocblas_layer_mode_log_trace)
                    logger.log_trace(handle,
                                     rocblas_hemv_name<T>,
                                     uplo,
                                     n,
                                     LOG_TRACE_SCALAR_VALUE(handle, alpha),
                                     A,
                                     lda,
                                     stride_A,
                                     x,
                                     incx,
                                     stride_x,
                                     LOG_TRACE_SCALAR_VALUE(handle, beta),
                                     y,
                                     incy,
                                     stride_y,
                                     batch_count);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f hemv_strided_batched -r",
                                     rocblas_precision_string<T>,
                                     "--uplo",
                                     uplo_letter,
                                     "-n",
                                     n,
                                     LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                     "--lda",
                                     lda,
                                     "--stride_a",
                                     stride_A,
                                     "--incx",
                                     incx,
                                     "--stride_x",
                                     stride_x,
                                     LOG_BENCH_SCALAR_VALUE(handle, beta),
                                     "--incy",
                                     incy,
                                     "--stride_y",
                                     stride_y,
                                     "--batch_count",
                                     batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                    logger.log_profile(handle,
                                       rocblas_hemv_name<T>,
                                       "uplo",
                                       uplo_letter,
                                       "N",
                                       n,
                                       "lda",
                                       lda,
                                       "stride_a",
                                       stride_A,
                                       "incx",
                                       incx,
                                       "stride_x",
                                       stride_x,
                                       "incy",
                                       incy,
                                       "stride_y",
                                       stride_y,
                                       "batch_count",
                                       batch_count);
            }
        }

        rocblas_status arg_status = rocblas_hemv_symv_arg_check<API_INT>(handle,
                                                                         uplo,
                                                                         n,
                                                                         alpha,
                                                                         0,
                                                                         A,
                                                                         0,
                                                                         lda,
                                                                         stride_A,
                                                                         x,
                                                                         0,
                                                                         incx,
                                                                         stride_x,
                                                                         beta,
                                                                         0,
                                                                         y,
                                                                         0,
                                                                         incy,
                                                                         stride_y,
                                                                         batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        size_t dev_bytes = rocblas_internal_hemv_symv_kernel_workspace_size<T>(n, batch_count);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        // flag to check whether the kernel function being called is for hemv or symv
        // For hemv IS_HEMV = true and for SYMV IS_HEMV = false
        static constexpr bool IS_HEMV = true;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status hemv_check_numerics_status
                = rocblas_hemv_check_numerics(rocblas_hemv_name<T>,
                                              handle,
                                              uplo,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              stride_A,
                                              x,
                                              0,
                                              incx,
                                              stride_x,
                                              y,
                                              0,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(hemv_check_numerics_status != rocblas_status_success)
                return hemv_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_hemv_template)(handle,
                                                                            uplo,
                                                                            n,
                                                                            alpha,
                                                                            0,
                                                                            A,
                                                                            0,
                                                                            lda,
                                                                            stride_A,
                                                                            x,
                                                                            0,
                                                                            incx,
                                                                            stride_x,
                                                                            beta,
                                                                            0,
                                                                            y,
                                                                            0,
                                                                            incy,
                                                                            stride_y,
                                                                            batch_count,
                                                                            (T*)w_mem);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status hemv_check_numerics_status
                = rocblas_hemv_check_numerics(rocblas_hemv_name<T>,
                                              handle,
                                              uplo,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              stride_A,
                                              x,
                                              0,
                                              incx,
                                              stride_x,
                                              y,
                                              0,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(hemv_check_numerics_status != rocblas_status_success)
                return hemv_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                    \
    rocblas_status routine_name_(rocblas_handle handle,                 \
                                 rocblas_fill   uplo,                   \
                                 TI_            n,                      \
                                 const T_*      alpha,                  \
                                 const T_*      A,                      \
                                 TI_            lda,                    \
                                 rocblas_stride strideA,                \
                                 const T_*      x,                      \
                                 TI_            incx,                   \
                                 rocblas_stride stridex,                \
                                 const T_*      beta,                   \
                                 T_*            y,                      \
                                 TI_            incy,                   \
                                 rocblas_stride stridey,                \
                                 TI_            batch_count)            \
    try                                                                 \
    {                                                                   \
        return rocblas_hemv_strided_batched_impl<TI_, T_>(handle,       \
                                                          uplo,         \
                                                          n,            \
                                                          alpha,        \
                                                          A,            \
                                                          lda,          \
                                                          strideA,      \
                                                          x,            \
                                                          incx,         \
                                                          stridex,      \
                                                          beta,         \
                                                          y,            \
                                                          incy,         \
                                                          stridey,      \
                                                          batch_count); \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

#define INST_HEMV_STRIDED_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                                   \
    IMPL(ROCBLAS_API(rocblas_chemv_strided_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zhemv_strided_batched), TI_, rocblas_double_complex); \
    } // extern "C"
