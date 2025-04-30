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
    constexpr char rocblas_hemv_name<rocblas_float_complex>[] = ROCBLAS_API_STR(rocblas_chemv);
    template <>
    constexpr char rocblas_hemv_name<rocblas_double_complex>[] = ROCBLAS_API_STR(rocblas_zhemv);

    template <typename API_INT, typename T>
    rocblas_status rocblas_hemv_impl(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     API_INT        n,
                                     const T*       alpha,
                                     const T*       A,
                                     API_INT        lda,
                                     const T*       x,
                                     API_INT        incx,
                                     const T*       beta,
                                     T*             y,
                                     API_INT        incy)
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
                                     x,
                                     incx,
                                     LOG_TRACE_SCALAR_VALUE(handle, beta),
                                     y,
                                     incy);

                if(layer_mode & rocblas_layer_mode_log_bench)
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f hemv -r",
                                     rocblas_precision_string<T>,
                                     "--uplo",
                                     uplo_letter,
                                     "-n",
                                     n,
                                     LOG_BENCH_SCALAR_VALUE(handle, alpha),
                                     "--lda",
                                     lda,
                                     "--incx",
                                     incx,
                                     LOG_BENCH_SCALAR_VALUE(handle, beta),
                                     "--incy",
                                     incy);

                if(layer_mode & rocblas_layer_mode_log_profile)
                    logger.log_profile(handle,
                                       rocblas_hemv_name<T>,
                                       "uplo",
                                       uplo_letter,
                                       "N",
                                       n,
                                       "lda",
                                       lda,
                                       "incx",
                                       incx,
                                       "incy",
                                       incy);
            }
        }

        rocblas_status arg_status = rocblas_hemv_symv_arg_check<API_INT>(
            handle, uplo, n, alpha, 0, A, 0, lda, 0, x, 0, incx, 0, beta, 0, y, 0, incy, 0, 1);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        size_t dev_bytes = rocblas_internal_hemv_symv_kernel_workspace_size<T>(n);
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
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              1,
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
                                                                            0,
                                                                            x,
                                                                            0,
                                                                            incx,
                                                                            0,
                                                                            beta,
                                                                            0,
                                                                            y,
                                                                            0,
                                                                            incy,
                                                                            0,
                                                                            1,
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
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              y,
                                              0,
                                              incy,
                                              0,
                                              1,
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

#define IMPL(routine_name_, TI_, T_)                                                               \
    rocblas_status routine_name_(rocblas_handle handle,                                            \
                                 rocblas_fill   uplo,                                              \
                                 TI_            n,                                                 \
                                 const T_*      alpha,                                             \
                                 const T_*      A,                                                 \
                                 TI_            lda,                                               \
                                 const T_*      x,                                                 \
                                 TI_            incx,                                              \
                                 const T_*      beta,                                              \
                                 T_*            y,                                                 \
                                 TI_            incy)                                              \
    try                                                                                            \
    {                                                                                              \
        return rocblas_hemv_impl<TI_, T_>(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy); \
    }                                                                                              \
    catch(...)                                                                                     \
    {                                                                                              \
        return exception_to_rocblas_status();                                                      \
    }

#define INST_HEMV_C_API(TI_)                                       \
    extern "C" {                                                   \
    IMPL(ROCBLAS_API(rocblas_chemv), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zhemv), TI_, rocblas_double_complex); \
    } // extern "C"
