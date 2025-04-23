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
#include "rocblas_gemv.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_gemv_name[] = "unknown";
    template <>
    constexpr char rocblas_gemv_name<float>[] = ROCBLAS_API_STR(rocblas_sgemv);
    template <>
    constexpr char rocblas_gemv_name<double>[] = ROCBLAS_API_STR(rocblas_dgemv);
    template <>
    constexpr char rocblas_gemv_name<rocblas_float_complex>[] = ROCBLAS_API_STR(rocblas_cgemv);
    template <>
    constexpr char rocblas_gemv_name<rocblas_double_complex>[] = ROCBLAS_API_STR(rocblas_zgemv);

    template <typename API_INT, typename T>
    rocblas_status rocblas_gemv_impl(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     API_INT           m,
                                     API_INT           n,
                                     const T*          alpha,
                                     const T*          A,
                                     API_INT           lda,
                                     const T*          x,
                                     API_INT           incx,
                                     const T*          beta,
                                     T*                y,
                                     API_INT           incy)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes
            = ROCBLAS_API(rocblas_internal_gemv_kernel_workspace_size)<T>(transA, m, n, 1);
        if(handle->is_device_memory_size_query())
            return handle->set_optimal_device_memory_size(dev_bytes);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto transA_letter = rocblas_transpose_letter(transA);

            if(layer_mode & rocblas_layer_mode_log_trace)
                logger.log_trace(handle,
                                 rocblas_gemv_name<T>,
                                 transA,
                                 m,
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
                                 ROCBLAS_API_BENCH " -f gemv -r",
                                 rocblas_precision_string<T>,
                                 "--transposeA",
                                 transA_letter,
                                 "-m",
                                 m,
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
                                   rocblas_gemv_name<T>,
                                   "transA",
                                   transA_letter,
                                   "M",
                                   m,
                                   "N",
                                   n,
                                   "lda",
                                   lda,
                                   "incx",
                                   incx,
                                   "incy",
                                   incy);
        }

        rocblas_status arg_status = rocblas_internal_gemv_arg_check<API_INT>(
            handle, transA, m, n, alpha, 0, A, 0, lda, 0, x, 0, incx, 0, beta, 0, y, 0, incy, 0, 1);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        rocblas_status perf_status = rocblas_status_success;
        auto           w_mem       = handle->device_malloc(dev_bytes);
        if(!w_mem)
            perf_status = rocblas_status_perf_degraded;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status gemv_check_numerics_status
                = rocblas_gemv_check_numerics(rocblas_gemv_name<T>,
                                              handle,
                                              transA,
                                              m,
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
            if(gemv_check_numerics_status != rocblas_status_success)
                return gemv_check_numerics_status;
        }

        // we don't instantiate _template for mixed types so directly calling launcher
        rocblas_status status = ROCBLAS_API(rocblas_internal_gemv_launcher)(handle,
                                                                            transA,
                                                                            m,
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
                                                                            (API_INT)1,
                                                                            (T*)w_mem);

        status = (status != rocblas_status_success) ? status : perf_status;
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status gemv_check_numerics_status
                = rocblas_gemv_check_numerics(rocblas_gemv_name<T>,
                                              handle,
                                              transA,
                                              m,
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
            if(gemv_check_numerics_status != rocblas_status_success)
                return gemv_check_numerics_status;
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

#define IMPL(name_, TI_, T_)                                              \
    rocblas_status name_(rocblas_handle    handle,                        \
                         rocblas_operation transA,                        \
                         TI_               m,                             \
                         TI_               n,                             \
                         const T_*         alpha,                         \
                         const T_*         A,                             \
                         TI_               lda,                           \
                         const T_*         x,                             \
                         TI_               incx,                          \
                         const T_*         beta,                          \
                         T_*               y,                             \
                         TI_               incy)                          \
    try                                                                   \
    {                                                                     \
        return rocblas_gemv_impl<TI_, T_>(                                \
            handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy); \
    }                                                                     \
    catch(...)                                                            \
    {                                                                     \
        return exception_to_rocblas_status();                             \
    }

#define INST_GEMV_C_API(TI_)                                       \
    extern "C" {                                                   \
    IMPL(ROCBLAS_API(rocblas_sgemv), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dgemv), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_cgemv), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zgemv), TI_, rocblas_double_complex); \
    } // extern "C"
