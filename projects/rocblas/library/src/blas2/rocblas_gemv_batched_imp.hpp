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
    template <typename, typename>
    constexpr char rocblas_gemv_name[] = "unknown";
    template <>
    constexpr char rocblas_gemv_name<float, float>[] = ROCBLAS_API_STR(rocblas_sgemv_batched);
    template <>
    constexpr char rocblas_gemv_name<double, double>[] = ROCBLAS_API_STR(rocblas_dgemv_batched);
    template <>
    constexpr char rocblas_gemv_name<rocblas_float_complex, rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cgemv_batched);
    template <>
    constexpr char rocblas_gemv_name<rocblas_double_complex, rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zgemv_batched);
    template <>
    constexpr char rocblas_gemv_name<rocblas_half, rocblas_half>[]
        = ROCBLAS_API_STR(rocblas_hshgemv_batched);
    template <>
    constexpr char rocblas_gemv_name<rocblas_half, float>[]
        = ROCBLAS_API_STR(rocblas_hssgemv_batched);
    template <>
    constexpr char rocblas_gemv_name<rocblas_bfloat16, rocblas_bfloat16>[]
        = ROCBLAS_API_STR(rocblas_tstgemv_batched);
    template <>
    constexpr char rocblas_gemv_name<rocblas_bfloat16, float>[]
        = ROCBLAS_API_STR(rocblas_tssgemv_batched);

    template <typename API_INT, typename Ti, typename Tex, typename To>
    rocblas_status rocblas_gemv_batched_impl(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             API_INT           m,
                                             API_INT           n,
                                             Tex const*        alpha,
                                             Ti const* const   A[],
                                             API_INT           lda,
                                             Ti const* const   x[],
                                             API_INT           incx,
                                             Tex const*        beta,
                                             To* const         y[],
                                             API_INT           incy,
                                             API_INT           batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        size_t dev_bytes = ROCBLAS_API(rocblas_internal_gemv_kernel_workspace_size)<Tex>(
            transA, m, n, batch_count);
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
                                 rocblas_gemv_name<Ti, To>,
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
                                 incy,
                                 batch_count);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if constexpr(std::is_same<Ti, rocblas_half>{}
                             || std::is_same<Ti, rocblas_bfloat16>{})
                {
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f gemv_batched --a_type",
                                     rocblas_precision_string<Ti>,
                                     "--c_type",
                                     rocblas_precision_string<To>,
                                     "--compute_type",
                                     rocblas_precision_string<Tex>,
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
                                     incy,
                                     "--batch_count",
                                     batch_count);
                }
                else
                {
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f gemv_batched -r",
                                     rocblas_precision_string<Ti>,
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
                                     incy,
                                     "--batch_count",
                                     batch_count);
                }
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_gemv_name<Ti, To>,
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
                                   incy,
                                   "batch_count",
                                   batch_count);
        }

        rocblas_status arg_status = rocblas_internal_gemv_arg_check(handle,
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
                                                                    batch_count);
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
                = rocblas_gemv_check_numerics(rocblas_gemv_name<Ti, To>,
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
                                              batch_count,
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
                                                                            batch_count,
                                                                            (Tex*)w_mem);

        status = (status != rocblas_status_success) ? status : perf_status;
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status gemv_check_numerics_status
                = rocblas_gemv_check_numerics(rocblas_gemv_name<Ti, To>,
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
                                              batch_count,
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

#define IMPL(name_, TI_, T_, TC_, TO_)                                                 \
    rocblas_status name_(rocblas_handle    handle,                                     \
                         rocblas_operation transA,                                     \
                         TI_               m,                                          \
                         TI_               n,                                          \
                         TC_ const*        alpha,                                      \
                         T_ const* const   A[],                                        \
                         TI_               lda,                                        \
                         T_ const* const   x[],                                        \
                         TI_               incx,                                       \
                         TC_ const*        beta,                                       \
                         TO_* const        y[],                                        \
                         TI_               incy,                                       \
                         TI_               batch_count)                                \
    try                                                                                \
    {                                                                                  \
        return rocblas_gemv_batched_impl<TI_, T_, TC_, TO_>(                           \
            handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count); \
    }                                                                                  \
    catch(...)                                                                         \
    {                                                                                  \
        return exception_to_rocblas_status();                                          \
    }

#define INST_GEMV_BATCHED_C_API(TI_)                                                            \
    extern "C" {                                                                                \
    IMPL(ROCBLAS_API(rocblas_sgemv_batched), TI_, float, float, float);                         \
    IMPL(ROCBLAS_API(rocblas_dgemv_batched), TI_, double, double, double);                      \
    IMPL(ROCBLAS_API(rocblas_cgemv_batched),                                                    \
         TI_,                                                                                   \
         rocblas_float_complex,                                                                 \
         rocblas_float_complex,                                                                 \
         rocblas_float_complex);                                                                \
    IMPL(ROCBLAS_API(rocblas_zgemv_batched),                                                    \
         TI_,                                                                                   \
         rocblas_double_complex,                                                                \
         rocblas_double_complex,                                                                \
         rocblas_double_complex);                                                               \
    IMPL(ROCBLAS_API(rocblas_hshgemv_batched), TI_, rocblas_half, float, rocblas_half);         \
    IMPL(ROCBLAS_API(rocblas_hssgemv_batched), TI_, rocblas_half, float, float);                \
    IMPL(ROCBLAS_API(rocblas_tstgemv_batched), TI_, rocblas_bfloat16, float, rocblas_bfloat16); \
    IMPL(ROCBLAS_API(rocblas_tssgemv_batched), TI_, rocblas_bfloat16, float, float);            \
    } // extern "C"
