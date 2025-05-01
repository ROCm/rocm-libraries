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
#include "rocblas_gbmv.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_gbmv_name[] = "unknown";
    template <>
    constexpr char rocblas_gbmv_name<float>[] = ROCBLAS_API_STR(rocblas_sgbmv_batched);
    template <>
    constexpr char rocblas_gbmv_name<double>[] = ROCBLAS_API_STR(rocblas_dgbmv_batched);
    template <>
    constexpr char rocblas_gbmv_name<rocblas_float_complex>[]
        = ROCBLAS_API_STR(rocblas_cgbmv_batched);
    template <>
    constexpr char rocblas_gbmv_name<rocblas_double_complex>[]
        = ROCBLAS_API_STR(rocblas_zgbmv_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_gbmv_batched_impl(rocblas_handle    handle,
                                             rocblas_operation transA,
                                             API_INT           m,
                                             API_INT           n,
                                             API_INT           kl,
                                             API_INT           ku,
                                             const T*          alpha,
                                             const T* const    A[],
                                             API_INT           lda,
                                             const T* const    x[],
                                             API_INT           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             API_INT           incy,
                                             API_INT           batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

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
                                 rocblas_gbmv_name<T>,
                                 transA,
                                 m,
                                 n,
                                 kl,
                                 ku,
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
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f gbmv_batched -r",
                                 rocblas_precision_string<T>,
                                 "--transposeA",
                                 transA_letter,
                                 "-m",
                                 m,
                                 "-n",
                                 n,
                                 "--kl",
                                 kl,
                                 "--ku",
                                 ku,
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

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_gbmv_name<T>,
                                   "transA",
                                   transA_letter,
                                   "M",
                                   m,
                                   "N",
                                   n,
                                   "kl",
                                   kl,
                                   "ku",
                                   ku,
                                   "lda",
                                   lda,
                                   "incx",
                                   incx,
                                   "incy",
                                   incy,
                                   "batch_count",
                                   batch_count);
        }

        rocblas_status arg_status = rocblas_gbmv_arg_check(handle,
                                                           transA,
                                                           m,
                                                           n,
                                                           kl,
                                                           ku,
                                                           alpha,
                                                           A,
                                                           0,
                                                           lda,
                                                           0,
                                                           x,
                                                           0,
                                                           incx,
                                                           0,
                                                           beta,
                                                           y,
                                                           0,
                                                           incy,
                                                           0,
                                                           batch_count);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status gbmv_check_numerics_status
                = rocblas_gbmv_check_numerics(rocblas_gbmv_name<T>,
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
            if(gbmv_check_numerics_status != rocblas_status_success)
                return gbmv_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_gbmv_launcher)(handle,
                                                                            transA,
                                                                            m,
                                                                            n,
                                                                            kl,
                                                                            ku,
                                                                            alpha,
                                                                            A,
                                                                            0,
                                                                            lda,
                                                                            0,
                                                                            x,
                                                                            0,
                                                                            incx,
                                                                            0,
                                                                            beta,
                                                                            y,
                                                                            0,
                                                                            incy,
                                                                            0,
                                                                            batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status gbmv_check_numerics_status
                = rocblas_gbmv_check_numerics(rocblas_gbmv_name<T>,
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
            if(gbmv_check_numerics_status != rocblas_status_success)
                return gbmv_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                                           \
    rocblas_status routine_name_(rocblas_handle    handle,                                     \
                                 rocblas_operation transA,                                     \
                                 TI_               m,                                          \
                                 TI_               n,                                          \
                                 TI_               kl,                                         \
                                 TI_               ku,                                         \
                                 const T_*         alpha,                                      \
                                 const T_* const   A[],                                        \
                                 TI_               lda,                                        \
                                 const T_* const   x[],                                        \
                                 TI_               incx,                                       \
                                 const T_*         beta,                                       \
                                 T_* const         y[],                                        \
                                 TI_               incy,                                       \
                                 TI_               batch_count)                                \
    try                                                                                        \
    {                                                                                          \
        return rocblas_gbmv_batched_impl<TI_, T_>(                                             \
            handle, transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, batch_count); \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        return exception_to_rocblas_status();                                                  \
    }

#define INST_GBMV_BATCHED_C_API(TI_)                                       \
    extern "C" {                                                           \
    IMPL(ROCBLAS_API(rocblas_sgbmv_batched), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dgbmv_batched), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_cgbmv_batched), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_zgbmv_batched), TI_, rocblas_double_complex); \
    } // extern "C"
