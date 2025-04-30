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
#include "rocblas_tbsv.hpp"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_tbsv_name[] = "unknown";
    template <>
    constexpr char rocblas_tbsv_name<float>[] = ROCBLAS_API_STR(rocblas_stbsv);
    template <>
    constexpr char rocblas_tbsv_name<double>[] = ROCBLAS_API_STR(rocblas_dtbsv);
    template <>
    constexpr char rocblas_tbsv_name<rocblas_float_complex>[] = ROCBLAS_API_STR(rocblas_ctbsv);
    template <>
    constexpr char rocblas_tbsv_name<rocblas_double_complex>[] = ROCBLAS_API_STR(rocblas_ztbsv);

    template <typename API_INT, typename T>
    rocblas_status rocblas_tbsv_impl(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     API_INT           n,
                                     API_INT           k,
                                     const T*          A,
                                     API_INT           lda,
                                     T*                x,
                                     API_INT           incx)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        rocblas_internal_logger logger;

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            logger.log_trace(
                handle, rocblas_tbsv_name<T>, uplo, transA, diag, n, k, A, lda, x, incx);

        if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto diag_letter   = rocblas_diag_letter(diag);

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                if(handle->pointer_mode == rocblas_pointer_mode_host) // TODO log both modes
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f tbsv -r",
                                     rocblas_precision_string<T>,
                                     "--uplo",
                                     uplo_letter,
                                     "--transposeA",
                                     transA_letter,
                                     "--diag",
                                     diag_letter,
                                     "-n",
                                     n,
                                     "-k",
                                     k,
                                     "--lda",
                                     lda,
                                     "--incx",
                                     incx);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(handle,
                                   rocblas_tbsv_name<T>,
                                   "uplo",
                                   uplo_letter,
                                   "transA",
                                   transA_letter,
                                   "diag",
                                   diag_letter,
                                   "N",
                                   n,
                                   "K",
                                   k,
                                   "lda",
                                   lda,
                                   "incx",
                                   incx);
        }

        rocblas_status arg_status
            = rocblas_tbsv_arg_check<API_INT>(handle, uplo, transA, diag, n, k, A, lda, x, incx, 1);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status tbsv_check_numerics_status
                = rocblas_tbsv_check_numerics(rocblas_tbsv_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(tbsv_check_numerics_status != rocblas_status_success)
                return tbsv_check_numerics_status;
        }

        rocblas_status status = ROCBLAS_API(rocblas_internal_tbsv_launcher)(
            handle, uplo, transA, diag, n, k, A, 0, lda, 0, x, 0, incx, 0, 1);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status tbsv_check_numerics_status
                = rocblas_tbsv_check_numerics(rocblas_tbsv_name<T>,
                                              handle,
                                              n,
                                              A,
                                              0,
                                              lda,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(tbsv_check_numerics_status != rocblas_status_success)
                return tbsv_check_numerics_status;
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

#define IMPL(routine_name_, TI_, T_)                                                      \
    rocblas_status routine_name_(rocblas_handle    handle,                                \
                                 rocblas_fill      uplo,                                  \
                                 rocblas_operation transA,                                \
                                 rocblas_diagonal  diag,                                  \
                                 TI_               n,                                     \
                                 TI_               k,                                     \
                                 const T_*         A,                                     \
                                 TI_               lda,                                   \
                                 T_*               x,                                     \
                                 TI_               incx)                                  \
    try                                                                                   \
    {                                                                                     \
        return rocblas_tbsv_impl<TI_>(handle, uplo, transA, diag, n, k, A, lda, x, incx); \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        return exception_to_rocblas_status();                                             \
    }

#define INST_TBSV_C_API(TI_)                                       \
    extern "C" {                                                   \
    IMPL(ROCBLAS_API(rocblas_stbsv), TI_, float);                  \
    IMPL(ROCBLAS_API(rocblas_dtbsv), TI_, double);                 \
    IMPL(ROCBLAS_API(rocblas_ctbsv), TI_, rocblas_float_complex);  \
    IMPL(ROCBLAS_API(rocblas_ztbsv), TI_, rocblas_double_complex); \
    } // extern "C"
