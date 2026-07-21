/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas1/rocblas_dot.hpp"

#include "handle.hpp"
#include "logging.hpp"

template <typename API_INT,
          int  NB,
          bool ISBATCHED,
          bool CONJ,
          typename Tx,
          typename Ty  = Tx,
          typename Tr  = Ty,
          typename Tex = Tr>
rocblas_status rocblas_dot_ex_typecasting(rocblas_handle __restrict__ handle,
                                          API_INT n,
                                          const void* __restrict__ x,
                                          API_INT        incx,
                                          rocblas_stride stride_x,
                                          const void* __restrict__ y,
                                          API_INT        incy,
                                          rocblas_stride stride_y,
                                          API_INT        batch_count,
                                          void* __restrict__ results,
                                          void* __restrict__ workspace)
{
    auto                            check_numerics = handle->check_numerics;
    rocblas_status                  status         = rocblas_status_success;
    static constexpr rocblas_stride offset_0       = 0;

    if(ISBATCHED)
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status dot_ex_check_numerics_status
                = rocblas_dot_check_numerics(ROCBLAS_API_STR(rocblas_dot_batched_ex),
                                             handle,
                                             n,
                                             (const Tx* const*)x,
                                             offset_0,
                                             incx,
                                             stride_x,
                                             (const Ty* const*)y,
                                             offset_0,
                                             incy,
                                             stride_y,
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(dot_ex_check_numerics_status != rocblas_status_success)
                return dot_ex_check_numerics_status;
        }
        status = ROCBLAS_API(rocblas_internal_dot_launcher)<API_INT, NB, CONJ>(handle,
                                                                               n,
                                                                               (const Tx* const*)x,
                                                                               offset_0,
                                                                               incx,
                                                                               stride_x,
                                                                               (const Ty* const*)y,
                                                                               offset_0,
                                                                               incy,
                                                                               stride_y,
                                                                               batch_count,
                                                                               (Tr*)results,
                                                                               (Tex*)workspace);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status dot_ex_check_numerics_status
                = rocblas_dot_check_numerics(ROCBLAS_API_STR(rocblas_dot_batched_ex),
                                             handle,
                                             n,
                                             (const Tx* const*)x,
                                             offset_0,
                                             incx,
                                             stride_x,
                                             (const Ty* const*)y,
                                             offset_0,
                                             incy,
                                             stride_y,
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(dot_ex_check_numerics_status != rocblas_status_success)
                return dot_ex_check_numerics_status;
        }
    }
    else
    {
        if(check_numerics)
        {
            bool           is_input                     = true;
            rocblas_status dot_ex_check_numerics_status = rocblas_dot_check_numerics(
                stride_x ? ROCBLAS_API_STR(rocblas_dot_strided_batched_ex)
                         : ROCBLAS_API_STR(rocblas_dot_ex),
                handle,
                n,
                (const Tx*)x,
                offset_0,
                incx,
                stride_x,
                (const Ty*)y,
                offset_0,
                incy,
                stride_y,
                batch_count,
                check_numerics,
                is_input);
            if(dot_ex_check_numerics_status != rocblas_status_success)
                return dot_ex_check_numerics_status;
        }

        status = ROCBLAS_API(rocblas_internal_dot_launcher)<API_INT, NB, CONJ>(handle,
                                                                               n,
                                                                               (const Tx*)x,
                                                                               offset_0,
                                                                               incx,
                                                                               stride_x,
                                                                               (const Ty*)y,
                                                                               offset_0,
                                                                               incy,
                                                                               stride_y,
                                                                               batch_count,
                                                                               (Tr*)results,
                                                                               (Tex*)workspace);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input                     = false;
            rocblas_status dot_ex_check_numerics_status = rocblas_dot_check_numerics(
                stride_x ? ROCBLAS_API_STR(rocblas_dot_strided_batched_ex)
                         : ROCBLAS_API_STR(rocblas_dot_ex),
                handle,
                n,
                (const Tx*)x,
                offset_0,
                incx,
                stride_x,
                (const Ty*)y,
                offset_0,
                incy,
                stride_y,
                batch_count,
                check_numerics,
                is_input);
            if(dot_ex_check_numerics_status != rocblas_status_success)
                return dot_ex_check_numerics_status;
        }
    }
    return status;
}

template <typename API_INT, int NB, bool ISBATCHED, bool CONJ>
rocblas_status rocblas_dot_ex_template(rocblas_handle __restrict__ handle,
                                       API_INT n,
                                       const void* __restrict__ x,
                                       rocblas_datatype x_type,
                                       API_INT          incx,
                                       rocblas_stride   stride_x,
                                       const void* __restrict__ y,
                                       rocblas_datatype y_type,
                                       API_INT          incy,
                                       rocblas_stride   stride_y,
                                       API_INT          batch_count,
                                       void* __restrict__ results,
                                       rocblas_datatype result_type,
                                       rocblas_datatype execution_type,
                                       void* __restrict__ workspace)
{
#define rocblas_dot_ex_typecasting_PARAM \
    handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, results, workspace

    if(x_type == rocblas_datatype_f16_r && y_type == rocblas_datatype_f16_r
       && result_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f16_r)
    {
        return rocblas_dot_ex_typecasting<API_INT, NB, ISBATCHED, CONJ, rocblas_half>(
            rocblas_dot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_bf16_r && y_type == rocblas_datatype_bf16_r
            && result_type == rocblas_datatype_bf16_r && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_dot_ex_typecasting<API_INT,
                                          NB,
                                          ISBATCHED,
                                          CONJ,
                                          rocblas_bfloat16,
                                          rocblas_bfloat16,
                                          rocblas_bfloat16,
                                          float>(rocblas_dot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f16_r && y_type == rocblas_datatype_f16_r
            && result_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_dot_ex_typecasting<API_INT,
                                          NB,
                                          ISBATCHED,
                                          CONJ,
                                          rocblas_half,
                                          rocblas_half,
                                          rocblas_half,
                                          float>(rocblas_dot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_r && y_type == rocblas_datatype_f32_r
            && result_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_dot_ex_typecasting<API_INT, NB, ISBATCHED, CONJ, float>(
            rocblas_dot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_r && y_type == rocblas_datatype_f64_r
            && result_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_r)
    {
        return rocblas_dot_ex_typecasting<API_INT, NB, ISBATCHED, CONJ, double>(
            rocblas_dot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && y_type == rocblas_datatype_f32_c
            && result_type == rocblas_datatype_f32_c && execution_type == rocblas_datatype_f32_c)
    {
        return rocblas_dot_ex_typecasting<API_INT, NB, ISBATCHED, CONJ, rocblas_float_complex>(
            rocblas_dot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && y_type == rocblas_datatype_f64_c
            && result_type == rocblas_datatype_f64_c && execution_type == rocblas_datatype_f64_c)
    {
        return rocblas_dot_ex_typecasting<API_INT, NB, ISBATCHED, CONJ, rocblas_double_complex>(
            rocblas_dot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_r && y_type == rocblas_datatype_f32_r
            && result_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_r)
    {
        return rocblas_dot_ex_typecasting<API_INT,
                                          NB,
                                          ISBATCHED,
                                          CONJ,
                                          float,
                                          float,
                                          double,
                                          double>(rocblas_dot_ex_typecasting_PARAM);
    }

    return rocblas_status_not_implemented;
}

#ifdef INSTANTIATE_DOT_EX_TEMPLATE
#error INSTANTIATE_DOT_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_DOT_EX_TEMPLATE(TI_, NB, ISBATCHED, CONJ)                  \
    template rocblas_status rocblas_dot_ex_template<TI_, NB, ISBATCHED, CONJ>( \
        rocblas_handle handle,                                                 \
        TI_            n,                                                      \
        const void* __restrict__ x,                                            \
        rocblas_datatype x_type,                                               \
        TI_              incx,                                                 \
        rocblas_stride   stride_x,                                             \
        const void* __restrict__ y,                                            \
        rocblas_datatype y_type,                                               \
        TI_              incy,                                                 \
        rocblas_stride   stride_y,                                             \
        TI_              batch_count,                                          \
        void* __restrict__ results,                                            \
        rocblas_datatype result_type,                                          \
        rocblas_datatype execution_type,                                       \
        void* __restrict__ workspace);
