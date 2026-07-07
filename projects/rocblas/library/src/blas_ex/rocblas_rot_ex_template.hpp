/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas1/rocblas_rot.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_rot_ex.hpp"

template <typename API_INT,
          rocblas_int NB,
          bool        ISBATCHED,
          typename Tx,
          typename Ty  = Tx,
          typename Tcs = Ty,
          typename Tex = Tcs>
rocblas_status rocblas_rot_ex_typecasting(rocblas_handle handle,
                                          API_INT        n,
                                          void*          x,
                                          API_INT        incx,
                                          rocblas_stride stride_x,
                                          void*          y,
                                          API_INT        incy,
                                          rocblas_stride stride_y,
                                          const void*    c,
                                          const void*    s,
                                          API_INT        batch_count)
{
    static constexpr rocblas_stride offset_0 = 0;
    static constexpr rocblas_stride stride_0 = 0;

    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;

    if(ISBATCHED)
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status rot_ex_check_numerics_status
                = rocblas_rot_check_numerics("rocblas_rot_batched_ex",
                                             handle,
                                             n,
                                             (Tx* const*)x,
                                             offset_0,
                                             incx,
                                             stride_x,
                                             (Ty* const*)y,
                                             offset_0,
                                             incy,
                                             stride_y,
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(rot_ex_check_numerics_status != rocblas_status_success)
                return rot_ex_check_numerics_status;
        }

        status = ROCBLAS_API(rocblas_internal_rot_launcher)<API_INT, NB, Tex>(handle,
                                                                              n,
                                                                              (Tx* const*)x,
                                                                              offset_0,
                                                                              incx,
                                                                              stride_x,
                                                                              (Ty* const*)y,
                                                                              offset_0,
                                                                              incy,
                                                                              stride_y,
                                                                              (const Tcs*)c,
                                                                              stride_0,
                                                                              (const Tcs*)s,
                                                                              stride_0,
                                                                              batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status rot_ex_check_numerics_status
                = rocblas_rot_check_numerics("rocblas_rot_batched_ex",
                                             handle,
                                             n,
                                             (Tx* const*)x,
                                             offset_0,
                                             incx,
                                             stride_x,
                                             (Ty* const*)y,
                                             offset_0,
                                             incy,
                                             stride_y,
                                             batch_count,
                                             check_numerics,
                                             is_input);
            if(rot_ex_check_numerics_status != rocblas_status_success)
                return rot_ex_check_numerics_status;
        }
    }
    else
    {
        if(check_numerics)
        {
            bool           is_input                     = true;
            rocblas_status rot_ex_check_numerics_status = rocblas_rot_check_numerics(
                stride_x ? "rocblas_rot_strided_batched_ex" : "rocblas_rot_ex",
                handle,
                n,
                (Tx*)x,
                offset_0,
                incx,
                stride_x,
                (Ty*)y,
                offset_0,
                incy,
                stride_y,
                batch_count,
                check_numerics,
                is_input);
            if(rot_ex_check_numerics_status != rocblas_status_success)
                return rot_ex_check_numerics_status;
        }

        status = ROCBLAS_API(rocblas_internal_rot_launcher)<API_INT, NB, Tex>(handle,
                                                                              n,
                                                                              (Tx*)x,
                                                                              offset_0,
                                                                              incx,
                                                                              stride_x,
                                                                              (Ty*)y,
                                                                              offset_0,
                                                                              incy,
                                                                              stride_y,
                                                                              (const Tcs*)c,
                                                                              stride_0,
                                                                              (const Tcs*)s,
                                                                              stride_0,
                                                                              batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input                     = false;
            rocblas_status rot_ex_check_numerics_status = rocblas_rot_check_numerics(
                stride_x ? "rocblas_rot_strided_batched_ex" : "rocblas_rot_ex",
                handle,
                n,
                (Tx*)x,
                offset_0,
                incx,
                stride_x,
                (Ty*)y,
                offset_0,
                incy,
                stride_y,
                batch_count,
                check_numerics,
                is_input);
            if(rot_ex_check_numerics_status != rocblas_status_success)
                return rot_ex_check_numerics_status;
        }
    }
    return status;
}

template <typename API_INT, rocblas_int NB, bool ISBATCHED>
rocblas_status rocblas_rot_ex_template(rocblas_handle   handle,
                                       API_INT          n,
                                       void*            x,
                                       rocblas_datatype x_type,
                                       API_INT          incx,
                                       rocblas_stride   stride_x,
                                       void*            y,
                                       rocblas_datatype y_type,
                                       API_INT          incy,
                                       rocblas_stride   stride_y,
                                       const void*      c,
                                       const void*      s,
                                       rocblas_datatype cs_type,
                                       API_INT          batch_count,
                                       rocblas_datatype execution_type)
{
#define rocblas_rot_ex_typecasting_PARAM \
    handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count

    if(x_type == rocblas_datatype_bf16_r && y_type == rocblas_datatype_bf16_r
       && cs_type == rocblas_datatype_bf16_r && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_rot_ex_typecasting<API_INT,
                                          NB,
                                          ISBATCHED,
                                          rocblas_bfloat16,
                                          rocblas_bfloat16,
                                          rocblas_bfloat16,
                                          float>(rocblas_rot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f16_r && y_type == rocblas_datatype_f16_r
            && cs_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_rot_ex_typecasting<API_INT,
                                          NB,
                                          ISBATCHED,
                                          rocblas_half,
                                          rocblas_half,
                                          rocblas_half,
                                          float>(rocblas_rot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_r && y_type == rocblas_datatype_f32_r
            && cs_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_rot_ex_typecasting<API_INT, NB, ISBATCHED, float>(
            rocblas_rot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_r && y_type == rocblas_datatype_f64_r
            && cs_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_r)
    {
        return rocblas_rot_ex_typecasting<API_INT, NB, ISBATCHED, double>(
            rocblas_rot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && y_type == rocblas_datatype_f32_c
            && cs_type == rocblas_datatype_f32_c && execution_type == rocblas_datatype_f32_c)
    {
        return rocblas_rot_ex_typecasting<API_INT, NB, ISBATCHED, rocblas_float_complex>(
            rocblas_rot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && y_type == rocblas_datatype_f32_c
            && cs_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_c)
    {
        return rocblas_rot_ex_typecasting<API_INT,
                                          NB,
                                          ISBATCHED,
                                          rocblas_float_complex,
                                          rocblas_float_complex,
                                          float,
                                          rocblas_float_complex>(rocblas_rot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && y_type == rocblas_datatype_f64_c
            && cs_type == rocblas_datatype_f64_c && execution_type == rocblas_datatype_f64_c)
    {
        return rocblas_rot_ex_typecasting<API_INT, NB, ISBATCHED, rocblas_double_complex>(
            rocblas_rot_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && y_type == rocblas_datatype_f64_c
            && cs_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_c)
    {
        return rocblas_rot_ex_typecasting<API_INT,
                                          NB,
                                          ISBATCHED,
                                          rocblas_double_complex,
                                          rocblas_double_complex,
                                          double,
                                          rocblas_double_complex>(rocblas_rot_ex_typecasting_PARAM);
    }

    return rocblas_status_not_implemented;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *rot_ex*.cpp

// clang-format off

#ifdef INSTANTIATE_ROT_EX_TEMPLATE
#error INSTANTIATE_ROT_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_ROT_EX_TEMPLATE(TI_ ,NB, ISBATCHED)                       \
template rocblas_status rocblas_rot_ex_template<TI_ , NB, ISBATCHED>           \
                                      (rocblas_handle   handle,          \
                                       TI_      n,               \
                                       void*            x,               \
                                       rocblas_datatype x_type,          \
                                       TI_      incx,            \
                                       rocblas_stride   stride_x,        \
                                       void*            y,               \
                                       rocblas_datatype y_type,          \
                                       TI_      incy,            \
                                       rocblas_stride   stride_y,        \
                                       const void*      c,               \
                                       const void*      s,               \
                                       rocblas_datatype cs_type,         \
                                       TI_      batch_count,     \
                                       rocblas_datatype execution_type);
// clang-format on
