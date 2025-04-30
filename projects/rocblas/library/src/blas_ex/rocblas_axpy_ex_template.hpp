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

#include "../blas1/rocblas_axpy.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"

template <typename API_INT,
          int  NB,
          bool BATCHED,
          typename Ta,
          typename Tx  = Ta,
          typename Ty  = Tx,
          typename Tex = Ty>
rocblas_status rocblas_axpy_ex_typecasting(rocblas_handle handle,
                                           API_INT        n,
                                           const void*    alpha,
                                           rocblas_stride stride_alpha,
                                           const void*    x,
                                           rocblas_stride offset_x,
                                           API_INT        incx,
                                           rocblas_stride stride_x,
                                           void*          y,
                                           rocblas_stride offset_y,
                                           API_INT        incy,
                                           rocblas_stride stride_y,
                                           API_INT        batch_count)
{
    auto check_numerics = handle->check_numerics;

    const Ta* alphat = (const Ta*)alpha;
    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alphat == 0)
            return rocblas_status_success;

        if(!x || !y)
            return rocblas_status_invalid_pointer;
    }

    if(BATCHED)
    {
        //Checking input batched vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status axpy_ex_check_numerics_status
                = rocblas_axpy_check_numerics(ROCBLAS_API_STR(rocblas_axpy_batched_ex),
                                              handle,
                                              n,
                                              (const Tx* const*)x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              (Ty* const*)y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }

        rocblas_status status
            = ROCBLAS_API(rocblas_internal_axpy_launcher)<API_INT, NB, Tex>(handle,
                                                                            n,
                                                                            alphat,
                                                                            stride_alpha,
                                                                            (const Tx* const*)x,
                                                                            offset_x,
                                                                            incx,
                                                                            stride_x,
                                                                            (Ty* const*)y,
                                                                            offset_y,
                                                                            incy,
                                                                            stride_y,
                                                                            batch_count);
        if(status != rocblas_status_success)
            return status;

        //Checking output batched vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status axpy_ex_check_numerics_status
                = rocblas_axpy_check_numerics(ROCBLAS_API_STR(rocblas_axpy_batched_ex),
                                              handle,
                                              n,
                                              (const Tx* const*)x,
                                              offset_x,
                                              incx,
                                              stride_x,
                                              (Ty* const*)y,
                                              offset_y,
                                              incy,
                                              stride_y,
                                              batch_count,
                                              check_numerics,
                                              is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }
        return status;
    }
    else
    {
        //Checking input vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input                      = true;
            rocblas_status axpy_ex_check_numerics_status = rocblas_axpy_check_numerics(
                stride_x ? ROCBLAS_API_STR(rocblas_axpy_strided_batched_ex)
                         : ROCBLAS_API_STR(rocblas_axpy_ex),
                handle,
                n,
                (const Tx*)x,
                offset_x,
                incx,
                stride_x,
                (Ty*)y,
                offset_y,
                incy,
                stride_y,
                batch_count,
                check_numerics,
                is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }

        rocblas_status status
            = ROCBLAS_API(rocblas_internal_axpy_launcher)<API_INT, NB, Tex>(handle,
                                                                            n,
                                                                            alphat,
                                                                            stride_alpha,
                                                                            (const Tx*)x,
                                                                            offset_x,
                                                                            incx,
                                                                            stride_x,
                                                                            (Ty*)y,
                                                                            offset_y,
                                                                            incy,
                                                                            stride_y,
                                                                            batch_count);

        if(status != rocblas_status_success)
            return status;

        //Checking output vectors for numerical abnormalities
        if(check_numerics)
        {
            bool           is_input                      = false;
            rocblas_status axpy_ex_check_numerics_status = rocblas_axpy_check_numerics(
                stride_x ? ROCBLAS_API_STR(rocblas_axpy_strided_batched_ex)
                         : ROCBLAS_API_STR(rocblas_axpy_ex),
                handle,
                n,
                (const Tx*)x,
                offset_x,
                incx,
                stride_x,
                (Ty*)y,
                offset_y,
                incy,
                stride_y,
                batch_count,
                check_numerics,
                is_input);
            if(axpy_ex_check_numerics_status != rocblas_status_success)
                return axpy_ex_check_numerics_status;
        }
        return status;
    }
}

template <typename API_INT, int NB, bool BATCHED>
rocblas_status rocblas_axpy_ex_template(rocblas_handle   handle,
                                        API_INT          n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        rocblas_stride   stride_alpha,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        rocblas_stride   offset_x,
                                        API_INT          incx,
                                        rocblas_stride   stride_x,
                                        void*            y,
                                        rocblas_datatype y_type,
                                        rocblas_stride   offset_y,
                                        API_INT          incy,
                                        rocblas_stride   stride_y,
                                        API_INT          batch_count,
                                        rocblas_datatype execution_type)
{
    // Error checking
    if(n <= 0 || batch_count <= 0) // Quick return if possible. Not Argument error
    {
        return rocblas_status_success;
    }

    if(!alpha)
        return rocblas_status_invalid_pointer;

    // Quick return (alpha == 0) check and other nullptr checks will be done
    // once we know the type (in rocblas_axpy_ex_typecasting).

    rocblas_status status = rocblas_status_not_implemented;

#define rocblas_axpy_ex_typecasting_PARAM                                                     \
    handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, y, offset_y, incy, stride_y, \
        batch_count

    if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
       && y_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        status = rocblas_axpy_ex_typecasting<API_INT,
                                             NB,
                                             BATCHED,
                                             rocblas_half,
                                             rocblas_half,
                                             rocblas_half,
                                             float>(rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f16_r
            && y_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f32_r)
    {
        status = rocblas_axpy_ex_typecasting<API_INT,
                                             NB,
                                             BATCHED,
                                             float,
                                             rocblas_half,
                                             rocblas_half,
                                             float>(rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_bf16_r && x_type == rocblas_datatype_bf16_r
            && y_type == rocblas_datatype_bf16_r && execution_type == rocblas_datatype_f32_r)
    {
        status = rocblas_axpy_ex_typecasting<API_INT,
                                             NB,
                                             BATCHED,
                                             rocblas_bfloat16,
                                             rocblas_bfloat16,
                                             rocblas_bfloat16,
                                             float>(rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_bf16_r
            && y_type == rocblas_datatype_bf16_r && execution_type == rocblas_datatype_f32_r)
    {
        status = rocblas_axpy_ex_typecasting<API_INT,
                                             NB,
                                             BATCHED,
                                             float,
                                             rocblas_bfloat16,
                                             rocblas_bfloat16,
                                             float>(rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
            && y_type == rocblas_datatype_f16_r && execution_type == rocblas_datatype_f16_r)
    {
        status = rocblas_axpy_ex_typecasting<API_INT, NB, BATCHED, rocblas_half>(
            rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f32_r
            && y_type == rocblas_datatype_f32_r && execution_type == rocblas_datatype_f32_r)
    {
        status = rocblas_axpy_ex_typecasting<API_INT, NB, BATCHED, float>(
            rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_r && x_type == rocblas_datatype_f64_r
            && y_type == rocblas_datatype_f64_r && execution_type == rocblas_datatype_f64_r)
    {
        status = rocblas_axpy_ex_typecasting<API_INT, NB, BATCHED, double>(
            rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_c && x_type == rocblas_datatype_f32_c
            && y_type == rocblas_datatype_f32_c && execution_type == rocblas_datatype_f32_c)
    {
        status = rocblas_axpy_ex_typecasting<API_INT, NB, BATCHED, rocblas_float_complex>(
            rocblas_axpy_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_c && x_type == rocblas_datatype_f64_c
            && y_type == rocblas_datatype_f64_c && execution_type == rocblas_datatype_f64_c)
    {
        status = rocblas_axpy_ex_typecasting<API_INT, NB, BATCHED, rocblas_double_complex>(
            rocblas_axpy_ex_typecasting_PARAM);
    }
    else
    {
        status = rocblas_status_not_implemented;
    }

    return status;

#undef rocblas_axpy_ex_typecasting_PARAM
}

#ifdef INSTANTIATE_AXPY_EX_TEMPLATE
#error INSTANTIATE_AXPY_EX_TEMPLATE already defined
#endif

#define INSTANTIATE_AXPY_EX_TEMPLATE(TI_, NB, BATCHED)                  \
    template rocblas_status rocblas_axpy_ex_template<TI_, NB, BATCHED>( \
        rocblas_handle   handle,                                        \
        TI_              n,                                             \
        const void*      alpha,                                         \
        rocblas_datatype alpha_type,                                    \
        rocblas_stride   stride_alpha,                                  \
        const void*      x,                                             \
        rocblas_datatype x_type,                                        \
        rocblas_stride   offset_x,                                      \
        TI_              incx,                                          \
        rocblas_stride   stride_x,                                      \
        void*            y,                                             \
        rocblas_datatype y_type,                                        \
        rocblas_stride   offset_y,                                      \
        TI_              incy,                                          \
        rocblas_stride   stride_y,                                      \
        TI_              batch_count,                                   \
        rocblas_datatype execution_type);
