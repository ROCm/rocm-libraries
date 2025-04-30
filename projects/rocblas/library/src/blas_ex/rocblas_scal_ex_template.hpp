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

#include "blas1/rocblas_scal.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"

template <typename API_INT, int NB, bool BATCHED, typename Ta, typename Tx = Ta, typename Tex = Tx>
rocblas_status rocblas_scal_ex_typecasting(rocblas_handle handle,
                                           API_INT        n,
                                           const void*    alpha_void,
                                           void*          x,
                                           API_INT        incx,
                                           rocblas_stride stride_x,
                                           API_INT        batch_count)
{
    const Ta*            alpha        = (const Ta*)alpha_void;
    const rocblas_stride stride_alpha = 0;
    const rocblas_stride offset_x     = 0;

    if(!alpha_void)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alpha == 1)
            return rocblas_status_success;
    }

    if(!x)
        return rocblas_status_invalid_pointer;

    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;

    if(BATCHED)
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status scal_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    ROCBLAS_API_STR(rocblas_scal_batched_ex),
                    handle,
                    n,
                    (Tx* const*)x,
                    offset_x,
                    incx,
                    stride_x,
                    batch_count,
                    check_numerics,
                    is_input);
            if(scal_ex_check_numerics_status != rocblas_status_success)
                return scal_ex_check_numerics_status;
        }

        status = ROCBLAS_API(rocblas_internal_scal_launcher)<API_INT, NB, Tx, Tex>(
            handle, n, alpha, stride_alpha, (Tx* const*)x, offset_x, incx, stride_x, batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status scal_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    ROCBLAS_API_STR(rocblas_scal_batched_ex),
                    handle,
                    n,
                    (Tx* const*)x,
                    offset_x,
                    incx,
                    stride_x,
                    batch_count,
                    check_numerics,
                    is_input);
            if(scal_ex_check_numerics_status != rocblas_status_success)
                return scal_ex_check_numerics_status;
        }
    }
    else
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status scal_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    stride_x ? ROCBLAS_API_STR(rocblas_scal_strided_batched_ex)
                             : ROCBLAS_API_STR(rocblas_scal_ex),
                    handle,
                    n,
                    (Tx*)x,
                    offset_x,
                    incx,
                    stride_x,
                    batch_count,
                    check_numerics,
                    is_input);
            if(scal_ex_check_numerics_status != rocblas_status_success)
                return scal_ex_check_numerics_status;
        }

        status = ROCBLAS_API(rocblas_internal_scal_launcher)<API_INT, NB, Tx, Tex>(
            handle, n, alpha, stride_alpha, (Tx*)x, offset_x, incx, stride_x, batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status scal_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    stride_x ? ROCBLAS_API_STR(rocblas_scal_strided_batched_ex)
                             : ROCBLAS_API_STR(rocblas_scal_ex),
                    handle,
                    n,
                    (Tx*)x,
                    offset_x,
                    incx,
                    stride_x,
                    batch_count,
                    check_numerics,
                    is_input);
            if(scal_ex_check_numerics_status != rocblas_status_success)
                return scal_ex_check_numerics_status;
        }
    }
    return status;
}

template <typename API_INT, int NB, bool BATCHED>
rocblas_status rocblas_scal_ex_template(rocblas_handle   handle,
                                        API_INT          n,
                                        const void*      alpha,
                                        rocblas_datatype alpha_type,
                                        void*            x,
                                        rocblas_datatype x_type,
                                        API_INT          incx,
                                        rocblas_stride   stride_x,
                                        API_INT          batch_count,
                                        rocblas_datatype execution_type)
{
    // Error checking
    if(n <= 0 || incx <= 0 || batch_count <= 0) // Quick return if possible. Not Argument error
        return rocblas_status_success;

    // Quick return (alpha == 1) check and other nullptr checks will be done
    // once we know the type (in rocblas_scal_ex_typecasting).

    rocblas_status status = rocblas_status_not_implemented;

#define rocblas_scal_ex_typecasting_PARAM handle, n, alpha, x, incx, stride_x, batch_count

    if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
       && execution_type == rocblas_datatype_f32_r)
    {
        // hscal with float computation
        status
            = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, rocblas_half, rocblas_half, float>(
                rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f16_r
            && execution_type == rocblas_datatype_f32_r)
    {
        // hscal with float computation & alpha
        status = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, float, rocblas_half, float>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f16_r && x_type == rocblas_datatype_f16_r
            && execution_type == rocblas_datatype_f16_r)
    {
        // hscal
        status = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, rocblas_half>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_bf16_r && x_type == rocblas_datatype_bf16_r
            && execution_type == rocblas_datatype_f32_r)
    {
        // bfloat16 with float computation
        status = rocblas_scal_ex_typecasting<API_INT,
                                             NB,
                                             BATCHED,
                                             rocblas_bfloat16,
                                             rocblas_bfloat16,
                                             float>(rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_bf16_r
            && execution_type == rocblas_datatype_f32_r)
    {
        // bfloat16 with float computation & alpha
        status = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, float, rocblas_bfloat16, float>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f32_r
            && execution_type == rocblas_datatype_f32_r)
    {
        // sscal
        status = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, float>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_r && x_type == rocblas_datatype_f64_r
            && execution_type == rocblas_datatype_f64_r)
    {
        // dscal
        status = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, double>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_c && x_type == rocblas_datatype_f32_c
            && execution_type == rocblas_datatype_f32_c)
    {
        // cscal
        status = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, rocblas_float_complex>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_c && x_type == rocblas_datatype_f64_c
            && execution_type == rocblas_datatype_f64_c)
    {
        // zscal
        status = rocblas_scal_ex_typecasting<API_INT, NB, BATCHED, rocblas_double_complex>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f32_r && x_type == rocblas_datatype_f32_c
            && execution_type == rocblas_datatype_f32_c)
    {
        // csscal
        status
            = rocblas_scal_ex_typecasting<API_INT,
                                          NB,
                                          BATCHED,
                                          float,
                                          rocblas_float_complex,
                                          rocblas_float_complex>(rocblas_scal_ex_typecasting_PARAM);
    }
    else if(alpha_type == rocblas_datatype_f64_r && x_type == rocblas_datatype_f64_c
            && execution_type == rocblas_datatype_f64_c)
    {
        // zdscal
        status = rocblas_scal_ex_typecasting<API_INT,
                                             NB,
                                             BATCHED,
                                             double,
                                             rocblas_double_complex,
                                             rocblas_double_complex>(
            rocblas_scal_ex_typecasting_PARAM);
    }
    else
    {
        status = rocblas_status_not_implemented;
    }

    return status;

#undef rocblas_scal_ex_typecasting_PARAM
}

#ifdef INSTANTIATE_SCAL_EX_TEMPLATE
#error INSTANTIATE_SCAL_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_SCAL_EX_TEMPLATE(TI_, NB, BATCHED)                  \
    template rocblas_status rocblas_scal_ex_template<TI_, NB, BATCHED>( \
        rocblas_handle   handle,                                        \
        TI_              n,                                             \
        const void*      alpha,                                         \
        rocblas_datatype alpha_type,                                    \
        void*            x,                                             \
        rocblas_datatype x_type,                                        \
        TI_              incx,                                          \
        rocblas_stride   stride_x,                                      \
        TI_              batch_count,                                   \
        rocblas_datatype execution_type);
