/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "internal/generic/rocsparse_spvv.h"
#include "control.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_dotci.hpp"
#include "rocsparse_doti.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename X, typename Y>
    rocsparse_status spvv_template_real(rocsparse_handle            handle,
                                        rocsparse_operation         trans,
                                        rocsparse_const_spvec_descr x,
                                        rocsparse_const_dnvec_descr y,
                                        void*                       result,
                                        rocsparse_datatype          compute_type,
                                        size_t*                     buffer_size,
                                        void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // If temp_buffer is nullptr, return buffer_size
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer
            *buffer_size = 4;

            return rocsparse_status_success;
        }

        // real precision
        if(compute_type == rocsparse_datatype_i32_r || compute_type == rocsparse_datatype_f32_r
           || compute_type == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::doti_template(handle,
                                                               (I)x->nnz,
                                                               (const X*)x->val_data,
                                                               (const I*)x->idx_data,
                                                               (const Y*)y->values,
                                                               (T*)result,
                                                               x->idx_base));
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    template <typename T, typename I, typename X, typename Y>
    rocsparse_status spvv_template_complex(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spvec_descr x,
                                           rocsparse_const_dnvec_descr y,
                                           void*                       result,
                                           rocsparse_datatype          compute_type,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // If temp_buffer is nullptr, return buffer_size
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer
            *buffer_size = 4;
            return rocsparse_status_success;
        }

        // complex precision
        if(compute_type == rocsparse_datatype_f32_c || compute_type == rocsparse_datatype_f64_c)
        {
            // non transpose
            if(trans == rocsparse_operation_none)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::doti_template(handle,
                                                                   (I)x->nnz,
                                                                   (const X*)x->val_data,
                                                                   (const I*)x->idx_data,
                                                                   (const Y*)y->values,
                                                                   (T*)result,
                                                                   x->idx_base));
                return rocsparse_status_success;
            }

            // conjugate transpose
            if(trans == rocsparse_operation_conjugate_transpose)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dotci_template(handle,
                                                                    (I)x->nnz,
                                                                    (const X*)x->val_data,
                                                                    (const I*)x->idx_data,
                                                                    (const Y*)y->values,
                                                                    (T*)result,
                                                                    x->idx_base));
                return rocsparse_status_success;
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spvv(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spvec_descr x,
                                           rocsparse_const_dnvec_descr y,
                                           void*                       result,
                                           rocsparse_datatype          compute_type,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_spvv",
                         trans,
                         (const void*&)x,
                         (const void*&)y,
                         (const void*&)result,
                         compute_type,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    // Check operation
    ROCSPARSE_CHECKARG_ENUM(1, trans);

    // Check compute type
    ROCSPARSE_CHECKARG_ENUM(5, compute_type);

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(2, x);
    ROCSPARSE_CHECKARG_POINTER(3, y);

    // Check for valid pointers
    ROCSPARSE_CHECKARG_POINTER(4, result);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        ROCSPARSE_CHECKARG_POINTER(6, buffer_size);
    }

    // Check if descriptors are initialized
    ROCSPARSE_CHECKARG(2, x, x->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(3, y, y->init == false, rocsparse_status_not_initialized);

    const rocsparse_indextype itype = x->idx_type;
    const rocsparse_datatype  xtype = x->data_type;
    const rocsparse_datatype  ytype = y->data_type;
    const rocsparse_datatype  ctype = compute_type;

#define PARAMS handle, trans, x, y, result, compute_type, buffer_size, temp_buffer

    if(ctype == rocsparse_datatype_f32_r && itype == rocsparse_indextype_i32
       && xtype == rocsparse_datatype_f16_r && ytype == rocsparse_datatype_f16_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<float, int32_t, _Float16, _Float16>(PARAMS)));
        return rocsparse_status_success;
    }

    if(ctype == rocsparse_datatype_f32_r && itype == rocsparse_indextype_i32
       && xtype == rocsparse_datatype_f32_r && ytype == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<float, int32_t, float, float>(PARAMS)));
        return rocsparse_status_success;
    }

    if(ctype == rocsparse_datatype_f32_r && itype == rocsparse_indextype_i32
       && xtype == rocsparse_datatype_i8_r && ytype == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<float, int32_t, int8_t, int8_t>(PARAMS)));
        return rocsparse_status_success;
    }

    if(ctype == rocsparse_datatype_f32_r && itype == rocsparse_indextype_i64
       && xtype == rocsparse_datatype_f16_r && ytype == rocsparse_datatype_f16_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<float, int64_t, _Float16, _Float16>(PARAMS)));
        return rocsparse_status_success;
    }
    if(ctype == rocsparse_datatype_f32_r && itype == rocsparse_indextype_i64
       && xtype == rocsparse_datatype_f32_r && ytype == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<float, int64_t, float, float>(PARAMS)));
        return rocsparse_status_success;
    }
    if(ctype == rocsparse_datatype_f32_r && itype == rocsparse_indextype_i64
       && xtype == rocsparse_datatype_i8_r && ytype == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<float, int64_t, int8_t, int8_t>(PARAMS)));
        return rocsparse_status_success;
    }

    if(ctype == rocsparse_datatype_f64_r && itype == rocsparse_indextype_i32
       && xtype == rocsparse_datatype_f64_r && ytype == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<double, int32_t, double, double>(PARAMS)));
        return rocsparse_status_success;
    }
    if(ctype == rocsparse_datatype_f64_r && itype == rocsparse_indextype_i64
       && xtype == rocsparse_datatype_f64_r && ytype == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<double, int64_t, double, double>(PARAMS)));
        return rocsparse_status_success;
    }

    if(ctype == rocsparse_datatype_f32_c && itype == rocsparse_indextype_i64
       && xtype == rocsparse_datatype_f32_c && ytype == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_complex<rocsparse_float_complex,
                                              int64_t,
                                              rocsparse_float_complex,
                                              rocsparse_float_complex>(PARAMS)));
        return rocsparse_status_success;
    }

    if(ctype == rocsparse_datatype_f32_c && itype == rocsparse_indextype_i32
       && xtype == rocsparse_datatype_f32_c && ytype == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_complex<rocsparse_float_complex,
                                              int32_t,
                                              rocsparse_float_complex,
                                              rocsparse_float_complex>(PARAMS)));
        return rocsparse_status_success;
    }
    if(ctype == rocsparse_datatype_f64_c && itype == rocsparse_indextype_i32
       && xtype == rocsparse_datatype_f64_c && ytype == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_complex<rocsparse_double_complex,
                                              int32_t,
                                              rocsparse_double_complex,
                                              rocsparse_double_complex>(PARAMS)));
        return rocsparse_status_success;
    }
    if(ctype == rocsparse_datatype_f64_c && itype == rocsparse_indextype_i64
       && xtype == rocsparse_datatype_f64_c && ytype == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_complex<rocsparse_double_complex,
                                              int64_t,
                                              rocsparse_double_complex,
                                              rocsparse_double_complex>(PARAMS)));

        return rocsparse_status_success;
    }
    if(ctype == rocsparse_datatype_i32_r && itype == rocsparse_indextype_i32
       && xtype == rocsparse_datatype_i8_r && ytype == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<int32_t, int32_t, int8_t, int8_t>(PARAMS)));
        return rocsparse_status_success;
    }
    if(ctype == rocsparse_datatype_i32_r && itype == rocsparse_indextype_i64
       && xtype == rocsparse_datatype_i8_r && ytype == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::spvv_template_real<int32_t, int64_t, int8_t, int8_t>(PARAMS)));
        return rocsparse_status_success;
    }
#undef PARAMS

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
