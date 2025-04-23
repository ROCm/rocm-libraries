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

#include "internal/generic/rocsparse_gather.h"
#include "control.h"
#include "handle.h"
#include "internal/level1/rocsparse_gthr.h"
#include "utility.h"

#include "rocsparse_gthr.hpp"

namespace rocsparse
{
    template <typename I, typename T>
    rocsparse_status gather_template(rocsparse_handle            handle,
                                     rocsparse_const_dnvec_descr y,
                                     rocsparse_spvec_descr       x)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gthr_template<I, T>)(handle,
                                                                   (I)x->nnz,
                                                                   (const T*)y->const_values,
                                                                   (T*)x->val_data,
                                                                   (const I*)x->const_idx_data,
                                                                   x->idx_base));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_gather(rocsparse_handle            handle,
                                             rocsparse_const_dnvec_descr y,
                                             rocsparse_spvec_descr       x)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_gather", (const void*&)y, (const void*&)x);

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(1, y);
    ROCSPARSE_CHECKARG_POINTER(2, x);

    // Check if descriptors are initialized
    ROCSPARSE_CHECKARG(1, y, y->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(2, x, x->init == false, rocsparse_status_not_initialized);

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(2, x, (x->data_type != y->data_type), rocsparse_status_not_implemented);

    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int32_t, int8_t>)(handle, y, x));
        return rocsparse_status_success;
    }
    // half real ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f16_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int32_t, _Float16>)(handle, y, x));
        return rocsparse_status_success;
    }
    // single real ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int32_t, float>)(handle, y, x));
        return rocsparse_status_success;
    }
    // double real ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int32_t, double>)(handle, y, x));
        return rocsparse_status_success;
    }
    // single complex ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::gather_template<int32_t, rocsparse_float_complex>)(handle, y, x));
        return rocsparse_status_success;
    }
    // double complex ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::gather_template<int32_t, rocsparse_double_complex>)(handle, y, x));
        return rocsparse_status_success;
    }
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int64_t, int8_t>)(handle, y, x));
        return rocsparse_status_success;
    }
    // half real ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f16_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int64_t, _Float16>)(handle, y, x));
        return rocsparse_status_success;
    }
    // single real ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int64_t, float>)(handle, y, x));
        return rocsparse_status_success;
    }
    // double real ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::gather_template<int64_t, double>)(handle, y, x));
        return rocsparse_status_success;
    }
    // single complex ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::gather_template<int64_t, rocsparse_float_complex>)(handle, y, x));
        return rocsparse_status_success;
    }
    // double complex ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::gather_template<int64_t, rocsparse_double_complex>)(handle, y, x));
        return rocsparse_status_success;
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
