/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_rot.h"
#include "rocsparse_control.hpp"
#include "rocsparse_roti.hpp"

namespace rocsparse
{
    template <typename I, typename T>
    rocsparse_status rot_template(rocsparse_handle      handle,
                                  const void*           c,
                                  const void*           s,
                                  rocsparse_spvec_descr x,
                                  rocsparse_dnvec_descr y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::roti_template<I, T>)(handle,
                                                                   (I)x->get_nnz(),
                                                                   (T*)x->get_val_data(),
                                                                   (const I*)x->get_idx_data(),
                                                                   (T*)y->get_values(),
                                                                   (const T*)c,
                                                                   (const T*)s,
                                                                   x->get_idx_base()));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_rot(rocsparse_handle      handle,
                                          const void*           c,
                                          const void*           s,
                                          rocsparse_spvec_descr x,
                                          rocsparse_dnvec_descr y)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_rot",
                         (const void*&)c,
                         (const void*&)s,
                         (const void*&)x,
                         (const void*&)y);

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(1, c);
    ROCSPARSE_CHECKARG_POINTER(2, s);
    ROCSPARSE_CHECKARG_POINTER(3, x);
    ROCSPARSE_CHECKARG_POINTER(4, y);

    // Check if descriptors are initialized
    ROCSPARSE_CHECKARG(3, x, x->get_init() == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, y, y->get_init() == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG(3, x, (x->get_batch_count() != 1), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4, y, (y->get_batch_count() != 1), rocsparse_status_not_implemented);

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(
        4, y, (y->get_data_type() != x->get_data_type()), rocsparse_status_not_implemented);

    // single real ; i32
    if(x->get_idx_type() == rocsparse_indextype_i32
       && x->get_data_type() == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::rot_template<int32_t, float>)(handle, c, s, x, y));
        return rocsparse_status_success;
    }
    // double real ; i32
    if(x->get_idx_type() == rocsparse_indextype_i32
       && x->get_data_type() == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::rot_template<int32_t, double>)(handle, c, s, x, y));
        return rocsparse_status_success;
    }
    // single complex ; i32
    if(x->get_idx_type() == rocsparse_indextype_i32
       && x->get_data_type() == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::rot_template<int32_t, rocsparse_float_complex>)(handle, c, s, x, y));
        return rocsparse_status_success;
    }
    // double complex ; i32
    if(x->get_idx_type() == rocsparse_indextype_i32
       && x->get_data_type() == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::rot_template<int32_t, rocsparse_double_complex>)(handle, c, s, x, y));
        return rocsparse_status_success;
    }
    // single real ; i64
    if(x->get_idx_type() == rocsparse_indextype_i64
       && x->get_data_type() == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::rot_template<int64_t, float>)(handle, c, s, x, y));
        return rocsparse_status_success;
    }
    // double real ; i64
    if(x->get_idx_type() == rocsparse_indextype_i64
       && x->get_data_type() == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::rot_template<int64_t, double>)(handle, c, s, x, y));
        return rocsparse_status_success;
    }
    // single complex ; i64
    if(x->get_idx_type() == rocsparse_indextype_i64
       && x->get_data_type() == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::rot_template<int64_t, rocsparse_float_complex>)(handle, c, s, x, y));
        return rocsparse_status_success;
    }
    // double complex ; i64
    if(x->get_idx_type() == rocsparse_indextype_i64
       && x->get_data_type() == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::rot_template<int64_t, rocsparse_double_complex>)(handle, c, s, x, y));
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
