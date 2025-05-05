/*! \file */
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
#include "internal/conversion/rocsparse_csr2dense.h"

#include "rocsparse_csx2dense_impl.hpp"

#define INSTANTIATE(DIRA, ITYPE, JTYPE, TTYPE)                                      \
    template rocsparse_status rocsparse::csx2dense_impl<DIRA, ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                           \
        JTYPE                     m,                                                \
        JTYPE                     n,                                                \
        const rocsparse_mat_descr descr,                                            \
        const TTYPE*              csx_val,                                          \
        const ITYPE*              csx_row_col_ptr,                                  \
        const JTYPE*              csx_col_row_ind,                                  \
        TTYPE*                    A,                                                \
        int64_t                   lda,                                              \
        rocsparse_order           order);

INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, _Float16);
INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, float);
INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, double);
INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_row, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, _Float16);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, float);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, double);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, _Float16);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, float);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, double);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_row, int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

extern "C" {

//
// Definition of the C-implementation.
//
#define CAPI_IMPL(name_, type_)                                                              \
    rocsparse_status name_(rocsparse_handle          handle,                                 \
                           rocsparse_int             m,                                      \
                           rocsparse_int             n,                                      \
                           const rocsparse_mat_descr descr,                                  \
                           const type_*              csr_val,                                \
                           const rocsparse_int*      csr_row_ptr,                            \
                           const rocsparse_int*      csr_col_ind,                            \
                           type_*                    A,                                      \
                           rocsparse_int             ld)                                     \
    {                                                                                        \
        try                                                                                  \
        {                                                                                    \
            ROCSPARSE_ROUTINE_TRACE;                                                         \
            RETURN_IF_ROCSPARSE_ERROR(                                                       \
                rocsparse::csx2dense_impl<rocsparse_direction_row>(handle,                   \
                                                                   m,                        \
                                                                   n,                        \
                                                                   descr,                    \
                                                                   csr_val,                  \
                                                                   csr_row_ptr,              \
                                                                   csr_col_ind,              \
                                                                   A,                        \
                                                                   ld,                       \
                                                                   rocsparse_order_column)); \
            return rocsparse_status_success;                                                 \
        }                                                                                    \
        catch(...)                                                                           \
        {                                                                                    \
            RETURN_ROCSPARSE_EXCEPTION();                                                    \
        }                                                                                    \
    }

//
// C-implementations.
//
CAPI_IMPL(rocsparse_scsr2dense, float);
CAPI_IMPL(rocsparse_dcsr2dense, double);
CAPI_IMPL(rocsparse_ccsr2dense, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zcsr2dense, rocsparse_double_complex);

//
// Undefine the macro CAPI_IMPL.
//
#undef CAPI_IMPL
}
