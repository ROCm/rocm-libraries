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
#include "internal/conversion/rocsparse_dense2csc.h"
#include "rocsparse_dense2csx_impl.hpp"

#define INSTANTIATE(DIRA, ITYPE, JTYPE, TTYPE)                                      \
    template rocsparse_status rocsparse::dense2csx_impl<DIRA, ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                           \
        rocsparse_order           order,                                            \
        JTYPE                     m,                                                \
        JTYPE                     n,                                                \
        const rocsparse_mat_descr descr_A,                                          \
        const TTYPE*              A,                                                \
        int64_t                   lda,                                              \
        const ITYPE*              nnz_per_row_column,                               \
        TTYPE*                    csx_val_A,                                        \
        ITYPE*                    csx_row_col_ptr_A,                                \
        JTYPE*                    csx_col_row_ind_A);

INSTANTIATE(rocsparse_direction_column, int32_t, int32_t, _Float16);
INSTANTIATE(rocsparse_direction_column, int32_t, int32_t, float);
INSTANTIATE(rocsparse_direction_column, int32_t, int32_t, double);
INSTANTIATE(rocsparse_direction_column, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_column, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(rocsparse_direction_column, int64_t, int32_t, _Float16);
INSTANTIATE(rocsparse_direction_column, int64_t, int32_t, float);
INSTANTIATE(rocsparse_direction_column, int64_t, int32_t, double);
INSTANTIATE(rocsparse_direction_column, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_column, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(rocsparse_direction_column, int64_t, int64_t, _Float16);
INSTANTIATE(rocsparse_direction_column, int64_t, int64_t, float);
INSTANTIATE(rocsparse_direction_column, int64_t, int64_t, double);
INSTANTIATE(rocsparse_direction_column, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(rocsparse_direction_column, int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

extern "C" {

//
// Check if the macro CAPI_IMPL already exists.
//
#ifdef CAPI_IMPL
#error macro CAPI_IMPL is already defined.
#endif

//
// Definition of the C-implementation.
//
#define CAPI_IMPL(name_, type_)                                                               \
    rocsparse_status name_(rocsparse_handle          handle,                                  \
                           rocsparse_int             m,                                       \
                           rocsparse_int             n,                                       \
                           const rocsparse_mat_descr descr,                                   \
                           const type_*              A,                                       \
                           rocsparse_int             ld,                                      \
                           const rocsparse_int*      nnz_per_Columns,                         \
                           type_*                    csc_val,                                 \
                           rocsparse_int*            csc_col_ptr,                             \
                           rocsparse_int*            csc_row_ind)                             \
    {                                                                                         \
        try                                                                                   \
        {                                                                                     \
            ROCSPARSE_ROUTINE_TRACE;                                                          \
            RETURN_IF_ROCSPARSE_ERROR(                                                        \
                rocsparse::dense2csx_impl<rocsparse_direction_column>(handle,                 \
                                                                      rocsparse_order_column, \
                                                                      m,                      \
                                                                      n,                      \
                                                                      descr,                  \
                                                                      A,                      \
                                                                      ld,                     \
                                                                      nnz_per_Columns,        \
                                                                      csc_val,                \
                                                                      csc_col_ptr,            \
                                                                      csc_row_ind));          \
            return rocsparse_status_success;                                                  \
        }                                                                                     \
        catch(...)                                                                            \
        {                                                                                     \
            RETURN_ROCSPARSE_EXCEPTION();                                                     \
        }                                                                                     \
    }

//
// C-implementations.
//
CAPI_IMPL(rocsparse_sdense2csc, float);
CAPI_IMPL(rocsparse_ddense2csc, double);
CAPI_IMPL(rocsparse_cdense2csc, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zdense2csc, rocsparse_double_complex);

//
// Undefine the macro CAPI_IMPL.
//
#undef CAPI_IMPL
}
