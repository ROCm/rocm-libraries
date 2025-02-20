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
#include "internal/level2/rocsparse_bsrmv.h"
#include "rocsparse_bsrmv.hpp"
#include "utility.h"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

// rocsparse_xbsrmv_ex_analysis
#define C_IMPL(NAME, TYPE)                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,            \
                                     rocsparse_direction       dir,               \
                                     rocsparse_operation       trans,             \
                                     rocsparse_int             mb,                \
                                     rocsparse_int             nb,                \
                                     rocsparse_int             nnzb,              \
                                     const rocsparse_mat_descr descr,             \
                                     const TYPE*               bsr_val,           \
                                     const rocsparse_int*      bsr_row_ptr,       \
                                     const rocsparse_int*      bsr_col_ind,       \
                                     rocsparse_int             block_dim,         \
                                     rocsparse_mat_info        info)              \
    try                                                                           \
    {                                                                             \
        ROCSPARSE_ROUTINE_TRACE;                                                  \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_analysis_template(handle,      \
                                                                     dir,         \
                                                                     trans,       \
                                                                     mb,          \
                                                                     nb,          \
                                                                     nnzb,        \
                                                                     descr,       \
                                                                     bsr_val,     \
                                                                     bsr_row_ptr, \
                                                                     bsr_col_ind, \
                                                                     block_dim,   \
                                                                     info));      \
        return rocsparse_status_success;                                          \
    }                                                                             \
    catch(...)                                                                    \
    {                                                                             \
        RETURN_ROCSPARSE_EXCEPTION();                                             \
    }

C_IMPL(rocsparse_sbsrmv_ex_analysis, float);
C_IMPL(rocsparse_dbsrmv_ex_analysis, double);
C_IMPL(rocsparse_cbsrmv_ex_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv_ex_analysis, rocsparse_double_complex);

#undef C_IMPL

// rocsparse_xbsrmv_ex
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nb,          \
                                     rocsparse_int             nnzb,        \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info,        \
                                     const TYPE*               x,           \
                                     const TYPE*               beta,        \
                                     TYPE*                     y)           \
    try                                                                     \
    {                                                                       \
        ROCSPARSE_ROUTINE_TRACE;                                            \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_template(handle,         \
                                                            dir,            \
                                                            trans,          \
                                                            mb,             \
                                                            nb,             \
                                                            nnzb,           \
                                                            alpha,          \
                                                            descr,          \
                                                            bsr_val,        \
                                                            bsr_row_ptr,    \
                                                            bsr_col_ind,    \
                                                            block_dim,      \
                                                            info,           \
                                                            x,              \
                                                            beta,           \
                                                            y));            \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_sbsrmv_ex, float);
C_IMPL(rocsparse_dbsrmv_ex, double);
C_IMPL(rocsparse_cbsrmv_ex, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrmv_ex, rocsparse_double_complex);

#undef C_IMPL

extern "C" rocsparse_status rocsparse_bsrmv_ex_clear(rocsparse_handle   handle,
                                                     rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsrmv_clear(handle, info));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
