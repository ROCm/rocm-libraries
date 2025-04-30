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

#include "internal/level2/rocsparse_bsrsv.h"
#include "rocsparse_bsrsv.hpp"

#include "control.h"
#include "utility.h"

#include "../level2/rocsparse_csrsv.hpp"

template <typename T>
rocsparse_status rocsparse::bsrsv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_operation       trans,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nnzb,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  bsr_val,
                                                    const rocsparse_int*      bsr_row_ptr,
                                                    const rocsparse_int*      bsr_col_ind,
                                                    rocsparse_int             block_dim,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_analysis_policy analysis,
                                                    rocsparse_solve_policy    solve,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    ROCSPARSE_CHECKARG_POINTER(5, descr);
    ROCSPARSE_CHECKARG_POINTER(10, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbsrsv_analysis"),
                         dir,
                         trans,
                         mb,
                         nnzb,
                         (const void*&)descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         block_dim,
                         (const void*&)info,
                         solve,
                         analysis,
                         (const void*&)temp_buffer);

    // Check direction
    ROCSPARSE_CHECKARG_ENUM(1, dir);

    // Check operation
    ROCSPARSE_CHECKARG_ENUM(2, trans);

    // Check analysis
    ROCSPARSE_CHECKARG_ENUM(11, analysis);

    // Check solve
    ROCSPARSE_CHECKARG_ENUM(12, solve);

    // Check operation type
    ROCSPARSE_CHECKARG(
        2,
        trans,
        (trans != rocsparse_operation_none && trans != rocsparse_operation_transpose),
        rocsparse_status_not_implemented);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        5, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check matrix sorting mode

    ROCSPARSE_CHECKARG(5,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(3, mb);
    ROCSPARSE_CHECKARG_SIZE(4, nnzb);
    ROCSPARSE_CHECKARG_SIZE(9, block_dim);
    ROCSPARSE_CHECKARG(9, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    // Quick return if possible
    if(mb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(7, bsr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(13, temp_buffer);

    ROCSPARSE_CHECKARG_ARRAY(6, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_col_ind);

    // Switch between lower and upper triangular analysis
    if(descr->fill_mode == rocsparse_fill_mode_upper)
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed upper part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If bsrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->bsrsv_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->bsrsvt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }

            //            // Check for other upper analysis meta data that could be used
            //            if(trans == rocsparse_operation_none && info->bsrsm_upper_info != nullptr)
            //            {
            //                // bsrsm meta data
            //                info->bsrsv_upper_info = info->bsrsm_upper_info;
            //                return rocsparse_status_success;
            //            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear bsrsv
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info((trans == rocsparse_operation_none)
                                                                  ? info->bsrsv_upper_info
                                                                  : info->bsrsvt_upper_info));

        // Create bsrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_trm_info((trans == rocsparse_operation_none)
                                                                 ? &info->bsrsv_upper_info
                                                                 : &info->bsrsvt_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::trm_analysis(
            handle,
            trans,
            mb,
            nnzb,
            descr,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind,
            (trans == rocsparse_operation_none) ? info->bsrsv_upper_info : info->bsrsvt_upper_info,
            (rocsparse_int**)&info->zero_pivot,
            temp_buffer));
    }
    else
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed lower part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If bsrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->bsrsv_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->bsrsvt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data that could be used
            if(trans == rocsparse_operation_none && info->bsric0_info != nullptr)
            {
                // bsric0 meta data
                info->bsrsv_lower_info = info->bsric0_info;
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_none && info->bsrilu0_info != nullptr)
            {
                // bsrilu0 meta data
                info->bsrsv_lower_info = info->bsrilu0_info;
                return rocsparse_status_success;
            }
            // else if(trans == rocsparse_operation_none && info->bsrsm_lower_info != nullptr)
            // {
            //     // bsrsm meta data
            //     info->bsrsv_lower_info = info->bsrsm_lower_info;
            //     return rocsparse_status_success;
            // }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear bsrsv
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info((trans == rocsparse_operation_none)
                                                                  ? info->bsrsv_lower_info
                                                                  : info->bsrsvt_lower_info));

        // Create bsrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_trm_info((trans == rocsparse_operation_none)
                                                                 ? &info->bsrsv_lower_info
                                                                 : &info->bsrsvt_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::trm_analysis(
            handle,
            trans,
            mb,
            nnzb,
            descr,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind,
            (trans == rocsparse_operation_none) ? info->bsrsv_lower_info : info->bsrsvt_lower_info,
            (rocsparse_int**)&info->zero_pivot,
            temp_buffer));
    }

    return rocsparse_status_success;
}

// bsrsv_analysis
#define C_IMPL(NAME, TYPE)                                                          \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,              \
                                     rocsparse_direction       dir,                 \
                                     rocsparse_operation       trans,               \
                                     rocsparse_int             mb,                  \
                                     rocsparse_int             nnzb,                \
                                     const rocsparse_mat_descr descr,               \
                                     const TYPE*               bsr_val,             \
                                     const rocsparse_int*      bsr_row_ptr,         \
                                     const rocsparse_int*      bsr_col_ind,         \
                                     rocsparse_int             block_dim,           \
                                     rocsparse_mat_info        info,                \
                                     rocsparse_analysis_policy analysis,            \
                                     rocsparse_solve_policy    solve,               \
                                     void*                     temp_buffer)         \
    try                                                                             \
    {                                                                               \
        ROCSPARSE_ROUTINE_TRACE;                                                    \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsv_analysis_template(handle,        \
                                                                     dir,           \
                                                                     trans,         \
                                                                     mb,            \
                                                                     nnzb,          \
                                                                     descr,         \
                                                                     bsr_val,       \
                                                                     bsr_row_ptr,   \
                                                                     bsr_col_ind,   \
                                                                     block_dim,     \
                                                                     info,          \
                                                                     analysis,      \
                                                                     solve,         \
                                                                     temp_buffer)); \
        return rocsparse_status_success;                                            \
    }                                                                               \
    catch(...)                                                                      \
    {                                                                               \
        RETURN_ROCSPARSE_EXCEPTION();                                               \
    }

C_IMPL(rocsparse_sbsrsv_analysis, float);
C_IMPL(rocsparse_dbsrsv_analysis, double);
C_IMPL(rocsparse_cbsrsv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsv_analysis, rocsparse_double_complex);
#undef C_IMPL
