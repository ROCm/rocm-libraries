/*! \file */
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

#include "internal/level2/rocsparse_bsrsv.h"
#include "rocsparse_bsrsv.hpp"

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "../level2/rocsparse_csrsv.hpp"

template <typename T, typename I, typename J>
rocsparse_status rocsparse::bsrsv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_operation       trans,
                                                    J                         mb,
                                                    I                         nnzb,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  bsr_val,
                                                    const I*                  bsr_row_ptr,
                                                    const J*                  bsr_col_ind,
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

    if(analysis == rocsparse_analysis_policy_reuse)
    {
        auto                     trm   = info->get_bsrsv_info(trans, descr->fill_mode);
        rocsparse::pivot_info_t* donor = nullptr;

        // donor is the container that owns a borrowed analysis. Its zero pivot holds
        // the structural zeros that analysis found, so the solve must report them.
        if((descr->fill_mode == rocsparse_fill_mode_lower) && (trans == rocsparse_operation_none))
        {
            if(trm == nullptr)
            {
                trm   = info->get_bsric0_info(trans, descr->fill_mode);
                donor = (trm != nullptr) ? info->get_bsric0_info() : nullptr;
            }
            if(trm == nullptr)
            {
                trm   = info->get_bsrilu0_info(trans, descr->fill_mode);
                donor = (trm != nullptr) ? info->get_bsrilu0_info() : nullptr;
            }
            if(trm == nullptr)
            {
                trm   = info->get_bsrsm_info(trans, descr->fill_mode);
                donor = (trm != nullptr) ? info->get_bsrsm_info() : nullptr;
            }
        }

        // Reuse only an analysis built with the same index types. A mismatched one is
        // rebuilt below, so the solve never reads a row map of another width.
        if(trm != nullptr
           && trm->has_indextypes(rocsparse::get_indextype<I>(), rocsparse::get_indextype<J>()))
        {
            // Copy or create the pivot before storing the alias. copy_pivot_info_async
            // can fail after it has already reset the destination to "no pivot". If the
            // alias were stored first, a retry would find this slot, leave donor null,
            // and keep that reset pivot, so a structural zero from the donor would be lost.
            auto bsrsv_info = info->get_bsrsv_info();
            if(donor != nullptr && donor->get_position() != nullptr)
            {
                // Copy the donor's pivot, with the donor's index type.
                RETURN_IF_ROCSPARSE_ERROR(bsrsv_info->drop_zero_pivot_if_indextype_async(
                    donor->get_indextype(), handle->stream));
                RETURN_IF_ROCSPARSE_ERROR(bsrsv_info->copy_pivot_info_async(donor, handle->stream));
            }
            else
            {
                // Reuse of this slot's own analysis keeps the pivot, including a
                // structural zero. The pivot is shared by every fill mode, so an
                // analysis of another width may have left a different index type.
                // Drop that first: create_position_async does not retype a buffer
                // that already exists.
                RETURN_IF_ROCSPARSE_ERROR(bsrsv_info->drop_zero_pivot_if_indextype_async(
                    rocsparse::get_indextype<J>(), handle->stream));
                if(bsrsv_info->get_position() == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(bsrsv_info->create_zero_pivot_async(
                        rocsparse::get_indextype<J>(), handle->stream));
                }
            }
            info->set_bsrsv_info(trans, descr->fill_mode, trm);
            return rocsparse_status_success;
        }
    }

    auto bsrsv_info = info->get_bsrsv_info();

    // Reuse stores a raw alias of a bsric0, bsrilu0 or bsrsm analysis in this
    // slot. recreate deletes whatever the slot holds, so drop the alias first
    // and leave the analysis with the routine that created it. An analysis
    // this routine owns stays in the slot and is replaced.
    info->detach_borrowed_bsrsv(trans, descr->fill_mode);

    // The new analysis writes its pivot with index type J. Drop a pivot of
    // another type first, since recreate does not retype it in place.
    RETURN_IF_ROCSPARSE_ERROR(bsrsv_info->drop_zero_pivot_if_indextype_async(
        rocsparse::get_indextype<J>(), handle->stream));

    RETURN_IF_ROCSPARSE_ERROR(bsrsv_info->recreate(
        handle, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, temp_buffer));

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

#define ROCSPARSE_BSRSV_ANALYSIS_INST(T, I, J)                             \
    template rocsparse_status rocsparse::bsrsv_analysis_template<T, I, J>( \
        rocsparse_handle,                                                  \
        rocsparse_direction,                                               \
        rocsparse_operation,                                               \
        J,                                                                 \
        I,                                                                 \
        const rocsparse_mat_descr,                                         \
        const T*,                                                          \
        const I*,                                                          \
        const J*,                                                          \
        rocsparse_int,                                                     \
        rocsparse_mat_info,                                                \
        rocsparse_analysis_policy,                                         \
        rocsparse_solve_policy,                                            \
        void*)

#define ROCSPARSE_BSRSV_ANALYSIS_INST_T(I, J)                     \
    ROCSPARSE_BSRSV_ANALYSIS_INST(float, I, J);                   \
    ROCSPARSE_BSRSV_ANALYSIS_INST(double, I, J);                  \
    ROCSPARSE_BSRSV_ANALYSIS_INST(rocsparse_float_complex, I, J); \
    ROCSPARSE_BSRSV_ANALYSIS_INST(rocsparse_double_complex, I, J)

ROCSPARSE_BSRSV_ANALYSIS_INST_T(int32_t, int32_t);
ROCSPARSE_BSRSV_ANALYSIS_INST_T(int64_t, int32_t);
ROCSPARSE_BSRSV_ANALYSIS_INST_T(int64_t, int64_t);

#undef ROCSPARSE_BSRSV_ANALYSIS_INST_T
#undef ROCSPARSE_BSRSV_ANALYSIS_INST

C_IMPL(rocsparse_sbsrsv_analysis, float);
C_IMPL(rocsparse_dbsrsv_analysis, double);
C_IMPL(rocsparse_cbsrsv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsv_analysis, rocsparse_double_complex);
#undef C_IMPL
