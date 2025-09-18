/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_bsrgemm.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_bsrgemm.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_csrgemm_mult.hpp"
#include "rocsparse_csrgemm_multadd.hpp"
#include "rocsparse_csrgemm_scal.hpp"

namespace rocsparse
{
    template <typename I, typename J>
    rocsparse_status bsrgemm_nnzb_checkarg(rocsparse_handle          handle, //0
                                           rocsparse_direction       dir, //1
                                           rocsparse_operation       trans_A, //2
                                           rocsparse_operation       trans_B, //3
                                           J                         mb, //4
                                           J                         nb, //5
                                           J                         kb, //6
                                           J                         block_dim, //7
                                           const rocsparse_mat_descr descr_A, //8
                                           I                         nnzb_A, //9
                                           const I*                  bsr_row_ptr_A, //10
                                           const J*                  bsr_col_ind_A, //11
                                           const rocsparse_mat_descr descr_B, //12
                                           I                         nnzb_B, //13
                                           const I*                  bsr_row_ptr_B, //14
                                           const J*                  bsr_col_ind_B, //15
                                           const rocsparse_mat_descr descr_D, //16
                                           I                         nnzb_D, //17
                                           const I*                  bsr_row_ptr_D, //18
                                           const J*                  bsr_col_ind_D, //19
                                           const rocsparse_mat_descr descr_C, //20
                                           I*                        bsr_row_ptr_C, //21
                                           I*                        nnzb_C, //22
                                           const rocsparse_mat_info  info_C, //23
                                           void*                     temp_buffer) //24
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_POINTER(23, info_C);
        ROCSPARSE_CHECKARG(
            23, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;

        if(mul == true && add == true)
        {
            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, dir);

            ROCSPARSE_CHECKARG_ENUM(2, trans_A);
            ROCSPARSE_CHECKARG_ENUM(3, trans_B);
            ROCSPARSE_CHECKARG_SIZE(4, mb);
            ROCSPARSE_CHECKARG_SIZE(5, nb);
            ROCSPARSE_CHECKARG_SIZE(6, kb);
            ROCSPARSE_CHECKARG_SIZE(7, block_dim);
            ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG_SIZE(9, nnzb_A);
            ROCSPARSE_CHECKARG_SIZE(13, nnzb_B);
            ROCSPARSE_CHECKARG_SIZE(17, nnzb_D);

            ROCSPARSE_CHECKARG_POINTER(8, descr_A);
            ROCSPARSE_CHECKARG_POINTER(12, descr_B);
            ROCSPARSE_CHECKARG_POINTER(16, descr_D);
            ROCSPARSE_CHECKARG_POINTER(20, descr_C);
            ROCSPARSE_CHECKARG_POINTER(22, nnzb_C);

            ROCSPARSE_CHECKARG_ARRAY(10, mb, bsr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(14, kb, bsr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(18, mb, bsr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(21, mb, bsr_row_ptr_C);

            ROCSPARSE_CHECKARG_ARRAY(11, nnzb_A, bsr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(15, nnzb_B, bsr_col_ind_B);
            ROCSPARSE_CHECKARG_ARRAY(19, nnzb_D, bsr_col_ind_D);

            ROCSPARSE_CHECKARG(8,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(12,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(16,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(20,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status
                = rocsparse::csrgemm_multadd_nnz_quickreturn(handle,
                                                             trans_A,
                                                             trans_B,
                                                             mb,
                                                             nb,
                                                             kb,
                                                             descr_A,
                                                             nnzb_A,
                                                             bsr_row_ptr_A,
                                                             bsr_col_ind_A,
                                                             descr_B,
                                                             nnzb_B,
                                                             bsr_row_ptr_B,
                                                             bsr_col_ind_B,
                                                             descr_D,
                                                             nnzb_D,
                                                             bsr_row_ptr_D,
                                                             bsr_col_ind_D,
                                                             descr_C,
                                                             bsr_row_ptr_C,
                                                             nnzb_C,
                                                             info_C,
                                                             temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(2,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(3,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            if(info_C->csrgemm_info->is_initialized)
            {
                ROCSPARSE_CHECKARG_ARRAY(23, info_C->csrgemm_info->buffer_size, temp_buffer);
            }
            else
            {
                ROCSPARSE_CHECKARG_POINTER(24, temp_buffer);
            }
            return rocsparse_status_continue;
        }
        else if(mul == true && add == false)
        {

            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, dir);
            ROCSPARSE_CHECKARG_ENUM(2, trans_A);
            ROCSPARSE_CHECKARG_ENUM(3, trans_B);
            ROCSPARSE_CHECKARG_SIZE(4, mb);
            ROCSPARSE_CHECKARG_SIZE(5, nb);
            ROCSPARSE_CHECKARG_SIZE(6, kb);
            ROCSPARSE_CHECKARG_SIZE(7, block_dim);
            ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG_SIZE(9, nnzb_A);
            ROCSPARSE_CHECKARG_SIZE(13, nnzb_B);
            ROCSPARSE_CHECKARG_ARRAY(10, mb, bsr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(14, kb, bsr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(21, mb, bsr_row_ptr_C);

            ROCSPARSE_CHECKARG_POINTER(8, descr_A);
            ROCSPARSE_CHECKARG_POINTER(12, descr_B);
            ROCSPARSE_CHECKARG_POINTER(20, descr_C);
            ROCSPARSE_CHECKARG_POINTER(22, nnzb_C);

            ROCSPARSE_CHECKARG_ARRAY(11, nnzb_A, bsr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(15, nnzb_B, bsr_col_ind_B);
            ROCSPARSE_CHECKARG(8,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(12,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(20,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::csrgemm_mult_nnz_quickreturn(handle,
                                                                                    trans_A,
                                                                                    trans_B,
                                                                                    mb,
                                                                                    nb,
                                                                                    kb,
                                                                                    descr_A,
                                                                                    nnzb_A,
                                                                                    bsr_row_ptr_A,
                                                                                    bsr_col_ind_A,
                                                                                    descr_B,
                                                                                    nnzb_B,
                                                                                    bsr_row_ptr_B,
                                                                                    bsr_col_ind_B,
                                                                                    descr_C,
                                                                                    bsr_row_ptr_C,
                                                                                    nnzb_C,
                                                                                    info_C,
                                                                                    temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(2,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(3,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG_ARRAY(24, info_C->csrgemm_info->buffer_size, temp_buffer);
            return rocsparse_status_continue;
        }
        else if(mul == false && add == true)
        {
            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, dir);
            ROCSPARSE_CHECKARG_ENUM(2, trans_A);
            ROCSPARSE_CHECKARG_ENUM(3, trans_B);
            ROCSPARSE_CHECKARG_SIZE(4, mb);
            ROCSPARSE_CHECKARG_SIZE(5, nb);
            ROCSPARSE_CHECKARG_SIZE(6, kb);
            ROCSPARSE_CHECKARG_SIZE(7, block_dim);
            ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG_SIZE(17, nnzb_D);
            ROCSPARSE_CHECKARG_POINTER(16, descr_D);
            ROCSPARSE_CHECKARG_POINTER(20, descr_C);
            ROCSPARSE_CHECKARG_POINTER(22, nnzb_C);
            ROCSPARSE_CHECKARG_ARRAY(18, mb, bsr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(21, mb, bsr_row_ptr_C);
            ROCSPARSE_CHECKARG_ARRAY(19, nnzb_D, bsr_col_ind_D);
            ROCSPARSE_CHECKARG(20,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(16,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::csrgemm_scal_nnz_quickreturn(handle,
                                                                                    mb,
                                                                                    nb,
                                                                                    descr_D,
                                                                                    nnzb_D,
                                                                                    bsr_row_ptr_D,
                                                                                    bsr_col_ind_D,
                                                                                    descr_C,
                                                                                    bsr_row_ptr_C,
                                                                                    nnzb_C,
                                                                                    info_C,
                                                                                    temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG_ARRAY(23, info_C->csrgemm_info->buffer_size, temp_buffer);
            return rocsparse_status_continue;
        }
        else
        {
            rocsparse_host_assert(mul == false && add == false, "Wrong logical dispatch.");
            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, dir);
            ROCSPARSE_CHECKARG_ENUM(2, trans_A);
            ROCSPARSE_CHECKARG_ENUM(3, trans_B);
            ROCSPARSE_CHECKARG_SIZE(4, mb);
            ROCSPARSE_CHECKARG_SIZE(5, nb);
            ROCSPARSE_CHECKARG_SIZE(6, kb);
            ROCSPARSE_CHECKARG_SIZE(7, block_dim);
            ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG_POINTER(23, info_C);
            ROCSPARSE_CHECKARG(
                23, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);
            ROCSPARSE_CHECKARG_ARRAY(21, mb, bsr_row_ptr_C);
            ROCSPARSE_CHECKARG_POINTER(22, nnzb_C);

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(nnzb_C, 0, sizeof(I), handle->stream));
            }
            else
            {
                *nnzb_C = 0;
            }

            if(bsr_row_ptr_C != nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                    handle, mb + 1, static_cast<I>(descr_C->base), bsr_row_ptr_C));
            }

            return rocsparse_status_success;
        }
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::bsrgemm_nnzb_template(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  J                         mb,
                                                  J                         nb,
                                                  J                         kb,
                                                  J                         block_dim,
                                                  const rocsparse_mat_descr descr_A,
                                                  I                         nnzb_A,
                                                  const I*                  bsr_row_ptr_A,
                                                  const J*                  bsr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  I                         nnzb_B,
                                                  const I*                  bsr_row_ptr_B,
                                                  const J*                  bsr_col_ind_B,
                                                  const rocsparse_mat_descr descr_D,
                                                  I                         nnzb_D,
                                                  const I*                  bsr_row_ptr_D,
                                                  const J*                  bsr_col_ind_D,
                                                  const rocsparse_mat_descr descr_C,
                                                  I*                        bsr_row_ptr_C,
                                                  I*                        nnzb_C,
                                                  const rocsparse_mat_info  info_C,
                                                  void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_nnz_template(handle,
                                                              trans_A,
                                                              trans_B,
                                                              mb,
                                                              nb,
                                                              kb,
                                                              descr_A,
                                                              nnzb_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              descr_B,
                                                              nnzb_B,
                                                              bsr_row_ptr_B,
                                                              bsr_col_ind_B,
                                                              descr_D,
                                                              nnzb_D,
                                                              bsr_row_ptr_D,
                                                              bsr_col_ind_D,
                                                              descr_C,
                                                              bsr_row_ptr_C,
                                                              nnzb_C,
                                                              info_C,
                                                              temp_buffer));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I, typename J>
    static rocsparse_status bsrgemm_nnzb_core(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              J                         mb,
                                              J                         nb,
                                              J                         kb,
                                              J                         block_dim,
                                              const rocsparse_mat_descr descr_A,
                                              I                         nnzb_A,
                                              const I*                  bsr_row_ptr_A,
                                              const J*                  bsr_col_ind_A,
                                              const rocsparse_mat_descr descr_B,
                                              I                         nnzb_B,
                                              const I*                  bsr_row_ptr_B,
                                              const J*                  bsr_col_ind_B,
                                              const rocsparse_mat_descr descr_D,
                                              I                         nnzb_D,
                                              const I*                  bsr_row_ptr_D,
                                              const J*                  bsr_col_ind_D,
                                              const rocsparse_mat_descr descr_C,
                                              I*                        bsr_row_ptr_C,
                                              I*                        nnzb_C,
                                              const rocsparse_mat_info  info_C,
                                              void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_nnz_core(handle,
                                                              trans_A,
                                                              trans_B,
                                                              mb,
                                                              nb,
                                                              kb,
                                                              descr_A,
                                                              nnzb_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              descr_B,
                                                              nnzb_B,
                                                              bsr_row_ptr_B,
                                                              bsr_col_ind_B,
                                                              descr_D,
                                                              nnzb_D,
                                                              bsr_row_ptr_D,
                                                              bsr_col_ind_D,
                                                              descr_C,
                                                              bsr_row_ptr_C,
                                                              nnzb_C,
                                                              info_C,
                                                              temp_buffer));
        return rocsparse_status_success;
    }

    template <typename... P>
    static inline rocsparse_status bsrgemm_nnzb_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_bsrgemm_nnzb", p...);

        const rocsparse_status status = rocsparse::bsrgemm_nnzb_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_nnzb_core(p...));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE)                                             \
    template rocsparse_status rocsparse::bsrgemm_nnzb_template<ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                     \
        rocsparse_direction       dir,                                        \
        rocsparse_operation       trans_A,                                    \
        rocsparse_operation       trans_B,                                    \
        JTYPE                     mb,                                         \
        JTYPE                     nb,                                         \
        JTYPE                     kb,                                         \
        JTYPE                     block_dim,                                  \
        const rocsparse_mat_descr descr_A,                                    \
        ITYPE                     nnzb_A,                                     \
        const ITYPE*              bsr_row_ptr_A,                              \
        const JTYPE*              bsr_col_ind_A,                              \
        const rocsparse_mat_descr descr_B,                                    \
        ITYPE                     nnzb_B,                                     \
        const ITYPE*              bsr_row_ptr_B,                              \
        const JTYPE*              bsr_col_ind_B,                              \
        const rocsparse_mat_descr descr_D,                                    \
        ITYPE                     nnzb_D,                                     \
        const ITYPE*              bsr_row_ptr_D,                              \
        const JTYPE*              bsr_col_ind_D,                              \
        const rocsparse_mat_descr descr_C,                                    \
        ITYPE*                    bsr_row_ptr_C,                              \
        ITYPE*                    nnzb_C,                                     \
        const rocsparse_mat_info  info_C,                                     \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

//
// rocsparse_xbsrgemm_nnz
//
extern "C" rocsparse_status rocsparse_bsrgemm_nnzb(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             kb,
                                                   rocsparse_int             block_dim,
                                                   const rocsparse_mat_descr descr_A,
                                                   rocsparse_int             nnzb_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   rocsparse_int             nnzb_B,
                                                   const rocsparse_int*      bsr_row_ptr_B,
                                                   const rocsparse_int*      bsr_col_ind_B,
                                                   const rocsparse_mat_descr descr_D,
                                                   rocsparse_int             nnzb_D,
                                                   const rocsparse_int*      bsr_row_ptr_D,
                                                   const rocsparse_int*      bsr_col_ind_D,
                                                   const rocsparse_mat_descr descr_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            nnzb_C,
                                                   const rocsparse_mat_info  info_C,
                                                   void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_nnzb_impl(handle,
                                                           dir,
                                                           trans_A,
                                                           trans_B,
                                                           mb,
                                                           nb,
                                                           kb,
                                                           block_dim,
                                                           descr_A,
                                                           nnzb_A,
                                                           bsr_row_ptr_A,
                                                           bsr_col_ind_A,
                                                           descr_B,
                                                           nnzb_B,
                                                           bsr_row_ptr_B,
                                                           bsr_col_ind_B,
                                                           descr_D,
                                                           nnzb_D,
                                                           bsr_row_ptr_D,
                                                           bsr_col_ind_D,
                                                           descr_C,
                                                           bsr_row_ptr_C,
                                                           nnzb_C,
                                                           info_C,
                                                           temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
