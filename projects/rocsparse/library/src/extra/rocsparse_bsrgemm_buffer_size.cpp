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
#include "internal/extra/rocsparse_bsrgemm.h"
#include "rocsparse_bsrgemm.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_csrgemm_mult.hpp"
#include "rocsparse_csrgemm_multadd.hpp"
#include "rocsparse_csrgemm_scal.hpp"

#include "rocsparse_bsrgemm_mult.hpp"
#include "rocsparse_bsrgemm_multadd.hpp"
#include "rocsparse_bsrgemm_scal.hpp"

namespace rocsparse
{
    static rocsparse_status bsrgemm_buffer_size_checkarg(rocsparse_handle          handle, //0
                                                         rocsparse_direction       dir, //1
                                                         rocsparse_operation       trans_A, //2
                                                         rocsparse_operation       trans_B, //3
                                                         int64_t                   mb, //4
                                                         int64_t                   nb, //5
                                                         int64_t                   kb, //6
                                                         int64_t                   block_dim, //7
                                                         const void*               alpha, //8
                                                         const rocsparse_mat_descr descr_A, //9
                                                         int64_t                   nnzb_A, //10
                                                         const void* bsr_row_ptr_A, //11
                                                         const void* bsr_col_ind_A, //12
                                                         const rocsparse_mat_descr descr_B, //13
                                                         int64_t                   nnzb_B, //14
                                                         const void* bsr_row_ptr_B, //15
                                                         const void* bsr_col_ind_B, //16
                                                         const void* beta, //17
                                                         const rocsparse_mat_descr descr_D, //18
                                                         int64_t                   nnzb_D, //19
                                                         const void*        bsr_row_ptr_D, //20
                                                         const void*        bsr_col_ind_D, //21
                                                         rocsparse_mat_info info_C, //22
                                                         size_t*            buffer_size) //23
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_POINTER(22, info_C);
        ROCSPARSE_CHECKARG(
            22, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_internal_error);
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

            ROCSPARSE_CHECKARG_SIZE(10, nnzb_A);
            ROCSPARSE_CHECKARG_SIZE(14, nnzb_B);
            ROCSPARSE_CHECKARG_SIZE(19, nnzb_D);

            ROCSPARSE_CHECKARG_POINTER(9, descr_A);
            ROCSPARSE_CHECKARG_POINTER(13, descr_B);
            ROCSPARSE_CHECKARG_POINTER(18, descr_D);
            ROCSPARSE_CHECKARG_POINTER(23, buffer_size);
            ROCSPARSE_CHECKARG_POINTER(8, alpha);
            ROCSPARSE_CHECKARG_POINTER(17, beta);

            ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(15, kb, bsr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(20, mb, bsr_row_ptr_D);

            ROCSPARSE_CHECKARG_ARRAY(12, nnzb_A, bsr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(16, nnzb_B, bsr_col_ind_B);
            ROCSPARSE_CHECKARG_ARRAY(21, nnzb_D, bsr_col_ind_D);

            ROCSPARSE_CHECKARG(9,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(13,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(18,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(2,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(3,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);

            const rocsparse_status status
                = rocsparse::bsrgemm_multadd_buffer_size_quickreturn(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     mb,
                                                                     nb,
                                                                     kb,
                                                                     alpha,
                                                                     descr_A,
                                                                     nnzb_A,
                                                                     bsr_row_ptr_A,
                                                                     bsr_col_ind_A,
                                                                     descr_B,
                                                                     nnzb_B,
                                                                     bsr_row_ptr_B,
                                                                     bsr_col_ind_B,
                                                                     beta,
                                                                     descr_D,
                                                                     nnzb_D,
                                                                     bsr_row_ptr_D,
                                                                     bsr_col_ind_D,
                                                                     info_C,
                                                                     buffer_size);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
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

            ROCSPARSE_CHECKARG_SIZE(10, nnzb_A);
            ROCSPARSE_CHECKARG_SIZE(14, nnzb_B);

            ROCSPARSE_CHECKARG_POINTER(9, descr_A);
            ROCSPARSE_CHECKARG_POINTER(13, descr_B);
            ROCSPARSE_CHECKARG_POINTER(23, buffer_size);
            ROCSPARSE_CHECKARG_POINTER(8, alpha);

            ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(15, kb, bsr_row_ptr_B);

            ROCSPARSE_CHECKARG_ARRAY(12, nnzb_A, bsr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(16, nnzb_B, bsr_col_ind_B);

            ROCSPARSE_CHECKARG(9,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(13,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(2,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(3,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);

            const rocsparse_status status
                = rocsparse::bsrgemm_mult_buffer_size_quickreturn(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  mb,
                                                                  nb,
                                                                  kb,
                                                                  alpha,
                                                                  descr_A,
                                                                  nnzb_A,
                                                                  bsr_row_ptr_A,
                                                                  bsr_col_ind_A,
                                                                  descr_B,
                                                                  nnzb_B,
                                                                  bsr_row_ptr_B,
                                                                  bsr_col_ind_B,
                                                                  info_C,
                                                                  buffer_size);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

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
            ROCSPARSE_CHECKARG_SIZE(7, block_dim);
            ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG_POINTER(23, buffer_size);
            ROCSPARSE_CHECKARG_POINTER(17, beta);
            ROCSPARSE_CHECKARG_ARRAY(20, mb, bsr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(21, nnzb_D, bsr_col_ind_D);

            ROCSPARSE_CHECKARG_POINTER(18, descr_D);
            ROCSPARSE_CHECKARG(18,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG_SIZE(19, nnzb_D);
            ROCSPARSE_CHECKARG_ARRAY(20, mb, bsr_row_ptr_D);
            ROCSPARSE_CHECKARG_POINTER(23, buffer_size);

            const rocsparse_status status
                = rocsparse::bsrgemm_scal_buffer_size_quickreturn(handle,
                                                                  mb,
                                                                  nb,
                                                                  beta,
                                                                  descr_D,
                                                                  nnzb_D,
                                                                  bsr_row_ptr_D,
                                                                  bsr_col_ind_D,
                                                                  info_C,
                                                                  buffer_size);

            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            return rocsparse_status_continue;
        }
        else
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
            ROCSPARSE_CHECKARG_POINTER(23, buffer_size);

            return rocsparse_status_continue;
        }
    }

    static rocsparse_status bsrgemm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                            rocsparse_direction       dir,
                                                            rocsparse_operation       trans_A,
                                                            rocsparse_operation       trans_B,
                                                            int64_t                   mb,
                                                            int64_t                   nb,
                                                            int64_t                   kb,
                                                            int64_t                   block_dim,
                                                            const void*               alpha,
                                                            const rocsparse_mat_descr descr_A,
                                                            int64_t                   nnzb_A,
                                                            const void*               bsr_row_ptr_A,
                                                            const void*               bsr_col_ind_A,
                                                            const rocsparse_mat_descr descr_B,
                                                            int64_t                   nnzb_B,
                                                            const void*               bsr_row_ptr_B,
                                                            const void*               bsr_col_ind_B,
                                                            const void*               beta,
                                                            const rocsparse_mat_descr descr_D,
                                                            int64_t                   nnzb_D,
                                                            const void*               bsr_row_ptr_D,
                                                            const void*               bsr_col_ind_D,
                                                            rocsparse_mat_info        info_C,
                                                            size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;

        if(mul == true && add == true)
        {
            const rocsparse_status status
                = rocsparse::bsrgemm_multadd_buffer_size_quickreturn(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     mb,
                                                                     nb,
                                                                     kb,
                                                                     alpha,
                                                                     descr_A,
                                                                     nnzb_A,
                                                                     bsr_row_ptr_A,
                                                                     bsr_col_ind_A,
                                                                     descr_B,
                                                                     nnzb_B,
                                                                     bsr_row_ptr_B,
                                                                     bsr_col_ind_B,
                                                                     beta,
                                                                     descr_D,
                                                                     nnzb_D,
                                                                     bsr_row_ptr_D,
                                                                     bsr_col_ind_D,
                                                                     info_C,
                                                                     buffer_size);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            return rocsparse_status_continue;
        }
        else if(mul == true && add == false)
        {
            const rocsparse_status status
                = rocsparse::bsrgemm_mult_buffer_size_quickreturn(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  mb,
                                                                  nb,
                                                                  kb,
                                                                  alpha,
                                                                  descr_A,
                                                                  nnzb_A,
                                                                  bsr_row_ptr_A,
                                                                  bsr_col_ind_A,
                                                                  descr_B,
                                                                  nnzb_B,
                                                                  bsr_row_ptr_B,
                                                                  bsr_col_ind_B,
                                                                  info_C,
                                                                  buffer_size);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }
            return rocsparse_status_continue;
        }
        else if(mul == false && add == true)
        {
            const rocsparse_status status
                = rocsparse::bsrgemm_scal_buffer_size_quickreturn(handle,
                                                                  mb,
                                                                  nb,
                                                                  beta,
                                                                  descr_D,
                                                                  nnzb_D,
                                                                  bsr_row_ptr_D,
                                                                  bsr_col_ind_D,
                                                                  info_C,
                                                                  buffer_size);

            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            return rocsparse_status_continue;
        }
        else
        {
            return rocsparse_status_continue;
        }
    }

    template <typename I, typename J, typename T>
    static rocsparse_status bsrgemm_buffer_size_core(rocsparse_handle          handle,
                                                     rocsparse_direction       dir,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     J                         mb,
                                                     J                         nb,
                                                     J                         kb,
                                                     J                         block_dim,
                                                     const T*                  alpha,
                                                     const rocsparse_mat_descr descr_A,
                                                     I                         nnzb_A,
                                                     const I*                  bsr_row_ptr_A,
                                                     const J*                  bsr_col_ind_A,
                                                     const rocsparse_mat_descr descr_B,
                                                     I                         nnzb_B,
                                                     const I*                  bsr_row_ptr_B,
                                                     const J*                  bsr_col_ind_B,
                                                     const T*                  beta,
                                                     const rocsparse_mat_descr descr_D,
                                                     I                         nnzb_D,
                                                     const I*                  bsr_row_ptr_D,
                                                     const J*                  bsr_col_ind_D,
                                                     rocsparse_mat_info        info_C,
                                                     size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;
        if(mul == true && add == true)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_multadd_buffer_size_core(handle,
                                                                                  trans_A,
                                                                                  trans_B,
                                                                                  mb,
                                                                                  nb,
                                                                                  kb,
                                                                                  alpha,
                                                                                  descr_A,
                                                                                  nnzb_A,
                                                                                  bsr_row_ptr_A,
                                                                                  bsr_col_ind_A,
                                                                                  descr_B,
                                                                                  nnzb_B,
                                                                                  bsr_row_ptr_B,
                                                                                  bsr_col_ind_B,
                                                                                  beta,
                                                                                  descr_D,
                                                                                  nnzb_D,
                                                                                  bsr_row_ptr_D,
                                                                                  bsr_col_ind_D,
                                                                                  info_C,
                                                                                  buffer_size));
            buffer_size[0] += ((sizeof(I) * nnzb_A - 1) / 256 + 1) * 256;
        }
        else if(mul == true && add == false)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_mult_buffer_size_core(handle,
                                                                               trans_A,
                                                                               trans_B,
                                                                               mb,
                                                                               nb,
                                                                               kb,
                                                                               alpha,
                                                                               descr_A,
                                                                               nnzb_A,
                                                                               bsr_row_ptr_A,
                                                                               bsr_col_ind_A,
                                                                               descr_B,
                                                                               nnzb_B,
                                                                               bsr_row_ptr_B,
                                                                               bsr_col_ind_B,
                                                                               info_C,
                                                                               buffer_size));

            buffer_size[0] += ((sizeof(I) * nnzb_A - 1) / 256 + 1) * 256;
        }
        else if(mul == false && add == true)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_scal_buffer_size_core(handle,
                                                                               mb,
                                                                               nb,
                                                                               beta,
                                                                               descr_D,
                                                                               nnzb_D,
                                                                               bsr_row_ptr_D,
                                                                               bsr_col_ind_D,
                                                                               info_C,
                                                                               buffer_size));
        }
        else
        {
            if(((beta != nullptr) && (nnzb_D == 0))
               || (alpha != nullptr && (kb == 0 || nnzb_A == 0 || nnzb_B == 0)))
            {
                *buffer_size = 0;
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }
        }

        info_C->csrgemm_info->buffer_size    = buffer_size[0];
        info_C->csrgemm_info->is_initialized = true;
        return rocsparse_status_success;
    }

    static rocsparse_status reinit_bsrgemm_info(rocsparse_mat_info info,
                                                const void*        alpha,
                                                const void*        beta,
                                                const int64_t      kb,
                                                const int64_t      nnzb_A,
                                                const int64_t      nnzb_B,
                                                const int64_t      nnzb_D)
    {
        ROCSPARSE_ROUTINE_TRACE;

        info->csrgemm_info->mul = (alpha != nullptr);
        info->csrgemm_info->add = (beta != nullptr);

        if(info->csrgemm_info->add && (nnzb_D == 0))
        {
            info->csrgemm_info->add = false;
        }

        if(info->csrgemm_info->mul && (kb == 0 || nnzb_A == 0 || nnzb_B == 0))
        {
            info->csrgemm_info->mul = false;
        }

        return rocsparse_status_success;
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::bsrgemm_buffer_size_template(rocsparse_handle          handle,
                                                         rocsparse_direction       dir,
                                                         rocsparse_operation       trans_A,
                                                         rocsparse_operation       trans_B,
                                                         J                         mb,
                                                         J                         nb,
                                                         J                         kb,
                                                         J                         block_dim,
                                                         const T*                  alpha,
                                                         const rocsparse_mat_descr descr_A,
                                                         I                         nnzb_A,
                                                         const I*                  bsr_row_ptr_A,
                                                         const J*                  bsr_col_ind_A,
                                                         const rocsparse_mat_descr descr_B,
                                                         I                         nnzb_B,
                                                         const I*                  bsr_row_ptr_B,
                                                         const J*                  bsr_col_ind_B,
                                                         const T*                  beta,
                                                         const rocsparse_mat_descr descr_D,
                                                         I                         nnzb_D,
                                                         const I*                  bsr_row_ptr_D,
                                                         const J*                  bsr_col_ind_D,
                                                         rocsparse_mat_info        info_C,
                                                         size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(20, info_C);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrgemm_info(info_C->csrgemm_info));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_csrgemm_info(&info_C->csrgemm_info));
    RETURN_IF_ROCSPARSE_ERROR(reinit_bsrgemm_info(info_C, alpha, beta, kb, nnzb_A, nnzb_B, nnzb_D));

    const rocsparse_status status = rocsparse::bsrgemm_buffer_size_quickreturn(handle,
                                                                               dir,
                                                                               trans_A,
                                                                               trans_B,
                                                                               mb,
                                                                               nb,
                                                                               kb,
                                                                               block_dim,
                                                                               alpha,
                                                                               descr_A,
                                                                               nnzb_A,
                                                                               bsr_row_ptr_A,
                                                                               bsr_col_ind_A,
                                                                               descr_B,
                                                                               nnzb_B,
                                                                               bsr_row_ptr_B,
                                                                               bsr_col_ind_B,
                                                                               beta,
                                                                               descr_D,
                                                                               nnzb_D,
                                                                               bsr_row_ptr_D,
                                                                               bsr_col_ind_D,
                                                                               info_C,
                                                                               buffer_size);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_buffer_size_core(handle,
                                                                  dir,
                                                                  trans_A,
                                                                  trans_B,
                                                                  mb,
                                                                  nb,
                                                                  kb,
                                                                  block_dim,
                                                                  alpha,
                                                                  descr_A,
                                                                  nnzb_A,
                                                                  bsr_row_ptr_A,
                                                                  bsr_col_ind_A,
                                                                  descr_B,
                                                                  nnzb_B,
                                                                  bsr_row_ptr_B,
                                                                  bsr_col_ind_B,
                                                                  beta,
                                                                  descr_D,
                                                                  nnzb_D,
                                                                  bsr_row_ptr_D,
                                                                  bsr_col_ind_D,
                                                                  info_C,
                                                                  buffer_size));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    static rocsparse_status bsrgemm_buffer_size_impl(rocsparse_handle          handle,
                                                     rocsparse_direction       dir,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     J                         mb,
                                                     J                         nb,
                                                     J                         kb,
                                                     J                         block_dim,
                                                     const T*                  alpha,
                                                     const rocsparse_mat_descr descr_A,
                                                     I                         nnzb_A,
                                                     const I*                  bsr_row_ptr_A,
                                                     const J*                  bsr_col_ind_A,
                                                     const rocsparse_mat_descr descr_B,
                                                     I                         nnzb_B,
                                                     const I*                  bsr_row_ptr_B,
                                                     const J*                  bsr_col_ind_B,
                                                     const T*                  beta,
                                                     const rocsparse_mat_descr descr_D,
                                                     I                         nnzb_D,
                                                     const I*                  bsr_row_ptr_D,
                                                     const J*                  bsr_col_ind_D,
                                                     rocsparse_mat_info        info_C,
                                                     size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsrgemm_buffer_size"),
                             dir,
                             trans_A,
                             trans_B,
                             mb,
                             nb,
                             kb,
                             block_dim,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr_A,
                             nnzb_A,
                             (const void*&)bsr_row_ptr_A,
                             (const void*&)bsr_col_ind_A,
                             (const void*&)descr_B,
                             nnzb_B,
                             (const void*&)bsr_row_ptr_B,
                             (const void*&)bsr_col_ind_B,
                             LOG_TRACE_SCALAR_VALUE(handle, beta),
                             (const void*&)descr_D,
                             nnzb_D,
                             (const void*&)bsr_row_ptr_D,
                             (const void*&)bsr_col_ind_D,
                             (const void*&)info_C,
                             (const void*&)buffer_size);

        ROCSPARSE_CHECKARG_POINTER(22, info_C);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrgemm_info(info_C->csrgemm_info));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_csrgemm_info(&info_C->csrgemm_info));
        RETURN_IF_ROCSPARSE_ERROR(
            reinit_bsrgemm_info(info_C, alpha, beta, kb, nnzb_A, nnzb_B, nnzb_D));

        const rocsparse_status status = rocsparse::bsrgemm_buffer_size_checkarg(handle,
                                                                                dir,
                                                                                trans_A,
                                                                                trans_B,
                                                                                mb,
                                                                                nb,
                                                                                kb,
                                                                                block_dim,
                                                                                alpha,
                                                                                descr_A,
                                                                                nnzb_A,
                                                                                bsr_row_ptr_A,
                                                                                bsr_col_ind_A,
                                                                                descr_B,
                                                                                nnzb_B,
                                                                                bsr_row_ptr_B,
                                                                                bsr_col_ind_B,
                                                                                beta,
                                                                                descr_D,
                                                                                nnzb_D,
                                                                                bsr_row_ptr_D,
                                                                                bsr_col_ind_D,
                                                                                info_C,
                                                                                buffer_size);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_buffer_size_core(handle,
                                                                      dir,
                                                                      trans_A,
                                                                      trans_B,
                                                                      mb,
                                                                      nb,
                                                                      kb,
                                                                      block_dim,
                                                                      alpha,
                                                                      descr_A,
                                                                      nnzb_A,
                                                                      bsr_row_ptr_A,
                                                                      bsr_col_ind_A,
                                                                      descr_B,
                                                                      nnzb_B,
                                                                      bsr_row_ptr_B,
                                                                      bsr_col_ind_B,
                                                                      beta,
                                                                      descr_D,
                                                                      nnzb_D,
                                                                      bsr_row_ptr_D,
                                                                      bsr_col_ind_D,
                                                                      info_C,
                                                                      buffer_size));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                    \
    template rocsparse_status rocsparse::bsrgemm_buffer_size_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                                   \
        rocsparse_direction       dir,                                                      \
        rocsparse_operation       trans_A,                                                  \
        rocsparse_operation       trans_B,                                                  \
        JTYPE                     mb,                                                       \
        JTYPE                     nb,                                                       \
        JTYPE                     kb,                                                       \
        JTYPE                     block_dim,                                                \
        const TTYPE*              alpha,                                                    \
        const rocsparse_mat_descr descr_A,                                                  \
        ITYPE                     nnzb_A,                                                   \
        const ITYPE*              bsr_row_ptr_A,                                            \
        const JTYPE*              bsr_col_ind_A,                                            \
        const rocsparse_mat_descr descr_B,                                                  \
        ITYPE                     nnzb_B,                                                   \
        const ITYPE*              bsr_row_ptr_B,                                            \
        const JTYPE*              bsr_col_ind_B,                                            \
        const TTYPE*              beta,                                                     \
        const rocsparse_mat_descr descr_D,                                                  \
        ITYPE                     nnzb_D,                                                   \
        const ITYPE*              bsr_row_ptr_D,                                            \
        const JTYPE*              bsr_col_ind_D,                                            \
        rocsparse_mat_info        info_C,                                                   \
        size_t*                   buffer_size);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                \
    template rocsparse_status rocsparse::bsrgemm_buffer_size_impl<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                               \
        rocsparse_direction       dir,                                                  \
        rocsparse_operation       trans_A,                                              \
        rocsparse_operation       trans_B,                                              \
        JTYPE                     mb,                                                   \
        JTYPE                     nb,                                                   \
        JTYPE                     kb,                                                   \
        JTYPE                     block_dim,                                            \
        const TTYPE*              alpha,                                                \
        const rocsparse_mat_descr descr_A,                                              \
        ITYPE                     nnzb_A,                                               \
        const ITYPE*              bsr_row_ptr_A,                                        \
        const JTYPE*              bsr_col_ind_A,                                        \
        const rocsparse_mat_descr descr_B,                                              \
        ITYPE                     nnzb_B,                                               \
        const ITYPE*              bsr_row_ptr_B,                                        \
        const JTYPE*              bsr_col_ind_B,                                        \
        const TTYPE*              beta,                                                 \
        const rocsparse_mat_descr descr_D,                                              \
        ITYPE                     nnzb_D,                                               \
        const ITYPE*              bsr_row_ptr_D,                                        \
        const JTYPE*              bsr_col_ind_D,                                        \
        rocsparse_mat_info        info_C,                                               \
        size_t*                   buffer_size);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

//
// rocsparse_xbsrgemm_buffer_size
//
#define C_IMPL(NAME, TYPE)                                                           \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,               \
                                     rocsparse_direction       dir,                  \
                                     rocsparse_operation       trans_A,              \
                                     rocsparse_operation       trans_B,              \
                                     rocsparse_int             mb,                   \
                                     rocsparse_int             nb,                   \
                                     rocsparse_int             kb,                   \
                                     rocsparse_int             block_dim,            \
                                     const TYPE*               alpha,                \
                                     const rocsparse_mat_descr descr_A,              \
                                     rocsparse_int             nnzb_A,               \
                                     const rocsparse_int*      bsr_row_ptr_A,        \
                                     const rocsparse_int*      bsr_col_ind_A,        \
                                     const rocsparse_mat_descr descr_B,              \
                                     rocsparse_int             nnzb_B,               \
                                     const rocsparse_int*      bsr_row_ptr_B,        \
                                     const rocsparse_int*      bsr_col_ind_B,        \
                                     const TYPE*               beta,                 \
                                     const rocsparse_mat_descr descr_D,              \
                                     rocsparse_int             nnzb_D,               \
                                     const rocsparse_int*      bsr_row_ptr_D,        \
                                     const rocsparse_int*      bsr_col_ind_D,        \
                                     rocsparse_mat_info        info_C,               \
                                     size_t*                   buffer_size)          \
    try                                                                              \
    {                                                                                \
        ROCSPARSE_ROUTINE_TRACE;                                                     \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_buffer_size_impl(handle,        \
                                                                      dir,           \
                                                                      trans_A,       \
                                                                      trans_B,       \
                                                                      mb,            \
                                                                      nb,            \
                                                                      kb,            \
                                                                      block_dim,     \
                                                                      alpha,         \
                                                                      descr_A,       \
                                                                      nnzb_A,        \
                                                                      bsr_row_ptr_A, \
                                                                      bsr_col_ind_A, \
                                                                      descr_B,       \
                                                                      nnzb_B,        \
                                                                      bsr_row_ptr_B, \
                                                                      bsr_col_ind_B, \
                                                                      beta,          \
                                                                      descr_D,       \
                                                                      nnzb_D,        \
                                                                      bsr_row_ptr_D, \
                                                                      bsr_col_ind_D, \
                                                                      info_C,        \
                                                                      buffer_size)); \
        return rocsparse_status_success;                                             \
    }                                                                                \
    catch(...)                                                                       \
    {                                                                                \
        RETURN_ROCSPARSE_EXCEPTION();                                                \
    }

C_IMPL(rocsparse_sbsrgemm_buffer_size, float);
C_IMPL(rocsparse_dbsrgemm_buffer_size, double);
C_IMPL(rocsparse_cbsrgemm_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrgemm_buffer_size, rocsparse_double_complex);

#undef C_IMPL
