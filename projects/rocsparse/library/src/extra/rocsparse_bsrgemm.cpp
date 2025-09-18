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

#include "internal/extra/rocsparse_bsrgemm.h"
#include "../conversion/rocsparse_identity.hpp"
#include "bsrgemm_device.h"
#include "csrgemm_device.h"
#include "rocsparse_bsrgemm.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_bsrgemm_mult.hpp"
#include "rocsparse_bsrgemm_multadd.hpp"
#include "rocsparse_bsrgemm_scal.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status bsrgemm_core(rocsparse_handle          handle,
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
                                  const T*                  bsr_val_A,
                                  const I*                  bsr_row_ptr_A,
                                  const J*                  bsr_col_ind_A,
                                  const rocsparse_mat_descr descr_B,
                                  I                         nnzb_B,
                                  const T*                  bsr_val_B,
                                  const I*                  bsr_row_ptr_B,
                                  const J*                  bsr_col_ind_B,
                                  const T*                  beta,
                                  const rocsparse_mat_descr descr_D,
                                  I                         nnzb_D,
                                  const T*                  bsr_val_D,
                                  const I*                  bsr_row_ptr_D,
                                  const J*                  bsr_col_ind_D,
                                  const rocsparse_mat_descr descr_C,
                                  T*                        bsr_val_C,
                                  const I*                  bsr_row_ptr_C,
                                  J*                        bsr_col_ind_C,
                                  const rocsparse_mat_info  info_C,
                                  void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(block_dim == 1)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_template(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  mb,
                                                                  nb,
                                                                  kb,
                                                                  alpha,
                                                                  descr_A,
                                                                  nnzb_A,
                                                                  bsr_val_A,
                                                                  bsr_row_ptr_A,
                                                                  bsr_col_ind_A,
                                                                  descr_B,
                                                                  nnzb_B,
                                                                  bsr_val_B,
                                                                  bsr_row_ptr_B,
                                                                  bsr_col_ind_B,
                                                                  beta,
                                                                  descr_D,
                                                                  nnzb_D,
                                                                  bsr_val_D,
                                                                  bsr_row_ptr_D,
                                                                  bsr_col_ind_D,
                                                                  descr_C,
                                                                  bsr_val_C,
                                                                  bsr_row_ptr_C,
                                                                  bsr_col_ind_C,
                                                                  info_C,
                                                                  temp_buffer));
            return rocsparse_status_success;
        }

        if((info_C->csrgemm_info->mul == false || kb == 0) && info_C->csrgemm_info->add == true)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_scal_core(handle,
                                                                   mb,
                                                                   nb,
                                                                   block_dim,
                                                                   beta,
                                                                   descr_D,
                                                                   nnzb_D,
                                                                   bsr_val_D,
                                                                   bsr_row_ptr_D,
                                                                   bsr_col_ind_D,
                                                                   descr_C,
                                                                   bsr_val_C,
                                                                   bsr_row_ptr_C,
                                                                   bsr_col_ind_C,
                                                                   info_C,
                                                                   temp_buffer));
            return rocsparse_status_success;
        }
        else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_multadd_core(handle,
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
                                                                      bsr_val_A,
                                                                      bsr_row_ptr_A,
                                                                      bsr_col_ind_A,
                                                                      descr_B,
                                                                      nnzb_B,
                                                                      bsr_val_B,
                                                                      bsr_row_ptr_B,
                                                                      bsr_col_ind_B,
                                                                      beta,
                                                                      descr_D,
                                                                      nnzb_D,
                                                                      bsr_val_D,
                                                                      bsr_row_ptr_D,
                                                                      bsr_col_ind_D,
                                                                      descr_C,
                                                                      bsr_val_C,
                                                                      bsr_row_ptr_C,
                                                                      bsr_col_ind_C,
                                                                      info_C,
                                                                      temp_buffer));
            return rocsparse_status_success;
        }
        else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_mult_core(handle,
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
                                                                   bsr_val_A,
                                                                   bsr_row_ptr_A,
                                                                   bsr_col_ind_A,
                                                                   descr_B,
                                                                   nnzb_B,
                                                                   bsr_val_B,
                                                                   bsr_row_ptr_B,
                                                                   bsr_col_ind_B,
                                                                   descr_C,
                                                                   bsr_val_C,
                                                                   bsr_row_ptr_C,
                                                                   bsr_col_ind_C,
                                                                   info_C,
                                                                   temp_buffer));
            return rocsparse_status_success;
        }
        else
        {
            // C = 0
            return rocsparse_status_success;
        }
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::bsrgemm_template(rocsparse_handle          handle,
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
                                             const T*                  bsr_val_A,
                                             const I*                  bsr_row_ptr_A,
                                             const J*                  bsr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             I                         nnzb_B,
                                             const T*                  bsr_val_B,
                                             const I*                  bsr_row_ptr_B,
                                             const J*                  bsr_col_ind_B,
                                             const T*                  beta,
                                             const rocsparse_mat_descr descr_D,
                                             I                         nnzb_D,
                                             const T*                  bsr_val_D,
                                             const I*                  bsr_row_ptr_D,
                                             const J*                  bsr_col_ind_D,
                                             const rocsparse_mat_descr descr_C,
                                             T*                        bsr_val_C,
                                             const I*                  bsr_row_ptr_C,
                                             J*                        bsr_col_ind_C,
                                             const rocsparse_mat_info  info_C,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(block_dim == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_template(handle,
                                                              trans_A,
                                                              trans_B,
                                                              mb,
                                                              nb,
                                                              kb,
                                                              alpha,
                                                              descr_A,
                                                              nnzb_A,
                                                              bsr_val_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              descr_B,
                                                              nnzb_B,
                                                              bsr_val_B,
                                                              bsr_row_ptr_B,
                                                              bsr_col_ind_B,
                                                              beta,
                                                              descr_D,
                                                              nnzb_D,
                                                              bsr_val_D,
                                                              bsr_row_ptr_D,
                                                              bsr_col_ind_D,
                                                              descr_C,
                                                              bsr_val_C,
                                                              bsr_row_ptr_C,
                                                              bsr_col_ind_C,
                                                              info_C,
                                                              temp_buffer));
        return rocsparse_status_success;
    }

    if((info_C->csrgemm_info->mul == false || kb == 0) && info_C->csrgemm_info->add == true)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_scal_template(handle,
                                                                   mb,
                                                                   nb,
                                                                   block_dim,
                                                                   beta,
                                                                   descr_D,
                                                                   nnzb_D,
                                                                   bsr_val_D,
                                                                   bsr_row_ptr_D,
                                                                   bsr_col_ind_D,
                                                                   descr_C,
                                                                   bsr_val_C,
                                                                   bsr_row_ptr_C,
                                                                   bsr_col_ind_C,
                                                                   info_C,
                                                                   temp_buffer));
        return rocsparse_status_success;
    }
    else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_multadd_template(handle,
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
                                                                      bsr_val_A,
                                                                      bsr_row_ptr_A,
                                                                      bsr_col_ind_A,
                                                                      descr_B,
                                                                      nnzb_B,
                                                                      bsr_val_B,
                                                                      bsr_row_ptr_B,
                                                                      bsr_col_ind_B,
                                                                      beta,
                                                                      descr_D,
                                                                      nnzb_D,
                                                                      bsr_val_D,
                                                                      bsr_row_ptr_D,
                                                                      bsr_col_ind_D,
                                                                      descr_C,
                                                                      bsr_val_C,
                                                                      bsr_row_ptr_C,
                                                                      bsr_col_ind_C,
                                                                      info_C,
                                                                      temp_buffer));
        return rocsparse_status_success;
    }
    else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_mult_template(handle,
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
                                                                   bsr_val_A,
                                                                   bsr_row_ptr_A,
                                                                   bsr_col_ind_A,
                                                                   descr_B,
                                                                   nnzb_B,
                                                                   bsr_val_B,
                                                                   bsr_row_ptr_B,
                                                                   bsr_col_ind_B,
                                                                   descr_C,
                                                                   bsr_val_C,
                                                                   bsr_row_ptr_C,
                                                                   bsr_col_ind_C,
                                                                   info_C,
                                                                   temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        // C = 0
        return rocsparse_status_success;
    }
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status bsrgemm_checkarg(rocsparse_handle          handle, //0
                                      rocsparse_direction       dir, //1
                                      rocsparse_operation       trans_A, //2
                                      rocsparse_operation       trans_B, //3
                                      J                         mb, //4
                                      J                         nb, //5
                                      J                         kb, //6
                                      J                         block_dim, //7
                                      const T*                  alpha, //8
                                      const rocsparse_mat_descr descr_A, //9
                                      I                         nnzb_A, //10
                                      const T*                  bsr_val_A, //11
                                      const I*                  bsr_row_ptr_A, //12
                                      const J*                  bsr_col_ind_A, //13
                                      const rocsparse_mat_descr descr_B, //14
                                      I                         nnzb_B, //15
                                      const T*                  bsr_val_B, //16
                                      const I*                  bsr_row_ptr_B, //17
                                      const J*                  bsr_col_ind_B, //18
                                      const T*                  beta, //19
                                      const rocsparse_mat_descr descr_D, //20
                                      I                         nnzb_D, //21
                                      const T*                  bsr_val_D, //22
                                      const I*                  bsr_row_ptr_D, //23
                                      const J*                  bsr_col_ind_D, //24
                                      const rocsparse_mat_descr descr_C, //25
                                      T*                        bsr_val_C, //26
                                      const I*                  bsr_row_ptr_C, //27
                                      J*                        bsr_col_ind_C, //28
                                      const rocsparse_mat_info  info_C, //29
                                      void*                     temp_buffer) //30
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(29, info_C);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans_A);
        ROCSPARSE_CHECKARG_ENUM(3, trans_B);
        ROCSPARSE_CHECKARG_SIZE(4, mb);
        ROCSPARSE_CHECKARG_SIZE(5, nb);
        ROCSPARSE_CHECKARG_SIZE(6, kb);
        ROCSPARSE_CHECKARG_SIZE(7, block_dim);
        ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            29, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);

        if(info_C->csrgemm_info->is_initialized)
        {
            ROCSPARSE_CHECKARG_ARRAY(30, info_C->csrgemm_info->buffer_size, temp_buffer);
        }
        else
        {
            ROCSPARSE_CHECKARG_POINTER(30, temp_buffer);
        }

        if((info_C->csrgemm_info->mul == false || kb == 0) && info_C->csrgemm_info->add == true)
        {
            ROCSPARSE_CHECKARG_SIZE(21, nnzb_D);
            ROCSPARSE_CHECKARG_POINTER(20, descr_D);
            ROCSPARSE_CHECKARG_POINTER(25, descr_C);
            ROCSPARSE_CHECKARG_POINTER(19, beta);
            ROCSPARSE_CHECKARG_ARRAY(22, nnzb_D, bsr_val_D);
            ROCSPARSE_CHECKARG_ARRAY(23, mb, bsr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(24, nnzb_D, bsr_col_ind_D);

            ROCSPARSE_CHECKARG_ARRAY(27, mb, bsr_row_ptr_C);
            if(bsr_val_C == nullptr || bsr_col_ind_C == nullptr)
            {
                I start = 0;
                I end   = 0;

                if(bsr_row_ptr_C != nullptr)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                                       &bsr_row_ptr_C[mb],
                                                       sizeof(I),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                       &bsr_row_ptr_C[0],
                                                       sizeof(I),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                }

                const I nnzb_C = (end - start);
                ROCSPARSE_CHECKARG_ARRAY(26, nnzb_C, bsr_val_C);
                ROCSPARSE_CHECKARG_ARRAY(28, nnzb_C, bsr_col_ind_C);
            }

            ROCSPARSE_CHECKARG(20,
                               descr_D,
                               descr_D->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(25,
                               descr_C,
                               descr_C->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::bsrgemm_scal_quickreturn(handle,
                                                                                mb,
                                                                                nb,
                                                                                block_dim,
                                                                                beta,
                                                                                descr_D,
                                                                                nnzb_D,
                                                                                bsr_val_D,
                                                                                bsr_row_ptr_D,
                                                                                bsr_col_ind_D,
                                                                                descr_C,
                                                                                bsr_val_C,
                                                                                bsr_row_ptr_C,
                                                                                bsr_col_ind_C,
                                                                                info_C,
                                                                                temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            return rocsparse_status_continue;
        }
        else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
        {
            ROCSPARSE_CHECKARG_SIZE(10, nnzb_A);
            ROCSPARSE_CHECKARG_SIZE(15, nnzb_B);
            ROCSPARSE_CHECKARG_SIZE(21, nnzb_D);

            ROCSPARSE_CHECKARG_POINTER(9, descr_A);
            ROCSPARSE_CHECKARG_POINTER(14, descr_B);
            ROCSPARSE_CHECKARG_POINTER(20, descr_D);
            ROCSPARSE_CHECKARG_POINTER(25, descr_C);
            ROCSPARSE_CHECKARG_POINTER(8, alpha);
            ROCSPARSE_CHECKARG_POINTER(19, beta);
            ROCSPARSE_CHECKARG_ARRAY(12, mb, bsr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(17, kb, bsr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(23, mb, bsr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(27, mb, bsr_row_ptr_C);

            ROCSPARSE_CHECKARG_ARRAY(11, nnzb_A, bsr_val_A);
            ROCSPARSE_CHECKARG_ARRAY(13, nnzb_A, bsr_col_ind_A);

            ROCSPARSE_CHECKARG_ARRAY(16, nnzb_B, bsr_val_B);
            ROCSPARSE_CHECKARG_ARRAY(18, nnzb_B, bsr_col_ind_B);

            ROCSPARSE_CHECKARG_ARRAY(22, nnzb_D, bsr_val_D);
            ROCSPARSE_CHECKARG_ARRAY(24, nnzb_D, bsr_col_ind_D);
            if(bsr_val_C == nullptr || bsr_col_ind_C == nullptr)
            {
                I start = 0;
                I end   = 0;

                if(bsr_row_ptr_C != nullptr)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                                       &bsr_row_ptr_C[mb],
                                                       sizeof(I),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                       &bsr_row_ptr_C[0],
                                                       sizeof(I),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                }

                const I nnzb_C = (end - start);
                ROCSPARSE_CHECKARG_ARRAY(26, nnzb_C, bsr_val_C);
                ROCSPARSE_CHECKARG_ARRAY(28, nnzb_C, bsr_col_ind_C);
            }
            ROCSPARSE_CHECKARG(9,
                               descr_A,
                               descr_A->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(14,
                               descr_B,
                               descr_B->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(25,
                               descr_C,
                               descr_C->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(20,
                               descr_D,
                               descr_D->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);
            const rocsparse_status status = rocsparse::bsrgemm_multadd_quickreturn(handle,
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
                                                                                   bsr_val_A,
                                                                                   bsr_row_ptr_A,
                                                                                   bsr_col_ind_A,
                                                                                   descr_B,
                                                                                   nnzb_B,
                                                                                   bsr_val_B,
                                                                                   bsr_row_ptr_B,
                                                                                   bsr_col_ind_B,
                                                                                   beta,
                                                                                   descr_D,
                                                                                   nnzb_D,
                                                                                   bsr_val_D,
                                                                                   bsr_row_ptr_D,
                                                                                   bsr_col_ind_D,
                                                                                   descr_C,
                                                                                   bsr_val_C,
                                                                                   bsr_row_ptr_C,
                                                                                   bsr_col_ind_C,
                                                                                   info_C,
                                                                                   temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(
                2, trans_A, trans_A != rocsparse_operation_none, rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(
                3, trans_B, trans_B != rocsparse_operation_none, rocsparse_status_not_implemented);

            if(info_C->csrgemm_info->is_initialized)
            {
                ROCSPARSE_CHECKARG_ARRAY(30, info_C->csrgemm_info->buffer_size, temp_buffer);
            }
            else
            {
                ROCSPARSE_CHECKARG_POINTER(30, temp_buffer);
            }

            return rocsparse_status_continue;
        }
        else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
        {

            ROCSPARSE_CHECKARG_SIZE(10, nnzb_A);
            ROCSPARSE_CHECKARG_SIZE(15, nnzb_B);

            ROCSPARSE_CHECKARG_POINTER(9, descr_A);
            ROCSPARSE_CHECKARG_POINTER(14, descr_B);
            ROCSPARSE_CHECKARG_POINTER(25, descr_C);

            ROCSPARSE_CHECKARG_POINTER(8, alpha);

            ROCSPARSE_CHECKARG_ARRAY(12, mb, bsr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(17, kb, bsr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(27, mb, bsr_row_ptr_C);

            ROCSPARSE_CHECKARG_ARRAY(11, nnzb_A, bsr_val_A);
            ROCSPARSE_CHECKARG_ARRAY(13, nnzb_A, bsr_col_ind_A);

            ROCSPARSE_CHECKARG_ARRAY(16, nnzb_B, bsr_val_B);
            ROCSPARSE_CHECKARG_ARRAY(18, nnzb_B, bsr_col_ind_B);

            if(bsr_val_C == nullptr || bsr_col_ind_C == nullptr)
            {
                I start = 0;
                I end   = 0;

                if(bsr_row_ptr_C != nullptr)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                                       &bsr_row_ptr_C[mb],
                                                       sizeof(I),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                       &bsr_row_ptr_C[0],
                                                       sizeof(I),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                }

                const I nnzb_C = (end - start);
                ROCSPARSE_CHECKARG_ARRAY(26, nnzb_C, bsr_val_C);
                ROCSPARSE_CHECKARG_ARRAY(28, nnzb_C, bsr_col_ind_C);
            }

            ROCSPARSE_CHECKARG(9,
                               descr_A,
                               descr_A->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(14,
                               descr_B,
                               descr_B->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(25,
                               descr_C,
                               descr_C->type != rocsparse_matrix_type_general,
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::bsrgemm_mult_quickreturn(handle,
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
                                                                                bsr_val_A,
                                                                                bsr_row_ptr_A,
                                                                                bsr_col_ind_A,
                                                                                descr_B,
                                                                                nnzb_B,
                                                                                bsr_val_B,
                                                                                bsr_row_ptr_B,
                                                                                bsr_col_ind_B,
                                                                                descr_C,
                                                                                bsr_val_C,
                                                                                bsr_row_ptr_C,
                                                                                bsr_col_ind_C,
                                                                                info_C,
                                                                                temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(
                2, trans_A, trans_A != rocsparse_operation_none, rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(
                3, trans_B, trans_B != rocsparse_operation_none, rocsparse_status_not_implemented);

            return rocsparse_status_continue;
        }
        else
        {
            return rocsparse_status_continue;
        }
    }

    template <typename... P>
    rocsparse_status bsrgemm_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_Xbsrgemm", p...);

        const rocsparse_status status = rocsparse::bsrgemm_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_core(p...));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                        \
    template rocsparse_status rocsparse::bsrgemm_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                       \
        rocsparse_direction       dir,                                          \
        rocsparse_operation       trans_A,                                      \
        rocsparse_operation       trans_B,                                      \
        JTYPE                     mb,                                           \
        JTYPE                     nb,                                           \
        JTYPE                     kb,                                           \
        JTYPE                     block_dim,                                    \
        const TTYPE*              alpha,                                        \
        const rocsparse_mat_descr descr_A,                                      \
        ITYPE                     nnzb_A,                                       \
        const TTYPE*              bsr_val_A,                                    \
        const ITYPE*              bsr_row_ptr_A,                                \
        const JTYPE*              bsr_col_ind_A,                                \
        const rocsparse_mat_descr descr_B,                                      \
        ITYPE                     nnzb_B,                                       \
        const TTYPE*              bsr_val_B,                                    \
        const ITYPE*              bsr_row_ptr_B,                                \
        const JTYPE*              bsr_col_ind_B,                                \
        const TTYPE*              beta,                                         \
        const rocsparse_mat_descr descr_D,                                      \
        ITYPE                     nnzb_D,                                       \
        const TTYPE*              bsr_val_D,                                    \
        const ITYPE*              bsr_row_ptr_D,                                \
        const JTYPE*              bsr_col_ind_D,                                \
        const rocsparse_mat_descr descr_C,                                      \
        TTYPE*                    bsr_val_C,                                    \
        const ITYPE*              bsr_row_ptr_C,                                \
        JTYPE*                    bsr_col_ind_C,                                \
        const rocsparse_mat_info  info_C,                                       \
        void*                     temp_buffer);

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

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_direction       dir,           \
                                     rocsparse_operation       trans_A,       \
                                     rocsparse_operation       trans_B,       \
                                     rocsparse_int             mb,            \
                                     rocsparse_int             nb,            \
                                     rocsparse_int             kb,            \
                                     rocsparse_int             block_dim,     \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnzb_A,        \
                                     const TYPE*               bsr_val_A,     \
                                     const rocsparse_int*      bsr_row_ptr_A, \
                                     const rocsparse_int*      bsr_col_ind_A, \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnzb_B,        \
                                     const TYPE*               bsr_val_B,     \
                                     const rocsparse_int*      bsr_row_ptr_B, \
                                     const rocsparse_int*      bsr_col_ind_B, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_D,       \
                                     rocsparse_int             nnzb_D,        \
                                     const TYPE*               bsr_val_D,     \
                                     const rocsparse_int*      bsr_row_ptr_D, \
                                     const rocsparse_int*      bsr_col_ind_D, \
                                     const rocsparse_mat_descr descr_C,       \
                                     TYPE*                     bsr_val_C,     \
                                     const rocsparse_int*      bsr_row_ptr_C, \
                                     rocsparse_int*            bsr_col_ind_C, \
                                     const rocsparse_mat_info  info_C,        \
                                     void*                     temp_buffer)   \
    try                                                                       \
    {                                                                         \
        ROCSPARSE_ROUTINE_TRACE;                                              \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_impl(handle,             \
                                                          dir,                \
                                                          trans_A,            \
                                                          trans_B,            \
                                                          mb,                 \
                                                          nb,                 \
                                                          kb,                 \
                                                          block_dim,          \
                                                          alpha,              \
                                                          descr_A,            \
                                                          nnzb_A,             \
                                                          bsr_val_A,          \
                                                          bsr_row_ptr_A,      \
                                                          bsr_col_ind_A,      \
                                                          descr_B,            \
                                                          nnzb_B,             \
                                                          bsr_val_B,          \
                                                          bsr_row_ptr_B,      \
                                                          bsr_col_ind_B,      \
                                                          beta,               \
                                                          descr_D,            \
                                                          nnzb_D,             \
                                                          bsr_val_D,          \
                                                          bsr_row_ptr_D,      \
                                                          bsr_col_ind_D,      \
                                                          descr_C,            \
                                                          bsr_val_C,          \
                                                          bsr_row_ptr_C,      \
                                                          bsr_col_ind_C,      \
                                                          info_C,             \
                                                          temp_buffer));      \
        return rocsparse_status_success;                                      \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                         \
    }

C_IMPL(rocsparse_sbsrgemm, float);
C_IMPL(rocsparse_dbsrgemm, double);
C_IMPL(rocsparse_cbsrgemm, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrgemm, rocsparse_double_complex);
#undef C_IMPL
