/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_common.h"
#include "rocsparse_primitives.hpp"

#include "rocsparse_csrgemm_mult.hpp"
#include "rocsparse_csrgemm_multadd.hpp"
#include "rocsparse_csrgemm_nnz_calc.hpp"
#include "rocsparse_csrgemm_scal.hpp"

namespace rocsparse
{
    template <typename I>
    rocsparse_status csrgemm_nnz_checkarg(rocsparse_handle          handle, //0
                                          rocsparse_operation       trans_A, //1
                                          rocsparse_operation       trans_B, //2
                                          int64_t                   m, //3
                                          int64_t                   n, //4
                                          int64_t                   k, //5
                                          const rocsparse_mat_descr descr_A, //6
                                          int64_t                   nnz_A, //7
                                          const void*               csr_row_ptr_A, //8
                                          const void*               csr_col_ind_A, //9
                                          const rocsparse_mat_descr descr_B, //10
                                          int64_t                   nnz_B, //11
                                          const void*               csr_row_ptr_B, //12
                                          const void*               csr_col_ind_B, //13
                                          const rocsparse_mat_descr descr_D, //14
                                          int64_t                   nnz_D, //15
                                          const void*               csr_row_ptr_D, //16
                                          const void*               csr_col_ind_D, //17
                                          const rocsparse_mat_descr descr_C, //18
                                          I*                        csr_row_ptr_C, //19
                                          I*                        nnz_C, //20
                                          const rocsparse_mat_info  info_C, //21
                                          void*                     temp_buffer) //22
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_POINTER(21, info_C);
        ROCSPARSE_CHECKARG(
            21, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;

        if(mul == true && add == true)
        {
            ROCSPARSE_CHECKARG_HANDLE(0, handle);

            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(7, nnz_A);
            ROCSPARSE_CHECKARG_SIZE(11, nnz_B);
            ROCSPARSE_CHECKARG_SIZE(15, nnz_D);

            ROCSPARSE_CHECKARG_POINTER(6, descr_A);
            ROCSPARSE_CHECKARG_POINTER(10, descr_B);
            ROCSPARSE_CHECKARG_POINTER(14, descr_D);
            ROCSPARSE_CHECKARG_POINTER(18, descr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);

            ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(12, k, csr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(16, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);

            ROCSPARSE_CHECKARG_ARRAY(9, nnz_A, csr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(13, nnz_B, csr_col_ind_B);
            ROCSPARSE_CHECKARG_ARRAY(17, nnz_D, csr_col_ind_D);

            ROCSPARSE_CHECKARG(6,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(10,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(14,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(18,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status
                = rocsparse::csrgemm_multadd_nnz_quickreturn(handle,
                                                             trans_A,
                                                             trans_B,
                                                             m,
                                                             n,
                                                             k,
                                                             descr_A,
                                                             nnz_A,
                                                             csr_row_ptr_A,
                                                             csr_col_ind_A,
                                                             descr_B,
                                                             nnz_B,
                                                             csr_row_ptr_B,
                                                             csr_col_ind_B,
                                                             descr_D,
                                                             nnz_D,
                                                             csr_row_ptr_D,
                                                             csr_col_ind_D,
                                                             descr_C,
                                                             csr_row_ptr_C,
                                                             nnz_C,
                                                             info_C,
                                                             temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(1,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(2,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG_POINTER(22, temp_buffer);
            return rocsparse_status_continue;
        }
        else if(mul == true && add == false)
        {
            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(7, nnz_A);
            ROCSPARSE_CHECKARG_SIZE(11, nnz_B);
            ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(12, k, csr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);

            ROCSPARSE_CHECKARG_POINTER(6, descr_A);
            ROCSPARSE_CHECKARG_POINTER(10, descr_B);
            ROCSPARSE_CHECKARG_POINTER(18, descr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);

            ROCSPARSE_CHECKARG_ARRAY(9, nnz_A, csr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(13, nnz_B, csr_col_ind_B);
            ROCSPARSE_CHECKARG(6,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(10,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(18,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::csrgemm_mult_nnz_quickreturn(handle,
                                                                                    trans_A,
                                                                                    trans_B,
                                                                                    m,
                                                                                    n,
                                                                                    k,
                                                                                    descr_A,
                                                                                    nnz_A,
                                                                                    csr_row_ptr_A,
                                                                                    csr_col_ind_A,
                                                                                    descr_B,
                                                                                    nnz_B,
                                                                                    csr_row_ptr_B,
                                                                                    csr_col_ind_B,
                                                                                    descr_C,
                                                                                    csr_row_ptr_C,
                                                                                    nnz_C,
                                                                                    info_C,
                                                                                    temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(1,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(2,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG_POINTER(22, temp_buffer);
            return rocsparse_status_continue;
        }
        else if(mul == false && add == true)
        {

            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(15, nnz_D);
            ROCSPARSE_CHECKARG_POINTER(14, descr_D);
            ROCSPARSE_CHECKARG_POINTER(18, descr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);
            ROCSPARSE_CHECKARG_ARRAY(16, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_ARRAY(17, nnz_D, csr_col_ind_D);
            ROCSPARSE_CHECKARG(18,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(14,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::csrgemm_scal_nnz_quickreturn(handle,
                                                                                    m,
                                                                                    n,
                                                                                    descr_D,
                                                                                    nnz_D,
                                                                                    csr_row_ptr_D,
                                                                                    csr_col_ind_D,
                                                                                    descr_C,
                                                                                    csr_row_ptr_C,
                                                                                    nnz_C,
                                                                                    info_C,
                                                                                    temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            return rocsparse_status_continue;
        }
        else
        {
            rocsparse_host_assert(mul == false && add == false, "Wrong logical dispatch.");
            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_POINTER(21, info_C);
            ROCSPARSE_CHECKARG(
                21, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), handle->stream));
            }
            else
            {
                *nnz_C = 0;
            }

            if(csr_row_ptr_C != nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::valset(handle, m + 1, static_cast<I>(descr_C->base), csr_row_ptr_C));
            }

            return rocsparse_status_success;
        }
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_nnz_template(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 J                         m,
                                                 J                         n,
                                                 J                         k,
                                                 const rocsparse_mat_descr descr_A,
                                                 I                         nnz_A,
                                                 const I*                  csr_row_ptr_A,
                                                 const J*                  csr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 I                         nnz_B,
                                                 const I*                  csr_row_ptr_B,
                                                 const J*                  csr_col_ind_B,
                                                 const rocsparse_mat_descr descr_D,
                                                 I                         nnz_D,
                                                 const I*                  csr_row_ptr_D,
                                                 const J*                  csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 I*                        csr_row_ptr_C,
                                                 I*                        nnz_C,
                                                 const rocsparse_mat_info  info_C,
                                                 void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;

    // Either mult, add or multadd need to be performed
    if(mul == true && add == true)
    {
        // C = alpha * A * B + beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_multadd_nnz_template(handle,
                                                                          trans_A,
                                                                          trans_B,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          descr_A,
                                                                          nnz_A,
                                                                          csr_row_ptr_A,
                                                                          csr_col_ind_A,
                                                                          descr_B,
                                                                          nnz_B,
                                                                          csr_row_ptr_B,
                                                                          csr_col_ind_B,
                                                                          descr_D,
                                                                          nnz_D,
                                                                          csr_row_ptr_D,
                                                                          csr_col_ind_D,
                                                                          descr_C,
                                                                          csr_row_ptr_C,
                                                                          nnz_C,
                                                                          info_C,
                                                                          temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == true && add == false)
    {
        // C = alpha * A * B
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_mult_nnz_template(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       descr_A,
                                                                       nnz_A,
                                                                       csr_row_ptr_A,
                                                                       csr_col_ind_A,
                                                                       descr_B,
                                                                       nnz_B,
                                                                       csr_row_ptr_B,
                                                                       csr_col_ind_B,
                                                                       descr_C,
                                                                       csr_row_ptr_C,
                                                                       nnz_C,
                                                                       info_C,
                                                                       temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == false && add == true)
    {
        // C = beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_scal_nnz_template(handle,
                                                                       m,
                                                                       n,
                                                                       descr_D,
                                                                       nnz_D,
                                                                       csr_row_ptr_D,
                                                                       csr_col_ind_D,
                                                                       descr_C,
                                                                       csr_row_ptr_C,
                                                                       nnz_C,
                                                                       info_C,
                                                                       temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        rocsparse_host_assert(mul == false && add == false, "Wrong logical dispatch.");

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), handle->stream));
        }
        else
        {
            *nnz_C = 0;
        }
        if(csr_row_ptr_C != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::valset(handle, m + 1, static_cast<I>(descr_C->base), csr_row_ptr_C));
        }
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE)                                            \
    template rocsparse_status rocsparse::csrgemm_nnz_template<ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_operation       trans_A,                                   \
        rocsparse_operation       trans_B,                                   \
        JTYPE                     m,                                         \
        JTYPE                     n,                                         \
        JTYPE                     k,                                         \
        const rocsparse_mat_descr descr_A,                                   \
        ITYPE                     nnz_A,                                     \
        const ITYPE*              csr_row_ptr_A,                             \
        const JTYPE*              csr_col_ind_A,                             \
        const rocsparse_mat_descr descr_B,                                   \
        ITYPE                     nnz_B,                                     \
        const ITYPE*              csr_row_ptr_B,                             \
        const JTYPE*              csr_col_ind_B,                             \
        const rocsparse_mat_descr descr_D,                                   \
        ITYPE                     nnz_D,                                     \
        const ITYPE*              csr_row_ptr_D,                             \
        const JTYPE*              csr_col_ind_D,                             \
        const rocsparse_mat_descr descr_C,                                   \
        ITYPE*                    csr_row_ptr_C,                             \
        ITYPE*                    nnz_C,                                     \
        const rocsparse_mat_info  info_C,                                    \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_nnz_core(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             J                         m,
                                             J                         n,
                                             J                         k,
                                             const rocsparse_mat_descr descr_A,
                                             I                         nnz_A,
                                             const I*                  csr_row_ptr_A,
                                             const J*                  csr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             I                         nnz_B,
                                             const I*                  csr_row_ptr_B,
                                             const J*                  csr_col_ind_B,
                                             const rocsparse_mat_descr descr_D,
                                             I                         nnz_D,
                                             const I*                  csr_row_ptr_D,
                                             const J*                  csr_col_ind_D,
                                             const rocsparse_mat_descr descr_C,
                                             I*                        csr_row_ptr_C,
                                             I*                        nnz_C,
                                             const rocsparse_mat_info  info_C,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    // Either mult, add or multadd need to be performed
    if(mul == true && add == true)
    {
        // C = alpha * A * B + beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_multadd_nnz_core(handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      descr_A,
                                                                      nnz_A,
                                                                      csr_row_ptr_A,
                                                                      csr_col_ind_A,
                                                                      descr_B,
                                                                      nnz_B,
                                                                      csr_row_ptr_B,
                                                                      csr_col_ind_B,
                                                                      descr_D,
                                                                      nnz_D,
                                                                      csr_row_ptr_D,
                                                                      csr_col_ind_D,
                                                                      descr_C,
                                                                      csr_row_ptr_C,
                                                                      nnz_C,
                                                                      info_C,
                                                                      temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == true && add == false)
    {
        // C = alpha * A * B
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_mult_nnz_core(handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   descr_A,
                                                                   nnz_A,
                                                                   csr_row_ptr_A,
                                                                   csr_col_ind_A,
                                                                   descr_B,
                                                                   nnz_B,
                                                                   csr_row_ptr_B,
                                                                   csr_col_ind_B,
                                                                   descr_C,
                                                                   csr_row_ptr_C,
                                                                   nnz_C,
                                                                   info_C,
                                                                   temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == false && add == true)
    {

        // C = beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_scal_nnz_core(handle,
                                                                   m,
                                                                   n,
                                                                   descr_D,
                                                                   nnz_D,
                                                                   csr_row_ptr_D,
                                                                   csr_col_ind_D,
                                                                   descr_C,
                                                                   csr_row_ptr_C,
                                                                   nnz_C,
                                                                   info_C,
                                                                   temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        return rocsparse_status_success;
    }
}

namespace rocsparse
{
    template <typename... P>
    static rocsparse_status csrgemm_nnz_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_csrgemm_nnz", p...);

        const rocsparse_status status = rocsparse::csrgemm_nnz_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_nnz_core(p...));
        return rocsparse_status_success;
    }
}

//
// rocsparse_xcsrgemm_nnz
//
extern "C" rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle          handle,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  rocsparse_int             k,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int             nnz_B,
                                                  const rocsparse_int*      csr_row_ptr_B,
                                                  const rocsparse_int*      csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_D,
                                                  rocsparse_int             nnz_D,
                                                  const rocsparse_int*      csr_row_ptr_D,
                                                  const rocsparse_int*      csr_col_ind_D,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int*            csr_row_ptr_C,
                                                  rocsparse_int*            nnz_C,
                                                  const rocsparse_mat_info  info_C,
                                                  void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_nnz_impl(handle,
                                                          trans_A,
                                                          trans_B,
                                                          m,
                                                          n,
                                                          k,
                                                          descr_A,
                                                          nnz_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          descr_B,
                                                          nnz_B,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          descr_D,
                                                          nnz_D,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          descr_C,
                                                          csr_row_ptr_C,
                                                          nnz_C,
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
