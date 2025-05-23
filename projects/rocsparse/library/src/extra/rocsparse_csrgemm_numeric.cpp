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

#include "rocsparse_csrgemm_numeric.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_csrgemm_numeric_mult.hpp"
#include "rocsparse_csrgemm_numeric_multadd.hpp"
#include "rocsparse_csrgemm_numeric_scal.hpp"
#include "utility.h"

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrgemm_numeric_core(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 J                         m,
                                                 J                         n,
                                                 J                         k,
                                                 const T*                  alpha_device_host,
                                                 const rocsparse_mat_descr descr_A,
                                                 I                         nnz_A,
                                                 const T*                  csr_val_A,
                                                 const I*                  csr_row_ptr_A,
                                                 const J*                  csr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 I                         nnz_B,
                                                 const T*                  csr_val_B,
                                                 const I*                  csr_row_ptr_B,
                                                 const J*                  csr_col_ind_B,
                                                 const T*                  beta_device_host,
                                                 const rocsparse_mat_descr descr_D,
                                                 I                         nnz_D,
                                                 const T*                  csr_val_D,
                                                 const I*                  csr_row_ptr_D,
                                                 const J*                  csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 I                         nnz_C,
                                                 T*                        csr_val_C,
                                                 const I*                  csr_row_ptr_C,
                                                 const J*                  csr_col_ind_C,
                                                 const rocsparse_mat_info  info_C,
                                                 void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(mul && add)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_numeric_multadd_core(handle,
                                                                          trans_A,
                                                                          trans_B,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          alpha_device_host,
                                                                          descr_A,
                                                                          nnz_A,
                                                                          csr_val_A,
                                                                          csr_row_ptr_A,
                                                                          csr_col_ind_A,
                                                                          descr_B,
                                                                          nnz_B,
                                                                          csr_val_B,
                                                                          csr_row_ptr_B,
                                                                          csr_col_ind_B,
                                                                          beta_device_host,
                                                                          descr_D,
                                                                          nnz_D,
                                                                          csr_val_D,
                                                                          csr_row_ptr_D,
                                                                          csr_col_ind_D,
                                                                          descr_C,
                                                                          nnz_C,
                                                                          csr_val_C,
                                                                          csr_row_ptr_C,
                                                                          csr_col_ind_C,
                                                                          info_C,
                                                                          temp_buffer));

        return rocsparse_status_success;
    }
    else if(mul && !add)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_numeric_mult_core(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       alpha_device_host,
                                                                       descr_A,
                                                                       nnz_A,
                                                                       csr_val_A,
                                                                       csr_row_ptr_A,
                                                                       csr_col_ind_A,
                                                                       descr_B,
                                                                       nnz_B,
                                                                       csr_val_B,
                                                                       csr_row_ptr_B,
                                                                       csr_col_ind_B,
                                                                       descr_C,
                                                                       nnz_C,
                                                                       csr_val_C,
                                                                       csr_row_ptr_C,
                                                                       csr_col_ind_C,
                                                                       info_C,
                                                                       temp_buffer));

        return rocsparse_status_success;
    }
    else if(!mul && add)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_numeric_scal_core(handle,
                                                                       m,
                                                                       n,
                                                                       beta_device_host,
                                                                       descr_D,
                                                                       nnz_D,
                                                                       csr_val_D,
                                                                       csr_row_ptr_D,
                                                                       csr_col_ind_D,
                                                                       descr_C,
                                                                       nnz_C,
                                                                       csr_val_C,
                                                                       csr_row_ptr_C,
                                                                       csr_col_ind_C,
                                                                       info_C,
                                                                       temp_buffer));

        return rocsparse_status_success;
    }
    else
    {
        rocsparse_host_assert(mul == false && add == false, "Wrong logical dispatch.");

        if(descr_C->type != rocsparse_matrix_type_general)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrgemm_numeric_quickreturn(rocsparse_handle          handle,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        int64_t                   m,
                                                        int64_t                   n,
                                                        int64_t                   k,
                                                        const void*               alpha,
                                                        const rocsparse_mat_descr descr_A,
                                                        int64_t                   nnz_A,
                                                        const void*               csr_val_A,
                                                        const void*               csr_row_ptr_A,
                                                        const void*               csr_col_ind_A,
                                                        const rocsparse_mat_descr descr_B,
                                                        int64_t                   nnz_B,
                                                        const void*               csr_val_B,
                                                        const void*               csr_row_ptr_B,
                                                        const void*               csr_col_ind_B,
                                                        const void*               beta,
                                                        const rocsparse_mat_descr descr_D,
                                                        int64_t                   nnz_D,
                                                        const void*               csr_val_D,
                                                        const void*               csr_row_ptr_D,
                                                        const void*               csr_col_ind_D,
                                                        const rocsparse_mat_descr descr_C,
                                                        int64_t                   nnz_C,
                                                        void*                     csr_val_C,
                                                        const void*               csr_row_ptr_C,
                                                        const void*               csr_col_ind_C,
                                                        const rocsparse_mat_info  info_C,
                                                        void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;

    if(mul == true && add == true)
    {
        const rocsparse_status status
            = rocsparse::csrgemm_numeric_multadd_quickreturn(handle,
                                                             trans_A,
                                                             trans_B,
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             descr_A,
                                                             nnz_A,
                                                             csr_val_A,
                                                             csr_row_ptr_A,
                                                             csr_col_ind_A,
                                                             descr_B,
                                                             nnz_B,
                                                             csr_val_B,
                                                             csr_row_ptr_B,
                                                             csr_col_ind_B,
                                                             beta,
                                                             descr_D,
                                                             nnz_D,
                                                             csr_val_D,
                                                             csr_row_ptr_D,
                                                             csr_col_ind_D,
                                                             descr_C,
                                                             nnz_C,
                                                             csr_val_C,
                                                             csr_row_ptr_C,
                                                             csr_col_ind_C,
                                                             info_C,
                                                             temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
    else if(mul == true && add == false)
    {
        const rocsparse_status status = rocsparse::csrgemm_numeric_mult_quickreturn(handle,
                                                                                    trans_A,
                                                                                    trans_B,
                                                                                    m,
                                                                                    n,
                                                                                    k,
                                                                                    alpha,
                                                                                    descr_A,
                                                                                    nnz_A,
                                                                                    csr_val_A,
                                                                                    csr_row_ptr_A,
                                                                                    csr_col_ind_A,
                                                                                    descr_B,
                                                                                    nnz_B,
                                                                                    csr_val_B,
                                                                                    csr_row_ptr_B,
                                                                                    csr_col_ind_B,
                                                                                    descr_C,
                                                                                    nnz_C,
                                                                                    csr_val_C,
                                                                                    csr_row_ptr_C,
                                                                                    csr_col_ind_C,
                                                                                    info_C,
                                                                                    temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
    else if(mul == false && add == true)
    {
        const rocsparse_status status = rocsparse::csrgemm_numeric_scal_quickreturn(handle,
                                                                                    m,
                                                                                    n,
                                                                                    beta,
                                                                                    descr_D,
                                                                                    nnz_D,
                                                                                    csr_val_D,
                                                                                    csr_row_ptr_D,
                                                                                    csr_col_ind_D,
                                                                                    descr_C,
                                                                                    nnz_C,
                                                                                    csr_val_C,
                                                                                    csr_row_ptr_C,
                                                                                    csr_col_ind_C,
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
        if(m == 0 || n == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    static rocsparse_status csrgemm_numeric_checkarg(rocsparse_handle          handle, //0
                                                     rocsparse_operation       trans_A, //1
                                                     rocsparse_operation       trans_B, //2
                                                     J                         m, //3
                                                     J                         n, //4
                                                     J                         k, //5
                                                     const T*                  alpha, //6
                                                     const rocsparse_mat_descr descr_A, //7
                                                     I                         nnz_A, //8
                                                     const T*                  csr_val_A, //9
                                                     const I*                  csr_row_ptr_A, //10
                                                     const J*                  csr_col_ind_A, //11
                                                     const rocsparse_mat_descr descr_B, //12
                                                     I                         nnz_B, //13
                                                     const T*                  csr_val_B, //14
                                                     const I*                  csr_row_ptr_B, //15
                                                     const J*                  csr_col_ind_B, //16
                                                     const T*                  beta, //17
                                                     const rocsparse_mat_descr descr_D, //18
                                                     I                         nnz_D, //19
                                                     const T*                  csr_val_D, //20
                                                     const I*                  csr_row_ptr_D, //21
                                                     const J*                  csr_col_ind_D, //22
                                                     const rocsparse_mat_descr descr_C, //23
                                                     I                         nnz_C, //24
                                                     T*                        csr_val_C, //25
                                                     const I*                  csr_row_ptr_C, //26
                                                     const J*                  csr_col_ind_C, //27
                                                     const rocsparse_mat_info  info_C, //28
                                                     void*                     temp_buffer) //29
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, n);
        ROCSPARSE_CHECKARG_SIZE(5, k);
        ROCSPARSE_CHECKARG_SIZE(8, nnz_A);
        ROCSPARSE_CHECKARG_SIZE(13, nnz_B);
        ROCSPARSE_CHECKARG_SIZE(19, nnz_D);
        ROCSPARSE_CHECKARG_SIZE(24, nnz_C);

        ROCSPARSE_CHECKARG_POINTER(28, info_C);
        ROCSPARSE_CHECKARG(
            28, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);

        const rocsparse_status status = rocsparse::csrgemm_numeric_quickreturn(handle,
                                                                               trans_A,
                                                                               trans_B,
                                                                               m,
                                                                               n,
                                                                               k,
                                                                               alpha,
                                                                               descr_A,
                                                                               nnz_A,
                                                                               csr_val_A,
                                                                               csr_row_ptr_A,
                                                                               csr_col_ind_A,
                                                                               descr_B,
                                                                               nnz_B,
                                                                               csr_val_B,
                                                                               csr_row_ptr_B,
                                                                               csr_col_ind_B,
                                                                               beta,
                                                                               descr_D,
                                                                               nnz_D,
                                                                               csr_val_D,
                                                                               csr_row_ptr_D,
                                                                               csr_col_ind_D,
                                                                               descr_C,
                                                                               nnz_C,
                                                                               csr_val_C,
                                                                               csr_row_ptr_C,
                                                                               csr_col_ind_C,
                                                                               info_C,
                                                                               temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;

        if(mul == true && add == true)
        {
            ROCSPARSE_CHECKARG_POINTER(7, descr_A);
            ROCSPARSE_CHECKARG_POINTER(12, descr_B);

            ROCSPARSE_CHECKARG_ARRAY(9, nnz_A, csr_val_A);
            ROCSPARSE_CHECKARG_ARRAY(
                10, ((trans_A == rocsparse_operation_none) ? m : k), csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(11, nnz_A, csr_col_ind_A);

            ROCSPARSE_CHECKARG_ARRAY(14, nnz_B, csr_val_B);
            ROCSPARSE_CHECKARG_ARRAY(
                15, ((trans_B == rocsparse_operation_none) ? k : n), csr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(16, nnz_B, csr_col_ind_B);

            ROCSPARSE_CHECKARG_POINTER(18, descr_D);
            ROCSPARSE_CHECKARG_ARRAY(20, nnz_D, csr_val_D);
            ROCSPARSE_CHECKARG_ARRAY(21, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(22, nnz_D, csr_col_ind_D);

            ROCSPARSE_CHECKARG_POINTER(23, descr_C);

            ROCSPARSE_CHECKARG_ARRAY(25, nnz_C, csr_val_C);
            ROCSPARSE_CHECKARG_ARRAY(26, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_ARRAY(27, nnz_C, csr_col_ind_C);

            ROCSPARSE_CHECKARG_POINTER(6, alpha);
            ROCSPARSE_CHECKARG_POINTER(17, beta);
            ROCSPARSE_CHECKARG_POINTER(29, temp_buffer);

            return rocsparse_status_continue;
        }
        else if(mul == true && add == false)
        {

            ROCSPARSE_CHECKARG_POINTER(7, descr_A);
            ROCSPARSE_CHECKARG_POINTER(12, descr_B);

            ROCSPARSE_CHECKARG_ARRAY(9, nnz_A, csr_val_A);
            ROCSPARSE_CHECKARG_ARRAY(
                10, ((trans_A == rocsparse_operation_none) ? m : k), csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(11, nnz_A, csr_col_ind_A);

            ROCSPARSE_CHECKARG_ARRAY(14, nnz_B, csr_val_B);
            ROCSPARSE_CHECKARG_ARRAY(16, nnz_B, csr_col_ind_B);
            ROCSPARSE_CHECKARG_ARRAY(
                15, ((trans_B == rocsparse_operation_none) ? k : n), csr_row_ptr_B);

            ROCSPARSE_CHECKARG_POINTER(23, descr_C);

            ROCSPARSE_CHECKARG_ARRAY(25, nnz_C, csr_val_C);
            ROCSPARSE_CHECKARG_ARRAY(26, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_ARRAY(27, nnz_C, csr_col_ind_C);
            ROCSPARSE_CHECKARG_POINTER(6, alpha);

            ROCSPARSE_CHECKARG_POINTER(29, temp_buffer);
            return rocsparse_status_continue;
        }
        else if(mul == false && add == true)
        {
            ROCSPARSE_CHECKARG_POINTER(17, beta);

            ROCSPARSE_CHECKARG_POINTER(18, descr_D);
            ROCSPARSE_CHECKARG_ARRAY(20, nnz_D, csr_val_D);
            ROCSPARSE_CHECKARG_ARRAY(21, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(22, nnz_D, csr_col_ind_D);

            ROCSPARSE_CHECKARG_POINTER(23, descr_C);
            ROCSPARSE_CHECKARG_ARRAY(25, nnz_C, csr_val_C);
            ROCSPARSE_CHECKARG_ARRAY(26, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_ARRAY(27, nnz_C, csr_col_ind_C);
            return rocsparse_status_continue;
        }
        else
        {
            rocsparse_host_assert(mul == false && add == false, "Wrong logical dispatch.");

            ROCSPARSE_CHECKARG_POINTER(23, descr_C);
            ROCSPARSE_CHECKARG_ARRAY(25, nnz_C, csr_val_C);
            ROCSPARSE_CHECKARG_ARRAY(26, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_ARRAY(27, nnz_C, csr_col_ind_C);
            return rocsparse_status_continue;
        }
    }

    template <typename... P>
    static rocsparse_status csrgemm_numeric_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_csrgemm_numeric", p...);

        const rocsparse_status status = rocsparse::csrgemm_numeric_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_numeric_core(p...));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                            \
    template rocsparse_status rocsparse::csrgemm_numeric_core<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                           \
        rocsparse_operation       trans_A,                                          \
        rocsparse_operation       trans_B,                                          \
        JTYPE                     m,                                                \
        JTYPE                     n,                                                \
        JTYPE                     k,                                                \
        const TTYPE*              alpha,                                            \
        const rocsparse_mat_descr descr_A,                                          \
        ITYPE                     nnz_A,                                            \
        const TTYPE*              csr_val_A,                                        \
        const ITYPE*              csr_row_ptr_A,                                    \
        const JTYPE*              csr_col_ind_A,                                    \
        const rocsparse_mat_descr descr_B,                                          \
        ITYPE                     nnz_B,                                            \
        const TTYPE*              csr_val_B,                                        \
        const ITYPE*              csr_row_ptr_B,                                    \
        const JTYPE*              csr_col_ind_B,                                    \
        const TTYPE*              beta,                                             \
        const rocsparse_mat_descr descr_D,                                          \
        ITYPE                     nnz_D,                                            \
        const TTYPE*              csr_val_D,                                        \
        const ITYPE*              csr_row_ptr_D,                                    \
        const JTYPE*              csr_col_ind_D,                                    \
        const rocsparse_mat_descr descr_C,                                          \
        ITYPE                     nnz_C,                                            \
        TTYPE*                    csr_val_C,                                        \
        const ITYPE*              csr_row_ptr_C,                                    \
        const JTYPE*              csr_col_ind_C,                                    \
        const rocsparse_mat_info  info_C,                                           \
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

#define C_IMPL(NAME, TYPE)                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,           \
                                     rocsparse_operation       trans_A,          \
                                     rocsparse_operation       trans_B,          \
                                     rocsparse_int             m,                \
                                     rocsparse_int             n,                \
                                     rocsparse_int             k,                \
                                     const TYPE*               alpha,            \
                                     const rocsparse_mat_descr descr_A,          \
                                     rocsparse_int             nnz_A,            \
                                     const TYPE*               csr_val_A,        \
                                     const rocsparse_int*      csr_row_ptr_A,    \
                                     const rocsparse_int*      csr_col_ind_A,    \
                                     const rocsparse_mat_descr descr_B,          \
                                     rocsparse_int             nnz_B,            \
                                     const TYPE*               csr_val_B,        \
                                     const rocsparse_int*      csr_row_ptr_B,    \
                                     const rocsparse_int*      csr_col_ind_B,    \
                                     const TYPE*               beta,             \
                                     const rocsparse_mat_descr descr_D,          \
                                     rocsparse_int             nnz_D,            \
                                     const TYPE*               csr_val_D,        \
                                     const rocsparse_int*      csr_row_ptr_D,    \
                                     const rocsparse_int*      csr_col_ind_D,    \
                                     const rocsparse_mat_descr descr_C,          \
                                     rocsparse_int             nnz_C,            \
                                     TYPE*                     csr_val_C,        \
                                     const rocsparse_int*      csr_row_ptr_C,    \
                                     const rocsparse_int*      csr_col_ind_C,    \
                                     const rocsparse_mat_info  info_C,           \
                                     void*                     temp_buffer)      \
    try                                                                          \
    {                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                 \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_numeric_impl(handle,        \
                                                                  trans_A,       \
                                                                  trans_B,       \
                                                                  m,             \
                                                                  n,             \
                                                                  k,             \
                                                                  alpha,         \
                                                                  descr_A,       \
                                                                  nnz_A,         \
                                                                  csr_val_A,     \
                                                                  csr_row_ptr_A, \
                                                                  csr_col_ind_A, \
                                                                  descr_B,       \
                                                                  nnz_B,         \
                                                                  csr_val_B,     \
                                                                  csr_row_ptr_B, \
                                                                  csr_col_ind_B, \
                                                                  beta,          \
                                                                  descr_D,       \
                                                                  nnz_D,         \
                                                                  csr_val_D,     \
                                                                  csr_row_ptr_D, \
                                                                  csr_col_ind_D, \
                                                                  descr_C,       \
                                                                  nnz_C,         \
                                                                  csr_val_C,     \
                                                                  csr_row_ptr_C, \
                                                                  csr_col_ind_C, \
                                                                  info_C,        \
                                                                  temp_buffer)); \
        return rocsparse_status_success;                                         \
    }                                                                            \
    catch(...)                                                                   \
    {                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                            \
    }

C_IMPL(rocsparse_scsrgemm_numeric, float);
C_IMPL(rocsparse_dcsrgemm_numeric, double);
C_IMPL(rocsparse_ccsrgemm_numeric, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrgemm_numeric, rocsparse_double_complex);
#undef C_IMPL
