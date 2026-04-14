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

#include "rocsparse_csrgemm_symbolic.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_csrgemm_symbolic_mult.hpp"
#include "rocsparse_csrgemm_symbolic_multadd.hpp"
#include "rocsparse_csrgemm_symbolic_scal.hpp"

rocsparse_status rocsparse::csrgemm_symbolic_quickreturn(rocsparse_handle          handle,
                                                         rocsparse_operation       trans_A,
                                                         rocsparse_operation       trans_B,
                                                         int64_t                   m,
                                                         int64_t                   n,
                                                         int64_t                   k,
                                                         const rocsparse_mat_descr descr_A,
                                                         int64_t                   nnz_A,
                                                         const void*               csr_row_ptr_A,
                                                         const void*               csr_col_ind_A,
                                                         const rocsparse_mat_descr descr_B,
                                                         int64_t                   nnz_B,
                                                         const void*               csr_row_ptr_B,
                                                         const void*               csr_col_ind_B,
                                                         const rocsparse_mat_descr descr_D,
                                                         int64_t                   nnz_D,
                                                         const void*               csr_row_ptr_D,
                                                         const void*               csr_col_ind_D,
                                                         const rocsparse_mat_descr descr_C,
                                                         int64_t                   nnz_C,
                                                         const void*               csr_row_ptr_C,
                                                         void*                     csr_col_ind_C,
                                                         const rocsparse_mat_info  info_C,
                                                         void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;

    if(true == mul && true == add)
    {
        const rocsparse_status status
            = rocsparse::csrgemm_symbolic_multadd_quickreturn(handle,
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
                                                              nnz_C,
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
    else if(true == mul && false == add)
    {
        const rocsparse_status status = rocsparse::csrgemm_symbolic_mult_quickreturn(handle,
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
                                                                                     nnz_C,
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
    else if(false == mul && true == add)
    {
        const rocsparse_status status = rocsparse::csrgemm_symbolic_scal_quickreturn(handle,
                                                                                     m,
                                                                                     n,
                                                                                     descr_D,
                                                                                     nnz_D,
                                                                                     csr_row_ptr_D,
                                                                                     csr_col_ind_D,
                                                                                     descr_C,
                                                                                     nnz_C,
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
        if(m == 0 || n == 0 || nnz_C == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

namespace rocsparse
{
    static rocsparse_status csrgemm_symbolic_checkarg(rocsparse_handle          handle, //0
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
                                                      int64_t                   nnz_C, //19
                                                      const void*               csr_row_ptr_C, //20
                                                      void*                     csr_col_ind_C, //21
                                                      const rocsparse_mat_info  info_C, //22
                                                      void*                     temp_buffer) //23
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool mul = (info_C) ? info_C->csrgemm_info->mul : false;
        const bool add = (info_C) ? info_C->csrgemm_info->add : false;
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(22, info_C);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, n);
        ROCSPARSE_CHECKARG_SIZE(5, k);
        ROCSPARSE_CHECKARG_SIZE(7, nnz_A);
        ROCSPARSE_CHECKARG_SIZE(11, nnz_B);
        ROCSPARSE_CHECKARG_SIZE(15, nnz_D);
        ROCSPARSE_CHECKARG_SIZE(19, nnz_C);
        ROCSPARSE_CHECKARG(
            22, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);

        const rocsparse_status status = rocsparse::csrgemm_symbolic_quickreturn(handle,
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
                                                                                nnz_C,
                                                                                csr_row_ptr_C,
                                                                                csr_col_ind_C,
                                                                                info_C,
                                                                                temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        if(mul)
        {
            ROCSPARSE_CHECKARG_POINTER(6, descr_A);
            ROCSPARSE_CHECKARG_POINTER(10, descr_B);

            ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(9, nnz_A, csr_col_ind_A);

            ROCSPARSE_CHECKARG_ARRAY(12, k, csr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(13, nnz_B, csr_col_ind_B);
        }

        if(add)
        {
            ROCSPARSE_CHECKARG_POINTER(14, descr_D);
            ROCSPARSE_CHECKARG_ARRAY(16, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(17, nnz_D, csr_col_ind_D);
        }

        ROCSPARSE_CHECKARG_POINTER(18, descr_C);
        ROCSPARSE_CHECKARG_POINTER(20, csr_row_ptr_C);
        ROCSPARSE_CHECKARG_POINTER(21, csr_col_ind_C);
        if(mul)
        {
            ROCSPARSE_CHECKARG_POINTER(23, temp_buffer);
        }

        return rocsparse_status_continue;
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_symbolic_core(rocsparse_handle          handle,
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
                                                  I                         nnz_C,
                                                  const I*                  csr_row_ptr_C,
                                                  J*                        csr_col_ind_C,
                                                  const rocsparse_mat_info  info_C,
                                                  void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;

    if(true == mul && true == add)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_symbolic_multadd_core(handle,
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
                                                                           nnz_C,
                                                                           csr_row_ptr_C,
                                                                           csr_col_ind_C,
                                                                           info_C,
                                                                           temp_buffer));
        return rocsparse_status_success;
    }
    else if(true == mul && false == add)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_symbolic_mult_core(handle,
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
                                                                        nnz_C,
                                                                        csr_row_ptr_C,
                                                                        csr_col_ind_C,
                                                                        info_C,
                                                                        temp_buffer));
        return rocsparse_status_success;
    }
    else if(false == mul && true == add)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_symbolic_scal_core(handle,
                                                                        m,
                                                                        n,
                                                                        descr_D,
                                                                        nnz_D,
                                                                        csr_row_ptr_D,
                                                                        csr_col_ind_D,
                                                                        descr_C,
                                                                        nnz_C,
                                                                        csr_row_ptr_C,
                                                                        csr_col_ind_C,
                                                                        info_C,
                                                                        temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        rocsparse_host_assert(mul == false && add == false, "Wrong logical dispatch.");

        return rocsparse_status_success;
    }
}

template <typename... P>
rocsparse_status rocsparse_csrgemm_symbolic_impl(P&&... p)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace("rocsparse_csrgemm_symbolic", p...);

    const rocsparse_status status = rocsparse::csrgemm_symbolic_checkarg(p...);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_symbolic_core(p...));
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE)                                             \
    template rocsparse_status rocsparse::csrgemm_symbolic_core<ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                     \
        rocsparse_operation       trans_A,                                    \
        rocsparse_operation       trans_B,                                    \
        JTYPE                     m,                                          \
        JTYPE                     n,                                          \
        JTYPE                     k,                                          \
        const rocsparse_mat_descr descr_A,                                    \
        ITYPE                     nnz_A,                                      \
        const ITYPE*              csr_row_ptr_A,                              \
        const JTYPE*              csr_col_ind_A,                              \
        const rocsparse_mat_descr descr_B,                                    \
        ITYPE                     nnz_B,                                      \
        const ITYPE*              csr_row_ptr_B,                              \
        const JTYPE*              csr_col_ind_B,                              \
        const rocsparse_mat_descr descr_D,                                    \
        ITYPE                     nnz_D,                                      \
        const ITYPE*              csr_row_ptr_D,                              \
        const JTYPE*              csr_col_ind_D,                              \
        const rocsparse_mat_descr descr_C,                                    \
        ITYPE                     nnz_C,                                      \
        const ITYPE*              csr_row_ptr_C,                              \
        JTYPE*                    csr_col_ind_C,                              \
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

#define C_IMPL(NAME)                                                             \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,           \
                                     rocsparse_operation       trans_A,          \
                                     rocsparse_operation       trans_B,          \
                                     rocsparse_int             m,                \
                                     rocsparse_int             n,                \
                                     rocsparse_int             k,                \
                                     const rocsparse_mat_descr descr_A,          \
                                     rocsparse_int             nnz_A,            \
                                     const rocsparse_int*      csr_row_ptr_A,    \
                                     const rocsparse_int*      csr_col_ind_A,    \
                                     const rocsparse_mat_descr descr_B,          \
                                     rocsparse_int             nnz_B,            \
                                     const rocsparse_int*      csr_row_ptr_B,    \
                                     const rocsparse_int*      csr_col_ind_B,    \
                                     const rocsparse_mat_descr descr_D,          \
                                     rocsparse_int             nnz_D,            \
                                     const rocsparse_int*      csr_row_ptr_D,    \
                                     const rocsparse_int*      csr_col_ind_D,    \
                                     const rocsparse_mat_descr descr_C,          \
                                     rocsparse_int             nnz_C,            \
                                     const rocsparse_int*      csr_row_ptr_C,    \
                                     rocsparse_int*            csr_col_ind_C,    \
                                     const rocsparse_mat_info  info_C,           \
                                     void*                     temp_buffer)      \
    try                                                                          \
    {                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                 \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrgemm_symbolic_impl(handle,        \
                                                                  trans_A,       \
                                                                  trans_B,       \
                                                                  m,             \
                                                                  n,             \
                                                                  k,             \
                                                                  descr_A,       \
                                                                  nnz_A,         \
                                                                  csr_row_ptr_A, \
                                                                  csr_col_ind_A, \
                                                                  descr_B,       \
                                                                  nnz_B,         \
                                                                  csr_row_ptr_B, \
                                                                  csr_col_ind_B, \
                                                                  descr_D,       \
                                                                  nnz_D,         \
                                                                  csr_row_ptr_D, \
                                                                  csr_col_ind_D, \
                                                                  descr_C,       \
                                                                  nnz_C,         \
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

C_IMPL(rocsparse_csrgemm_symbolic);

#undef C_IMPL
