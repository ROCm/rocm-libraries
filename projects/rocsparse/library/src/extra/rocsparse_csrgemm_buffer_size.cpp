/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrgemm_mult.hpp"
#include "rocsparse_csrgemm_multadd.hpp"
#include "rocsparse_csrgemm_scal.hpp"

namespace rocsparse
{
    static rocsparse_status reinit_csrgemm_info(rocsparse_mat_info info,
                                                const void*        alpha,
                                                const void*        beta,
                                                const int64_t      k,
                                                const int64_t      nnz_A,
                                                const int64_t      nnz_B,
                                                const int64_t      nnz_D)
    {
        ROCSPARSE_ROUTINE_TRACE;

        info->csrgemm_info->mul = (alpha != nullptr);
        info->csrgemm_info->add = (beta != nullptr);

        if(info->csrgemm_info->add && (nnz_D == 0))
        {
            info->csrgemm_info->add = false;
        }

        if(info->csrgemm_info->mul && (k == 0 || nnz_A == 0 || nnz_B == 0))
        {
            info->csrgemm_info->mul = false;
        }

        return rocsparse_status_success;
    }

    static rocsparse_status csrgemm_buffer_size_checkarg(rocsparse_handle          handle, //0
                                                         rocsparse_operation       trans_A, //1
                                                         rocsparse_operation       trans_B, //2
                                                         int64_t                   m, //3
                                                         int64_t                   n, //4
                                                         int64_t                   k, //5
                                                         const void*               alpha, //6
                                                         const rocsparse_mat_descr descr_A, //7
                                                         int64_t                   nnz_A, //8
                                                         const void* csr_row_ptr_A, //9
                                                         const void* csr_col_ind_A, //10
                                                         const rocsparse_mat_descr descr_B, //11
                                                         int64_t                   nnz_B, //12
                                                         const void* csr_row_ptr_B, //13
                                                         const void* csr_col_ind_B, //14
                                                         const void* beta, //15
                                                         const rocsparse_mat_descr descr_D, //16
                                                         int64_t                   nnz_D, //17
                                                         const void*        csr_row_ptr_D, //18
                                                         const void*        csr_col_ind_D, //19
                                                         rocsparse_mat_info info_C, //20
                                                         size_t*            buffer_size) //21
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(20, info_C);
        ROCSPARSE_CHECKARG(
            20, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_internal_error);

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;

        if(mul == true && add == true)
        {
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(8, nnz_A);
            ROCSPARSE_CHECKARG_SIZE(12, nnz_B);
            ROCSPARSE_CHECKARG_SIZE(17, nnz_D);

            ROCSPARSE_CHECKARG_POINTER(7, descr_A);
            ROCSPARSE_CHECKARG_POINTER(11, descr_B);
            ROCSPARSE_CHECKARG_POINTER(16, descr_D);
            ROCSPARSE_CHECKARG_POINTER(21, buffer_size);
            ROCSPARSE_CHECKARG_POINTER(6, alpha);
            ROCSPARSE_CHECKARG_POINTER(15, beta);

            ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(13, k, csr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(18, m, csr_row_ptr_D);

            ROCSPARSE_CHECKARG_ARRAY(10, nnz_A, csr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(14, nnz_B, csr_col_ind_B);
            ROCSPARSE_CHECKARG_ARRAY(19, nnz_D, csr_col_ind_D);

            ROCSPARSE_CHECKARG(7,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(11,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(16,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(1,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(2,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);

            const rocsparse_status status
                = rocsparse::csrgemm_multadd_buffer_size_quickreturn(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     alpha,
                                                                     descr_A,
                                                                     nnz_A,
                                                                     csr_row_ptr_A,
                                                                     csr_col_ind_A,
                                                                     descr_B,
                                                                     nnz_B,
                                                                     csr_row_ptr_B,
                                                                     csr_col_ind_B,
                                                                     beta,
                                                                     descr_D,
                                                                     nnz_D,
                                                                     csr_row_ptr_D,
                                                                     csr_col_ind_D,
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
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(8, nnz_A);
            ROCSPARSE_CHECKARG_SIZE(12, nnz_B);

            ROCSPARSE_CHECKARG_POINTER(7, descr_A);
            ROCSPARSE_CHECKARG_POINTER(11, descr_B);
            ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(13, k, csr_row_ptr_B);

            ROCSPARSE_CHECKARG_POINTER(6, alpha);
            ROCSPARSE_CHECKARG_POINTER(21, buffer_size);

            ROCSPARSE_CHECKARG_ARRAY(10, nnz_A, csr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(14, nnz_B, csr_col_ind_B);

            ROCSPARSE_CHECKARG(7,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(11,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(1,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(2,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);

            const rocsparse_status status
                = rocsparse::csrgemm_mult_buffer_size_quickreturn(handle,
                                                                  trans_A,
                                                                  trans_B,
                                                                  m,
                                                                  n,
                                                                  k,
                                                                  alpha,
                                                                  descr_A,
                                                                  nnz_A,
                                                                  csr_row_ptr_A,
                                                                  csr_col_ind_A,
                                                                  descr_B,
                                                                  nnz_B,
                                                                  csr_row_ptr_B,
                                                                  csr_col_ind_B,
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
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_POINTER(15, beta);
            ROCSPARSE_CHECKARG_POINTER(16, descr_D);
            ROCSPARSE_CHECKARG(16,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG_SIZE(17, nnz_D);
            ROCSPARSE_CHECKARG_ARRAY(18, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_POINTER(21, buffer_size);
            ROCSPARSE_CHECKARG_ARRAY(19, nnz_D, csr_col_ind_D);

            const rocsparse_status status
                = rocsparse::csrgemm_scal_buffer_size_quickreturn(handle,
                                                                  m,
                                                                  n,
                                                                  beta,
                                                                  descr_D,
                                                                  nnz_D,
                                                                  csr_row_ptr_D,
                                                                  csr_col_ind_D,
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

            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_POINTER(21, buffer_size);
            return rocsparse_status_continue;
        }
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrgemm_buffer_size_template(rocsparse_handle          handle,
                                                         rocsparse_operation       trans_A,
                                                         rocsparse_operation       trans_B,
                                                         J                         m,
                                                         J                         n,
                                                         J                         k,
                                                         const T*                  alpha,
                                                         const rocsparse_mat_descr descr_A,
                                                         I                         nnz_A,
                                                         const I*                  csr_row_ptr_A,
                                                         const J*                  csr_col_ind_A,
                                                         const rocsparse_mat_descr descr_B,
                                                         I                         nnz_B,
                                                         const I*                  csr_row_ptr_B,
                                                         const J*                  csr_col_ind_B,
                                                         const T*                  beta,
                                                         const rocsparse_mat_descr descr_D,
                                                         I                         nnz_D,
                                                         const I*                  csr_row_ptr_D,
                                                         const J*                  csr_col_ind_D,
                                                         rocsparse_mat_info        info_C,
                                                         size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrgemm_info(info_C->csrgemm_info));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_csrgemm_info(&info_C->csrgemm_info));
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::reinit_csrgemm_info(info_C, alpha, beta, k, nnz_A, nnz_B, nnz_D));

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(mul == true && add == true)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_multadd_buffer_size_template(handle,
                                                                                  trans_A,
                                                                                  trans_B,
                                                                                  m,
                                                                                  n,
                                                                                  k,
                                                                                  alpha,
                                                                                  descr_A,
                                                                                  nnz_A,
                                                                                  csr_row_ptr_A,
                                                                                  csr_col_ind_A,
                                                                                  descr_B,
                                                                                  nnz_B,
                                                                                  csr_row_ptr_B,
                                                                                  csr_col_ind_B,
                                                                                  beta,
                                                                                  descr_D,
                                                                                  nnz_D,
                                                                                  csr_row_ptr_D,
                                                                                  csr_col_ind_D,
                                                                                  info_C,
                                                                                  buffer_size));
        return rocsparse_status_success;
    }
    else if(mul == true && add == false)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_mult_buffer_size_template(handle,
                                                                               trans_A,
                                                                               trans_B,
                                                                               m,
                                                                               n,
                                                                               k,
                                                                               alpha,
                                                                               descr_A,
                                                                               nnz_A,
                                                                               csr_row_ptr_A,
                                                                               csr_col_ind_A,
                                                                               descr_B,
                                                                               nnz_B,
                                                                               csr_row_ptr_B,
                                                                               csr_col_ind_B,
                                                                               info_C,
                                                                               buffer_size));
        return rocsparse_status_success;
    }
    else if(mul == false && add == true)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_scal_buffer_size_template(
            handle, m, n, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size));
        return rocsparse_status_success;
    }
    else
    {
        if((beta != nullptr && nnz_D == 0)
           || (alpha != nullptr && (k == 0 || nnz_A == 0 || nnz_B == 0)))
        {
            *buffer_size                         = 0;
            info_C->csrgemm_info->buffer_size    = buffer_size[0];
            info_C->csrgemm_info->is_initialized = true;
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }
    }
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    static rocsparse_status csrgemm_buffer_size_impl(rocsparse_handle          handle,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     J                         m,
                                                     J                         n,
                                                     J                         k,
                                                     const T*                  alpha,
                                                     const rocsparse_mat_descr descr_A,
                                                     I                         nnz_A,
                                                     const I*                  csr_row_ptr_A,
                                                     const J*                  csr_col_ind_A,
                                                     const rocsparse_mat_descr descr_B,
                                                     I                         nnz_B,
                                                     const I*                  csr_row_ptr_B,
                                                     const J*                  csr_col_ind_B,
                                                     const T*                  beta,
                                                     const rocsparse_mat_descr descr_D,
                                                     I                         nnz_D,
                                                     const I*                  csr_row_ptr_D,
                                                     const J*                  csr_col_ind_D,
                                                     rocsparse_mat_info        info_C,
                                                     size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrgemm_buffer_size"),
                             trans_A,
                             trans_B,
                             m,
                             n,
                             k,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr_A,
                             nnz_A,
                             (const void*&)csr_row_ptr_A,
                             (const void*&)csr_col_ind_A,
                             (const void*&)descr_B,
                             nnz_B,
                             (const void*&)csr_row_ptr_B,
                             (const void*&)csr_col_ind_B,
                             LOG_TRACE_SCALAR_VALUE(handle, beta),
                             (const void*&)descr_D,
                             nnz_D,
                             (const void*&)csr_row_ptr_D,
                             (const void*&)csr_col_ind_D,
                             (const void*&)info_C,
                             (const void*&)buffer_size);

        ROCSPARSE_CHECKARG_POINTER(20, info_C);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrgemm_info(info_C->csrgemm_info));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_csrgemm_info(&info_C->csrgemm_info));
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::reinit_csrgemm_info(info_C, alpha, beta, k, nnz_A, nnz_B, nnz_D));

        const rocsparse_status status = rocsparse::csrgemm_buffer_size_checkarg(handle,
                                                                                trans_A,
                                                                                trans_B,
                                                                                m,
                                                                                n,
                                                                                k,
                                                                                alpha,
                                                                                descr_A,
                                                                                nnz_A,
                                                                                csr_row_ptr_A,
                                                                                csr_col_ind_A,
                                                                                descr_B,
                                                                                nnz_B,
                                                                                csr_row_ptr_B,
                                                                                csr_col_ind_B,
                                                                                beta,
                                                                                descr_D,
                                                                                nnz_D,
                                                                                csr_row_ptr_D,
                                                                                csr_col_ind_D,
                                                                                info_C,
                                                                                buffer_size);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;
        if(mul == true && add == true)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_multadd_buffer_size_core(handle,
                                                                                  trans_A,
                                                                                  trans_B,
                                                                                  m,
                                                                                  n,
                                                                                  k,
                                                                                  alpha,
                                                                                  descr_A,
                                                                                  nnz_A,
                                                                                  csr_row_ptr_A,
                                                                                  csr_col_ind_A,
                                                                                  descr_B,
                                                                                  nnz_B,
                                                                                  csr_row_ptr_B,
                                                                                  csr_col_ind_B,
                                                                                  beta,
                                                                                  descr_D,
                                                                                  nnz_D,
                                                                                  csr_row_ptr_D,
                                                                                  csr_col_ind_D,
                                                                                  info_C,
                                                                                  buffer_size));

            return rocsparse_status_success;
        }
        else if(mul == true && add == false)
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_mult_buffer_size_core(handle,
                                                                               trans_A,
                                                                               trans_B,
                                                                               m,
                                                                               n,
                                                                               k,
                                                                               alpha,
                                                                               descr_A,
                                                                               nnz_A,
                                                                               csr_row_ptr_A,
                                                                               csr_col_ind_A,
                                                                               descr_B,
                                                                               nnz_B,
                                                                               csr_row_ptr_B,
                                                                               csr_col_ind_B,
                                                                               info_C,
                                                                               buffer_size));

            return rocsparse_status_success;
        }
        else if(mul == false && add == true)
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_scal_buffer_size_core(handle,
                                                                               m,
                                                                               n,
                                                                               beta,
                                                                               descr_D,
                                                                               nnz_D,
                                                                               csr_row_ptr_D,
                                                                               csr_col_ind_D,
                                                                               info_C,
                                                                               buffer_size));

            return rocsparse_status_success;
        }
        else
        {
            if((beta != nullptr && nnz_D == 0)
               || (alpha != nullptr && (k == 0 || nnz_A == 0 || nnz_B == 0)))
            {
                *buffer_size                         = 0;
                info_C->csrgemm_info->buffer_size    = buffer_size[0];
                info_C->csrgemm_info->is_initialized = true;
                return rocsparse_status_success;
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }
        }
    }
}

#define INSTANTIATE(I, J, T)                                           \
    template rocsparse_status rocsparse::csrgemm_buffer_size_template( \
        rocsparse_handle          handle,                              \
        rocsparse_operation       trans_A,                             \
        rocsparse_operation       trans_B,                             \
        J                         m,                                   \
        J                         n,                                   \
        J                         k,                                   \
        const T*                  alpha,                               \
        const rocsparse_mat_descr descr_A,                             \
        I                         nnz_A,                               \
        const I*                  csr_row_ptr_A,                       \
        const J*                  csr_col_ind_A,                       \
        const rocsparse_mat_descr descr_B,                             \
        I                         nnz_B,                               \
        const I*                  csr_row_ptr_B,                       \
        const J*                  csr_col_ind_B,                       \
        const T*                  beta,                                \
        const rocsparse_mat_descr descr_D,                             \
        I                         nnz_D,                               \
        const I*                  csr_row_ptr_D,                       \
        const J*                  csr_col_ind_D,                       \
        rocsparse_mat_info        info_C,                              \
        size_t*                   buffer_size)

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
// rocsparse_xcsrgemm_buffer_size
//
#define C_IMPL(NAME, TYPE)                                                           \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,               \
                                     rocsparse_operation       trans_A,              \
                                     rocsparse_operation       trans_B,              \
                                     rocsparse_int             m,                    \
                                     rocsparse_int             n,                    \
                                     rocsparse_int             k,                    \
                                     const TYPE*               alpha,                \
                                     const rocsparse_mat_descr descr_A,              \
                                     rocsparse_int             nnz_A,                \
                                     const rocsparse_int*      csr_row_ptr_A,        \
                                     const rocsparse_int*      csr_col_ind_A,        \
                                     const rocsparse_mat_descr descr_B,              \
                                     rocsparse_int             nnz_B,                \
                                     const rocsparse_int*      csr_row_ptr_B,        \
                                     const rocsparse_int*      csr_col_ind_B,        \
                                     const TYPE*               beta,                 \
                                     const rocsparse_mat_descr descr_D,              \
                                     rocsparse_int             nnz_D,                \
                                     const rocsparse_int*      csr_row_ptr_D,        \
                                     const rocsparse_int*      csr_col_ind_D,        \
                                     rocsparse_mat_info        info_C,               \
                                     size_t*                   buffer_size)          \
    try                                                                              \
    {                                                                                \
        ROCSPARSE_ROUTINE_TRACE;                                                     \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_buffer_size_impl(handle,        \
                                                                      trans_A,       \
                                                                      trans_B,       \
                                                                      m,             \
                                                                      n,             \
                                                                      k,             \
                                                                      alpha,         \
                                                                      descr_A,       \
                                                                      nnz_A,         \
                                                                      csr_row_ptr_A, \
                                                                      csr_col_ind_A, \
                                                                      descr_B,       \
                                                                      nnz_B,         \
                                                                      csr_row_ptr_B, \
                                                                      csr_col_ind_B, \
                                                                      beta,          \
                                                                      descr_D,       \
                                                                      nnz_D,         \
                                                                      csr_row_ptr_D, \
                                                                      csr_col_ind_D, \
                                                                      info_C,        \
                                                                      buffer_size)); \
        return rocsparse_status_success;                                             \
    }                                                                                \
    catch(...)                                                                       \
    {                                                                                \
        RETURN_ROCSPARSE_EXCEPTION();                                                \
    }

C_IMPL(rocsparse_scsrgemm_buffer_size, float);
C_IMPL(rocsparse_dcsrgemm_buffer_size, double);
C_IMPL(rocsparse_ccsrgemm_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrgemm_buffer_size, rocsparse_double_complex);

#undef C_IMPL
