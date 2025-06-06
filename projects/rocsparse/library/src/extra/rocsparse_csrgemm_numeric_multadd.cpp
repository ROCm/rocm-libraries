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
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_csrgemm.hpp"

#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm_numeric_calc.hpp"
#include "rocsparse_csrgemm_numeric_multadd.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrgemm_numeric_multadd_quickreturn(rocsparse_handle    handle,
                                                                rocsparse_operation trans_A,
                                                                rocsparse_operation trans_B,
                                                                int64_t             m,
                                                                int64_t             n,
                                                                int64_t             k,
                                                                const void* alpha_device_host,
                                                                const rocsparse_mat_descr descr_A,
                                                                int64_t                   nnz_A,
                                                                const void*               csr_val_A,
                                                                const void* csr_row_ptr_A,
                                                                const void* csr_col_ind_A,
                                                                const rocsparse_mat_descr descr_B,
                                                                int64_t                   nnz_B,
                                                                const void*               csr_val_B,
                                                                const void* csr_row_ptr_B,
                                                                const void* csr_col_ind_B,
                                                                const void* beta_device_host,
                                                                const rocsparse_mat_descr descr_D,
                                                                int64_t                   nnz_D,
                                                                const void*               csr_val_D,
                                                                const void* csr_row_ptr_D,
                                                                const void* csr_col_ind_D,
                                                                const rocsparse_mat_descr descr_C,
                                                                int64_t                   nnz_C,
                                                                void*                     csr_val_C,
                                                                const void* csr_row_ptr_C,
                                                                const void* csr_col_ind_C,
                                                                const rocsparse_mat_info info_C,
                                                                void* temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(mul == true && add == true)
    {
        if((m == 0 || n == 0) || ((k == 0 || nnz_A == 0 || nnz_B == 0) && (nnz_D == 0)))
        {
            return rocsparse_status_success;
        }
        else
        {
            return rocsparse_status_continue;
        }
    }
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                               "failed condition (mul -- true && add == true)");
    }
}

template <typename I, typename J, typename T>
inline rocsparse_status rocsparse::csrgemm_numeric_multadd_core(rocsparse_handle    handle,
                                                                rocsparse_operation trans_A,
                                                                rocsparse_operation trans_B,
                                                                J                   m,
                                                                J                   n,
                                                                J                   k,
                                                                const T* alpha_device_host,
                                                                const rocsparse_mat_descr descr_A,
                                                                I                         nnz_A,
                                                                const T*                  csr_val_A,
                                                                const I* csr_row_ptr_A,
                                                                const J* csr_col_ind_A,
                                                                const rocsparse_mat_descr descr_B,
                                                                I                         nnz_B,
                                                                const T*                  csr_val_B,
                                                                const I* csr_row_ptr_B,
                                                                const J* csr_col_ind_B,
                                                                const T* beta_device_host,
                                                                const rocsparse_mat_descr descr_D,
                                                                I                         nnz_D,
                                                                const T*                  csr_val_D,
                                                                const I* csr_row_ptr_D,
                                                                const J* csr_col_ind_D,
                                                                const rocsparse_mat_descr descr_C,
                                                                I                         nnz_C,
                                                                T*                        csr_val_C,
                                                                const I* csr_row_ptr_C,
                                                                const J* csr_col_ind_C,
                                                                const rocsparse_mat_info info_C,
                                                                void* temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(mul == true && add == true)
    {
        if((k == 0 || nnz_A == 0 || nnz_B == 0) && (nnz_D == 0))
        {
            return rocsparse_status_success;
        }

        if(descr_A->type != rocsparse_matrix_type_general)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (descr_A->type != rocsparse_matrix_type_general)");
        }

        if(descr_B->type != rocsparse_matrix_type_general)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (descr_B->type != rocsparse_matrix_type_general)");
        }

        if(descr_C->type != rocsparse_matrix_type_general)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (descr_C->type != rocsparse_matrix_type_general)");
        }

        if(descr_D->type != rocsparse_matrix_type_general)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (descr_D->type != rocsparse_matrix_type_general)");
        }

        if(trans_A != rocsparse_operation_none)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (trans_A != rocsparse_operation_none)");
        }

        if(trans_B != rocsparse_operation_none)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                " failed on condition (trans_B != rocsparse_operation_none)");
        }

        if(descr_D->type != rocsparse_matrix_type_general)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        if(((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none)))
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        if(descr_A->type != rocsparse_matrix_type_general)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        if(descr_B->type != rocsparse_matrix_type_general)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_numeric_calc_template(handle,
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
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                               "failed condition (mul == true && add == true)");
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                    \
    template rocsparse_status rocsparse::csrgemm_numeric_multadd_core<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                                   \
        rocsparse_operation       trans_A,                                                  \
        rocsparse_operation       trans_B,                                                  \
        JTYPE                     m,                                                        \
        JTYPE                     n,                                                        \
        JTYPE                     k,                                                        \
        const TTYPE*              alpha,                                                    \
        const rocsparse_mat_descr descr_A,                                                  \
        ITYPE                     nnz_A,                                                    \
        const TTYPE*              csr_val_A,                                                \
        const ITYPE*              csr_row_ptr_A,                                            \
        const JTYPE*              csr_col_ind_A,                                            \
        const rocsparse_mat_descr descr_B,                                                  \
        ITYPE                     nnz_B,                                                    \
        const TTYPE*              csr_val_B,                                                \
        const ITYPE*              csr_row_ptr_B,                                            \
        const JTYPE*              csr_col_ind_B,                                            \
        const TTYPE*              beta,                                                     \
        const rocsparse_mat_descr descr_D,                                                  \
        ITYPE                     nnz_D,                                                    \
        const TTYPE*              csr_val_D,                                                \
        const ITYPE*              csr_row_ptr_D,                                            \
        const JTYPE*              csr_col_ind_D,                                            \
        const rocsparse_mat_descr descr_C,                                                  \
        ITYPE                     nnz_C,                                                    \
        TTYPE*                    csr_val_C,                                                \
        const ITYPE*              csr_row_ptr_C,                                            \
        const JTYPE*              csr_col_ind_C,                                            \
        const rocsparse_mat_info  info_C,                                                   \
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
