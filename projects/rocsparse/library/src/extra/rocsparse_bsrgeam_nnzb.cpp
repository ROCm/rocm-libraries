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

#include "internal/extra/rocsparse_bsrgeam.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csrgeam.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    static rocsparse_status bsrgeam_nnzb_core(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nb,
                                              rocsparse_int             block_dim,
                                              const rocsparse_mat_descr descr_A,
                                              rocsparse_int             nnzb_A,
                                              const rocsparse_int*      bsr_row_ptr_A,
                                              const rocsparse_int*      bsr_col_ind_A,
                                              const rocsparse_mat_descr descr_B,
                                              rocsparse_int             nnzb_B,
                                              const rocsparse_int*      bsr_row_ptr_B,
                                              const rocsparse_int*      bsr_col_ind_B,
                                              const rocsparse_mat_descr descr_C,
                                              rocsparse_int*            bsr_row_ptr_C,
                                              rocsparse_int*            nnzb_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_spgeam_descr descr       = nullptr;
        void*                        temp_buffer = nullptr;
        RETURN_IF_ROCSPARSE_ERROR(
            (csrgeam_nnz_template<rocsparse_int, rocsparse_int>(handle,
                                                                descr,
                                                                rocsparse_operation_none,
                                                                rocsparse_operation_none,
                                                                mb,
                                                                nb,
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
                                                                temp_buffer,
                                                                false)));

        return rocsparse_status_success;
    }

    static rocsparse_status bsrgeam_nnzb_quickreturn(rocsparse_handle          handle,
                                                     rocsparse_direction       dir,
                                                     rocsparse_int             mb,
                                                     rocsparse_int             nb,
                                                     rocsparse_int             block_dim,
                                                     const rocsparse_mat_descr descr_A,
                                                     rocsparse_int             nnzb_A,
                                                     const rocsparse_int*      bsr_row_ptr_A,
                                                     const rocsparse_int*      bsr_col_ind_A,
                                                     const rocsparse_mat_descr descr_B,
                                                     rocsparse_int             nnzb_B,
                                                     const rocsparse_int*      bsr_row_ptr_B,
                                                     const rocsparse_int*      bsr_col_ind_B,
                                                     const rocsparse_mat_descr descr_C,
                                                     rocsparse_int*            bsr_row_ptr_C,
                                                     rocsparse_int*            nnzb_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(mb == 0 || nb == 0 || (nnzb_A == 0 && nnzb_B == 0))
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_host)
            {
                *nnzb_C = 0;
            }
            else
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(nnzb_C, 0, sizeof(rocsparse_int), handle->stream));
            }

            if(nnzb_A == 0 && nnzb_B == 0)
            {
                if(bsr_row_ptr_C != nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                        handle, mb + 1, static_cast<rocsparse_int>(descr_C->base), bsr_row_ptr_C));
                }
            }

            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    static rocsparse_status bsrgeam_nnzb_checkarg(rocsparse_handle          handle, //0
                                                  rocsparse_direction       dir, //1
                                                  rocsparse_int             mb, //2
                                                  rocsparse_int             nb, //3
                                                  rocsparse_int             block_dim, //4
                                                  const rocsparse_mat_descr descr_A, //5
                                                  rocsparse_int             nnzb_A, //6
                                                  const rocsparse_int*      bsr_row_ptr_A, //7
                                                  const rocsparse_int*      bsr_col_ind_A, //8
                                                  const rocsparse_mat_descr descr_B, //9
                                                  rocsparse_int             nnzb_B, //10
                                                  const rocsparse_int*      bsr_row_ptr_B, //11
                                                  const rocsparse_int*      bsr_col_ind_B, //12
                                                  const rocsparse_mat_descr descr_C, //13
                                                  rocsparse_int*            bsr_row_ptr_C, //14
                                                  rocsparse_int*            nnzb_C) //15
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(5, descr_A);
        ROCSPARSE_CHECKARG_POINTER(9, descr_B);
        ROCSPARSE_CHECKARG_POINTER(13, descr_C);

        ROCSPARSE_CHECKARG(5,
                           descr_A,
                           (descr_A->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(9,
                           descr_B,
                           (descr_B->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(13,
                           descr_C,
                           (descr_C->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(5,
                           descr_A,
                           (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(9,
                           descr_B,
                           (descr_B->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(13,
                           descr_C,
                           (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_SIZE(2, mb);
        ROCSPARSE_CHECKARG_SIZE(3, nb);

        ROCSPARSE_CHECKARG_SIZE(4, block_dim);
        ROCSPARSE_CHECKARG(4, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG_SIZE(6, nnzb_A);
        ROCSPARSE_CHECKARG_SIZE(10, nnzb_B);

        ROCSPARSE_CHECKARG_ARRAY(14, mb, bsr_row_ptr_C);
        ROCSPARSE_CHECKARG_POINTER(15, nnzb_C);

        const rocsparse_status status = rocsparse::bsrgeam_nnzb_quickreturn(handle,
                                                                            dir,
                                                                            mb,
                                                                            nb,
                                                                            block_dim,
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
                                                                            nnzb_C);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_ARRAY(7, mb, bsr_row_ptr_A);
        ROCSPARSE_CHECKARG_ARRAY(8, nnzb_A, bsr_col_ind_A);

        ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr_B);
        ROCSPARSE_CHECKARG_ARRAY(12, nnzb_B, bsr_col_ind_B);

        return rocsparse_status_continue;
    }

    template <typename... P>
    static rocsparse_status bsrgeam_nnzb_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_csrgeam_nnz", p...);

        const rocsparse_status status = rocsparse::bsrgeam_nnzb_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgeam_nnzb_core(p...));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_bsrgeam_nnzb(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             block_dim,
                                                   const rocsparse_mat_descr descr_A,
                                                   rocsparse_int             nnzb_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   rocsparse_int             nnzb_B,
                                                   const rocsparse_int*      bsr_row_ptr_B,
                                                   const rocsparse_int*      bsr_col_ind_B,
                                                   const rocsparse_mat_descr descr_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            nnzb_C)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgeam_nnzb_impl(handle,
                                                           dir,
                                                           mb,
                                                           nb,
                                                           block_dim,
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
                                                           nnzb_C));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
