/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <sstream>

#include "rocsparse.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_determine_indextype.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spsm_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spsm_alg_default);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spsm_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spsm_stage_buffer_size);
        CASE(rocsparse_spsm_stage_preprocess);
        CASE(rocsparse_spsm_stage_compute);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsm_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsm_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsm_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsm_stage_buffer_size:
    case rocsparse_spsm_stage_preprocess:
    case rocsparse_spsm_stage_compute:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{
    enum class spsm_case
    {
        NT_NT,
        T_NT,
        NT_T,
        T_T
    };

    static spsm_case
        spsm_get_case(rocsparse_operation trans_B, rocsparse_order order_B, rocsparse_order order_C)
    {
        const bool B_is_transposed
            = (((trans_B == rocsparse_operation_none) && (order_B == rocsparse_order_row))
               || ((trans_B != rocsparse_operation_none) && (order_B == rocsparse_order_column)));
        const bool C_is_transposed = (order_C == rocsparse_order_row);

        if(B_is_transposed && C_is_transposed)
        {
            // 1) B col order + transposed and C row order
            // 2) B row order + non-transposed and C row order
            return spsm_case::T_T;
        }
        else if(B_is_transposed && !C_is_transposed)
        {
            // 1) B col order + transposed and C col order
            // 2) B row order + non-transposed and C col order
            return spsm_case::T_NT;
        }
        else if(!B_is_transposed && C_is_transposed)
        {
            // 1) B row order + transposed and C row order
            // 2) B col order + non-transposed and C row order
            return spsm_case::NT_T;
        }
        else
        {
            // 1) B row order + transposed and C col order
            // 2) B col order + non-transposed and C col order
            return spsm_case::NT_NT;
        }
    }

    static rocsparse_status spsm_solve_T_T(rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           rocsparse_const_spmat_descr matA,
                                           rocsparse_const_dnmat_descr matB,
                                           const rocsparse_dnmat_descr matC,
                                           rocsparse_spsm_alg          alg,
                                           void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype alpha_datatype = matA->data_type;
        // 1) B col order + transposed and C row order
        // 2) B row order + non-transposed and C row order
        void* csrsm_buffer = temp_buffer;
        if(matB->rows > 0 && matB->cols > 0)
        {
            const size_t sizeof_datatype = rocsparse::datatype_sizeof(matC->data_type);
            if(matB->order == rocsparse_order_column)
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(matC->values,
                                                     sizeof_datatype * matC->ld,
                                                     matB->const_values,
                                                     sizeof_datatype * matB->ld,
                                                     sizeof_datatype * matB->rows,
                                                     matB->cols,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
            else
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(matC->values,
                                                     sizeof_datatype * matC->ld,
                                                     matB->const_values,
                                                     sizeof_datatype * matB->ld,
                                                     sizeof_datatype * matB->cols,
                                                     matB->rows,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
        }

        switch(matA->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             matC->values,
                                                             matC->ld,
                                                             matC->order,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             matC->values,
                                                             matC->ld,
                                                             matC->order,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        return rocsparse_status_success;
    }

    static rocsparse_status spsm_solve_T_NT(rocsparse_handle            handle,
                                            rocsparse_operation         trans_A,
                                            rocsparse_operation         trans_B,
                                            const void*                 alpha,
                                            rocsparse_const_spmat_descr matA,
                                            rocsparse_const_dnmat_descr matB,
                                            const rocsparse_dnmat_descr matC,
                                            rocsparse_spsm_alg          alg,
                                            void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype alpha_datatype = matA->data_type;

        const size_t sizeof_datatype = rocsparse::datatype_sizeof(matB->data_type);

        // 1) B col order + transposed and C col order
        // 2) B row order + non-transposed and C col order
        void* spsm_buffer  = temp_buffer;
        void* csrsm_buffer = ((char*)temp_buffer)
                             + ((sizeof_datatype * matB->rows * matB->cols - 1) / 256 + 1) * 256;

        if(matB->rows > 0 && matB->cols > 0)
        {
            if(matB->order == rocsparse_order_column)
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(spsm_buffer,
                                                     sizeof_datatype * matB->rows,
                                                     matB->const_values,
                                                     sizeof_datatype * matB->ld,
                                                     sizeof_datatype * matB->rows,
                                                     matB->cols,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
            else
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(spsm_buffer,
                                                     sizeof_datatype * matB->cols,
                                                     matB->const_values,
                                                     sizeof_datatype * matB->ld,
                                                     sizeof_datatype * matB->cols,
                                                     matB->rows,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
        }

        switch(matA->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             spsm_buffer,
                                                             matC->cols,
                                                             rocsparse_order_row,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             spsm_buffer,
                                                             matC->cols,
                                                             rocsparse_order_row,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        if(matB->rows > 0 && matB->cols > 0)
        {
            if(matB->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->rows,
                                                                     matB->cols,
                                                                     matB->data_type,
                                                                     spsm_buffer,
                                                                     matB->rows,
                                                                     matC->data_type,
                                                                     matC->values,
                                                                     matC->ld));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->cols,
                                                                     matB->rows,
                                                                     matB->data_type,
                                                                     spsm_buffer,
                                                                     matB->cols,
                                                                     matC->data_type,
                                                                     matC->values,
                                                                     matC->ld));
            }
        }

        return rocsparse_status_success;
    }

    static rocsparse_status spsm_solve_NT_T(rocsparse_handle            handle,
                                            rocsparse_operation         trans_A,
                                            rocsparse_operation         trans_B,
                                            const void*                 alpha,
                                            rocsparse_const_spmat_descr matA,
                                            rocsparse_const_dnmat_descr matB,
                                            const rocsparse_dnmat_descr matC,
                                            rocsparse_spsm_alg          alg,
                                            void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_datatype alpha_datatype = matA->data_type;
        // 1) B row order + transposed and C row order
        // 2) B col order + non-transposed and C row order
        void* csrsm_buffer = temp_buffer;
        if(matB->rows > 0 && matB->cols > 0)
        {
            if(matB->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->rows,
                                                                     matB->cols,
                                                                     matB->data_type,
                                                                     matB->const_values,
                                                                     matB->ld,
                                                                     matC->data_type,
                                                                     matC->values,
                                                                     matC->ld));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->cols,
                                                                     matB->rows,
                                                                     matB->data_type,
                                                                     matB->const_values,
                                                                     matB->ld,
                                                                     matC->data_type,
                                                                     matC->values,
                                                                     matC->ld));
            }
        }

        switch(matA->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             matC->values,
                                                             matC->ld,
                                                             matC->order,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             matC->values,
                                                             matC->ld,
                                                             matC->order,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        return rocsparse_status_success;
    }

    static rocsparse_status spsm_solve_NT_NT(rocsparse_handle            handle,
                                             rocsparse_operation         trans_A,
                                             rocsparse_operation         trans_B,
                                             const void*                 alpha,
                                             rocsparse_const_spmat_descr matA,
                                             rocsparse_const_dnmat_descr matB,
                                             const rocsparse_dnmat_descr matC,
                                             rocsparse_spsm_alg          alg,
                                             void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype alpha_datatype = matA->data_type;

        // 1) B row order + transposed and C col order
        // 2) B col order + non-transposed and C col order
        void* spsm_buffer = temp_buffer;
        void* csrsm_buffer
            = ((char*)temp_buffer)
              + ((rocsparse::datatype_sizeof(matB->data_type) * matB->rows * matB->cols - 1) / 256
                 + 1)
                    * 256;

        if(matB->rows > 0 && matB->cols > 0)
        {
            if(matB->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->rows,
                                                                     matB->cols,
                                                                     matB->data_type,
                                                                     matB->const_values,
                                                                     matB->ld,
                                                                     matB->data_type,
                                                                     spsm_buffer,
                                                                     matB->cols));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->cols,
                                                                     matB->rows,
                                                                     matB->data_type,
                                                                     matB->const_values,
                                                                     matB->ld,
                                                                     matB->data_type,
                                                                     spsm_buffer,
                                                                     matB->rows));
            }
        }

        switch(matA->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             spsm_buffer,
                                                             matC->cols,
                                                             rocsparse_order_row,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             trans_A,
                                                             trans_B,
                                                             matA->rows,
                                                             matC->cols,
                                                             matA->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             matA->descr,
                                                             matA->data_type,
                                                             matA->const_val_data,
                                                             matA->row_type,
                                                             matA->const_row_data,
                                                             matA->col_type,
                                                             matA->const_col_data,
                                                             matC->data_type,
                                                             spsm_buffer,
                                                             matC->cols,
                                                             rocsparse_order_row,
                                                             matA->info,
                                                             rocsparse_solve_policy_auto,
                                                             matA->info->get_csrsm_info(),
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        if(matB->rows > 0 && matB->cols > 0)
        {
            if(matB->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->cols,
                                                                     matB->rows,
                                                                     matB->data_type,
                                                                     spsm_buffer,
                                                                     matB->cols,
                                                                     matC->data_type,
                                                                     matC->values,
                                                                     matC->ld));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     matB->rows,
                                                                     matB->cols,
                                                                     matB->data_type,
                                                                     spsm_buffer,
                                                                     matB->rows,
                                                                     matC->data_type,
                                                                     matC->values,
                                                                     matC->ld));
            }
        }

        return rocsparse_status_success;
    }

    rocsparse_status spsm(rocsparse_handle            handle,
                          rocsparse_operation         trans_A,
                          rocsparse_operation         trans_B,
                          const void*                 alpha,
                          rocsparse_const_spmat_descr matA,
                          rocsparse_const_dnmat_descr matB,
                          const rocsparse_dnmat_descr matC,
                          rocsparse_spsm_alg          alg,
                          rocsparse_spsm_stage        stage,
                          size_t*                     buffer_size,
                          void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype alpha_datatype = matA->data_type;

        rocsparse::spsm_case spsm_case = spsm_get_case(trans_B, matB->order, matC->order);

        switch(stage)
        {
        case rocsparse_spsm_stage_buffer_size:
        {
            switch(matA->format)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_buffer_size(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       matA->rows,
                                                                       matC->cols,
                                                                       matA->nnz,
                                                                       alpha_datatype,
                                                                       matA->descr,
                                                                       matA->data_type,
                                                                       matA->const_val_data,
                                                                       matA->row_type,
                                                                       matA->const_row_data,
                                                                       matA->col_type,
                                                                       matA->const_col_data,
                                                                       matC->data_type,
                                                                       matC->order,
                                                                       matA->info,
                                                                       rocsparse_solve_policy_auto,
                                                                       buffer_size));

                if(spsm_case == rocsparse::spsm_case::NT_NT
                   || spsm_case == rocsparse::spsm_case::T_NT)
                {
                    *buffer_size
                        += ((rocsparse::datatype_sizeof(matB->data_type) * matB->rows * matB->cols
                             - 1)
                                / 256
                            + 1)
                           * 256;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_buffer_size(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       matA->rows,
                                                                       matC->cols,
                                                                       matA->nnz,
                                                                       alpha_datatype,
                                                                       matA->descr,
                                                                       matA->data_type,
                                                                       matA->const_val_data,
                                                                       matA->row_type,
                                                                       matA->const_row_data,
                                                                       matA->col_type,
                                                                       matA->const_col_data,
                                                                       matC->data_type,
                                                                       matC->order,
                                                                       matA->info,
                                                                       rocsparse_solve_policy_auto,
                                                                       buffer_size));

                if(spsm_case == rocsparse::spsm_case::NT_NT
                   || spsm_case == rocsparse::spsm_case::T_NT)
                {
                    *buffer_size
                        += ((rocsparse::datatype_sizeof(matB->data_type) * matB->rows * matB->cols
                             - 1)
                                / 256
                            + 1)
                           * 256;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spsm_stage_preprocess:
        {
            void* csrsm_buffer = temp_buffer;
            if(spsm_case == rocsparse::spsm_case::NT_NT || spsm_case == rocsparse::spsm_case::T_NT)
            {
                csrsm_buffer
                    = reinterpret_cast<char*>(temp_buffer)
                      + ((rocsparse::datatype_sizeof(matB->data_type) * matB->rows * matB->cols - 1)
                             / 256
                         + 1)
                            * 256;
            }

            switch(matA->format)
            {
            case rocsparse_format_csr:
            {
                if(matA->analysed == false)
                {
                    rocsparse_csrsm_info csrsm_info = matA->info->get_csrsm_info();
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::csrsm_analysis(handle,
                                                   trans_A,
                                                   trans_B,
                                                   matA->rows,
                                                   matC->cols,
                                                   matA->nnz,
                                                   alpha_datatype,
                                                   alpha,
                                                   matA->descr,
                                                   matA->data_type,
                                                   matA->const_val_data,
                                                   matA->row_type,
                                                   matA->const_row_data,
                                                   matA->col_type,
                                                   matA->const_col_data,
                                                   matC->data_type,
                                                   matC->values,
                                                   matC->ld,
                                                   matA->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   &csrsm_info,
                                                   csrsm_buffer)));

                    matA->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                if(matA->analysed == false)
                {
                    rocsparse_csrsm_info csrsm_info = matA->info->get_csrsm_info();
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coosm_analysis(handle,
                                                   trans_A,
                                                   trans_B,
                                                   matA->rows,
                                                   matC->cols,
                                                   matA->nnz,
                                                   alpha_datatype,
                                                   alpha,
                                                   matA->descr,
                                                   matA->data_type,
                                                   matA->const_val_data,
                                                   matA->row_type,
                                                   matA->const_row_data,
                                                   matA->col_type,
                                                   matA->const_col_data,
                                                   matC->data_type,
                                                   matC->values,
                                                   matC->ld,
                                                   matA->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   &csrsm_info,
                                                   csrsm_buffer)));
                    matA->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spsm_stage_compute:
        {
            switch(spsm_case)
            {
            case rocsparse::spsm_case::T_T:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spsm_solve_T_T(
                    handle, trans_A, trans_B, alpha, matA, matB, matC, alg, temp_buffer)));
                return rocsparse_status_success;
            }
            case rocsparse::spsm_case::T_NT:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spsm_solve_T_NT(
                    handle, trans_A, trans_B, alpha, matA, matB, matC, alg, temp_buffer)));
                return rocsparse_status_success;
            }
            case rocsparse::spsm_case::NT_T:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spsm_solve_NT_T(
                    handle, trans_A, trans_B, alpha, matA, matB, matC, alg, temp_buffer)));
                return rocsparse_status_success;
            }
            case rocsparse::spsm_case::NT_NT:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::spsm_solve_NT_NT(
                    handle, trans_A, trans_B, alpha, matA, matB, matC, alg, temp_buffer)));
                return rocsparse_status_success;
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spsm(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans_A, //1
                                           rocsparse_operation         trans_B, //2
                                           const void*                 alpha, //3
                                           rocsparse_const_spmat_descr matA, //4
                                           rocsparse_const_dnmat_descr matB, //5
                                           const rocsparse_dnmat_descr matC, //6
                                           rocsparse_datatype          compute_type, //7
                                           rocsparse_spsm_alg          alg, //8
                                           rocsparse_spsm_stage        stage, //9
                                           size_t*                     buffer_size, //10
                                           void*                       temp_buffer) //11
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, matA);
    ROCSPARSE_CHECKARG(4, matA, matA->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(5, matB);
    ROCSPARSE_CHECKARG(5, matB, matB->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(6, matC);
    ROCSPARSE_CHECKARG(6, matC, matC->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_ENUM(7, compute_type);
    ROCSPARSE_CHECKARG(7,
                       compute_type,
                       (compute_type != matA->data_type || compute_type != matB->data_type
                        || compute_type != matC->data_type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ENUM(8, alg);
    ROCSPARSE_CHECKARG_ENUM(9, stage);

    switch(stage)
    {
    case rocsparse_spsm_stage_buffer_size:
    {
        ROCSPARSE_CHECKARG_POINTER(10, buffer_size);
        break;
    }
    case rocsparse_spsm_stage_preprocess:
    {
        break;
    }
    case rocsparse_spsm_stage_compute:
    {
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsm(
        handle, trans_A, trans_B, alpha, matA, matB, matC, alg, stage, buffer_size, temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
