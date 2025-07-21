/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_sptrsm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_common.h"
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_sptrsm_descr.hpp"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsm_stage value)
{
    switch(value)
    {
    case rocsparse_sptrsm_stage_analysis:
    case rocsparse_sptrsm_stage_compute:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsm_alg value)
{
    switch(value)
    {
    case rocsparse_sptrsm_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsm_input value)
{
    switch(value)
    {
    case rocsparse_sptrsm_input_alg:
    case rocsparse_sptrsm_input_operation_A:
    case rocsparse_sptrsm_input_operation_X:
    case rocsparse_sptrsm_input_scalar_datatype:
    case rocsparse_sptrsm_input_scalar_alpha:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsm_output value)
{
    switch(value)
    {
    case rocsparse_sptrsm_output_zero_pivot_position:
    {
        return false;
    }
    }
    return true;
};

extern "C" rocsparse_status rocsparse_sptrsm_set_input(rocsparse_handle       handle,
                                                       rocsparse_sptrsm_descr descr,
                                                       rocsparse_sptrsm_input input,
                                                       const void*            data,
                                                       size_t                 data_size_in_bytes,
                                                       rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, input);
    ROCSPARSE_CHECKARG_POINTER(3, data);

    switch(input)
    {
    case rocsparse_sptrsm_input_alg:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_sptrsm_alg),
                           rocsparse_status_invalid_size);
        const rocsparse_sptrsm_alg alg = *reinterpret_cast<const rocsparse_sptrsm_alg*>(data);
        descr->set_alg(alg);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_scalar_alpha:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(const void*),
                           rocsparse_status_invalid_size);
        descr->set_scalar_alpha(data);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_scalar_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        descr->set_scalar_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_operation_A:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op = *reinterpret_cast<const rocsparse_operation*>(data);
        descr->set_operation_A(op);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_operation_X:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op = *reinterpret_cast<const rocsparse_operation*>(data);
        descr->set_operation_X(op);
        return rocsparse_status_success;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_sptrsm_get_output(rocsparse_handle        handle,
                                                        rocsparse_sptrsm_descr  descr,
                                                        rocsparse_sptrsm_output output,
                                                        void*                   data,
                                                        size_t                  data_size_in_bytes,
                                                        rocsparse_error*        p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, output);
    ROCSPARSE_CHECKARG_POINTER(3, data);
    ROCSPARSE_CHECKARG(
        4, data_size_in_bytes, data_size_in_bytes == 0, rocsparse_status_invalid_size);

    switch(output)
    {
    case rocsparse_sptrsm_output_zero_pivot_position:
    {
        if(sizeof(int32_t) == data_size_in_bytes)
        {
            const int64_t zero_pivot_position = descr->get_zero_pivot_position();
            *reinterpret_cast<int32_t*>(data) = zero_pivot_position;
        }
        else if(sizeof(int64_t) == data_size_in_bytes)
        {
            *reinterpret_cast<int64_t*>(data) = descr->get_zero_pivot_position();
        }
        else
        {
        }
        return rocsparse_status_success;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

namespace rocsparse
{
    enum class sptrsm_case
    {
        NT_NT,
        T_NT,
        NT_T,
        T_T
    };

    static sptrsm_case sptrsm_get_case(rocsparse_operation X_operation,
                                       rocsparse_order     order_B,
                                       rocsparse_order     order_C)
    {
        const bool B_is_transposed
            = (((X_operation == rocsparse_operation_none) && (order_B == rocsparse_order_row))
               || ((X_operation != rocsparse_operation_none)
                   && (order_B == rocsparse_order_column)));
        const bool C_is_transposed = (order_C == rocsparse_order_row);

        if(B_is_transposed && C_is_transposed)
        {
            // 1) B col order + transposed and C row order
            // 2) B row order + non-transposed and C row order
            return sptrsm_case::T_T;
        }
        else if(B_is_transposed && !C_is_transposed)
        {
            // 1) B col order + transposed and C col order
            // 2) B row order + non-transposed and C col order
            return sptrsm_case::T_NT;
        }
        else if(!B_is_transposed && C_is_transposed)
        {
            // 1) B row order + transposed and C row order
            // 2) B col order + non-transposed and C row order
            return sptrsm_case::NT_T;
        }
        else
        {
            // 1) B row order + transposed and C col order
            // 2) B col order + non-transposed and C col order
            return sptrsm_case::NT_NT;
        }
    }

    static rocsparse_status sptrsm_solve_T_T(rocsparse_handle            handle,
                                             rocsparse_operation         operation,
                                             rocsparse_operation         X_operation,
                                             const void*                 alpha,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_dnmat_descr X,
                                             const rocsparse_dnmat_descr Y,
                                             rocsparse_sptrsm_alg        alg,
                                             void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype alpha_datatype = A->data_type;
        // 1) B col order + transposed and C row order
        // 2) B row order + non-transposed and C row order
        void* csrsm_buffer = buffer;
        if(X->rows > 0 && X->cols > 0)
        {
            const size_t sizeof_datatype = rocsparse::datatype_sizeof(Y->data_type);
            if(X->order == rocsparse_order_column)
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(Y->values,
                                                     sizeof_datatype * Y->ld,
                                                     X->const_values,
                                                     sizeof_datatype * X->ld,
                                                     sizeof_datatype * X->rows,
                                                     X->cols,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
            else
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(Y->values,
                                                     sizeof_datatype * Y->ld,
                                                     X->const_values,
                                                     sizeof_datatype * X->ld,
                                                     sizeof_datatype * X->cols,
                                                     X->rows,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
        }

        switch(A->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             Y->values,
                                                             Y->ld,
                                                             Y->order,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             Y->values,
                                                             Y->ld,
                                                             Y->order,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
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

    static rocsparse_status sptrsm_solve_T_NT(rocsparse_handle            handle,
                                              rocsparse_operation         operation,
                                              rocsparse_operation         X_operation,
                                              const void*                 alpha,
                                              rocsparse_const_spmat_descr A,
                                              rocsparse_const_dnmat_descr X,
                                              const rocsparse_dnmat_descr Y,
                                              rocsparse_sptrsm_alg        alg,
                                              void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype alpha_datatype = A->data_type;

        const size_t sizeof_datatype = rocsparse::datatype_sizeof(X->data_type);

        // 1) B col order + transposed and C col order
        // 2) B row order + non-transposed and C col order
        void* sptrsm_buffer = buffer;
        void* csrsm_buffer
            = ((char*)buffer) + ((sizeof_datatype * X->rows * X->cols - 1) / 256 + 1) * 256;

        if(X->rows > 0 && X->cols > 0)
        {
            if(X->order == rocsparse_order_column)
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(sptrsm_buffer,
                                                     sizeof_datatype * X->rows,
                                                     X->const_values,
                                                     sizeof_datatype * X->ld,
                                                     sizeof_datatype * X->rows,
                                                     X->cols,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
            else
            {
                RETURN_IF_HIP_ERROR(hipMemcpy2DAsync(sptrsm_buffer,
                                                     sizeof_datatype * X->cols,
                                                     X->const_values,
                                                     sizeof_datatype * X->ld,
                                                     sizeof_datatype * X->cols,
                                                     X->rows,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
            }
        }

        switch(A->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             sptrsm_buffer,
                                                             Y->cols,
                                                             rocsparse_order_row,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             sptrsm_buffer,
                                                             Y->cols,
                                                             rocsparse_order_row,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
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

        if(X->rows > 0 && X->cols > 0)
        {
            if(X->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->rows,
                                                                     X->cols,
                                                                     X->data_type,
                                                                     sptrsm_buffer,
                                                                     X->rows,
                                                                     Y->data_type,
                                                                     Y->values,
                                                                     Y->ld));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->cols,
                                                                     X->rows,
                                                                     X->data_type,
                                                                     sptrsm_buffer,
                                                                     X->cols,
                                                                     Y->data_type,
                                                                     Y->values,
                                                                     Y->ld));
            }
        }

        return rocsparse_status_success;
    }

    static rocsparse_status sptrsm_solve_NT_T(rocsparse_handle            handle,
                                              rocsparse_operation         operation,
                                              rocsparse_operation         X_operation,
                                              const void*                 alpha,
                                              rocsparse_const_spmat_descr A,
                                              rocsparse_const_dnmat_descr X,
                                              const rocsparse_dnmat_descr Y,
                                              rocsparse_sptrsm_alg        alg,
                                              void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_datatype alpha_datatype = A->data_type;
        // 1) B row order + transposed and C row order
        // 2) B col order + non-transposed and C row order
        void* csrsm_buffer = buffer;
        if(X->rows > 0 && X->cols > 0)
        {
            if(X->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->rows,
                                                                     X->cols,
                                                                     X->data_type,
                                                                     X->const_values,
                                                                     X->ld,
                                                                     Y->data_type,
                                                                     Y->values,
                                                                     Y->ld));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->cols,
                                                                     X->rows,
                                                                     X->data_type,
                                                                     X->const_values,
                                                                     X->ld,
                                                                     Y->data_type,
                                                                     Y->values,
                                                                     Y->ld));
            }
        }

        switch(A->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             Y->values,
                                                             Y->ld,
                                                             Y->order,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             Y->values,
                                                             Y->ld,
                                                             Y->order,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
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

    static rocsparse_status sptrsm_solve_NT_NT(rocsparse_handle            handle,
                                               rocsparse_operation         operation,
                                               rocsparse_operation         X_operation,
                                               const void*                 alpha,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_const_dnmat_descr X,
                                               const rocsparse_dnmat_descr Y,
                                               rocsparse_sptrsm_alg        alg,
                                               void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype alpha_datatype = A->data_type;

        // 1) B row order + transposed and C col order
        // 2) B col order + non-transposed and C col order
        void* sptrsm_buffer = buffer;
        void* csrsm_buffer
            = ((char*)buffer)
              + ((rocsparse::datatype_sizeof(X->data_type) * X->rows * X->cols - 1) / 256 + 1)
                    * 256;

        if(X->rows > 0 && X->cols > 0)
        {
            if(X->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->rows,
                                                                     X->cols,
                                                                     X->data_type,
                                                                     X->const_values,
                                                                     X->ld,
                                                                     X->data_type,
                                                                     sptrsm_buffer,
                                                                     X->cols));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->cols,
                                                                     X->rows,
                                                                     X->data_type,
                                                                     X->const_values,
                                                                     X->ld,
                                                                     X->data_type,
                                                                     sptrsm_buffer,
                                                                     X->rows));
            }
        }

        switch(A->format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             sptrsm_buffer,
                                                             Y->cols,
                                                             rocsparse_order_row,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
                                                             csrsm_buffer));
            break;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve(handle,
                                                             operation,
                                                             X_operation,
                                                             A->rows,
                                                             Y->cols,
                                                             A->nnz,
                                                             alpha_datatype,
                                                             alpha,
                                                             A->descr,
                                                             A->data_type,
                                                             A->const_val_data,
                                                             A->row_type,
                                                             A->const_row_data,
                                                             A->col_type,
                                                             A->const_col_data,
                                                             Y->data_type,
                                                             sptrsm_buffer,
                                                             Y->cols,
                                                             rocsparse_order_row,
                                                             A->info,
                                                             rocsparse_solve_policy_auto,
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

        if(X->rows > 0 && X->cols > 0)
        {
            if(X->order == rocsparse_order_column)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->cols,
                                                                     X->rows,
                                                                     X->data_type,
                                                                     sptrsm_buffer,
                                                                     X->cols,
                                                                     Y->data_type,
                                                                     Y->values,
                                                                     Y->ld));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                                     X->rows,
                                                                     X->cols,
                                                                     X->data_type,
                                                                     sptrsm_buffer,
                                                                     X->rows,
                                                                     Y->data_type,
                                                                     Y->values,
                                                                     Y->ld));
            }
        }

        return rocsparse_status_success;
    }

    static rocsparse_status sptrsm_buffer_size(rocsparse_handle            handle,
                                               rocsparse_sptrsm_descr      sptrsm_descr,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_sptrsm_stage      sptrsm_stage,
                                               size_t*                     buffer_size_in_bytes)
    {

        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_operation operation_X = sptrsm_descr->get_operation_X();

        const rocsparse::sptrsm_case sptrsm_case = sptrsm_get_case(
            operation_X, sptrsm_descr->get_X_order(), sptrsm_descr->get_Y_order());

        const rocsparse_operation operation      = sptrsm_descr->get_operation_A();
        const rocsparse_datatype  alpha_datatype = sptrsm_descr->get_scalar_datatype();
        const rocsparse_format    format         = A->format;
        const int64_t             nrhs           = sptrsm_descr->get_nrhs();
        const int64_t             n              = A->rows;

        switch(format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_buffer_size(handle,
                                                                   operation,
                                                                   operation_X,
                                                                   A->rows,
                                                                   nrhs,
                                                                   A->nnz,
                                                                   alpha_datatype,
                                                                   A->descr,
                                                                   A->data_type,
                                                                   A->const_val_data,
                                                                   A->row_type,
                                                                   A->const_row_data,
                                                                   A->col_type,
                                                                   A->const_col_data,
                                                                   sptrsm_descr->get_Y_datatype(),
                                                                   sptrsm_descr->get_Y_order(),
                                                                   A->info,
                                                                   rocsparse_solve_policy_auto,
                                                                   buffer_size_in_bytes));

            switch(sptrsm_case)
            {
            case rocsparse::sptrsm_case::NT_NT:
            case rocsparse::sptrsm_case::T_NT:
            {
                *buffer_size_in_bytes
                    += ((rocsparse::datatype_sizeof(sptrsm_descr->get_X_datatype()) * nrhs * n - 1)
                            / 256
                        + 1)
                       * 256;
                break;
            }
            case rocsparse::sptrsm_case::T_T:
            case rocsparse::sptrsm_case::NT_T:
            {
                break;
            }
            }
            return rocsparse_status_success;
        }

        case rocsparse_format_coo:
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_buffer_size(handle,
                                                                   operation,
                                                                   operation_X,
                                                                   A->rows,
                                                                   nrhs,
                                                                   A->nnz,
                                                                   alpha_datatype,
                                                                   A->descr,
                                                                   A->data_type,
                                                                   A->const_val_data,
                                                                   A->row_type,
                                                                   A->const_row_data,
                                                                   A->col_type,
                                                                   A->const_col_data,
                                                                   sptrsm_descr->get_Y_datatype(),
                                                                   sptrsm_descr->get_Y_order(),
                                                                   A->info,
                                                                   rocsparse_solve_policy_auto,
                                                                   buffer_size_in_bytes));

            switch(sptrsm_case)
            {
            case rocsparse::sptrsm_case::NT_NT:
            case rocsparse::sptrsm_case::T_NT:
            {
                *buffer_size_in_bytes
                    += ((rocsparse::datatype_sizeof(sptrsm_descr->get_X_datatype()) * nrhs * n - 1)
                            / 256
                        + 1)
                       * 256;
                break;
            }
            case rocsparse::sptrsm_case::T_T:
            case rocsparse::sptrsm_case::NT_T:
            {
                break;
            }
            }
            return rocsparse_status_success;
        }

        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_coo_aos:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    static rocsparse_status sptrsm(rocsparse_handle            handle,
                                   rocsparse_sptrsm_descr      sptrsm_descr,
                                   const void*                 alpha,
                                   rocsparse_const_spmat_descr A,
                                   rocsparse_const_dnmat_descr X,
                                   const rocsparse_dnmat_descr Y,
                                   rocsparse_sptrsm_stage      sptrsm_stage,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_operation    operation      = sptrsm_descr->get_operation_A();
        const rocsparse_operation    X_operation    = sptrsm_descr->get_operation_X();
        const rocsparse_datatype     alpha_datatype = A->data_type;
        const rocsparse::sptrsm_case sptrsm_case = sptrsm_get_case(X_operation, X->order, Y->order);

        switch(sptrsm_stage)
        {
        case rocsparse_sptrsm_stage_analysis:
        {
            void* csrsm_buffer = buffer;

            switch(sptrsm_case)
            {
            case rocsparse::sptrsm_case::NT_NT:
            case rocsparse::sptrsm_case::T_NT:
            {
                csrsm_buffer
                    = reinterpret_cast<char*>(buffer)
                      + ((rocsparse::datatype_sizeof(X->data_type) * X->rows * X->cols - 1) / 256
                         + 1)
                            * 256;
                break;
            }
            case rocsparse::sptrsm_case::T_T:
            case rocsparse::sptrsm_case::NT_T:
            {
                break;
            }
            }

            switch(A->format)
            {
            case rocsparse_format_csr:
            {
                if(A->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::csrsm_analysis(handle,
                                                   operation,
                                                   X_operation,
                                                   A->rows,
                                                   Y->cols,
                                                   A->nnz,
                                                   alpha_datatype,
                                                   alpha,
                                                   A->descr,
                                                   A->data_type,
                                                   A->const_val_data,
                                                   A->row_type,
                                                   A->const_row_data,
                                                   A->col_type,
                                                   A->const_col_data,
                                                   Y->data_type,
                                                   Y->values,
                                                   Y->ld,
                                                   A->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   csrsm_buffer)));

                    A->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                if(A->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coosm_analysis(handle,
                                                   operation,
                                                   X_operation,
                                                   A->rows,
                                                   Y->cols,
                                                   A->nnz,
                                                   alpha_datatype,
                                                   alpha,
                                                   A->descr,
                                                   A->data_type,
                                                   A->const_val_data,
                                                   A->row_type,
                                                   A->const_row_data,
                                                   A->col_type,
                                                   A->const_col_data,
                                                   Y->data_type,
                                                   Y->values,
                                                   Y->ld,
                                                   A->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   csrsm_buffer)));
                    A->analysed = true;
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

        case rocsparse_sptrsm_stage_compute:
        {
            switch(sptrsm_case)
            {
            case rocsparse::sptrsm_case::T_T:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::sptrsm_solve_T_T(handle,
                                                                       operation,
                                                                       X_operation,
                                                                       alpha,
                                                                       A,
                                                                       X,
                                                                       Y,
                                                                       sptrsm_descr->get_alg(),
                                                                       buffer)));
                return rocsparse_status_success;
            }
            case rocsparse::sptrsm_case::T_NT:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::sptrsm_solve_T_NT(handle,
                                                                        operation,
                                                                        X_operation,
                                                                        alpha,
                                                                        A,
                                                                        X,
                                                                        Y,
                                                                        sptrsm_descr->get_alg(),
                                                                        buffer)));
                return rocsparse_status_success;
            }
            case rocsparse::sptrsm_case::NT_T:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::sptrsm_solve_NT_T(handle,
                                                                        operation,
                                                                        X_operation,
                                                                        alpha,
                                                                        A,
                                                                        X,
                                                                        Y,
                                                                        sptrsm_descr->get_alg(),
                                                                        buffer)));
                return rocsparse_status_success;
            }
            case rocsparse::sptrsm_case::NT_NT:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::sptrsm_solve_NT_NT(handle,
                                                                         operation,
                                                                         X_operation,
                                                                         alpha,
                                                                         A,
                                                                         X,
                                                                         Y,
                                                                         sptrsm_descr->get_alg(),
                                                                         buffer)));
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
extern "C" rocsparse_status rocsparse_sptrsm_buffer_size(rocsparse_handle       handle, // 0
                                                         rocsparse_sptrsm_descr sptrsm_descr, // 1
                                                         rocsparse_const_spmat_descr A, // 2
                                                         rocsparse_const_dnmat_descr X, // 3
                                                         rocsparse_const_dnmat_descr Y, // 4
                                                         rocsparse_sptrsm_stage sptrsm_stage, // 5
                                                         size_t*          buffer_size_in_bytes, // 6
                                                         rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsm_descr);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, X);
    ROCSPARSE_CHECKARG_POINTER(4, Y);
    ROCSPARSE_CHECKARG_ENUM(5, sptrsm_stage);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size_in_bytes);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::sptrsm_buffer_size(handle, sptrsm_descr, A, sptrsm_stage, buffer_size_in_bytes));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sptrsm(rocsparse_handle            handle, // 0
                                             rocsparse_sptrsm_descr      sptrsm_descr, // 1
                                             const void*                 alpha, // 2
                                             rocsparse_const_spmat_descr A, // 3
                                             rocsparse_const_dnmat_descr X, // 4
                                             rocsparse_dnmat_descr       Y, // 5
                                             rocsparse_sptrsm_stage      sptrsm_stage, // 6
                                             size_t                      buffer_size_in_bytes, // 7
                                             void*                       buffer, // 8
                                             rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsm_descr);
    ROCSPARSE_CHECKARG_POINTER(2, alpha);
    ROCSPARSE_CHECKARG_POINTER(3, A);
    ROCSPARSE_CHECKARG_POINTER(4, X);
    ROCSPARSE_CHECKARG_POINTER(5, Y);

    ROCSPARSE_CHECKARG_ENUM(6, sptrsm_stage);

    ROCSPARSE_CHECKARG(7,
                       buffer_size_in_bytes,
                       (buffer_size_in_bytes == 0) && (buffer != nullptr),
                       rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG(8,
                       buffer,
                       (buffer == nullptr) && (buffer_size_in_bytes == 0),
                       rocsparse_status_invalid_pointer);

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    ROCSPARSE_CHECKARG(3, A, (A->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, X, (X->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(5, Y, (Y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    const rocsparse_datatype compute_type = sptrsm_descr->get_scalar_datatype();

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(3, A, (A->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4, X, (X->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(5, Y, (Y->data_type != compute_type), rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsm(
        handle, sptrsm_descr, alpha, A, X, Y, sptrsm_stage, buffer_size_in_bytes, buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
