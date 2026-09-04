/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_utility.hpp"

#include "../rocsparse_sptrsm_descr.hpp"
#include "rocsparse_coosm.hpp"
#include "rocsparse_cscsm.hpp"
#include "rocsparse_csrsm.hpp"

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
    case rocsparse_sptrsm_input_compute_datatype:
    case rocsparse_sptrsm_input_scalar_datatype:
    case rocsparse_sptrsm_input_scalar_alpha:
    case rocsparse_sptrsm_input_analysis_policy:
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
    case rocsparse_sptrsm_output_singularity_position:
    case rocsparse_sptrsm_output_singularity:
    {
        return false;
    }
    }
    return true;
};

namespace rocsparse
{
    static rocsparse_status sptrsm_buffer_size(rocsparse_handle            handle,
                                               rocsparse_sptrsm_descr      sptrsm_descr,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_const_dnmat_descr X,
                                               rocsparse_const_dnmat_descr Y,
                                               rocsparse_sptrsm_stage      sptrsm_stage,
                                               size_t*                     p_buffer_size_in_bytes)
    {

        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_operation A_op = sptrsm_descr->get_operation_A();
        const rocsparse_operation X_op = sptrsm_descr->get_operation_X();

        const bool             is_Y_column_order = (Y->order == rocsparse_order_column);
        const rocsparse_format A_format          = A->format;

        _rocsparse_dnvec_descr alpha_st(1,
                                        1,
                                        sptrsm_descr->get_compute_datatype(),
                                        sptrsm_descr->get_scalar_alpha(),
                                        nullptr,
                                        1,
                                        0);

        _rocsparse_dnmat_descr Z_st{true,
                                    Y->rows,
                                    Y->cols,
                                    Y->cols,
                                    (void*)0x4,
                                    (const void*)0x4,
                                    Y->data_type,
                                    rocsparse_order_row,
                                    Y->batch_count,
                                    Y->rows * Y->cols};

        rocsparse_dnvec_descr alpha = &alpha_st;
        rocsparse_dnmat_descr Z     = &Z_st;

        const size_t Z_data_size_in_nbytes = rocsparse::align_size(
            rocsparse::datatype_sizeof(Z->data_type) * Z->rows * Z->cols * Z->batch_count);

        switch(A_format)
        {
        case rocsparse_format_csr:
        {
            switch(sptrsm_stage)
            {
            case rocsparse_sptrsm_stage_analysis:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis_buffer_size(
                    handle, Y->cols, A_op, X_op, alpha, A, Z, p_buffer_size_in_bytes, nullptr));
                return rocsparse_status_success;
            }

            case rocsparse_sptrsm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve_buffer_size(
                    handle, Y->cols, A_op, X_op, alpha, A, Z, p_buffer_size_in_bytes, nullptr));

                if(is_Y_column_order)
                {
                    p_buffer_size_in_bytes[0] += Z_data_size_in_nbytes;
                }
                return rocsparse_status_success;
            }
            }

            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        case rocsparse_format_coo:
        {
            switch(sptrsm_stage)
            {
            case rocsparse_sptrsm_stage_analysis:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_analysis_buffer_size(
                    handle, Y->cols, A_op, X_op, alpha, A, Z, p_buffer_size_in_bytes, nullptr));
                return rocsparse_status_success;
            }

            case rocsparse_sptrsm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve_buffer_size(
                    handle, Y->cols, A_op, X_op, alpha, A, Z, p_buffer_size_in_bytes, nullptr));
                if(is_Y_column_order)
                {
                    p_buffer_size_in_bytes[0] += Z_data_size_in_nbytes;
                }
                return rocsparse_status_success;
            }
            }

            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        case rocsparse_format_csc:
        {
            switch(sptrsm_stage)
            {
            case rocsparse_sptrsm_stage_analysis:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_analysis_buffer_size(
                    handle, Y->cols, A_op, X_op, alpha, A, Z, p_buffer_size_in_bytes, nullptr));
                return rocsparse_status_success;
            }

            case rocsparse_sptrsm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_solve_buffer_size(
                    handle, Y->cols, A_op, X_op, alpha, A, Z, p_buffer_size_in_bytes, nullptr));
                if(is_Y_column_order)
                {
                    p_buffer_size_in_bytes[0] += Z_data_size_in_nbytes;
                }
                return rocsparse_status_success;
            }
            }

            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_sell:
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

    switch(sptrsm_stage)
    {
    case rocsparse_sptrsm_stage_analysis:
    {
        //
        // Let's record X order and X datatype.
        //
        sptrsm_descr->set_X_datatype(X->data_type);
        sptrsm_descr->set_X_order(X->order);
        sptrsm_descr->set_Y_datatype(Y->data_type);
        sptrsm_descr->set_Y_order(Y->order);
        sptrsm_descr->set_nrhs(Y->cols);
        break;
    }

    case rocsparse_sptrsm_stage_compute:
    {
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsm_buffer_size(
        handle, sptrsm_descr, A, X, Y, sptrsm_stage, buffer_size_in_bytes));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
