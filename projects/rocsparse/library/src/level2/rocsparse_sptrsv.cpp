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

#include <map>
#include <sstream>

#include "internal/generic/rocsparse_sptrsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosv.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_sptrsv_descr.hpp"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsv_stage value)
{
    switch(value)
    {
    case rocsparse_sptrsv_stage_analysis:
    case rocsparse_sptrsv_stage_compute:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsv_alg value)
{
    switch(value)
    {
    case rocsparse_sptrsv_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsv_input value)
{
    switch(value)
    {
    case rocsparse_sptrsv_input_alg:
    case rocsparse_sptrsv_input_scalar_alpha:
    case rocsparse_sptrsv_input_operation:
    case rocsparse_sptrsv_input_scalar_datatype:
    case rocsparse_sptrsv_input_compute_datatype:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsv_output value)
{
    switch(value)
    {
    case rocsparse_sptrsv_output_zero_pivot_position:
    {
        return false;
    }
    }
    return true;
};

extern "C" rocsparse_status rocsparse_sptrsv_set_input(rocsparse_handle       handle,
                                                       rocsparse_sptrsv_descr descr,
                                                       rocsparse_sptrsv_input input,
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
    case rocsparse_sptrsv_input_alg:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_sptrsv_alg),
                           rocsparse_status_invalid_size);
        const rocsparse_sptrsv_alg alg = *reinterpret_cast<const rocsparse_sptrsv_alg*>(data);
        descr->set_alg(alg);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_scalar_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        descr->set_scalar_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_scalar_alpha:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(const void*),
                           rocsparse_status_invalid_size);
        descr->set_scalar_alpha(data);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_compute_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        descr->set_compute_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_operation:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op = *reinterpret_cast<const rocsparse_operation*>(data);
        descr->set_operation(op);
        return rocsparse_status_success;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_sptrsv_get_output(rocsparse_handle        handle,
                                                        rocsparse_sptrsv_descr  descr,
                                                        rocsparse_sptrsv_output output,
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
    case rocsparse_sptrsv_output_zero_pivot_position:
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
    static rocsparse_status sptrsv_buffer_size(rocsparse_handle            handle,
                                               rocsparse_sptrsv_descr      sptrsv_descr,
                                               rocsparse_const_spmat_descr spmat_descr,
                                               rocsparse_sptrsv_stage      sptrsv_stage,
                                               size_t*                     buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format    format    = spmat_descr->format;
        const rocsparse_operation operation = sptrsv_descr->get_operation();
        switch(format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size(handle,
                                                                   operation,
                                                                   spmat_descr->rows,
                                                                   spmat_descr->nnz,
                                                                   spmat_descr->descr,
                                                                   spmat_descr->data_type,
                                                                   spmat_descr->const_val_data,
                                                                   spmat_descr->row_type,
                                                                   spmat_descr->const_row_data,
                                                                   spmat_descr->col_type,
                                                                   spmat_descr->const_col_data,
                                                                   spmat_descr->info,
                                                                   buffer_size_in_bytes));

            return rocsparse_status_success;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosv_buffer_size(handle,
                                                                   operation,
                                                                   spmat_descr->rows,
                                                                   spmat_descr->nnz,
                                                                   spmat_descr->descr,
                                                                   spmat_descr->data_type,
                                                                   spmat_descr->const_val_data,
                                                                   spmat_descr->row_type,
                                                                   spmat_descr->const_row_data,
                                                                   spmat_descr->col_type,
                                                                   spmat_descr->const_col_data,
                                                                   spmat_descr->info,
                                                                   buffer_size_in_bytes));
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

    static rocsparse_status sptrsv(rocsparse_handle            handle,
                                   rocsparse_sptrsv_descr      sptrsv_descr,
                                   rocsparse_const_spmat_descr spmat_descr,
                                   rocsparse_const_dnvec_descr dnvec_descr_x,
                                   const rocsparse_dnvec_descr dnvec_descr_y,
                                   rocsparse_sptrsv_stage      sptrsv_stage,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format    format    = spmat_descr->format;
        const rocsparse_operation operation = sptrsv_descr->get_operation();
        const void*               alpha     = sptrsv_descr->get_scalar_alpha();
        switch(format)
        {
        case rocsparse_format_csr:
        {
            switch(sptrsv_stage)
            {
            case rocsparse_sptrsv_stage_analysis:
            {
                if(spmat_descr->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::csrsv_analysis(handle,
                                                   operation,
                                                   spmat_descr->rows,
                                                   spmat_descr->nnz,
                                                   spmat_descr->descr,
                                                   spmat_descr->data_type,
                                                   spmat_descr->const_val_data,
                                                   spmat_descr->row_type,
                                                   spmat_descr->const_row_data,
                                                   spmat_descr->col_type,
                                                   spmat_descr->const_col_data,
                                                   spmat_descr->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   buffer)));
                    spmat_descr->analysed = true;
                }

                return rocsparse_status_success;
            }
            case rocsparse_sptrsv_stage_compute:
            {
                const rocsparse_datatype datatype = spmat_descr->data_type;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                                 operation,
                                                                 spmat_descr->rows,
                                                                 spmat_descr->nnz,
                                                                 datatype,
                                                                 alpha,
                                                                 spmat_descr->descr,
                                                                 datatype,
                                                                 spmat_descr->const_val_data,
                                                                 spmat_descr->row_type,
                                                                 spmat_descr->const_row_data,
                                                                 spmat_descr->col_type,
                                                                 spmat_descr->const_col_data,
                                                                 spmat_descr->info,
                                                                 datatype,
                                                                 dnvec_descr_x->const_values,
                                                                 static_cast<int64_t>(1),
                                                                 datatype,
                                                                 dnvec_descr_y->values,
                                                                 rocsparse_solve_policy_auto,
                                                                 buffer));
                return rocsparse_status_success;
            }
            }

            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
            break;
        }
        case rocsparse_format_coo:
        {
            switch(sptrsv_stage)
            {
            case rocsparse_sptrsv_stage_analysis:
            {
                if(spmat_descr->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coosv_analysis(handle,
                                                   operation,
                                                   spmat_descr->rows,
                                                   spmat_descr->nnz,
                                                   spmat_descr->descr,
                                                   spmat_descr->data_type,
                                                   spmat_descr->const_val_data,
                                                   spmat_descr->row_type,
                                                   spmat_descr->const_row_data,
                                                   spmat_descr->col_type,
                                                   spmat_descr->const_col_data,
                                                   spmat_descr->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   buffer)));
                    spmat_descr->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_sptrsv_stage_compute:
            {
                const rocsparse_datatype datatype = spmat_descr->data_type;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosv_solve(handle,
                                                                 operation,
                                                                 spmat_descr->rows,
                                                                 spmat_descr->nnz,
                                                                 datatype,
                                                                 alpha,
                                                                 spmat_descr->descr,
                                                                 datatype,
                                                                 spmat_descr->const_val_data,
                                                                 spmat_descr->row_type,
                                                                 spmat_descr->const_row_data,
                                                                 spmat_descr->col_type,
                                                                 spmat_descr->const_col_data,
                                                                 spmat_descr->info,
                                                                 datatype,
                                                                 dnvec_descr_x->const_values,
                                                                 datatype,
                                                                 dnvec_descr_y->values,
                                                                 rocsparse_solve_policy_auto,
                                                                 buffer));
                return rocsparse_status_success;
            }
            }

            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
            break;
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

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_sptrsv_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_sptrsv_descr      sptrsv_descr,
                                                         rocsparse_const_spmat_descr spmat_descr,
                                                         rocsparse_const_dnvec_descr x,
                                                         rocsparse_const_dnvec_descr y,
                                                         rocsparse_sptrsv_stage      sptrsv_stage,
                                                         size_t*          buffer_size_in_bytes,
                                                         rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_HANDLE(1, sptrsv_descr);
    ROCSPARSE_CHECKARG_POINTER(2, spmat_descr);
    ROCSPARSE_CHECKARG_ENUM(3, sptrsv_stage);
    ROCSPARSE_CHECKARG_POINTER(4, buffer_size_in_bytes);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsv_buffer_size(
        handle, sptrsv_descr, spmat_descr, sptrsv_stage, buffer_size_in_bytes));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sptrsv(rocsparse_handle            handle, // 0
                                             rocsparse_sptrsv_descr      sptrsv_descr, // 1
                                             rocsparse_const_spmat_descr spmat_descr, // 2
                                             rocsparse_const_dnvec_descr dnvec_descr_x, // 3
                                             const rocsparse_dnvec_descr dnvec_descr_y, // 4
                                             rocsparse_sptrsv_stage      sptrsv_stage, // 5
                                             size_t                      buffer_size_in_bytes, // 6
                                             void*                       buffer, // 7
                                             rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsv_descr);
    ROCSPARSE_CHECKARG_POINTER(2, spmat_descr);
    ROCSPARSE_CHECKARG_POINTER(3, dnvec_descr_x);
    ROCSPARSE_CHECKARG_POINTER(4, dnvec_descr_y);

    ROCSPARSE_CHECKARG_ENUM(5, sptrsv_stage);

    ROCSPARSE_CHECKARG(6,
                       buffer_size_in_bytes,
                       (buffer_size_in_bytes == 0) && (buffer != nullptr),
                       rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG(7,
                       buffer,
                       (buffer == nullptr) && (buffer_size_in_bytes != 0),
                       rocsparse_status_invalid_pointer);

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    ROCSPARSE_CHECKARG(
        2, spmat_descr, (spmat_descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(
        3, dnvec_descr_x, (dnvec_descr_x->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(
        4, dnvec_descr_y, (dnvec_descr_y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(2,
                       spmat_descr,
                       (spmat_descr->data_type != sptrsv_descr->get_scalar_datatype()),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       dnvec_descr_x,
                       (dnvec_descr_x->data_type != sptrsv_descr->get_scalar_datatype()),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4,
                       dnvec_descr_y,
                       (dnvec_descr_y->data_type != sptrsv_descr->get_scalar_datatype()),
                       rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsv(handle,
                                                sptrsv_descr,
                                                spmat_descr,
                                                dnvec_descr_x,
                                                dnvec_descr_y,
                                                sptrsv_stage,
                                                buffer_size_in_bytes,
                                                buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
