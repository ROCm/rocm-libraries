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

#include "internal/generic/rocsparse_spsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosv.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_determine_indextype.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spsv_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spsv_alg_default);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spsv_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spsv_stage_buffer_size);
        CASE(rocsparse_spsv_stage_preprocess);
        CASE(rocsparse_spsv_stage_compute);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsv_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsv_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsv_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsv_stage_buffer_size:
    case rocsparse_spsv_stage_preprocess:
    case rocsparse_spsv_stage_compute:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{
    static rocsparse_status spsv(rocsparse_handle            handle,
                                 rocsparse_operation         trans,
                                 const void*                 alpha,
                                 rocsparse_const_spmat_descr mat,
                                 rocsparse_const_dnvec_descr x,
                                 rocsparse_dnvec_descr       y,
                                 rocsparse_spsv_alg          alg,
                                 rocsparse_spsv_stage        stage,
                                 size_t*                     buffer_size,
                                 void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format format = mat->format;
        switch(format)
        {
        case rocsparse_format_csr:
        {
            switch(stage)
            {
            case rocsparse_spsv_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size(handle,
                                                                       trans,
                                                                       mat->rows,
                                                                       mat->nnz,
                                                                       mat->descr,
                                                                       mat->data_type,
                                                                       mat->const_val_data,
                                                                       mat->row_type,
                                                                       mat->const_row_data,
                                                                       mat->col_type,
                                                                       mat->const_col_data,
                                                                       mat->info,
                                                                       buffer_size));

                *buffer_size = rocsparse::max(static_cast<size_t>(4), *buffer_size);
                return rocsparse_status_success;
            }
            case rocsparse_spsv_stage_preprocess:
            {
                if(mat->analysed == false)
                {
                    rocsparse_csrsv_info csrsv_info = mat->info->get_csrsv_info();
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::csrsv_analysis(handle,
                                                   trans,
                                                   mat->rows,
                                                   mat->nnz,
                                                   mat->descr,
                                                   mat->data_type,
                                                   mat->const_val_data,
                                                   mat->row_type,
                                                   mat->const_row_data,
                                                   mat->col_type,
                                                   mat->const_col_data,
                                                   mat->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   &csrsv_info,
                                                   temp_buffer)));
                    mat->analysed = true;
                }

                return rocsparse_status_success;
            }
            case rocsparse_spsv_stage_compute:
            {
                const rocsparse_datatype datatype = mat->data_type;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                                 trans,
                                                                 mat->rows,
                                                                 mat->nnz,
                                                                 datatype,
                                                                 alpha,
                                                                 mat->descr,
                                                                 datatype,
                                                                 mat->const_val_data,
                                                                 mat->row_type,
                                                                 mat->const_row_data,
                                                                 mat->col_type,
                                                                 mat->const_col_data,
                                                                 mat->info,
                                                                 datatype,
                                                                 x->const_values,
                                                                 (int64_t)1,
                                                                 datatype,
                                                                 y->values,
                                                                 rocsparse_solve_policy_auto,
                                                                 mat->info->get_csrsv_info(),
                                                                 temp_buffer));
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

            switch(stage)
            {
            case rocsparse_spsv_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosv_buffer_size(handle,
                                                                       trans,
                                                                       mat->rows,
                                                                       mat->nnz,
                                                                       mat->descr,
                                                                       mat->data_type,
                                                                       mat->const_val_data,
                                                                       mat->row_type,
                                                                       mat->const_row_data,
                                                                       mat->col_type,
                                                                       mat->const_col_data,
                                                                       mat->info,
                                                                       buffer_size));
                *buffer_size = rocsparse::max(static_cast<size_t>(4), *buffer_size);
                return rocsparse_status_success;
            }
            case rocsparse_spsv_stage_preprocess:
            {
                if(mat->analysed == false)
                {
                    rocsparse_csrsv_info csrsv_info = mat->info->get_csrsv_info();
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coosv_analysis(handle,
                                                   trans,
                                                   mat->rows,
                                                   mat->nnz,
                                                   mat->descr,
                                                   mat->data_type,
                                                   mat->const_val_data,
                                                   mat->row_type,
                                                   mat->const_row_data,
                                                   mat->col_type,
                                                   mat->const_col_data,
                                                   mat->info,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   &csrsv_info,
                                                   temp_buffer)));
                    mat->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_spsv_stage_compute:
            {
                const rocsparse_datatype datatype = mat->data_type;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosv_solve(handle,
                                                                 trans,
                                                                 mat->rows,
                                                                 mat->nnz,
                                                                 datatype,
                                                                 alpha,
                                                                 mat->descr,
                                                                 datatype,
                                                                 mat->const_val_data,
                                                                 mat->row_type,
                                                                 mat->const_row_data,
                                                                 mat->col_type,
                                                                 mat->const_col_data,
                                                                 mat->info,
                                                                 datatype,
                                                                 x->const_values,
                                                                 datatype,
                                                                 y->values,
                                                                 rocsparse_solve_policy_auto,
                                                                 mat->info->get_csrsv_info(),
                                                                 temp_buffer));
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

extern "C" rocsparse_status rocsparse_spsv(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans, //1
                                           const void*                 alpha, //2
                                           rocsparse_const_spmat_descr mat, //3
                                           rocsparse_const_dnvec_descr x, //4
                                           const rocsparse_dnvec_descr y, //5
                                           rocsparse_datatype          compute_type, //6
                                           rocsparse_spsv_alg          alg, //7
                                           rocsparse_spsv_stage        stage, //8
                                           size_t*                     buffer_size, //9
                                           void*                       temp_buffer) // 10
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, alpha);
    ROCSPARSE_CHECKARG_POINTER(3, mat);
    ROCSPARSE_CHECKARG_POINTER(4, x);
    ROCSPARSE_CHECKARG_POINTER(5, y);
    ROCSPARSE_CHECKARG_ENUM(6, compute_type);
    ROCSPARSE_CHECKARG_ENUM(7, alg);
    ROCSPARSE_CHECKARG_ENUM(8, stage);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        ROCSPARSE_CHECKARG_POINTER(9, buffer_size);
    }

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    ROCSPARSE_CHECKARG(3, mat, (mat->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, x, (x->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(5, y, (y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(3, mat, (mat->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4, x, (x->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(5, y, (y->data_type != compute_type), rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spsv(handle, trans, alpha, mat, x, y, alg, stage, buffer_size, temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
