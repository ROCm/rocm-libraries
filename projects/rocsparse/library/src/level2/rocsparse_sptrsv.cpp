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

#include "../conversion/rocsparse_convert_array.hpp"
#include "../conversion/rocsparse_convert_scalar.hpp"
#include "internal/level2/rocsparse_csrsv.h"
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
    case rocsparse_sptrsv_input_analysis_policy:
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
                                                       rocsparse_sptrsv_descr sptrsv_descr,
                                                       rocsparse_sptrsv_input input,
                                                       const void*            data,
                                                       size_t                 data_size_in_bytes,
                                                       rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsv_descr);
    ROCSPARSE_CHECKARG_ENUM(2, input);
    ROCSPARSE_CHECKARG_POINTER(3, data);

    switch(input)
    {
    case rocsparse_sptrsv_input_alg:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsv_descr->get_stage() != ((rocsparse_sptrsv_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsv_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsv_stage was executed");

        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_sptrsv_alg),
                           rocsparse_status_invalid_size);

        const rocsparse_sptrsv_alg alg = *reinterpret_cast<const rocsparse_sptrsv_alg*>(data);
        sptrsv_descr->set_alg(alg);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_analysis_policy:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsv_descr->get_stage() != ((rocsparse_sptrsv_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsv_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsv_stage was executed");

        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_analysis_policy),
                           rocsparse_status_invalid_size);
        const auto analysis_policy = *reinterpret_cast<const rocsparse_analysis_policy*>(data);
        sptrsv_descr->set_analysis_policy(analysis_policy);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_scalar_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        sptrsv_descr->set_scalar_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_scalar_alpha:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(const void*),
                           rocsparse_status_invalid_size);
        sptrsv_descr->set_scalar_alpha(data);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_compute_datatype:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsv_descr->get_stage() != ((rocsparse_sptrsv_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsv_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsv_stage was executed");
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        sptrsv_descr->set_compute_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsv_input_operation:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsv_descr->get_stage() != ((rocsparse_sptrsv_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsv_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsv_stage was executed");

        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op = *reinterpret_cast<const rocsparse_operation*>(data);
        sptrsv_descr->set_operation(op);
        return rocsparse_status_success;
    }
        // LCOV_EXCL_START
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sptrsv_get_output(rocsparse_handle        handle,
                                                        rocsparse_sptrsv_descr  sptrsv_descr,
                                                        rocsparse_sptrsv_output output,
                                                        void*                   data,
                                                        size_t                  data_size_in_bytes,
                                                        rocsparse_error*        p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsv_descr);
    ROCSPARSE_CHECKARG_ENUM(2, output);
    ROCSPARSE_CHECKARG_POINTER(3, data);
    ROCSPARSE_CHECKARG(
        4, data_size_in_bytes, data_size_in_bytes == 0, rocsparse_status_invalid_size);

    switch(output)
    {
    case rocsparse_sptrsv_output_zero_pivot_position:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(int64_t),
                           rocsparse_status_invalid_size);

        auto csrsv_info = sptrsv_descr->get_csrsv_info();
        auto status
            = rocsparse::csrsv_zero_pivot(handle, csrsv_info, rocsparse_indextype_i64, data);
        if(status == rocsparse_status_zero_pivot)
        {
            return status;
        }
        RETURN_IF_ROCSPARSE_ERROR(status);

        return rocsparse_status_success;
    }
        // LCOV_EXCL_START
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    static rocsparse_status sptrsv_buffer_size(rocsparse_handle            handle,
                                               rocsparse_sptrsv_descr      sptrsv_descr,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_sptrsv_stage      sptrsv_stage,
                                               size_t*                     buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format    format    = A->format;
        const rocsparse_operation operation = sptrsv_descr->get_operation();
        switch(format)
        {
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size(handle,
                                                                   operation,
                                                                   A->rows,
                                                                   A->nnz,
                                                                   A->descr,
                                                                   A->data_type,
                                                                   A->const_val_data,
                                                                   A->row_type,
                                                                   A->const_row_data,
                                                                   A->col_type,
                                                                   A->const_col_data,
                                                                   A->info,
                                                                   buffer_size_in_bytes));

            return rocsparse_status_success;
        }

        case rocsparse_format_coo:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosv_buffer_size(handle,
                                                                   operation,
                                                                   A->rows,
                                                                   A->nnz,
                                                                   A->descr,
                                                                   A->data_type,
                                                                   A->const_val_data,
                                                                   A->row_type,
                                                                   A->const_row_data,
                                                                   A->col_type,
                                                                   A->const_col_data,
                                                                   A->info,
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
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    static rocsparse_status convert_scalars(rocsparse_handle             handle,
                                            const rocsparse_sptrsv_descr descr,
                                            const void*                  alpha,
                                            const void**                 local_alpha)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype scalar_datatype  = descr->get_scalar_datatype();
        const rocsparse_datatype compute_datatype = descr->get_compute_datatype();

        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((rocsparse::enum_utils::is_invalid(scalar_datatype))
                                                   ? rocsparse_status_invalid_value
                                                   : rocsparse_status_success,
                                               "invalid scalar datatype");

        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((rocsparse::enum_utils::is_invalid(compute_datatype))
                                                   ? rocsparse_status_invalid_value
                                                   : rocsparse_status_success,
                                               "invalid compute datatype");

        *local_alpha = alpha;
        if(scalar_datatype != compute_datatype)
        {
            // Convert scalars from scalar_datatype to compute_datatype.
            switch(handle->pointer_mode)
            {
            case rocsparse_pointer_mode_host:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_host_scalars(
                    scalar_datatype, compute_datatype, alpha, descr->get_local_host_alpha()));

                *local_alpha = descr->get_local_host_alpha();
                break;
            }
            case rocsparse_pointer_mode_device:
            {
                *local_alpha = handle->alpha;
                break;
            }
                // LCOV_EXCL_START
            }
            // LCOV_EXCL_STOP
        }

        return rocsparse_status_success;
    }

    static rocsparse_status sptrsv(rocsparse_handle            handle,
                                   rocsparse_sptrsv_descr      sptrsv_descr,
                                   rocsparse_const_spmat_descr A,
                                   rocsparse_const_dnvec_descr dnvec_descr_x,
                                   const rocsparse_dnvec_descr dnvec_descr_y,
                                   rocsparse_sptrsv_stage      sptrsv_stage,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format       format         = A->format;
        const rocsparse_operation    operation      = sptrsv_descr->get_operation();
        const rocsparse_sptrsv_stage previous_stage = sptrsv_descr->get_stage();
        const rocsparse_sptrsv_alg   alg            = sptrsv_descr->get_alg();

        ROCSPARSE_CHECKARG(1,
                           sptrsv_descr,
                           rocsparse::enum_utils::is_invalid(alg),
                           rocsparse_status_invalid_value);

        switch(sptrsv_stage)
        {
        case rocsparse_sptrsv_stage_analysis:
        {
            switch(previous_stage)
            {
            case rocsparse_sptrsv_stage_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_sptrsv_stage_analysis has already "
                    "been "
                    "executed");
                // LCOV_EXCL_START
            }
                // LCOV_EXCL_STOP

            case rocsparse_sptrsv_stage_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_sptrsv_stage_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_sptrsv_stage_compute");
                // LCOV_EXCL_START
            }
            }
            // LCOV_EXCL_STOP

            const rocsparse_analysis_policy analysis_policy = sptrsv_descr->get_analysis_policy();
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                (rocsparse::enum_utils::is_invalid(analysis_policy))
                    ? rocsparse_status_invalid_value
                    : rocsparse_status_success,
                "invalid analysis_policy");

            switch(format)
            {
            case rocsparse_format_csr:
            {
                rocsparse_csrsv_info csrsv_info{};
                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    sptrsv_descr->set_shared_csrsv_info(A->info->get_shared_csrsv_info());
                    csrsv_info = sptrsv_descr->get_csrsv_info();
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    csrsv_info = nullptr;
                    break;
                }
                }

                RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_analysis(handle,
                                                                     operation,
                                                                     A->rows,
                                                                     A->nnz,
                                                                     A->descr,
                                                                     A->data_type,
                                                                     A->const_val_data,
                                                                     A->row_type,
                                                                     A->const_row_data,
                                                                     A->col_type,
                                                                     A->const_col_data,
                                                                     A->info,
                                                                     analysis_policy,
                                                                     rocsparse_solve_policy_auto,
                                                                     &csrsv_info,
                                                                     buffer)));
                sptrsv_descr->set_stage(rocsparse_sptrsv_stage_analysis);
                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    sptrsv_descr->set_csrsv_info(csrsv_info);
                    break;
                }
                }

                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                rocsparse_csrsv_info csrsv_info{};
                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    sptrsv_descr->set_shared_csrsv_info(A->info->get_shared_csrsv_info());
                    csrsv_info = sptrsv_descr->get_csrsv_info();
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    csrsv_info = nullptr;
                    break;
                }
                }

                RETURN_IF_ROCSPARSE_ERROR((rocsparse::coosv_analysis(handle,
                                                                     operation,
                                                                     A->rows,
                                                                     A->nnz,
                                                                     A->descr,
                                                                     A->data_type,
                                                                     A->const_val_data,
                                                                     A->row_type,
                                                                     A->const_row_data,
                                                                     A->col_type,
                                                                     A->const_col_data,
                                                                     A->info,
                                                                     analysis_policy,
                                                                     rocsparse_solve_policy_auto,
                                                                     &csrsv_info,
                                                                     buffer)));

                switch(analysis_policy)
                {
                case rocsparse_analysis_policy_reuse:
                {
                    break;
                }
                case rocsparse_analysis_policy_force:
                {
                    sptrsv_descr->set_csrsv_info(csrsv_info);
                    break;
                }
                }

                sptrsv_descr->set_stage(rocsparse_sptrsv_stage_analysis);

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_coo_aos:
            {
                // LCOV_EXCL_START
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_STOP
        }
        case rocsparse_sptrsv_stage_compute:
        {

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                (previous_stage == ((rocsparse_sptrsv_stage)-1)) ? rocsparse_status_invalid_value
                                                                 : rocsparse_status_success,
                "invalid stage, the stage rocsparse_sptrsv_stage_analysis must be executed "
                "before "
                "the stage rocsparse_sptrsv_stage_compute");

            const void* alpha = sptrsv_descr->get_scalar_alpha();

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                (alpha == nullptr) ? rocsparse_status_invalid_pointer : rocsparse_status_success,
                "rocsparse_sptrsv_input_scalar_alpha must be set up.");

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_scalars(
                handle, sptrsv_descr, sptrsv_descr->get_scalar_alpha(), &alpha));

            const rocsparse_datatype alpha_datatype = sptrsv_descr->get_compute_datatype();

            switch(format)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                                 operation,
                                                                 A->rows,
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
                                                                 A->info,
                                                                 dnvec_descr_x->data_type,
                                                                 dnvec_descr_x->const_values,
                                                                 static_cast<int64_t>(1),
                                                                 dnvec_descr_y->data_type,
                                                                 dnvec_descr_y->values,
                                                                 rocsparse_solve_policy_auto,
                                                                 sptrsv_descr->get_csrsv_info(),
                                                                 buffer));
                sptrsv_descr->set_stage(rocsparse_sptrsv_stage_compute);
                return rocsparse_status_success;
            }
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosv_solve(handle,
                                                                 operation,
                                                                 A->rows,
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
                                                                 A->info,
                                                                 dnvec_descr_x->data_type,
                                                                 dnvec_descr_x->const_values,
                                                                 dnvec_descr_y->data_type,
                                                                 dnvec_descr_y->values,
                                                                 rocsparse_solve_policy_auto,
                                                                 sptrsv_descr->get_csrsv_info(),
                                                                 buffer));
                sptrsv_descr->set_stage(rocsparse_sptrsv_stage_compute);
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
            }
            }
        }
        }
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
                                                         rocsparse_const_spmat_descr A,
                                                         rocsparse_const_dnvec_descr x,
                                                         rocsparse_const_dnvec_descr y,
                                                         rocsparse_sptrsv_stage      sptrsv_stage,
                                                         size_t*          buffer_size_in_bytes,
                                                         rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsv_descr);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, x);
    ROCSPARSE_CHECKARG_POINTER(4, y);
    ROCSPARSE_CHECKARG_ENUM(5, sptrsv_stage);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size_in_bytes);
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::sptrsv_buffer_size(handle, sptrsv_descr, A, sptrsv_stage, buffer_size_in_bytes));

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
                                             rocsparse_const_spmat_descr A, // 2
                                             rocsparse_const_dnvec_descr x, // 3
                                             const rocsparse_dnvec_descr y, // 4
                                             rocsparse_sptrsv_stage      sptrsv_stage, // 5
                                             size_t                      buffer_size_in_bytes, // 6
                                             void*                       buffer, // 7
                                             rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsv_descr);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, x);
    ROCSPARSE_CHECKARG_POINTER(4, y);

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
    ROCSPARSE_CHECKARG(2, A, (A->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(3, x, (x->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, y, (y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(2,
                       A,
                       (A->data_type != sptrsv_descr->get_compute_datatype()),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       x,
                       (x->data_type != sptrsv_descr->get_compute_datatype()),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4,
                       y,
                       (y->data_type != sptrsv_descr->get_compute_datatype()),
                       rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsv(
        handle, sptrsv_descr, A, x, y, sptrsv_stage, buffer_size_in_bytes, buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
