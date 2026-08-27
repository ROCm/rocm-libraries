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

#include "../../conversion/rocsparse_convert_array.hpp"
#include "../../conversion/rocsparse_convert_scalar.hpp"
#include "rocsparse_common.h"
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"

#ifdef ROCSPARSE_WITH_CSC_TRSM
#include "rocsparse_cscsm.hpp"
#endif

#include "../rocsparse_sptrsm_descr.hpp"

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

namespace rocsparse
{

    static rocsparse_status sptrsm_convert_scalars(rocsparse_handle             handle,
                                                   const rocsparse_sptrsm_descr descr,
                                                   const void*                  alpha,
                                                   const void**                 local_alpha)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype scalar_datatype  = descr->get_scalar_datatype();
        const rocsparse_datatype compute_datatype = descr->get_compute_datatype();

        *local_alpha = alpha;
        if(scalar_datatype != compute_datatype)
        {
            // Convert scalars from scalar_datatype to compute_datatype
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
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_device_scalars(
                    handle->stream, scalar_datatype, compute_datatype, alpha, handle->alpha));

                *local_alpha = handle->alpha;
                break;
            }
            }
        }

        return rocsparse_status_success;
    }

    static rocsparse_status sptrsm_analysis(rocsparse_handle            handle,
                                            rocsparse_sptrsm_descr      sptrsm_descr,
                                            rocsparse_dnvec_descr       alpha,
                                            rocsparse_const_spmat_descr A,
                                            rocsparse_const_dnmat_descr X,
                                            rocsparse_const_dnmat_descr Y,
                                            rocsparse_sptrsm_stage      sptrsm_stage,
                                            size_t                      buffer_size_in_bytes,
                                            void*                       buffer,
                                            rocsparse_error*            p_error)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_operation       A_op            = sptrsm_descr->get_operation_A();
        const rocsparse_operation       X_op            = sptrsm_descr->get_operation_X();
        const rocsparse_analysis_policy analysis_policy = sptrsm_descr->get_analysis_policy();
        if(rocsparse::enum_utils::is_invalid(analysis_policy))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   "invalid analysis_policy");
        }

        //    const rocsparse_datatype  alpha_datatype = sptrsm_descr->get_compute_datatype();

        const auto nrhs = Y->cols;
        //
        // Preprocess.
        //
        _rocsparse_dnmat_descr local_Z{true,
                                       A->rows,
                                       nrhs,
                                       nrhs,
                                       (void*)0x4,
                                       (const void*)0x4,
                                       X->data_type,
                                       rocsparse_order_row,
                                       X->batch_count,
                                       (X->batch_stride == 0) ? 0 : (A->rows * Y->cols)};

        const bool is_Y_column_order = (Y->order == rocsparse_order_column);

        rocsparse_const_dnmat_descr Z = (is_Y_column_order) ? &local_Z : Y;

        //
        // Check this is called Only once and before the compute phase.
        //
        const rocsparse_sptrsm_stage previous_stage             = sptrsm_descr->get_stage();
        auto                         csrsm_buffer               = reinterpret_cast<char*>(buffer);
        size_t                       csrsm_buffer_size_in_bytes = buffer_size_in_bytes;
        rocsparse_csrsm_info         csrsm_info{};

        switch(previous_stage)
        {
        case rocsparse_sptrsm_stage_analysis:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_sptrsm_stage_analysis has already "
                "been "
                "executed");
            break;
        }

        case rocsparse_sptrsm_stage_compute:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_sptrsm_stage_analysis cannot be "
                "called "
                "after "
                "the stage rocsparse_sptrsm_stage_compute");
            break;
        }
        }

        //
        // Grab csrsm_info.
        //
        switch(analysis_policy)
        {
        case rocsparse_analysis_policy_reuse:
        {
            sptrsm_descr->set_shared_csrsm_info(A->info->get_shared_csrsm_info());
            csrsm_info = sptrsm_descr->get_csrsm_info();
            break;
        }
        case rocsparse_analysis_policy_force:
        {
            csrsm_info = nullptr;
            break;
        }
        }

        //
        // Call the corresponding analysis.
        //
        switch(A->format)
        {
#ifndef ROCSPARSE_WITH_CSC_TRSM
        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

#else
        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_analysis(handle,
                                                                nrhs,
                                                                A_op,
                                                                X_op,
                                                                alpha,
                                                                A,
                                                                Z,
                                                                analysis_policy,
                                                                &csrsm_info,
                                                                csrsm_buffer_size_in_bytes,
                                                                csrsm_buffer,
                                                                p_error));
            break;
        }

#endif

        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis(handle,
                                                                nrhs,
                                                                A_op,
                                                                X_op,
                                                                alpha,
                                                                A,
                                                                Z,
                                                                analysis_policy,
                                                                &csrsm_info,
                                                                csrsm_buffer_size_in_bytes,
                                                                csrsm_buffer,
                                                                p_error));
            break;
        }

        case rocsparse_format_coo:
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_analysis(handle,
                                                                nrhs,
                                                                A_op,
                                                                X_op,
                                                                alpha,
                                                                A,
                                                                X,
                                                                analysis_policy,
                                                                &csrsm_info,
                                                                csrsm_buffer_size_in_bytes,
                                                                csrsm_buffer,
                                                                p_error));

            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_sell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        switch(analysis_policy)
        {
        case rocsparse_analysis_policy_reuse:
        {
            break;
        }

        case rocsparse_analysis_policy_force:
        {
            sptrsm_descr->set_csrsm_info(csrsm_info);
            break;
        }
        }

        sptrsm_descr->set_stage(rocsparse_sptrsm_stage_analysis);
        return rocsparse_status_success;
    }

    static rocsparse_status sptrsm_compute(rocsparse_handle            handle,
                                           rocsparse_sptrsm_descr      sptrsm_descr,
                                           rocsparse_dnvec_descr       alpha,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_const_dnmat_descr X,
                                           const rocsparse_dnmat_descr Y,
                                           rocsparse_sptrsm_stage      sptrsm_stage,
                                           size_t                      buffer_size_in_bytes,
                                           void*                       buffer,
                                           rocsparse_error*            p_error)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_sptrsm_stage previous_stage = sptrsm_descr->get_stage();
        if(previous_stage == ((rocsparse_sptrsm_stage)-1))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_sptrsm_stage_analysis must be executed "
                "before "
                "the stage rocsparse_sptrsm_stage_compute");
        }

        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            ((alpha == nullptr) || (alpha->const_values == nullptr))
                ? rocsparse_status_invalid_pointer
                : rocsparse_status_success,
            "rocsparse_sptrsm_input_scalar_alpha must be set up.");

        const auto A_op               = sptrsm_descr->get_operation_A();
        const auto X_op               = sptrsm_descr->get_operation_X();
        const bool X_is_row_order     = (X->order == rocsparse_order_row);
        const bool Y_is_column_order  = (Y->order == rocsparse_order_column);
        const bool X_op_is_transposed = (X_op != rocsparse_operation_none);
        const auto nrhs               = Y->cols;

        sptrsm_descr->set_batch_count(Y->batch_count);

        //
        // Maybe we need to convert the scalar.
        //
        {
            const void* values = alpha->const_values;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsm_convert_scalars(
                handle, sptrsm_descr, sptrsm_descr->get_scalar_alpha(), &values));
            alpha->const_values = values;
        }

        _rocsparse_dnmat_descr Z_st{true,
                                    Y->rows,
                                    Y->cols,
                                    Y->cols,
                                    buffer,
                                    buffer,
                                    Y->data_type,
                                    rocsparse_order_row,
                                    Y->batch_count,

                                    (Y->batch_stride == 0) ? 0 : (Y->rows * Y->cols)};

        rocsparse_dnmat_descr Z = (Y_is_column_order) ? &Z_st : Y;

        if(Y_is_column_order)
        {
            const size_t nbytes = rocsparse::align_size(rocsparse::datatype_sizeof(Z->data_type)
                                                        * Z->rows * Z->cols * Z->batch_count);
            buffer              = reinterpret_cast<char*>(buffer) + nbytes;
            buffer_size_in_bytes -= nbytes;
        }

        static constexpr rocsparse_const_dnvec_descr no_scale = nullptr;

        if(X_op_is_transposed)
        {
            //
            // Matrices with opposite dimension and arbitrary order.
            //
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::dnmat_transpose(handle, no_scale, X, Z, p_error));
        }
        else if(X_is_row_order)
        {
            //
            // Matrices with same dimension and same order.
            //
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::dnmat_copy_data(handle, no_scale, X, Z, p_error));
        }
        else
        {
            //
            // Matrices with same dimension and different order.
            //
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::dnmat_switch_order(handle, no_scale, X, Z, p_error));
        }

        //
        // Compute
        //
        switch(A->format)
        {
#ifndef ROCSPARSE_WITH_CSC_TRSM
        case rocsparse_format_csc:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

#else
        case rocsparse_format_csc:
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_compute(handle,
                                                               nrhs,
                                                               A_op,
                                                               X_op,
                                                               alpha,
                                                               A,
                                                               Z,
                                                               sptrsm_descr->get_csrsm_info(),
                                                               buffer_size_in_bytes,
                                                               buffer,
                                                               p_error));
            break;
        }
#endif
        case rocsparse_format_csr:
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_compute(handle,
                                                               nrhs,
                                                               A_op,
                                                               X_op,
                                                               alpha,
                                                               A,
                                                               Z,
                                                               sptrsm_descr->get_csrsm_info(),
                                                               buffer_size_in_bytes,
                                                               buffer,
                                                               p_error));
            break;
        }

        case rocsparse_format_coo:
        {

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_compute(handle,
                                                               nrhs,
                                                               A_op,
                                                               X_op,
                                                               alpha,
                                                               A,
                                                               Z,
                                                               sptrsm_descr->get_csrsm_info(),
                                                               buffer_size_in_bytes,
                                                               buffer,
                                                               p_error));

            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_sell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        if(Y_is_column_order)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::dnmat_switch_order(handle, no_scale, Z, Y, p_error));
        }

        sptrsm_descr->set_stage(rocsparse_sptrsm_stage_compute);
        return rocsparse_status_success;
    }

    static rocsparse_status sptrsm(rocsparse_handle            handle,
                                   rocsparse_sptrsm_descr      sptrsm_descr,
                                   rocsparse_const_spmat_descr A,
                                   rocsparse_const_dnmat_descr X,
                                   const rocsparse_dnmat_descr Y,
                                   rocsparse_sptrsm_stage      sptrsm_stage,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer,
                                   rocsparse_error*            p_error)
    {
        ROCSPARSE_ROUTINE_TRACE;

        //
        // Define the vector alpha.
        //
        _rocsparse_dnvec_descr alpha_st(1,
                                        1,
                                        sptrsm_descr->get_scalar_datatype(),
                                        sptrsm_descr->get_scalar_alpha(),
                                        nullptr,
                                        1,
                                        0,
                                        handle->pointer_mode);

        rocsparse_dnvec_descr alpha = &alpha_st;

        //
        // To pass the number of right-hand-sides explicitly.
        //
        switch(sptrsm_stage)
        {
        case rocsparse_sptrsm_stage_analysis:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsm_analysis(handle,
                                                                 sptrsm_descr,
                                                                 alpha,
                                                                 A,
                                                                 X,
                                                                 Y,
                                                                 sptrsm_stage,
                                                                 buffer_size_in_bytes,
                                                                 buffer,
                                                                 p_error));

            return rocsparse_status_success;
        }

        case rocsparse_sptrsm_stage_compute:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsm_compute(handle,
                                                                sptrsm_descr,
                                                                alpha,
                                                                A,
                                                                X,
                                                                Y,
                                                                sptrsm_stage,
                                                                buffer_size_in_bytes,
                                                                buffer,
                                                                p_error));

            return rocsparse_status_success;
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

}

extern "C" rocsparse_status rocsparse_sptrsm(rocsparse_handle            handle, // 0
                                             rocsparse_sptrsm_descr      sptrsm_descr, // 1
                                             rocsparse_const_spmat_descr A, // 2
                                             rocsparse_const_dnmat_descr X, // 3
                                             rocsparse_dnmat_descr       Y, // 4
                                             rocsparse_sptrsm_stage      sptrsm_stage, // 5
                                             size_t                      buffer_size_in_bytes, // 6
                                             void*                       buffer, // 7
                                             rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsm_descr);
    ROCSPARSE_CHECKARG_POINTER(2, A);
    ROCSPARSE_CHECKARG_POINTER(3, X);
    ROCSPARSE_CHECKARG_POINTER(4, Y);

    ROCSPARSE_CHECKARG_ENUM(5, sptrsm_stage);

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
    ROCSPARSE_CHECKARG(3, X, (X->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, Y, (Y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    //
    // Batch count is driven by Y, batch_stride = 0 is not allowed if batch_count > 1.
    //
    ROCSPARSE_CHECKARG(
        4, Y, (Y->batch_count != 1) && (Y->batch_stride == 0), rocsparse_status_not_implemented);

    //
    // A is allowed to have a batch_count = 1
    //
    ROCSPARSE_CHECKARG(2,
                       A,
                       (A->batch_count != Y->batch_count) && (A->batch_count != 1),
                       rocsparse_status_not_implemented);

    //
    // X is allowed to have a batch_count = 1
    //
    ROCSPARSE_CHECKARG(3,
                       X,
                       (X->batch_count != Y->batch_count) && (X->batch_count != 1),
                       rocsparse_status_not_implemented);

    const rocsparse_datatype compute_type = sptrsm_descr->get_compute_datatype();

    //
    // Check for matching types while we do not support mixed precision computation
    //
    ROCSPARSE_CHECKARG(2, A, (A->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3, X, (X->data_type != compute_type), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(4, Y, (Y->data_type != compute_type), rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sptrsm(
        handle, sptrsm_descr, A, X, Y, sptrsm_stage, buffer_size_in_bytes, buffer, p_error));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
