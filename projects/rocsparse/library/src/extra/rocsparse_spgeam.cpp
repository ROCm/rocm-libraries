/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_spgeam.h"
#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include "../conversion/rocsparse_convert_array.hpp"
#include "../conversion/rocsparse_convert_scalar.hpp"
#include "rocsparse_csrgeam.hpp"
#include "rocsparse_csrgeam_numeric.hpp"
#include "rocsparse_csrgeam_symbolic.hpp"

rocsparse_status _rocsparse_spgeam_descr::csrgeam_allocate_descr_memory(
    rocsparse_handle handle, int64_t m, int64_t n, int64_t nnz_A, int64_t nnz_B)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Clean up row pointer array
    if(this->csr_row_ptr_C != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->csr_row_ptr_C, handle->stream));
    }

    // Clean up rocprim buffer
    if(this->rocprim_buffer != nullptr && this->rocprim_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->rocprim_buffer, handle->stream));
    }

    this->indextype = (nnz_A + nnz_B) <= std::numeric_limits<int32_t>::max()
                          ? rocsparse_indextype_i32
                          : rocsparse_indextype_i64;

    // We do not know how many nonzeros will exist in C yet therefore use an int64_t row ptr array
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&this->csr_row_ptr_C,
                                 rocsparse::indextype_sizeof(this->indextype) * (m + 1),
                                 handle->stream));

    this->rocprim_alloc  = false;
    this->rocprim_buffer = nullptr;
    this->rocprim_size   = 0;

    this->m = m;

    return rocsparse_status_success;
}

rocsparse_status
    _rocsparse_spgeam_descr::csrgeam_copy_row_pointer(rocsparse_handle          handle,
                                                      int64_t                   m,
                                                      int64_t                   n,
                                                      const rocsparse_mat_descr descr_C,
                                                      rocsparse_indextype csr_row_ptr_C_indextype,
                                                      void*               csr_row_ptr_C,
                                                      int64_t*            nnz_C)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(this->csr_row_ptr_C != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_array(handle,
                                                           m + 1,
                                                           csr_row_ptr_C_indextype,
                                                           csr_row_ptr_C,
                                                           descr_C->base,
                                                           this->indextype,
                                                           this->csr_row_ptr_C,
                                                           rocsparse_index_base_zero));
        return rocsparse_status_success;
    }

    return rocsparse_status_success;
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spgeam_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spgeam_alg_default);
#undef CASE
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spgeam_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spgeam_stage_analysis);
        CASE(rocsparse_spgeam_stage_compute);
        CASE(rocsparse_spgeam_stage_symbolic_compute);
        CASE(rocsparse_spgeam_stage_numeric_compute);
        CASE(rocsparse_spgeam_stage_symbolic_analysis);
        CASE(rocsparse_spgeam_stage_numeric_analysis);
#undef CASE
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spgeam_input value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spgeam_input_alg);
        CASE(rocsparse_spgeam_input_scalar_datatype);
        CASE(rocsparse_spgeam_input_compute_datatype);
        CASE(rocsparse_spgeam_input_operation_A);
        CASE(rocsparse_spgeam_input_operation_B);
        CASE(rocsparse_spgeam_input_scalar_alpha);
        CASE(rocsparse_spgeam_input_scalar_beta);
#undef CASE
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spgeam_output value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spgeam_output_nnz);
#undef CASE
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spgeam_alg value_)
{
    switch(value_)
    {
    case rocsparse_spgeam_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spgeam_stage value_)
{
    switch(value_)
    {
    case rocsparse_spgeam_stage_analysis:
    case rocsparse_spgeam_stage_compute:
    case rocsparse_spgeam_stage_symbolic_compute:
    case rocsparse_spgeam_stage_numeric_compute:
    case rocsparse_spgeam_stage_symbolic_analysis:
    case rocsparse_spgeam_stage_numeric_analysis:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spgeam_input value_)
{
    switch(value_)
    {
    case rocsparse_spgeam_input_alg:
    case rocsparse_spgeam_input_scalar_datatype:
    case rocsparse_spgeam_input_compute_datatype:
    case rocsparse_spgeam_input_operation_A:
    case rocsparse_spgeam_input_operation_B:
    case rocsparse_spgeam_input_scalar_alpha:
    case rocsparse_spgeam_input_scalar_beta:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spgeam_output value_)
{
    switch(value_)
    {
    case rocsparse_spgeam_output_nnz:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{
    static rocsparse_status convert_scalars(rocsparse_handle             handle,
                                            const rocsparse_spgeam_descr descr,
                                            const void*                  alpha,
                                            const void*                  beta,
                                            const void**                 local_alpha,
                                            const void**                 local_beta)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_datatype scalar_datatype  = descr->get_scalar_datatype();
        const rocsparse_datatype compute_datatype = descr->get_compute_datatype();

        *local_alpha = alpha;
        *local_beta  = beta;

        if(scalar_datatype != compute_datatype)
        {
            // Convert scalars from scalar_datatype to compute_datatype
            switch(handle->pointer_mode)
            {
            case rocsparse_pointer_mode_host:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::convert_host_scalars(scalar_datatype,
                                                    compute_datatype,
                                                    alpha,
                                                    descr->get_local_host_alpha(),
                                                    beta,
                                                    descr->get_local_host_beta()));

                *local_alpha = descr->get_local_host_alpha();
                *local_beta  = descr->get_local_host_beta();

                break;
            }
            case rocsparse_pointer_mode_device:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::convert_device_scalars(handle->stream,
                                                                            scalar_datatype,
                                                                            compute_datatype,
                                                                            alpha,
                                                                            handle->alpha,
                                                                            beta,
                                                                            handle->beta));

                *local_alpha = handle->alpha;
                *local_beta  = handle->beta;

                break;
            }
            }
        }

        return rocsparse_status_success;
    }

    static rocsparse_status spgeam_buffer_size(rocsparse_handle            handle,
                                               rocsparse_spgeam_descr      descr,
                                               rocsparse_const_spmat_descr mat_A,
                                               rocsparse_const_spmat_descr mat_B,
                                               rocsparse_const_spmat_descr mat_C,
                                               rocsparse_spgeam_stage      stage,
                                               size_t*                     buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format format_A = mat_A->format;
        switch(stage)
        {
        case rocsparse_spgeam_stage_numeric_analysis:
        {
            *buffer_size = 0;
        }
        case rocsparse_spgeam_stage_symbolic_analysis:
        case rocsparse_spgeam_stage_analysis:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_buffer_size(
                    handle,
                    descr->get_operation_A(),
                    descr->get_operation_B(),
                    mat_A->rows,
                    mat_B->cols,
                    mat_A->descr,
                    mat_A->nnz,
                    mat_A->const_row_data,
                    mat_A->const_col_data,
                    mat_B->descr,
                    mat_B->nnz,
                    mat_B->const_row_data,
                    mat_B->const_col_data,
                    mat_C != nullptr ? mat_C->const_row_data : nullptr,
                    buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        case rocsparse_spgeam_stage_compute:
        case rocsparse_spgeam_stage_symbolic_compute:
        case rocsparse_spgeam_stage_numeric_compute:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        }
        return rocsparse_status_success;
    }

    static rocsparse_status spgeam_buffer_size_checkarg(rocsparse_handle            handle, //0
                                                        rocsparse_spgeam_descr      descr, //1
                                                        rocsparse_const_spmat_descr mat_A, //2
                                                        rocsparse_const_spmat_descr mat_B, //3
                                                        rocsparse_const_spmat_descr mat_C, //4
                                                        rocsparse_spgeam_stage      stage, //5
                                                        size_t*                     buffer_size) //6
    {
        ROCSPARSE_ROUTINE_TRACE;
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(2, mat_A);
        ROCSPARSE_CHECKARG_POINTER(3, mat_B);
        ROCSPARSE_CHECKARG(4,
                           mat_C,
                           ((stage == rocsparse_spgeam_stage_compute) && mat_C == nullptr),
                           rocsparse_status_invalid_pointer);

        ROCSPARSE_CHECKARG_ENUM(5, stage);

        ROCSPARSE_CHECKARG(2, mat_A, (mat_A->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(3, mat_B, (mat_B->init == false), rocsparse_status_not_initialized);

        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->format != mat_A->format), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(2,
                           mat_A,
                           (mat_A->data_type != descr->get_compute_datatype()),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           mat_B,
                           (mat_B->data_type != descr->get_compute_datatype()),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->col_type != mat_A->col_type), rocsparse_status_type_mismatch);

        if(stage == rocsparse_spgeam_stage_compute || mat_C != nullptr)
        {
            ROCSPARSE_CHECKARG(4, mat_C, (mat_C->init == false), rocsparse_status_not_initialized);

            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->format != mat_A->format), rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(4,
                               mat_C,
                               (mat_C->data_type != descr->get_compute_datatype()),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->col_type != mat_A->col_type), rocsparse_status_type_mismatch);
        }

        return rocsparse_status_continue;
    }

    static rocsparse_status spgeam_checkarg(rocsparse_handle            handle, //0
                                            rocsparse_spgeam_descr      descr, //1
                                            rocsparse_const_spmat_descr mat_A, //2
                                            rocsparse_const_spmat_descr mat_B, //3
                                            rocsparse_spmat_descr       mat_C, //4
                                            rocsparse_spgeam_stage      stage, //5
                                            size_t                      buffer_size, //6
                                            void*                       temp_buffer) //7
    {
        ROCSPARSE_ROUTINE_TRACE;
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(2, mat_A);
        ROCSPARSE_CHECKARG_POINTER(3, mat_B);
        if(stage == rocsparse_spgeam_stage_compute
           || stage == rocsparse_spgeam_stage_symbolic_compute
           || stage == rocsparse_spgeam_stage_numeric_compute)
        {
            ROCSPARSE_CHECKARG_POINTER(4, mat_C);
        }
        ROCSPARSE_CHECKARG(7,
                           temp_buffer,
                           (buffer_size > 0 && temp_buffer == nullptr),
                           rocsparse_status_invalid_pointer);

        ROCSPARSE_CHECKARG_ENUM(5, stage);

        ROCSPARSE_CHECKARG(2, mat_A, (mat_A->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(3, mat_B, (mat_B->init == false), rocsparse_status_not_initialized);

        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->format != mat_A->format), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(2,
                           mat_A,
                           (mat_A->data_type != descr->get_compute_datatype()),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           mat_B,
                           (mat_B->data_type != descr->get_compute_datatype()),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            3, mat_B, (mat_B->col_type != mat_A->col_type), rocsparse_status_type_mismatch);

        if(stage == rocsparse_spgeam_stage_compute
           || stage == rocsparse_spgeam_stage_symbolic_compute
           || stage == rocsparse_spgeam_stage_numeric_compute || mat_C != nullptr)
        {
            ROCSPARSE_CHECKARG(4, mat_C, (mat_C->init == false), rocsparse_status_not_initialized);

            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->format != mat_A->format), rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(4,
                               mat_C,
                               (mat_C->data_type != descr->get_compute_datatype()),
                               rocsparse_status_not_implemented);

            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
            ROCSPARSE_CHECKARG(
                4, mat_C, (mat_C->col_type != mat_A->col_type), rocsparse_status_type_mismatch);
        }

        // Validate spgeam descriptor inputs.
        ROCSPARSE_CHECKARG(1,
                           descr,
                           rocsparse::enum_utils::is_invalid(descr->get_alg()),
                           rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(1,
                           descr,
                           rocsparse::enum_utils::is_invalid(descr->get_operation_A()),
                           rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(1,
                           descr,
                           rocsparse::enum_utils::is_invalid(descr->get_operation_B()),
                           rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(1,
                           descr,
                           rocsparse::enum_utils::is_invalid(descr->get_compute_datatype()),
                           rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(1,
                           descr,
                           rocsparse::enum_utils::is_invalid(descr->get_scalar_datatype()),
                           rocsparse_status_invalid_value);

        const rocsparse_spgeam_stage previous_stage = descr->get_stage();
        // Validate the stage.
        switch(stage)
        {
        case rocsparse_spgeam_stage_symbolic_analysis:
        {

            switch(previous_stage)
            {

            case rocsparse_spgeam_stage_symbolic_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_analysis has already "
                    "been "
                    "executed");
            }

            case rocsparse_spgeam_stage_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_analysis");
            }

            case rocsparse_spgeam_stage_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_compute");
            }

            case rocsparse_spgeam_stage_numeric_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_numeric_analysis");
            }

            case rocsparse_spgeam_stage_numeric_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_numeric_compute");
            }

            case rocsparse_spgeam_stage_symbolic_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_symbolic_compute");
            }
            }

            break;
        }

        case rocsparse_spgeam_stage_numeric_analysis:
        {

            switch(previous_stage)
            {

            case rocsparse_spgeam_stage_numeric_compute:
            case rocsparse_spgeam_stage_symbolic_compute:
            {
                break;
            }

            case rocsparse_spgeam_stage_symbolic_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_numeric_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_symbolic_analysis");
            }

            case rocsparse_spgeam_stage_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_numeric_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_analysis");
            }

            case rocsparse_spgeam_stage_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_numeric_analysis cannot be "
                    "called "
                    "after "
                    "the stage rocsparse_spgeam_stage_compute");
            }

            case rocsparse_spgeam_stage_numeric_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_numeric_analysis has already "
                    "been "
                    "executed");
            }
            }

            break;
        }

        case rocsparse_spgeam_stage_symbolic_compute:
        {

            if(previous_stage == ((rocsparse_spgeam_stage)-1))
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis must be executed "
                    "before "
                    "the stage rocsparse_spgeam_stage_symbolic_compute");
            }

            switch(previous_stage)
            {
            case rocsparse_spgeam_stage_symbolic_analysis:
            {
                break;
            }

            case rocsparse_spgeam_stage_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_compute cannot be "
                    "called "
                    "after the stage rocsparse_spgeam_stage_analysis");
                break;
            }

            case rocsparse_spgeam_stage_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_compute cannot be "
                    "called "
                    "after the stage rocsparse_spgeam_stage_compute");
            }

            case rocsparse_spgeam_stage_numeric_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_compute cannot be "
                    "called "
                    "after the stage rocsparse_spgeam_stage_numeric_analysis");
            }

            case rocsparse_spgeam_stage_numeric_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_compute cannot be "
                    "called "
                    "after the stage rocsparse_spgeam_stage_numeric_compute");
            }

            case rocsparse_spgeam_stage_symbolic_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_symbolic_compute has already "
                    "been "
                    "executed");
            }
            }
            break;
        }

        case rocsparse_spgeam_stage_numeric_compute:
        {
            switch(previous_stage)
            {
            case rocsparse_spgeam_stage_symbolic_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_numeric_compute cannot be "
                    "called "
                    "after the stage rocsparse_spgeam_stage_symbolic_analysis");
            }

            case rocsparse_spgeam_stage_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_numeric_compute cannot be "
                    "called "
                    "after the stage rocsparse_spgeam_stage_analysis");
            }

            case rocsparse_spgeam_stage_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_numeric_compute cannot be "
                    "called "
                    "after the stage rocsparse_spgeam_stage_compute");
            }

            case rocsparse_spgeam_stage_numeric_analysis:
            case rocsparse_spgeam_stage_numeric_compute:
            case rocsparse_spgeam_stage_symbolic_compute:
            {
                break;
            }
            }

            break;
        }

        case rocsparse_spgeam_stage_analysis:
        {
            switch(previous_stage)
            {
            case rocsparse_spgeam_stage_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis has already been "
                    "executed");
            }

            case rocsparse_spgeam_stage_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis cannot be called "
                    "after "
                    "the stage rocsparse_spgeam_stage_compute");
            }

            case rocsparse_spgeam_stage_numeric_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis cannot be called "
                    "after "
                    "the stage rocsparse_spgeam_stage_numeric_analysis");
            }

            case rocsparse_spgeam_stage_symbolic_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis cannot be called "
                    "after "
                    "the stage rocsparse_spgeam_stage_symbolic_analysis");
            }

            case rocsparse_spgeam_stage_numeric_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis cannot be called "
                    "after "
                    "the stage rocsparse_spgeam_stage_numeric_compute");
            }

            case rocsparse_spgeam_stage_symbolic_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis cannot be called "
                    "after "
                    "the stage rocsparse_spgeam_stage_symbolic_compute");
            }
            }

            break;
        }

        case rocsparse_spgeam_stage_compute:
        {
            if(previous_stage == ((rocsparse_spgeam_stage)-1))
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_analysis must be executed "
                    "before "
                    "the stage rocsparse_spgeam_stage_compute");
            }

            switch(previous_stage)
            {
            case rocsparse_spgeam_stage_analysis:
            case rocsparse_spgeam_stage_compute:
            {
                break;
            }

            case rocsparse_spgeam_stage_numeric_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_compute cannot be called "
                    "after the stage rocsparse_spgeam_stage_numeric_compute");
            }

            case rocsparse_spgeam_stage_symbolic_compute:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_compute cannot be called "
                    "after the stage rocsparse_spgeam_stage_symbolic_compute");
            }

            case rocsparse_spgeam_stage_numeric_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_compute cannot be called "
                    "after the stage rocsparse_spgeam_stage_numeric_analysis");
            }

            case rocsparse_spgeam_stage_symbolic_analysis:
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                    rocsparse_status_invalid_value,
                    "invalid stage, the stage rocsparse_spgeam_stage_compute cannot be called "
                    "after the stage rocsparse_spgeam_stage_symbolic_analysis");
            }
            }

            break;
        }
        }

        return rocsparse_status_continue;
    }

    static rocsparse_status spgeam(rocsparse_handle             handle,
                                   const rocsparse_spgeam_descr descr,
                                   rocsparse_const_spmat_descr  mat_A,
                                   rocsparse_const_spmat_descr  mat_B,
                                   rocsparse_spmat_descr        mat_C,
                                   rocsparse_spgeam_stage       stage,
                                   size_t                       buffer_size,
                                   void*                        temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format format_A = mat_A->format;
        switch(stage)
        {
        case rocsparse_spgeam_stage_numeric_analysis:
        {
            //
            // do nothing.
            //
            return rocsparse_status_success;
        }

        case rocsparse_spgeam_stage_symbolic_analysis:
        case rocsparse_spgeam_stage_analysis:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                if(mat_C == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(descr->csrgeam_allocate_descr_memory(
                        handle, mat_A->rows, mat_B->cols, mat_A->nnz, mat_B->nnz));
                }

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz(
                    handle,
                    descr,
                    descr->get_operation_A(),
                    descr->get_operation_B(),
                    mat_A->rows,
                    mat_B->cols,
                    mat_A->descr,
                    mat_A->nnz,
                    mat_A->row_type,
                    mat_A->const_row_data,
                    mat_A->col_type,
                    mat_A->const_col_data,
                    mat_B->descr,
                    mat_B->nnz,
                    mat_B->row_type,
                    mat_B->const_row_data,
                    mat_B->col_type,
                    mat_B->const_col_data,
                    mat_C != nullptr ? mat_C->descr : nullptr,
                    mat_C != nullptr ? mat_C->row_type : ((rocsparse_indextype)-1),
                    mat_C != nullptr ? mat_C->row_data : nullptr,
                    mat_C != nullptr ? &mat_C->nnz : nullptr,
                    temp_buffer,
                    true));

                if(mat_C != nullptr && mat_C->row_type == rocsparse_indextype_i32)
                {
                    // Temporary hack to handle the fact that we pass mat_C->nnz as an int64_t* but internally in order to match
                    // the legacy API we handle nnz_C using int32_t* when csr_row_ptr_C is int32_t*
                    mat_C->nnz = *reinterpret_cast<const int32_t*>(&mat_C->nnz);
                }

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spgeam_stage_compute:
        {

            const void* local_alpha = descr->get_scalar_A();
            const void* local_beta  = descr->get_scalar_B();

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                (local_alpha == nullptr) ? rocsparse_status_invalid_pointer
                                         : rocsparse_status_success,
                "rocsparse_spgeam_input_scalar_alpha must be set up.");
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                (local_beta == nullptr) ? rocsparse_status_invalid_pointer
                                        : rocsparse_status_success,
                "rocsparse_spgeam_input_scalar_beta must be set up.");

            RETURN_IF_ROCSPARSE_ERROR(convert_scalars(handle,
                                                      descr,
                                                      descr->get_scalar_A(),
                                                      descr->get_scalar_B(),
                                                      &local_alpha,
                                                      &local_beta));

            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(descr->csrgeam_copy_row_pointer(handle,
                                                                          mat_A->rows,
                                                                          mat_B->cols,
                                                                          mat_C->descr,
                                                                          mat_C->row_type,
                                                                          mat_C->row_data,
                                                                          &mat_C->nnz));

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam(handle,
                                                             descr->get_operation_A(),
                                                             descr->get_operation_B(),
                                                             mat_A->rows,
                                                             mat_B->cols,
                                                             descr->get_scalar_datatype(),
                                                             local_alpha,
                                                             mat_A->descr,
                                                             mat_A->nnz,
                                                             mat_A->data_type,
                                                             mat_A->const_val_data,
                                                             mat_A->row_type,
                                                             mat_A->const_row_data,
                                                             mat_A->col_type,
                                                             mat_A->const_col_data,
                                                             descr->get_scalar_datatype(),
                                                             local_beta,
                                                             mat_B->descr,
                                                             mat_B->nnz,
                                                             mat_B->data_type,
                                                             mat_B->const_val_data,
                                                             mat_B->row_type,
                                                             mat_B->const_row_data,
                                                             mat_B->col_type,
                                                             mat_B->const_col_data,
                                                             mat_C->descr,
                                                             mat_C->data_type,
                                                             mat_C->val_data,
                                                             mat_C->row_type,
                                                             mat_C->const_row_data,
                                                             mat_C->col_type,
                                                             mat_C->col_data,
                                                             temp_buffer));

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spgeam_stage_symbolic_compute:
        {
            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(descr->csrgeam_copy_row_pointer(handle,
                                                                          mat_A->rows,
                                                                          mat_B->cols,
                                                                          mat_C->descr,
                                                                          mat_C->row_type,
                                                                          mat_C->row_data,
                                                                          &mat_C->nnz));

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_symbolic(handle,
                                                                      descr->get_operation_A(),
                                                                      descr->get_operation_B(),
                                                                      mat_A->rows,
                                                                      mat_B->cols,
                                                                      mat_A->descr,
                                                                      mat_A->nnz,
                                                                      mat_A->row_type,
                                                                      mat_A->const_row_data,
                                                                      mat_A->col_type,
                                                                      mat_A->const_col_data,
                                                                      mat_B->descr,
                                                                      mat_B->nnz,
                                                                      mat_B->row_type,
                                                                      mat_B->const_row_data,
                                                                      mat_B->col_type,
                                                                      mat_B->const_col_data,
                                                                      mat_C->descr,
                                                                      mat_C->row_type,
                                                                      mat_C->const_row_data,
                                                                      mat_C->col_type,
                                                                      mat_C->col_data,
                                                                      temp_buffer));

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spgeam_stage_numeric_compute:
        {

            const void* local_alpha = descr->get_scalar_A();
            const void* local_beta  = descr->get_scalar_B();

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                (local_alpha == nullptr) ? rocsparse_status_invalid_pointer
                                         : rocsparse_status_success,
                "rocsparse_spgeam_input_scalar_alpha must be set up.");

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                (local_beta == nullptr) ? rocsparse_status_invalid_pointer
                                        : rocsparse_status_success,
                "rocsparse_spgeam_input_scalar_beta must be set up.");

            RETURN_IF_ROCSPARSE_ERROR(convert_scalars(handle,
                                                      descr,
                                                      descr->get_scalar_A(),
                                                      descr->get_scalar_B(),
                                                      &local_alpha,
                                                      &local_beta));

            switch(format_A)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_numeric(handle,
                                                                     descr->get_operation_A(),
                                                                     descr->get_operation_B(),
                                                                     mat_A->rows,
                                                                     mat_B->cols,
                                                                     descr->get_scalar_datatype(),
                                                                     local_alpha,
                                                                     mat_A->descr,
                                                                     mat_A->nnz,
                                                                     mat_A->data_type,
                                                                     mat_A->const_val_data,
                                                                     mat_A->row_type,
                                                                     mat_A->const_row_data,
                                                                     mat_A->col_type,
                                                                     mat_A->const_col_data,
                                                                     descr->get_scalar_datatype(),
                                                                     local_beta,
                                                                     mat_B->descr,
                                                                     mat_B->nnz,
                                                                     mat_B->data_type,
                                                                     mat_B->const_val_data,
                                                                     mat_B->row_type,
                                                                     mat_B->const_row_data,
                                                                     mat_B->col_type,
                                                                     mat_B->const_col_data,
                                                                     mat_C->descr,
                                                                     mat_C->data_type,
                                                                     mat_C->val_data,
                                                                     mat_C->row_type,
                                                                     mat_C->const_row_data,
                                                                     mat_C->col_type,
                                                                     mat_C->col_data,
                                                                     temp_buffer));

                return rocsparse_status_success;
            }
            case rocsparse_format_bsr:
            case rocsparse_format_coo:
            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

extern "C" rocsparse_status rocsparse_spgeam_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_spgeam_descr      descr,
                                                         rocsparse_const_spmat_descr mat_A,
                                                         rocsparse_const_spmat_descr mat_B,
                                                         rocsparse_const_spmat_descr mat_C,
                                                         rocsparse_spgeam_stage      stage,
                                                         size_t*          buffer_size_in_bytes,
                                                         rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::log_trace("rocsparse_spgeam_buffer_size",
                         handle,
                         descr,
                         mat_A,
                         mat_B,
                         mat_C,
                         stage,
                         buffer_size_in_bytes);

    const rocsparse_status status = rocsparse::spgeam_buffer_size_checkarg(
        handle, descr, mat_A, mat_B, mat_C, stage, buffer_size_in_bytes);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::spgeam_buffer_size(
        handle, descr, mat_A, mat_B, mat_C, stage, buffer_size_in_bytes)));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_spgeam(rocsparse_handle            handle,
                                             rocsparse_spgeam_descr      descr,
                                             rocsparse_const_spmat_descr mat_A,
                                             rocsparse_const_spmat_descr mat_B,
                                             rocsparse_spmat_descr       mat_C,
                                             rocsparse_spgeam_stage      stage,
                                             size_t                      buffer_size,
                                             void*                       temp_buffer,
                                             rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::log_trace(
        "rocsparse_spgeam", handle, descr, mat_A, mat_B, mat_C, stage, buffer_size, temp_buffer);

    const rocsparse_status status = rocsparse::spgeam_checkarg(
        handle, descr, mat_A, mat_B, mat_C, stage, buffer_size, temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spgeam(handle, descr, mat_A, mat_B, mat_C, stage, buffer_size, temp_buffer));

    // Record the stage that has been executed.
    descr->set_stage(stage);

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
