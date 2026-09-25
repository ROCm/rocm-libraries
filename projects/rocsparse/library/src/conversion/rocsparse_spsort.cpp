/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosort.hpp"
#include "rocsparse_cscsort.hpp"
#include "rocsparse_csrsort.hpp"
#include "rocsparse_spsort.hpp"

rocsparse_status rocsparse::spsort_check_arguments(rocsparse_spsort_descr      descr,
                                                   rocsparse_const_spmat_descr source,
                                                   rocsparse_const_spmat_descr target)
{
    // The target receives the sorted source, so both must describe a matrix with the same shape
    // and layout.
    ROCSPARSE_CHECKARG(
        3, target, (target->format != source->format), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(3,
                       target,
                       (target->rows != source->rows || target->cols != source->cols
                        || target->nnz != source->nnz),
                       rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(3,
                       target,
                       (target->row_type != source->row_type || target->col_type != source->col_type
                        || target->data_type != source->data_type
                        || target->idx_base != source->idx_base),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(
        3, target, (target->batch_count != source->batch_count), rocsparse_status_invalid_value);

    if(source->batch_count > 1)
    {
        // Batches that overlap cannot be sorted independently.
        switch(source->format)
        {
        case rocsparse_format_coo:
        {
            ROCSPARSE_CHECKARG(
                2, source, (source->batch_stride < source->nnz), rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG(
                3, target, (target->batch_stride < target->nnz), rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_csr:
        case rocsparse_format_csc:
        {
            // The offsets may be shared by all batches with a stride of zero, but the target
            // can only share its offsets if the source does too.
            const int64_t offsets_size
                = ((source->format == rocsparse_format_csr) ? source->rows : source->cols) + 1;
            ROCSPARSE_CHECKARG(2,
                               source,
                               (source->columns_values_batch_stride < source->nnz
                                || (source->offsets_batch_stride != 0
                                    && source->offsets_batch_stride < offsets_size)),
                               rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG(
                3,
                target,
                (target->columns_values_batch_stride < target->nnz
                 || (target->offsets_batch_stride != 0
                     && target->offsets_batch_stride < offsets_size)
                 || (target->offsets_batch_stride == 0 && source->offsets_batch_stride != 0)),
                rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_coo_aos:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_sell:
        {
            ROCSPARSE_CHECKARG(2, source, true, rocsparse_status_not_implemented);
        }
        }
    }

    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_alg()),
                       rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_dir()),
                       rocsparse_status_invalid_value);

    // A CSR matrix can only have the column indices within each row sorted, and a CSC matrix
    // can only have the row indices within each column sorted.
    ROCSPARSE_CHECKARG(
        1,
        descr,
        (source->format == rocsparse_format_csr && descr->get_dir() != rocsparse_direction_row),
        rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(
        1,
        descr,
        (source->format == rocsparse_format_csc && descr->get_dir() != rocsparse_direction_column),
        rocsparse_status_invalid_value);

    return rocsparse_status_success;
}

namespace rocsparse
{
    static rocsparse_status spsort(rocsparse_handle            handle,
                                   rocsparse_spsort_descr      spsort_descr,
                                   rocsparse_const_spmat_descr source,
                                   rocsparse_spmat_descr       target,
                                   rocsparse_spsort_stage      stage,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_format format = source->format;

        const rocsparse_direction dir = spsort_descr->get_dir();

        switch(stage)
        {
        case rocsparse_spsort_stage_analysis:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            {
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
            // LCOV_EXCL_STOP

        case rocsparse_spsort_stage_compute:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort(handle,
                                                             rocsparse_coosort_alg_default,
                                                             dir,
                                                             source,
                                                             target,
                                                             buffer_size_in_bytes,
                                                             buffer));
                return rocsparse_status_success;
            }
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort(handle,
                                                             rocsparse_csrsort_alg_default,
                                                             source,
                                                             target,
                                                             buffer_size_in_bytes,
                                                             buffer));
                return rocsparse_status_success;
            }
            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsort(handle,
                                                             rocsparse_cscsort_alg_default,
                                                             source,
                                                             target,
                                                             buffer_size_in_bytes,
                                                             buffer));
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

extern "C" rocsparse_status rocsparse_spsort(rocsparse_handle            handle, //0
                                             rocsparse_spsort_descr      descr, //1
                                             rocsparse_const_spmat_descr source, //2
                                             rocsparse_spmat_descr       target, //3
                                             rocsparse_spsort_stage      stage, //4
                                             size_t                      buffer_size_in_bytes, //5
                                             void*                       temp_buffer, //6
                                             rocsparse_error*            error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, source);
    ROCSPARSE_CHECKARG_POINTER(3, target);
    ROCSPARSE_CHECKARG_ENUM(4, stage);
    ROCSPARSE_CHECKARG(5,
                       buffer_size_in_bytes,
                       (buffer_size_in_bytes == 0 && temp_buffer != nullptr),
                       rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(6,
                       temp_buffer,
                       (temp_buffer == nullptr && buffer_size_in_bytes > 0),
                       rocsparse_status_invalid_pointer);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsort_check_arguments(descr, source, target));

    // Validate the stage.
    const rocsparse_spsort_stage current_stage = descr->get_stage();
    switch(stage)
    {
    case rocsparse_spsort_stage_analysis:
    {
        if(current_stage == rocsparse_spsort_stage_compute)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_spsort_stage_analysis cannot be called after "
                "the stage rocsparse_spsort_stage_compute");
        }
        else if(current_stage == rocsparse_spsort_stage_analysis)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_spsort_stage_analysis has already been "
                "executed");
        }
        break;
    }
    case rocsparse_spsort_stage_compute:
    {
        if(current_stage == ((rocsparse_spsort_stage)-1))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "invalid stage, the stage rocsparse_spsort_stage_analysis must be executed before "
                "the stage rocsparse_spsort_stage_compute");
        }
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spsort(handle, descr, source, target, stage, buffer_size_in_bytes, temp_buffer));

    // Record the stage that has been executed.
    descr->set_stage(stage);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
