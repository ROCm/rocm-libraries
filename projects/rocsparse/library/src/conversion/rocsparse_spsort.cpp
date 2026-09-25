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

#include "internal/generic/rocsparse_spsort.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosort.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsort_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_input value_)
{
    switch(value_)
    {
    case rocsparse_spsort_input_alg:
    case rocsparse_spsort_input_direction:
    {
        return false;
    }
    }
    return true;
};

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsort_stage_analysis:
    case rocsparse_spsort_stage_compute:
    {
        return false;
    }
    }
    return true;
};

struct _rocsparse_spsort_descr
{
protected:
    rocsparse_spsort_stage m_stage;
    rocsparse_spsort_alg   m_alg;
    rocsparse_direction    m_dir;

public:
    ~_rocsparse_spsort_descr() {}

    _rocsparse_spsort_descr()
        : m_stage((rocsparse_spsort_stage)-1)
        , m_alg((rocsparse_spsort_alg)-1)
        , m_dir((rocsparse_direction)-1)
    {
    }

    rocsparse_spsort_stage get_stage() const
    {
        return this->m_stage;
    }
    rocsparse_spsort_alg get_alg() const
    {
        return this->m_alg;
    }
    rocsparse_direction get_dir() const
    {
        return this->m_dir;
    }

    void set_stage(rocsparse_spsort_stage value)
    {
        this->m_stage = value;
    }
    void set_alg(rocsparse_spsort_alg value)
    {
        this->m_alg = value;
    }
    void set_dir(rocsparse_direction value)
    {
        this->m_dir = value;
    }
};

extern "C" rocsparse_status rocsparse_spsort_descr_create(rocsparse_handle        handle,
                                                          rocsparse_spsort_descr* p_spsort_descr,
                                                          rocsparse_error*        p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_spsort_descr);
    *p_spsort_descr = new _rocsparse_spsort_descr();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spsort_descr_destroy(rocsparse_handle       handle,
                                                           rocsparse_spsort_descr spsort_descr,
                                                           rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    if(spsort_descr != nullptr)
    {
        delete spsort_descr;
    }
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spsort_set_input(rocsparse_handle       handle,
                                                       rocsparse_spsort_descr descr,
                                                       rocsparse_spsort_input input,
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
    case rocsparse_spsort_input_alg:
    {
        switch(descr->get_stage())
        {
        case rocsparse_spsort_stage_analysis:
        case rocsparse_spsort_stage_compute:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_internal_error,
                "The field 'rocsparse_spsort_input_alg' must be set before the stage "
                "'rocsparse_spsort_stage_analysis' is executed.");
        }
        }

        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_spsort_alg),
                           rocsparse_status_invalid_size);
        const rocsparse_spsort_alg alg = *reinterpret_cast<const rocsparse_spsort_alg*>(data);
        descr->set_alg(alg);
        return rocsparse_status_success;
    }
    case rocsparse_spsort_input_direction:
    {
        switch(descr->get_stage())
        {
        case rocsparse_spsort_stage_analysis:
        case rocsparse_spsort_stage_compute:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_internal_error,
                "The field 'rocsparse_spsort_input_direction' must be set before the stage "
                "'rocsparse_spsort_stage_analysis' is executed.");
        }
        }

        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_direction),
                           rocsparse_status_invalid_size);
        const rocsparse_direction dir = *reinterpret_cast<const rocsparse_direction*>(data);
        ROCSPARSE_CHECKARG(
            3, data, rocsparse::enum_utils::is_invalid(dir), rocsparse_status_invalid_value);
        descr->set_dir(dir);
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
    // The target receives the sorted source, so both must describe a matrix with the same shape
    // and layout.
    static rocsparse_status spsort_check_matrices(rocsparse_const_spmat_descr source,
                                                  rocsparse_spmat_descr       target)
    {
        ROCSPARSE_CHECKARG(
            3, target, (target->format != source->format), rocsparse_status_invalid_value);
        ROCSPARSE_CHECKARG(3,
                           target,
                           (target->rows != source->rows || target->cols != source->cols
                            || target->nnz != source->nnz),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            3,
            target,
            (target->row_type != source->row_type || target->col_type != source->col_type
             || target->data_type != source->data_type || target->idx_base != source->idx_base),
            rocsparse_status_invalid_value);

        ROCSPARSE_CHECKARG(3,
                           target,
                           (target->batch_count != source->batch_count),
                           rocsparse_status_invalid_value);

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

        return rocsparse_status_success;
    }

    rocsparse_status spsort_buffer_size(rocsparse_handle            handle,
                                        rocsparse_spsort_descr      descr,
                                        rocsparse_const_spmat_descr source,
                                        rocsparse_const_spmat_descr target,
                                        rocsparse_spsort_stage      stage,
                                        size_t*                     buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_format format = source->format;

        const rocsparse_direction dir = descr->get_dir();

        switch(stage)
        {
        case rocsparse_spsort_stage_analysis:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            {
                *buffer_size_in_bytes = 0;
                return rocsparse_status_success;
            }
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        case rocsparse_spsort_stage_compute:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::coosort_buffer_size(handle,
                                                    rocsparse_coosort_alg_default,
                                                    dir,
                                                    source->rows,
                                                    source->cols,
                                                    source->nnz,
                                                    source->row_type,
                                                    source->const_row_data,
                                                    source->col_type,
                                                    source->const_col_data,
                                                    source->data_type,
                                                    source->const_val_data,
                                                    target->row_type,
                                                    target->const_row_data,
                                                    target->col_type,
                                                    target->const_col_data,
                                                    target->data_type,
                                                    target->const_val_data,
                                                    buffer_size_in_bytes)));
                return rocsparse_status_success;
            }
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    rocsparse_status spsort(rocsparse_handle            handle,
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
            {
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_sell:
            case rocsparse_format_coo_aos:
            case rocsparse_format_bell:
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
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::coosort(handle,
                                                              rocsparse_coosort_alg_default,
                                                              dir,
                                                              source->rows,
                                                              source->cols,
                                                              source->nnz,
                                                              source->batch_count,
                                                              source->batch_stride,
                                                              source->row_type,
                                                              source->const_row_data,
                                                              source->col_type,
                                                              source->const_col_data,
                                                              source->data_type,
                                                              source->const_val_data,
                                                              target->batch_count,
                                                              target->batch_stride,
                                                              target->row_type,
                                                              target->row_data,
                                                              target->col_type,
                                                              target->col_data,
                                                              target->data_type,
                                                              target->val_data,
                                                              buffer)));
                return rocsparse_status_success;
            }

                // LCOV_EXCL_START
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_coo_aos:
            case rocsparse_format_sell:
            case rocsparse_format_bell:
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

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_spsort_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_spsort_descr      descr,
                                                         rocsparse_const_spmat_descr source,
                                                         rocsparse_spmat_descr       target,
                                                         rocsparse_spsort_stage      stage,
                                                         size_t*          buffer_size_in_bytes,
                                                         rocsparse_error* error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, source);
    ROCSPARSE_CHECKARG_POINTER(3, target);
    ROCSPARSE_CHECKARG_ENUM(4, stage);
    ROCSPARSE_CHECKARG_POINTER(5, buffer_size_in_bytes);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsort_check_matrices(source, target));

    // Validate spmv_inputs.
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_alg()),
                       rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_dir()),
                       rocsparse_status_invalid_value);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spsort_buffer_size(handle, descr, source, target, stage, buffer_size_in_bytes));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsort_check_matrices(source, target));

    // Validate spmv_inputs.
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_alg()),
                       rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(1,
                       descr,
                       rocsparse::enum_utils::is_invalid(descr->get_dir()),
                       rocsparse_status_invalid_value);

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
