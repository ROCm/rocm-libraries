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

#include "rocsparse_spsort.hpp"

_rocsparse_spsort_descr::~_rocsparse_spsort_descr() {}

_rocsparse_spsort_descr::_rocsparse_spsort_descr()
    : m_stage((rocsparse_spsort_stage)-1)
    , m_alg((rocsparse_spsort_alg)-1)
    , m_dir((rocsparse_direction)-1)
{
}

rocsparse_spsort_stage _rocsparse_spsort_descr::get_stage() const
{
    return this->m_stage;
}

rocsparse_spsort_alg _rocsparse_spsort_descr::get_alg() const
{
    return this->m_alg;
}

rocsparse_direction _rocsparse_spsort_descr::get_dir() const
{
    return this->m_dir;
}

void _rocsparse_spsort_descr::set_stage(rocsparse_spsort_stage value)
{
    this->m_stage = value;
}

void _rocsparse_spsort_descr::set_alg(rocsparse_spsort_alg value)
{
    this->m_alg = value;
}

void _rocsparse_spsort_descr::set_dir(rocsparse_direction value)
{
    this->m_dir = value;
}

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
    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                           "the rocsparse_spsort_input value is not supported");
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
