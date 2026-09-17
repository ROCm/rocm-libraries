/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_spildlt0_descr.hpp"
#include "rocsparse_utility.hpp"

#include "internal/generic/rocsparse_spildlt0.h"

rocsparse::numeric_boost* _rocsparse_spildlt0_descr::get_boost()
{
    return &this->m_boost;
}

void _rocsparse_spildlt0_descr::set_csrildlt0_info(rocsparse_csrildlt0_info value)
{
    this->m_csrildlt0_info = std::shared_ptr<_rocsparse_csric0_info>(value);
}

rocsparse_csrildlt0_info _rocsparse_spildlt0_descr::get_csrildlt0_info()
{
    return this->m_csrildlt0_info.get();
}

void _rocsparse_spildlt0_descr::set_shared_csrildlt0_info(
    std::shared_ptr<_rocsparse_csric0_info> value)
{
    this->m_csrildlt0_info = value;
}

_rocsparse_spildlt0_descr::~_rocsparse_spildlt0_descr()
{
    this->m_stage                  = ((rocsparse_spildlt0_stage)-1);
    this->m_alg                    = ((rocsparse_spildlt0_alg)-1);
    this->m_compute_datatype       = ((rocsparse_datatype)-1);
    this->m_analysis_policy        = ((rocsparse_analysis_policy)-1);
    this->m_format                 = ((rocsparse_format)-1);
    this->m_tolerance_pointer      = nullptr;
    this->m_tolerance_pointer_mode = ((rocsparse_pointer_mode)-1);
    this->m_diag                   = nullptr;

    this->m_csrildlt0_info.reset();
}

_rocsparse_spildlt0_descr::_rocsparse_spildlt0_descr()
    : m_stage((rocsparse_spildlt0_stage)-1)
    , m_alg((rocsparse_spildlt0_alg)-1)
    , m_compute_datatype((rocsparse_datatype)-1)
    , m_analysis_policy((rocsparse_analysis_policy)-1)
    , m_format((rocsparse_format)-1)
    , m_tolerance_pointer(nullptr)
    , m_tolerance_pointer_mode((rocsparse_pointer_mode)-1)
    , m_diag(nullptr)
{
}

rocsparse_spildlt0_stage _rocsparse_spildlt0_descr::get_stage() const
{
    return this->m_stage;
}

rocsparse_spildlt0_alg _rocsparse_spildlt0_descr::get_alg() const
{
    return this->m_alg;
}

rocsparse_analysis_policy _rocsparse_spildlt0_descr::get_analysis_policy() const
{
    return this->m_analysis_policy;
}

void _rocsparse_spildlt0_descr::set_analysis_policy(rocsparse_analysis_policy value)
{
    this->m_analysis_policy = value;
}

rocsparse_datatype _rocsparse_spildlt0_descr::get_compute_datatype() const
{
    return this->m_compute_datatype;
}

void _rocsparse_spildlt0_descr::set_stage(rocsparse_spildlt0_stage value)
{
    this->m_stage = value;
}

void _rocsparse_spildlt0_descr::set_alg(rocsparse_spildlt0_alg value)
{
    this->m_alg = value;
}

void _rocsparse_spildlt0_descr::set_compute_datatype(rocsparse_datatype value)
{
    this->m_compute_datatype = value;
}

const double* _rocsparse_spildlt0_descr::get_tolerance_pointer() const
{
    return this->m_tolerance_pointer;
}

rocsparse_pointer_mode _rocsparse_spildlt0_descr::get_tolerance_pointer_mode() const
{
    return this->m_tolerance_pointer_mode;
}

void _rocsparse_spildlt0_descr::set_tolerance_pointer(const void* value)
{
    this->m_tolerance_pointer = reinterpret_cast<const double*>(value);
}

void _rocsparse_spildlt0_descr::set_tolerance_pointer_mode(rocsparse_pointer_mode value)
{
    this->m_tolerance_pointer_mode = value;
}

rocsparse_format _rocsparse_spildlt0_descr::get_format() const
{
    return this->m_format;
}

void _rocsparse_spildlt0_descr::set_format(rocsparse_format value)
{
    this->m_format = value;
}

void* _rocsparse_spildlt0_descr::get_diag() const
{
    return this->m_diag;
}

void _rocsparse_spildlt0_descr::set_diag(void* diag)
{
    this->m_diag = diag;
}

extern "C" rocsparse_status rocsparse_spildlt0_descr_create(rocsparse_handle          handle,
                                                            rocsparse_spildlt0_descr* descr,
                                                            rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    *descr = new _rocsparse_spildlt0_descr();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spildlt0_descr_destroy(rocsparse_handle         handle,
                                                             rocsparse_spildlt0_descr descr,
                                                             rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    if(descr != nullptr)
    {
        delete descr;
    }
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
