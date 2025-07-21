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

#include "rocsparse_sptrsv_descr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_logging.hpp"

_rocsparse_sptrsv_descr::_rocsparse_sptrsv_descr()
    : m_stage((rocsparse_sptrsv_stage)-1)
    , m_alg((rocsparse_sptrsv_alg)-1)
    , m_operation((rocsparse_operation)-1)
    , m_scalar_datatype((rocsparse_datatype)-1)
    , m_compute_datatype((rocsparse_datatype)-1)
{
}

int64_t _rocsparse_sptrsv_descr::get_zero_pivot_position() const
{
    return this->m_zero_pivot_position;
}

rocsparse_sptrsv_stage _rocsparse_sptrsv_descr::get_stage() const
{
    return this->m_stage;
}

rocsparse_sptrsv_alg _rocsparse_sptrsv_descr::get_alg() const
{
    return this->m_alg;
}

rocsparse_operation _rocsparse_sptrsv_descr::get_operation() const
{
    return this->m_operation;
}

rocsparse_datatype _rocsparse_sptrsv_descr::get_scalar_datatype() const
{
    return this->m_scalar_datatype;
}

rocsparse_datatype _rocsparse_sptrsv_descr::get_compute_datatype() const
{
    return this->m_compute_datatype;
}

void _rocsparse_sptrsv_descr::set_stage(rocsparse_sptrsv_stage value)
{
    this->m_stage = value;
}
void _rocsparse_sptrsv_descr::set_alg(rocsparse_sptrsv_alg value)
{
    this->m_alg = value;
}

void _rocsparse_sptrsv_descr::set_operation(rocsparse_operation value)
{
    this->m_operation = value;
}

const void* _rocsparse_sptrsv_descr::get_scalar_alpha() const
{
    return this->m_scalar_alpha;
}

void _rocsparse_sptrsv_descr::set_scalar_alpha(const void* value)
{
    this->m_scalar_alpha = value;
}

void _rocsparse_sptrsv_descr::set_scalar_datatype(rocsparse_datatype value)
{
    this->m_scalar_datatype = value;
}

void _rocsparse_sptrsv_descr::set_compute_datatype(rocsparse_datatype value)
{
    this->m_compute_datatype = value;
}

void _rocsparse_sptrsv_descr::set_zero_pivot_position(int64_t value)
{
    this->m_zero_pivot_position = value;
}

extern "C" rocsparse_status rocsparse_create_sptrsv_descr(rocsparse_sptrsv_descr* descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_POINTER(0, descr);
    *descr = new _rocsparse_sptrsv_descr();
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_destroy_sptrsv_descr(rocsparse_sptrsv_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    if(descr != nullptr)
    {
        delete descr;
    }
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
