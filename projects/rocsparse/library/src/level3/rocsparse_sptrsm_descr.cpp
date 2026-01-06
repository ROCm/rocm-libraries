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

#include "rocsparse_sptrsm_descr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_logging.hpp"

void _rocsparse_sptrsm_descr::set_csrsm_info(rocsparse_csrsm_info value)
{
    this->m_csrsm_info = std::shared_ptr<_rocsparse_csrsm_info>(value);
}

rocsparse_csrsm_info _rocsparse_sptrsm_descr::get_csrsm_info()
{
    return this->m_csrsm_info.get();
}
void _rocsparse_sptrsm_descr::set_shared_csrsm_info(std::shared_ptr<_rocsparse_csrsm_info> value)
{
    this->m_csrsm_info = value;
}

_rocsparse_sptrsm_descr::~_rocsparse_sptrsm_descr()
{
    m_stage            = ((rocsparse_sptrsm_stage)-1);
    m_alg              = ((rocsparse_sptrsm_alg)-1);
    m_operation_A      = ((rocsparse_operation)-1);
    m_operation_X      = ((rocsparse_operation)-1);
    m_X_datatype       = ((rocsparse_datatype)-1);
    m_Y_datatype       = ((rocsparse_datatype)-1);
    m_X_order          = ((rocsparse_order)-1);
    m_Y_order          = ((rocsparse_order)-1);
    m_scalar_datatype  = ((rocsparse_datatype)-1);
    m_compute_datatype = ((rocsparse_datatype)-1);
    m_nrhs             = -1;
    m_scalar_alpha     = nullptr;
    m_analysis_policy  = ((rocsparse_analysis_policy)-1);
    this->m_csrsm_info.reset();
    this->m_scalar_alpha = nullptr;
}
_rocsparse_sptrsm_descr::_rocsparse_sptrsm_descr()
    : m_stage((rocsparse_sptrsm_stage)-1)
    , m_alg((rocsparse_sptrsm_alg)-1)
    , m_operation_A((rocsparse_operation)-1)
    , m_operation_X((rocsparse_operation)-1)
    , m_X_datatype((rocsparse_datatype)-1)
    , m_Y_datatype((rocsparse_datatype)-1)
    , m_X_order((rocsparse_order)-1)
    , m_Y_order((rocsparse_order)-1)
    , m_scalar_datatype((rocsparse_datatype)-1)
    , m_compute_datatype((rocsparse_datatype)-1)
    , m_nrhs(-1)
    , m_scalar_alpha(nullptr)
    , m_analysis_policy((rocsparse_analysis_policy)-1)
{
}

rocsparse_analysis_policy _rocsparse_sptrsm_descr::get_analysis_policy() const
{
    return this->m_analysis_policy;
}

void _rocsparse_sptrsm_descr::set_analysis_policy(rocsparse_analysis_policy value)
{
    this->m_analysis_policy = value;
}

rocsparse_sptrsm_stage _rocsparse_sptrsm_descr::get_stage() const
{
    return this->m_stage;
}

rocsparse_sptrsm_alg _rocsparse_sptrsm_descr::get_alg() const
{
    return this->m_alg;
}

int64_t _rocsparse_sptrsm_descr::get_nrhs() const
{
    return this->m_nrhs;
}

rocsparse_operation _rocsparse_sptrsm_descr::get_operation_A() const
{
    return this->m_operation_A;
}

rocsparse_operation _rocsparse_sptrsm_descr::get_operation_X() const
{
    return this->m_operation_X;
}

rocsparse_datatype _rocsparse_sptrsm_descr::get_scalar_datatype() const
{
    return this->m_scalar_datatype;
}

rocsparse_datatype _rocsparse_sptrsm_descr::get_compute_datatype() const
{
    return this->m_compute_datatype;
}

const void* _rocsparse_sptrsm_descr::get_scalar_alpha() const
{
    return this->m_scalar_alpha;
}

rocsparse_datatype _rocsparse_sptrsm_descr::get_X_datatype() const
{
    return this->m_X_datatype;
}

rocsparse_datatype _rocsparse_sptrsm_descr::get_Y_datatype() const
{
    return this->m_Y_datatype;
}

rocsparse_order _rocsparse_sptrsm_descr::get_X_order() const
{
    return this->m_X_order;
}

rocsparse_order _rocsparse_sptrsm_descr::get_Y_order() const
{
    return this->m_Y_order;
}

void _rocsparse_sptrsm_descr::set_nrhs(int64_t value)
{
    this->m_nrhs = value;
}

void _rocsparse_sptrsm_descr::set_stage(rocsparse_sptrsm_stage value)
{
    this->m_stage = value;
}

void _rocsparse_sptrsm_descr::set_alg(rocsparse_sptrsm_alg value)
{
    this->m_alg = value;
}

void _rocsparse_sptrsm_descr::set_operation_A(rocsparse_operation value)
{
    this->m_operation_A = value;
}

void _rocsparse_sptrsm_descr::set_operation_X(rocsparse_operation value)
{
    this->m_operation_X = value;
}

void _rocsparse_sptrsm_descr::set_scalar_datatype(rocsparse_datatype value)
{
    this->m_scalar_datatype = value;
}

void _rocsparse_sptrsm_descr::set_compute_datatype(rocsparse_datatype value)
{
    this->m_compute_datatype = value;
}

void _rocsparse_sptrsm_descr::set_scalar_alpha(const void* value)
{
    this->m_scalar_alpha = value;
}

void _rocsparse_sptrsm_descr::set_X_datatype(rocsparse_datatype value)
{
    this->m_X_datatype = value;
}
void _rocsparse_sptrsm_descr::set_Y_datatype(rocsparse_datatype value)
{
    this->m_Y_datatype = value;
}

void _rocsparse_sptrsm_descr::set_X_order(rocsparse_order value)
{
    this->m_X_order = value;
}
void _rocsparse_sptrsm_descr::set_Y_order(rocsparse_order value)
{
    this->m_Y_order = value;
}

extern "C" rocsparse_status rocsparse_create_sptrsm_descr(rocsparse_sptrsm_descr* descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_POINTER(0, descr);
    *descr = new _rocsparse_sptrsm_descr();
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_destroy_sptrsm_descr(rocsparse_sptrsm_descr descr)
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
