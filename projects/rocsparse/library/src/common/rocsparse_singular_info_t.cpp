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

#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_utility.hpp"

void rocsparse::singular_info_t::set_tolerance_pointer(const void*            p,
                                                       rocsparse_pointer_mode pointer_mode,
                                                       rocsparse_datatype     datatype)
{
    this->m_singular_tol_pointer      = reinterpret_cast<const double*>(p);
    this->m_singular_tol_pointer_mode = pointer_mode;
    this->m_singular_tol_datatype     = datatype;
}

rocsparse_status
    rocsparse::singular_info_t::copy_singular_info_async(const rocsparse::singular_info_t* that,
                                                         hipStream_t                       stream)
{
    if(that != nullptr && that != this)
    {
        this->m_singular_tol_host_value[0] = that->m_singular_tol_host_value[0];
        this->m_singular_tol_pointer       = that->m_singular_tol_pointer;
        this->m_singular_tol_pointer_mode  = that->m_singular_tol_pointer_mode;
    }
    RETURN_IF_ROCSPARSE_ERROR(this->copy_position_async(that, stream));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::singular_info_t::create_singular_pivot_async(
    int64_t batch_count, rocsparse_indextype indextype, hipStream_t stream)
{
    RETURN_IF_ROCSPARSE_ERROR(this->create_position_async(batch_count, indextype, stream));
    return rocsparse_status_success;
}

double rocsparse::singular_info_t::get_tolerance_legacy() const
{
    return this->m_singular_tol_host_value[0];
}

void rocsparse::singular_info_t::set_tolerance_legacy(double value)
{
    this->m_singular_tol_host_value[0] = value;
    this->m_singular_tol_pointer       = &this->m_singular_tol_host_value[0];
    this->m_singular_tol_pointer_mode  = rocsparse_pointer_mode_host;
}

rocsparse::singular_info_t::~singular_info_t() {}

const void* rocsparse::singular_info_t::get_tolerance_pointer() const
{
    return this->m_singular_tol_pointer;
}

rocsparse_pointer_mode rocsparse::singular_info_t::get_tolerance_pointer_mode() const
{
    return this->m_singular_tol_pointer_mode;
}

rocsparse_datatype rocsparse::singular_info_t::get_tolerance_datatype() const
{
    return this->m_singular_tol_datatype;
}
