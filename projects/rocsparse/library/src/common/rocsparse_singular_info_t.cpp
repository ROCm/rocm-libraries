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

#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status
    rocsparse::singular_info_t::copy_singular_pivot_async(rocsparse_pointer_mode pointer_mode,
                                                          rocsparse_indextype    position_indextype,
                                                          void*                  position,
                                                          hipStream_t            stream) const
{
    auto status = this->copy_position_async(pointer_mode, position_indextype, position, stream);
    if(status == rocsparse_status_zero_pivot)
    {
        return status;
    }
    RETURN_IF_ROCSPARSE_ERROR(status);
    return rocsparse_status_success;
}

void rocsparse::singular_info_t::copy_singular_info_async(const rocsparse::singular_info_t* that,
                                                          hipStream_t                       stream)
{
    if(that != nullptr && that != this)
    {
        this->m_singular_tol = that->m_singular_tol;
    }
    return this->copy_position_async(that, stream);
}

void* rocsparse::singular_info_t::get_singular_pivot()
{
    return this->get_position();
}

const void* rocsparse::singular_info_t::get_singular_pivot() const
{
    return this->get_position();
}

void rocsparse::singular_info_t::create_singular_pivot_async(rocsparse_indextype indextype,
                                                             hipStream_t         stream)
{
    this->create_position_async(indextype, stream);
}

rocsparse_indextype rocsparse::singular_info_t::get_singular_pivot_indextype() const
{
    return this->get_position_indextype();
}

double rocsparse::singular_info_t::get_singular_tol() const
{
    return this->m_singular_tol;
}

void rocsparse::singular_info_t::set_singular_tol(double value)
{
    this->m_singular_tol = value;
}

rocsparse::singular_info_t::~singular_info_t() {}
