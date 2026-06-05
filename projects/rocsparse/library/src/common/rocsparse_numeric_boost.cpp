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

#include "rocsparse_numeric_boost.hpp"

void rocsparse::numeric_boost::define(int32_t                enable,
                                      rocsparse_pointer_mode tol_pointer_mode,
                                      rocsparse_datatype     tol_datatype,
                                      const void*            tol,
                                      rocsparse_pointer_mode val_pointer_mode,
                                      const void*            val)
{
    this->m_enable           = enable;
    this->m_tol_pointer_mode = tol_pointer_mode;
    this->m_tol_datatype     = tol_datatype;
    this->m_tol              = tol;
    this->m_val_pointer_mode = val_pointer_mode;
    this->m_val              = val;
}

rocsparse::numeric_boost::~numeric_boost()
{
    this->m_tol_pointer_mode = (rocsparse_pointer_mode)-1;
    this->m_val_pointer_mode = (rocsparse_pointer_mode)-1;
    this->m_enable           = 0;
    this->m_tol_datatype     = (rocsparse_datatype)-1;
    this->m_tol              = nullptr;
    this->m_val              = nullptr;
}

rocsparse::numeric_boost::numeric_boost()
    : m_enable(0)
    , m_tol_pointer_mode((rocsparse_pointer_mode)-1)
    , m_tol_datatype(rocsparse_datatype_f32_r)
    , m_tol(nullptr)
    , m_val_pointer_mode((rocsparse_pointer_mode)-1)
    , m_val(nullptr)
{
}

void rocsparse::numeric_boost::copy(const rocsparse::numeric_boost& that)
{
    this->m_tol_pointer_mode = that.m_tol_pointer_mode;
    this->m_val_pointer_mode = that.m_val_pointer_mode;
    this->m_enable           = that.m_enable;
    this->m_tol_datatype     = that.m_tol_datatype;
    this->m_tol              = that.m_tol;
    this->m_val              = that.m_val;
}

const void* rocsparse::numeric_boost::get_tol() const
{
    return this->m_tol;
}

void rocsparse::numeric_boost::set_tol(const void* value)
{
    this->m_tol = value;
}

const void* rocsparse::numeric_boost::get_val() const
{
    return this->m_val;
}

void rocsparse::numeric_boost::set_val(const void* value)
{
    this->m_val = value;
}

int32_t rocsparse::numeric_boost::get_enable() const
{
    return this->m_enable;
}

void rocsparse::numeric_boost::set_enable(int32_t value)
{
    this->m_enable = value;
}

rocsparse_datatype rocsparse::numeric_boost::get_tol_datatype() const
{
    return this->m_tol_datatype;
}
void rocsparse::numeric_boost::set_tol_datatype(rocsparse_datatype value)
{
    this->m_tol_datatype = value;
}

rocsparse_pointer_mode rocsparse::numeric_boost::get_tol_pointer_mode() const
{
    return this->m_tol_pointer_mode;
}

void rocsparse::numeric_boost::set_tol_pointer_mode(rocsparse_pointer_mode value)
{
    this->m_tol_pointer_mode = value;
}
rocsparse_pointer_mode rocsparse::numeric_boost::get_val_pointer_mode() const
{
    return this->m_val_pointer_mode;
}

void rocsparse::numeric_boost::set_val_pointer_mode(rocsparse_pointer_mode value)
{
    this->m_val_pointer_mode = value;
}
