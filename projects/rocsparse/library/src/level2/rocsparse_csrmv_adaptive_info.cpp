/*! \file */
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

#include "rocsparse_adaptive_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

_rocsparse_adaptive_info::~_rocsparse_adaptive_info()
{

    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->row_blocks));
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->wg_flags));
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->wg_ids));

    this->row_blocks = nullptr;
    this->wg_flags   = nullptr;
    this->wg_ids     = nullptr;
    this->size       = 0;
    this->first_row  = 0;
    this->last_row   = 0;
}

void _rocsparse_adaptive_info::clear()
{
    THROW_IF_HIP_ERROR(rocsparse_hipFree(this->row_blocks));
    THROW_IF_HIP_ERROR(rocsparse_hipFree(this->wg_flags));
    THROW_IF_HIP_ERROR(rocsparse_hipFree(this->wg_ids));
    this->row_blocks = nullptr;
    this->wg_flags   = nullptr;
    this->wg_ids     = nullptr;
    this->size       = 0;
    this->first_row  = 0;
    this->last_row   = 0;
}
