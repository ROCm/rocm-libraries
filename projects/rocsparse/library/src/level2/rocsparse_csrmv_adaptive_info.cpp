/*! \file */
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

#include "rocsparse_adaptive_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status _rocsparse_adaptive_info::destroy(hipStream_t stream)
{
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->row_blocks, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->wg_flags, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->wg_ids, stream));

    this->row_blocks = nullptr;
    this->wg_flags   = nullptr;
    this->wg_ids     = nullptr;
    this->size       = 0;
    this->first_row  = 0;
    this->last_row   = 0;
    return rocsparse_status_success;
}

void _rocsparse_adaptive_info::clear(hipStream_t stream)
{
    WARNING_IF_ROCSPARSE_ERROR(this->destroy(stream));
}
