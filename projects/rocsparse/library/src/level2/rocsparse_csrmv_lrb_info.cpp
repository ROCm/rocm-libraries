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

#include "rocsparse_control.hpp"
#include "rocsparse_lrb_info.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status _rocsparse_lrb_info::destroy(hipStream_t stream)
{
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->wg_flags, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->rows_offsets_scratch, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->rows_bins, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->n_rows_bins, stream));

    this->wg_flags             = nullptr;
    this->rows_offsets_scratch = nullptr;
    this->rows_bins            = nullptr;
    this->n_rows_bins          = nullptr;
    return rocsparse_status_success;
}

void _rocsparse_lrb_info::clear(hipStream_t stream)
{
    WARNING_IF_ROCSPARSE_ERROR(this->destroy(stream));
    this->size = 0;
    memset(this->nRowsBins, 0, sizeof(int64_t) * 32);
}
