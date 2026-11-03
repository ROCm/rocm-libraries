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

#include "rocsparse_control.hpp"
#include "rocsparse_nnzsplit_info.hpp"
#include "rocsparse_utility.hpp"

_rocsparse_nnzsplit_info::~_rocsparse_nnzsplit_info()
{
    WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

    // starting_block_ids points into the starting_ids allocation, so only one free is needed.
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->starting_ids));

    this->starting_ids                     = nullptr;
    this->starting_block_ids               = nullptr;
    this->starting_ids_size_in_bytes       = 0;
    this->starting_block_ids_size_in_bytes = 0;
    this->size                             = 0;
    this->use_starting_block_ids           = false;
    this->block_size                       = 0;
    this->nnz_per_thread                   = 0;
    this->max_row_nnz                      = 0;
}

void _rocsparse_nnzsplit_info::clear()
{
    WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

    // starting_block_ids points into the starting_ids allocation, so only one free is needed.
    THROW_IF_HIP_ERROR(rocsparse_hipFree(this->starting_ids));

    this->starting_ids                     = nullptr;
    this->starting_block_ids               = nullptr;
    this->starting_ids_size_in_bytes       = 0;
    this->starting_block_ids_size_in_bytes = 0;
    this->size                             = 0;
    this->use_starting_block_ids           = false;
    this->block_size                       = 0;
    this->nnz_per_thread                   = 0;
    this->max_row_nnz                      = 0;
}

rocsparse_status _rocsparse_nnzsplit_info::allocate_starting_ids(hipStream_t stream,
                                                                 size_t      required_blocks,
                                                                 size_t      index_size)
{
    this->size = required_blocks;

    const size_t row_ids = required_blocks + 1;
    this->starting_block_ids_size_in_bytes
        = this->use_starting_block_ids ? index_size * row_ids : 0;
    this->starting_ids_size_in_bytes
        = index_size * row_ids + this->starting_block_ids_size_in_bytes;

    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&this->starting_ids, this->starting_ids_size_in_bytes, stream));
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMemsetAsync(this->starting_ids, 0, this->starting_ids_size_in_bytes, stream));

    this->starting_block_ids = static_cast<char*>(this->starting_ids) + index_size * row_ids;
    return rocsparse_status_success;
}
