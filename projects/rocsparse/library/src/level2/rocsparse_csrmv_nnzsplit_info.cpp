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

void _rocsparse_nnzsplit_info::clear()
{
    WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

    THROW_IF_HIP_ERROR(rocsparse_hipFree(this->id_buffer));

    this->id_buffer              = nullptr;
    this->size                   = 0;
    this->use_starting_block_ids = false;
}

rocsparse_status _rocsparse_nnzsplit_info::allocate_id_buffer(size_t      size,
                                                              bool        use_starting_block_ids,
                                                              size_t      index_size,
                                                              hipStream_t stream)
{
    const size_t buffer_size = index_size * (use_starting_block_ids ? 2 : 1) * (size + 1);

    this->size                   = size;
    this->use_starting_block_ids = use_starting_block_ids;

    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->id_buffer, buffer_size, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(this->id_buffer, 0, buffer_size, stream));

    return rocsparse_status_success;
}
