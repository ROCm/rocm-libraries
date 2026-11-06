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

#include "rocsparse_csrmv_nnzsplit_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

void _rocsparse_csrmv_nnzsplit_info::clear()
{
    WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

    THROW_IF_HIP_ERROR(rocsparse_hipFree(this->id_buffer));

    this->id_buffer              = nullptr;
    this->size                   = 0;
    this->use_starting_block_ids = false;
    this->indextype              = deprecated_rocsparse_indextype_u16;
    this->block_size             = 0;
    this->nnz_per_thread         = 0;
    this->max_row_nnz            = 0;
}

rocsparse_status _rocsparse_csrmv_nnzsplit_info::allocate(size_t size,
                                                          bool   use_starting_block_ids,
                                                          rocsparse_indextype indextype,
                                                          hipStream_t         stream)
{
    const size_t buffer_size
        = rocsparse::indextype_sizeof(indextype) * (use_starting_block_ids ? 2 : 1) * (size + 1);

    this->size                   = size;
    this->use_starting_block_ids = use_starting_block_ids;
    this->indextype              = indextype;

    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->id_buffer, buffer_size, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(this->id_buffer, 0, buffer_size, stream));

    return rocsparse_status_success;
}

rocsparse_status
    _rocsparse_csrmv_nnzsplit_info::copy_from(const _rocsparse_csrmv_nnzsplit_info& src)
{
    if(src.id_buffer != nullptr)
    {
        const size_t buffer_size = src.get_size_in_bytes();
        if(this->id_buffer == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&this->id_buffer, buffer_size));
        }
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpy(
            this->id_buffer, src.id_buffer, buffer_size, hipMemcpyDeviceToDevice));
    }

    this->size                   = src.size;
    this->use_starting_block_ids = src.use_starting_block_ids;
    this->indextype              = src.indextype;
    this->block_size             = src.block_size;
    this->nnz_per_thread         = src.nnz_per_thread;
    this->max_row_nnz            = src.max_row_nnz;

    return rocsparse_status_success;
}

size_t _rocsparse_csrmv_nnzsplit_info::get_size() const
{
    return this->size;
}

bool _rocsparse_csrmv_nnzsplit_info::get_use_starting_block_ids() const
{
    return this->use_starting_block_ids;
}

rocsparse_indextype _rocsparse_csrmv_nnzsplit_info::get_indextype() const
{
    return this->indextype;
}

void* _rocsparse_csrmv_nnzsplit_info::get_starting_ids() const
{
    return this->id_buffer;
}

void* _rocsparse_csrmv_nnzsplit_info::get_starting_block_ids() const
{
    if(!this->use_starting_block_ids)
    {
        return nullptr;
    }

    return static_cast<char*>(this->id_buffer)
           + rocsparse::indextype_sizeof(this->indextype) * (this->size + 1);
}

size_t _rocsparse_csrmv_nnzsplit_info::get_size_in_bytes() const
{
    if(this->id_buffer == nullptr)
    {
        return 0;
    }

    return rocsparse::indextype_sizeof(this->indextype) * (this->use_starting_block_ids ? 2 : 1)
           * (this->size + 1);
}
