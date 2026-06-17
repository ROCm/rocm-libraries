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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_pivot_info_t.hpp"
#include "rocsparse_utility.hpp"

rocsparse::position_t::position_t()
    : m_position_indextype((rocsparse_indextype)-1) // set invalid value.
    , m_batch_count(1)
    , m_position(nullptr)
{
}

int64_t rocsparse::position_t::get_stride() const
{
    return 1;
}

rocsparse::position_t::~position_t()
{
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_position));
}

rocsparse_status rocsparse::position_t::free_position_async(hipStream_t stream)
{
    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->m_position, stream));
    this->m_position = nullptr;
    return rocsparse_status_success;
}

void* rocsparse::position_t::get_position()
{
    return this->m_position;
}

const void* rocsparse::position_t::get_position() const
{
    return this->m_position;
}

int64_t rocsparse::position_t::get_batch_count() const
{
    return this->m_batch_count;
}

rocsparse_status rocsparse::position_t::set_max_position_async(hipStream_t stream)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(
        this->m_batch_count, this->m_position_indextype, this->m_position, stream));

    return rocsparse_status_success;
}

rocsparse_indextype rocsparse::position_t::get_indextype() const
{
    return this->m_position_indextype;
}

rocsparse_status rocsparse::position_t::create_position_async(int64_t             batch_count,
                                                              rocsparse_indextype indextype,
                                                              hipStream_t         stream)
{
    if((this->m_position != nullptr) && (this->m_batch_count != batch_count))
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->m_position, stream));
        this->m_position = nullptr;
    }

    if(this->m_position == nullptr)
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&this->m_position, sizeof(int64_t) * batch_count, stream));
        if(indextype == rocsparse_indextype_i32)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
                this->m_position, 0, sizeof(int64_t) * batch_count, stream));
        }
        this->m_batch_count        = batch_count;
        this->m_position_indextype = indextype;
    }
    RETURN_IF_ROCSPARSE_ERROR(this->set_max_position_async(stream));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::position_t::copy_position_async(const position_t* that,
                                                            hipStream_t       stream)
{
    if(that->m_position != nullptr)
    {
        const size_t J_size = rocsparse::indextype_sizeof(that->m_position_indextype);
        this->create_position_async(that->m_batch_count, that->m_position_indextype, stream);
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(this->m_position,
                                           that->m_position,
                                           J_size * this->m_batch_count,
                                           hipMemcpyDeviceToDevice,
                                           stream));
    }
    return rocsparse_status_success;
}
