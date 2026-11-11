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

#include "rocsparse_idnvec_descr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_utility.hpp"

rocsparse::idnvec_descr_t::~idnvec_descr_t()
{
    this->free();
}

void rocsparse::idnvec_descr_t::free()
{
    if(this->m_values != nullptr)
    {
        WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_values));
        this->m_values = nullptr;
    }
    this->m_size      = 0;
    this->m_indextype = (rocsparse_indextype)-1;
}

rocsparse_status rocsparse::idnvec_descr_t::create_async(int64_t             size,
                                                         rocsparse_indextype indextype,
                                                         hipStream_t         stream)
{
    // Keep the allocation when it already fits so that existing references stay valid.
    if(this->m_values != nullptr && this->m_size == size && this->m_indextype == indextype)
    {
        return rocsparse_status_success;
    }
    this->free();
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
        &this->m_values, rocsparse::indextype_sizeof(indextype) * size, stream));
    this->m_size      = size;
    this->m_indextype = indextype;
    return rocsparse_status_success;
}

void rocsparse::idnvec_descr_t::copy(const rocsparse::idnvec_descr_t& that)
{
    if(&that == this)
    {
        return;
    }

    if(that.m_values == nullptr)
    {
        this->free();
        return;
    }

    const size_t nbytes = rocsparse::indextype_sizeof(that.m_indextype) * that.m_size;
    if(this->m_values == nullptr || this->m_size != that.m_size
       || this->m_indextype != that.m_indextype)
    {
        this->free();
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&this->m_values, nbytes));
        this->m_size      = that.m_size;
        this->m_indextype = that.m_indextype;
    }
    THROW_IF_HIP_ERROR(
        rocsparse_hipMemcpy(this->m_values, that.m_values, nbytes, hipMemcpyDeviceToDevice));
}

int64_t rocsparse::idnvec_descr_t::get_size() const
{
    return this->m_size;
}

rocsparse_indextype rocsparse::idnvec_descr_t::get_indextype() const
{
    return this->m_indextype;
}

const void* rocsparse::idnvec_descr_t::get_values() const
{
    return this->m_values;
}

void* rocsparse::idnvec_descr_t::get_values()
{
    return this->m_values;
}
