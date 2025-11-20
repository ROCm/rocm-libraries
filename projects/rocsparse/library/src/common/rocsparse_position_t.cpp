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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_pivot_info_t.hpp"
#include "rocsparse_utility.hpp"

rocsparse::position_t::position_t()
    : m_position_indextype((rocsparse_indextype)-1)
    , m_position(nullptr)
{
}

rocsparse::position_t::~position_t()
{
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->m_position));
}

void rocsparse::position_t::free_position_async(hipStream_t stream)
{
    THROW_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->m_position, stream));
}

void* rocsparse::position_t::get_position()
{
    return this->m_position;
}

const void* rocsparse::position_t::get_position() const
{
    return this->m_position;
}

void rocsparse::position_t::set_max_position_async(hipStream_t stream)
{
    if(this->m_position_indextype == rocsparse_indextype_i32)
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse::assign_async(
            (int32_t*)this->m_position, std::numeric_limits<int32_t>::max(), stream));
    }
    else
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse::assign_async(
            (int64_t*)this->m_position, std::numeric_limits<int64_t>::max(), stream));
    }
}

rocsparse_indextype rocsparse::position_t::get_position_indextype() const
{
    return this->m_position_indextype;
}

void rocsparse::position_t::create_position_async(rocsparse_indextype indextype, hipStream_t stream)
{
    this->m_position_indextype = indextype;
    if(this->m_position == nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_position, sizeof(int64_t), stream));
        if(indextype == rocsparse_indextype_i32)
        {
            THROW_IF_HIP_ERROR(hipMemsetAsync(this->m_position, 0, sizeof(int64_t), stream));
        }
    }
    this->set_max_position_async(stream);
}

void rocsparse::position_t::copy_position_async(const position_t* that, hipStream_t stream)
{
    if(that->m_position != nullptr)
    {
        // m position for csrsv, csrsm, csrilu0, csric0
        const size_t J_size = rocsparse::indextype_sizeof(that->m_position_indextype);
        this->create_position_async(this->m_position_indextype, stream);
        THROW_IF_HIP_ERROR(hipMemcpyAsync(
            this->m_position, that->m_position, J_size, hipMemcpyDeviceToDevice, stream));
    }
}

void rocsparse::position_t::copy_async(hipStream_t            stream,
                                       rocsparse_pointer_mode mode,
                                       rocsparse_indextype    indextype,
                                       void*                  value) const
{
    if(this->m_position_indextype == indextype)
    {
        THROW_IF_HIP_ERROR(hipMemcpyAsync(value,
                                          this->m_position,
                                          rocsparse::indextype_sizeof(indextype),
                                          hipMemcpyDefault,
                                          stream));
    }
    else
    {
        if(rocsparse_indextype_i64 == indextype)
        {
            if(mode == rocsparse_pointer_mode_device)
            {
                THROW_IF_HIP_ERROR(hipMemsetAsync(value, 0, sizeof(int64_t), stream));
            }
            else
            {
                *reinterpret_cast<int64_t*>(value) = 0;
            }

            THROW_IF_HIP_ERROR(hipMemcpyAsync(value,
                                              this->m_position,
                                              rocsparse::indextype_sizeof(m_position_indextype),
                                              hipMemcpyDefault,
                                              stream));
        }
        else
        {
            THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                  "cannot reach this state");
        }
    }
}

void rocsparse::position_t::copy_value(hipStream_t            stream,
                                       rocsparse_pointer_mode mode,
                                       int64_t*               value) const
{
    if(this->m_position_indextype == rocsparse_indextype_i32)
    {
        if(mode == rocsparse_pointer_mode_device)
        {
            THROW_IF_HIP_ERROR(hipMemsetAsync(value, 0, sizeof(int64_t), stream));
        }
        else
        {
            *value = 0;
        }
    }
    THROW_IF_HIP_ERROR(hipMemcpyAsync(value,
                                      this->m_position,
                                      rocsparse::indextype_sizeof(this->m_position_indextype),
                                      hipMemcpyDefault,
                                      stream));
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));
}

rocsparse_status rocsparse::position_t::copy_position_async(rocsparse_pointer_mode pointer_mode,
                                                            rocsparse_indextype position_indextype,
                                                            void*               position,
                                                            hipStream_t         stream) const
{
    ROCSPARSE_ROUTINE_TRACE;
    // Stream

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(this->m_position == nullptr)
    {
        rocsparse::set_minus_one_async(stream, pointer_mode, position_indextype, position);
        return rocsparse_status_success;
    }

    // Differentiate between pointer modes
    int64_t value_bytes[1]{0};
    this->copy_value(stream, rocsparse_pointer_mode_host, value_bytes);

    const int64_t value = (this->m_position_indextype == rocsparse_indextype_i32)
                              ? *reinterpret_cast<const int32_t*>(value_bytes)
                              : *reinterpret_cast<const int64_t*>(value_bytes);

    const int64_t mx = (this->m_position_indextype == rocsparse_indextype_i32)
                           ? std::numeric_limits<int32_t>::max()
                           : std::numeric_limits<int64_t>::max();

    if(value == mx)
    {
        rocsparse::set_minus_one_async(stream, pointer_mode, position_indextype, position);
        return rocsparse_status_success;
    }

    this->copy_async(stream, pointer_mode, position_indextype, position);
    return rocsparse_status_zero_pivot;
}
