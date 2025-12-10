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

#include "rocsparse_pivot_info_t.hpp"
#include "rocsparse_control.hpp"

rocsparse_status rocsparse::pivot_info_t::set_pivot_batch_count(const int64_t value,
                                                                hipStream_t   stream)
{
    RETURN_IF_ROCSPARSE_ERROR(this->set_position_batch_count(value, stream));
    return rocsparse_status_success;
}

int64_t rocsparse::pivot_info_t::get_pivot_batch_count() const
{
    return this->get_position_batch_count();
}

rocsparse_status
    rocsparse::pivot_info_t::copy_zero_pivot_async(int64_t                batch_count,
                                                   rocsparse_pointer_mode pointer_mode,
                                                   rocsparse_indextype    position_indextype,
                                                   void*                  position,
                                                   hipStream_t            stream) const
{
    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((batch_count != this->get_pivot_batch_count())
                                               ? rocsparse_status_invalid_value
                                               : rocsparse_status_success,
                                           "incompatible batch_count");
    const rocsparse_status status
        = this->copy_position_async(pointer_mode, position_indextype, position, stream);

    if(status == rocsparse_status_zero_pivot)
    {
        return status;
    }

    RETURN_IF_ROCSPARSE_ERROR(status);
    return rocsparse_status_success;
}

rocsparse_status rocsparse::pivot_info_t::create_zero_pivot_async(int64_t             batch_count,
                                                                  rocsparse_indextype indextype,
                                                                  hipStream_t         stream)
{
    RETURN_IF_ROCSPARSE_ERROR(this->create_position_async(batch_count, indextype, stream));
    return rocsparse_status_success;
}

const void* rocsparse::pivot_info_t::get_zero_pivot() const
{
    return this->get_position();
}

void* rocsparse::pivot_info_t::get_zero_pivot()
{
    return this->get_position();
}

rocsparse_indextype rocsparse::pivot_info_t::get_zero_pivot_indextype() const
{
    return this->get_position_indextype();
}

rocsparse_status rocsparse::pivot_info_t::copy_pivot_info_async(const pivot_info_t* that,
                                                                hipStream_t         stream)
{
    RETURN_IF_ROCSPARSE_ERROR(this->copy_position_async(that, stream));
    return rocsparse_status_success;
}

int64_t rocsparse::pivot_info_t::get_zero_pivot_stride() const
{
    return this->get_position_stride();
}
