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

#include "rocsparse_csritsv_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

/********************************************************************************
 * \brief Copy csritsv info.
 *******************************************************************************/
void _rocsparse_csritsv_info::copy(const _rocsparse_csritsv_info* that, hipStream_t stream)
{
    ROCSPARSE_ROUTINE_TRACE;
    //
    // this == nullptr
    //
    if(that == nullptr || this == that)
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }
    this->is_submatrix      = that->is_submatrix;
    this->ptr_end_size      = that->ptr_end_size;
    this->ptr_end_indextype = that->ptr_end_indextype;
    this->ptr_end           = that->ptr_end;
    this->rocsparse::pivot_info_t::copy_pivot_info_async(that, stream);
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));
}

/********************************************************************************
 * \brief Destroy csritsv info.
 *******************************************************************************/
rocsparse_status _rocsparse_csritsv_info::destroy(hipStream_t stream)
{
    ROCSPARSE_ROUTINE_TRACE;
    this->rocsparse::pivot_info_t::destroy(stream);
    if(this->ptr_end != nullptr && this->is_submatrix)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(this->ptr_end, stream));
        this->ptr_end = nullptr;
    }

    if(this->m_csrmv_info)
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_csrmv_info->destroy(stream));
        delete this->m_csrmv_info;
        this->m_csrmv_info = nullptr;
    }
    return rocsparse_status_success;
}
