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

#include "rocsparse_sorted_coo2csr_info.hpp"
#include "../conversion/rocsparse_gcoo2csr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

const void* rocsparse::sorted_coo2csr_info_t::get_row_ptr() const
{
    return this->m_row_ptr;
}

rocsparse::sorted_coo2csr_info_t::sorted_coo2csr_info_t(int64_t             num_rows,
                                                        rocsparse_indextype ptr_indextype,
                                                        hipStream_t         stream)
    : m_num_rows(num_rows)
    , m_row_ptr_indextype(ptr_indextype)
{
    const size_t num_bytes
        = rocsparse::indextype_sizeof(this->m_row_ptr_indextype) * (this->m_num_rows + 1);
    THROW_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_row_ptr, num_bytes, stream));
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));
}

hipError_t rocsparse::sorted_coo2csr_info_t::free_memory(hipStream_t stream)
{
    auto e = rocsparse_hipFreeAsync(this->m_row_ptr, stream);
    if(e == hipSuccess)
    {
        this->m_row_ptr = nullptr;
        return hipSuccess;
    }
    else
    {
        return e;
    }
}

rocsparse::sorted_coo2csr_info_t::~sorted_coo2csr_info_t()
{
    hipStream_t default_stream = 0;
    std::ignore                = this->free_memory(default_stream);
}

rocsparse_status
    rocsparse::sorted_coo2csr_info_t::calculate(rocsparse_handle     handle,
                                                int64_t              coo_row_ind_size,
                                                const void*          coo_row_ind,
                                                rocsparse_indextype  coo_row_ind_indextype,
                                                rocsparse_index_base coo_row_ind_index_base)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcoo2csr(handle,
                                                  coo_row_ind_indextype,
                                                  coo_row_ind,
                                                  coo_row_ind_size,
                                                  this->m_num_rows,
                                                  this->m_row_ptr_indextype,
                                                  this->m_row_ptr,
                                                  coo_row_ind_index_base));
    return rocsparse_status_success;
}
