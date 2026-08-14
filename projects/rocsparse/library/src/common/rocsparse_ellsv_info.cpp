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

#include "rocsparse_ellsv_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_utility.hpp"

rocsparse::ellsv_info_t::ellsv_info_t(int64_t             num_rows,
                                      rocsparse_indextype index_type,
                                      rocsparse_datatype  value_type,
                                      hipStream_t         stream)
    : m_num_rows(num_rows)
    , m_index_type(index_type)
    , m_value_type(value_type)
{
    const size_t num_bytes = rocsparse::indextype_sizeof(this->m_index_type) * this->m_num_rows;
    THROW_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_row_map, num_bytes, stream));
    THROW_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
}

hipError_t rocsparse::ellsv_info_t::free_transposed(hipStream_t stream)
{
    const hipError_t e0 = rocsparse_hipFreeAsync(this->m_transposed_col_ind, stream);
    const hipError_t e1 = rocsparse_hipFreeAsync(this->m_transposed_val, stream);

    this->m_transposed_col_ind = nullptr;
    this->m_transposed_val     = nullptr;
    this->m_transposed_width   = 0;

    return (e0 != hipSuccess) ? e0 : e1;
}

hipError_t rocsparse::ellsv_info_t::free_memory(hipStream_t stream)
{
    const hipError_t e0 = rocsparse_hipFreeAsync(this->m_row_map, stream);
    this->m_row_map      = nullptr;
    const hipError_t e1  = this->free_transposed(stream);
    return (e0 != hipSuccess) ? e0 : e1;
}

rocsparse::ellsv_info_t::~ellsv_info_t()
{
    WARNING_IF_HIP_ERROR(this->free_memory(nullptr));
}

rocsparse_status rocsparse::ellsv_info_t::allocate_transposed(int64_t width, hipStream_t stream)
{
    RETURN_IF_HIP_ERROR(this->free_transposed(stream));

    this->m_transposed_width = width;

    const int64_t count     = rocsparse::max(this->m_num_rows * width, static_cast<int64_t>(1));
    const size_t  col_bytes = rocsparse::indextype_sizeof(this->m_index_type) * count;
    const size_t  val_bytes = rocsparse::datatype_sizeof(this->m_value_type) * count;

    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_transposed_col_ind, col_bytes, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_transposed_val, val_bytes, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

    return rocsparse_status_success;
}

bool rocsparse::ellsv_info_t::matches(rocsparse_operation trans,
                                      rocsparse_fill_mode fill_mode,
                                      rocsparse_diag_type diag_type) const
{
    return this->m_computed && this->m_trans == trans && this->m_fill_mode == fill_mode
           && this->m_diag_type == diag_type;
}

void rocsparse::ellsv_info_t::set_config(rocsparse_operation trans,
                                         rocsparse_fill_mode fill_mode,
                                         rocsparse_diag_type diag_type)
{
    this->m_trans     = trans;
    this->m_fill_mode = fill_mode;
    this->m_diag_type = diag_type;
    this->m_computed  = true;
}

int64_t rocsparse::ellsv_info_t::get_num_rows() const
{
    return this->m_num_rows;
}

rocsparse_indextype rocsparse::ellsv_info_t::get_index_type() const
{
    return this->m_index_type;
}

rocsparse_datatype rocsparse::ellsv_info_t::get_value_type() const
{
    return this->m_value_type;
}

void* rocsparse::ellsv_info_t::get_row_map()
{
    return this->m_row_map;
}

const void* rocsparse::ellsv_info_t::get_row_map() const
{
    return this->m_row_map;
}

int64_t rocsparse::ellsv_info_t::get_transposed_width() const
{
    return this->m_transposed_width;
}

void* rocsparse::ellsv_info_t::get_transposed_col_ind()
{
    return this->m_transposed_col_ind;
}

const void* rocsparse::ellsv_info_t::get_transposed_col_ind() const
{
    return this->m_transposed_col_ind;
}

void* rocsparse::ellsv_info_t::get_transposed_val()
{
    return this->m_transposed_val;
}

const void* rocsparse::ellsv_info_t::get_transposed_val() const
{
    return this->m_transposed_val;
}
