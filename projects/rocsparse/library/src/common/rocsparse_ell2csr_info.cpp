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

#include "rocsparse_ell2csr_info.hpp"
#include "../conversion/rocsparse_gell2csr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_utility.hpp"

rocsparse::ell2csr_info_t::ell2csr_info_t(int64_t             num_rows,
                                          int64_t             num_cols,
                                          rocsparse_indextype csr_row_ptr_indextype,
                                          rocsparse_indextype csr_col_ind_indextype,
                                          rocsparse_datatype  csr_val_datatype,
                                          hipStream_t         stream)
    : m_num_rows(num_rows)
    , m_num_cols(num_cols)
    , m_csr_row_ptr_indextype(csr_row_ptr_indextype)
    , m_csr_col_ind_indextype(csr_col_ind_indextype)
    , m_csr_val_datatype(csr_val_datatype)
{
    const size_t num_bytes
        = rocsparse::indextype_sizeof(this->m_csr_row_ptr_indextype) * (this->m_num_rows + 1);
    THROW_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_csr_row_ptr, num_bytes, stream));
    THROW_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
}

hipError_t rocsparse::ell2csr_info_t::free_memory(hipStream_t stream)
{
    hipError_t status = hipSuccess;

    const hipError_t e0 = rocsparse_hipFreeAsync(this->m_csr_row_ptr, stream);
    const hipError_t e1 = rocsparse_hipFreeAsync(this->m_csr_col_ind, stream);
    const hipError_t e2 = rocsparse_hipFreeAsync(this->m_csr_val, stream);

    this->m_csr_row_ptr = nullptr;
    this->m_csr_col_ind = nullptr;
    this->m_csr_val     = nullptr;

    if(e0 != hipSuccess)
    {
        status = e0;
    }
    if(e1 != hipSuccess)
    {
        status = e1;
    }
    if(e2 != hipSuccess)
    {
        status = e2;
    }
    return status;
}

rocsparse::ell2csr_info_t::~ell2csr_info_t()
{
    WARNING_IF_HIP_ERROR(this->free_memory(nullptr));
}

rocsparse_status rocsparse::ell2csr_info_t::calculate(rocsparse_handle            handle,
                                                      rocsparse_const_spmat_descr ell)
{
    const hipStream_t stream = handle->stream;

    // Compute the CSR row pointer array and the number of non-zeros.
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gell2csr_nnz(handle,
                                                      this->m_num_rows,
                                                      this->m_num_cols,
                                                      ell->descr,
                                                      ell->ell_width,
                                                      ell->col_type,
                                                      ell->const_col_data,
                                                      ell->descr,
                                                      this->m_csr_row_ptr_indextype,
                                                      this->m_csr_row_ptr,
                                                      &this->m_csr_nnz));

    // Allocate the column indices and values now that the exact number of
    // non-zeros is known. Allocate at least one element so that the arrays are
    // never null (which some downstream argument checks reject).
    const int64_t alloc_count = rocsparse::max(this->m_csr_nnz, static_cast<int64_t>(1));

    const size_t col_ind_bytes
        = rocsparse::indextype_sizeof(this->m_csr_col_ind_indextype) * alloc_count;
    const size_t val_bytes = rocsparse::datatype_sizeof(this->m_csr_val_datatype) * alloc_count;

    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_csr_col_ind, col_ind_bytes, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&this->m_csr_val, val_bytes, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

    // Fill the CSR column indices and values.
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gell2csr(handle,
                                                  this->m_num_rows,
                                                  this->m_num_cols,
                                                  ell->descr,
                                                  ell->ell_width,
                                                  ell->data_type,
                                                  ell->const_val_data,
                                                  ell->col_type,
                                                  ell->const_col_data,
                                                  ell->descr,
                                                  this->m_csr_val_datatype,
                                                  this->m_csr_val,
                                                  this->m_csr_row_ptr_indextype,
                                                  this->m_csr_row_ptr,
                                                  this->m_csr_col_ind_indextype,
                                                  this->m_csr_col_ind));

    return rocsparse_status_success;
}

int64_t rocsparse::ell2csr_info_t::get_csr_nnz() const
{
    return this->m_csr_nnz;
}

rocsparse_indextype rocsparse::ell2csr_info_t::get_csr_row_ptr_indextype() const
{
    return this->m_csr_row_ptr_indextype;
}

rocsparse_indextype rocsparse::ell2csr_info_t::get_csr_col_ind_indextype() const
{
    return this->m_csr_col_ind_indextype;
}

rocsparse_datatype rocsparse::ell2csr_info_t::get_csr_val_datatype() const
{
    return this->m_csr_val_datatype;
}

const void* rocsparse::ell2csr_info_t::get_csr_row_ptr() const
{
    return this->m_csr_row_ptr;
}

const void* rocsparse::ell2csr_info_t::get_csr_col_ind() const
{
    return this->m_csr_col_ind;
}

const void* rocsparse::ell2csr_info_t::get_csr_val() const
{
    return this->m_csr_val;
}
