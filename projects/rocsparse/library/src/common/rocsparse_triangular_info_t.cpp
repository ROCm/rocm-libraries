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

#include "rocsparse_triangular_info_t.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

rocsparse::triangular_info_t::~triangular_info_t()
{
    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    WARNING_IF_HIP_ERROR(rocsparse_hipDeviceSynchronize());
}

void rocsparse::triangular_info_t::set_max_nnz(const int64_t value)
{
    this->max_nnz = value;
}
int64_t rocsparse::triangular_info_t::get_max_nnz() const
{
    return this->max_nnz;
}

const rocsparse::idnvec_descr_t* rocsparse::triangular_info_t::get_row_map() const
{
    return &this->row_map;
}

rocsparse::idnvec_descr_t* rocsparse::triangular_info_t::get_row_map()
{
    return &this->row_map;
}

const rocsparse::idnvec_descr_t* rocsparse::triangular_info_t::get_diag_ind() const
{
    return this->diag_ind;
}

void rocsparse::triangular_info_t::set_diag_ind(const rocsparse::idnvec_descr_t* value)
{
    this->diag_ind = value;
}

const rocsparse::transposed_sparsity_pattern_t* rocsparse::triangular_info_t::get_transposed() const
{
    return this->transposed;
}

void rocsparse::triangular_info_t::set_transposed(
    const rocsparse::transposed_sparsity_pattern_t* value)
{
    this->transposed = value;
}

void rocsparse::triangular_info_t::set_m(const int64_t value)
{
    this->m = value;
}
int64_t rocsparse::triangular_info_t::get_m() const
{
    return this->m;
}

void rocsparse::triangular_info_t::set_nnz(const int64_t value)
{
    this->nnz = value;
}
int64_t rocsparse::triangular_info_t::get_nnz() const
{
    return this->nnz;
}

const _rocsparse_mat_descr* rocsparse::triangular_info_t::get_descr() const
{
    return this->descr;
}
void rocsparse::triangular_info_t::set_descr(const _rocsparse_mat_descr* const value)
{
    this->descr = value;
}

void rocsparse::triangular_info_t::destroy(rocsparse::triangular_info_t* const p_that)
{
    if(p_that != nullptr)
    {
        delete p_that;
    }
}

rocsparse::triangular_info_t::triangular_info_t() {}

rocsparse::triangular_info_t::triangular_info_t(const rocsparse::triangular_info_t& that)
{
    this->max_nnz = that.max_nnz;
    this->m       = that.m;
    this->nnz     = that.nnz;

    // Not owned by the info struct. Just pointers to externally allocated memory
    this->descr = that.descr;

    this->row_map.copy(that.row_map);

    // Not owned either: the triangular data rewires them to its own storage after a copy.
    this->diag_ind   = that.diag_ind;
    this->transposed = that.transposed;
}

rocsparse::triangular_info_t&
    rocsparse::triangular_info_t::operator=(const rocsparse::triangular_info_t& that)
{
    bool invalid = false;
    invalid |= (this->max_nnz != that.max_nnz);
    invalid |= (this->m != that.m);
    invalid |= (this->nnz != that.nnz);
    if(invalid)
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    if(that.row_map.get_values() != nullptr)
    {
        this->row_map.copy(that.row_map);
    }

    this->diag_ind   = that.diag_ind;
    this->transposed = that.transposed;
    this->descr      = that.descr;
    return *this;
}

void rocsparse::triangular_info_t::copy(rocsparse::triangular_info_t* __restrict__* const p_dest,
                                        const rocsparse::triangular_info_t* const __restrict__ that)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(p_dest[0] == nullptr)
    {
        p_dest[0] = new rocsparse::triangular_info_t(that[0]);
    }
    else
    {
        if(p_dest[0] != that)
        {
            p_dest[0][0] = that[0];
        }
    }
}
