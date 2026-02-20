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

#include "rocsparse_trm_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

rocsparse::trm_info_t::~trm_info_t()
{
    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    WARNING_IF_HIP_ERROR(hipDeviceSynchronize());

    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->row_map));
    this->row_map = nullptr;

    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->diag_ind));
    this->diag_ind = nullptr;

    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->transposed_perm));
    this->transposed_perm = nullptr;

    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->transposed_row_ptr));
    this->transposed_row_ptr = nullptr;

    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->transposed_col_ind));
    this->transposed_col_ind = nullptr;
}

void rocsparse::trm_info_t::set_max_nnz(const int64_t value)
{
    this->max_nnz = value;
}
int64_t rocsparse::trm_info_t::get_max_nnz() const
{
    return this->max_nnz;
}

const void* rocsparse::trm_info_t::get_row_map() const
{
    return this->row_map;
}

void* rocsparse::trm_info_t::get_row_map()
{
    return this->row_map;
}

void** rocsparse::trm_info_t::get_ref_row_map()
{
    return &this->row_map;
}

const void* rocsparse::trm_info_t::get_diag_ind() const
{
    return this->diag_ind;
}

void* rocsparse::trm_info_t::get_diag_ind()
{
    return this->diag_ind;
}

void** rocsparse::trm_info_t::get_ref_diag_ind()
{
    return &this->diag_ind;
}

const void* rocsparse::trm_info_t::get_transposed_perm() const
{
    return this->transposed_perm;
}

void* rocsparse::trm_info_t::get_transposed_perm()
{
    return this->transposed_perm;
}

void** rocsparse::trm_info_t::get_ref_transposed_perm()
{
    return &this->transposed_perm;
}

const void* rocsparse::trm_info_t::get_transposed_row_ptr() const
{
    return this->transposed_row_ptr;
}

void* rocsparse::trm_info_t::get_transposed_row_ptr()
{
    return this->transposed_row_ptr;
}

void** rocsparse::trm_info_t::get_ref_transposed_row_ptr()
{
    return &this->transposed_row_ptr;
}

const void* rocsparse::trm_info_t::get_transposed_col_ind() const
{
    return this->transposed_col_ind;
}

void* rocsparse::trm_info_t::get_transposed_col_ind()
{
    return this->transposed_col_ind;
}

void** rocsparse::trm_info_t::get_ref_transposed_col_ind()
{
    return &this->transposed_col_ind;
}

void rocsparse::trm_info_t::set_m(const int64_t value)
{
    this->m = value;
}
int64_t rocsparse::trm_info_t::get_m() const
{
    return this->m;
}

void rocsparse::trm_info_t::set_nnz(const int64_t value)
{
    this->nnz = value;
}
int64_t rocsparse::trm_info_t::get_nnz() const
{
    return this->nnz;
}

void rocsparse::trm_info_t::set_row_ptr(const void* const value)
{
    this->row_ptr = value;
}
const void* rocsparse::trm_info_t::get_row_ptr()
{
    return this->row_ptr;
}

void rocsparse::trm_info_t::set_col_ind(const void* const value)
{
    this->col_ind = value;
}
const void* rocsparse::trm_info_t::get_col_ind()
{
    return this->col_ind;
}

const _rocsparse_mat_descr* rocsparse::trm_info_t::get_descr() const
{
    return this->descr;
}
void rocsparse::trm_info_t::set_descr(const _rocsparse_mat_descr* const value)
{
    this->descr = value;
}

rocsparse_indextype rocsparse::trm_info_t::get_offset_indextype() const
{
    return this->index_type_I;
}
rocsparse_indextype rocsparse::trm_info_t::get_index_indextype() const
{
    return this->index_type_J;
}

void rocsparse::trm_info_t::set_offset_indextype(const rocsparse_indextype value)
{
    this->index_type_I = value;
}
void rocsparse::trm_info_t::set_index_indextype(const rocsparse_indextype value)
{
    this->index_type_J = value;
}

void rocsparse::trm_info_t::destroy(rocsparse::trm_info_t* const p_that)
{
    if(p_that != nullptr)
    {
        delete p_that;
    }
}

rocsparse::trm_info_t::trm_info_t() {}

rocsparse::trm_info_t::trm_info_t(const rocsparse::trm_info_t& that)
{

    this->max_nnz      = that.max_nnz;
    this->m            = that.m;
    this->nnz          = that.nnz;
    this->index_type_I = that.index_type_I;
    this->index_type_J = that.index_type_J;

    // Not owned by the info struct. Just pointers to externally allocated memory
    this->descr   = that.descr;
    this->row_ptr = that.row_ptr;
    this->col_ind = that.col_ind;

    const size_t I_size = rocsparse::indextype_sizeof(that.index_type_I);
    const size_t J_size = rocsparse::indextype_sizeof(that.index_type_J);

    if(that.row_map != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&this->row_map, J_size * that.m));
        THROW_IF_HIP_ERROR(
            hipMemcpy(this->row_map, that.row_map, J_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.diag_ind != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->diag_ind), I_size * that.m));
        THROW_IF_HIP_ERROR(
            hipMemcpy(this->diag_ind, that.diag_ind, I_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.transposed_perm != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_perm), I_size * that.nnz));
        THROW_IF_HIP_ERROR(hipMemcpy(this->transposed_perm,
                                     that.transposed_perm,
                                     I_size * that.nnz,
                                     hipMemcpyDeviceToDevice));
    }

    if(that.transposed_row_ptr != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_row_ptr), I_size * (that.m + 1)));
        THROW_IF_HIP_ERROR(hipMemcpy(this->transposed_row_ptr,
                                     that.transposed_row_ptr,
                                     I_size * (that.m + 1),
                                     hipMemcpyDeviceToDevice));
    }

    if(that.transposed_col_ind != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_col_ind), J_size * that.nnz));

        THROW_IF_HIP_ERROR(hipMemcpy(this->transposed_col_ind,
                                     that.transposed_col_ind,
                                     J_size * that.nnz,
                                     hipMemcpyDeviceToDevice));
    }
}

rocsparse::trm_info_t& rocsparse::trm_info_t::operator=(const rocsparse::trm_info_t& that)
{
    bool invalid = false;
    invalid |= (this->max_nnz != that.max_nnz);
    invalid |= (this->m != that.m);
    invalid |= (this->nnz != that.nnz);
    invalid |= (this->index_type_I != that.index_type_I);
    invalid |= (this->index_type_J != that.index_type_J);
    if(invalid)
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    const size_t I_size = rocsparse::indextype_sizeof(that.index_type_I);
    const size_t J_size = rocsparse::indextype_sizeof(that.index_type_J);

    if(that.row_map != nullptr)
    {
        if(this->row_map == nullptr)
        {
            THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->row_map), J_size * that.m));
        }
        THROW_IF_HIP_ERROR(
            hipMemcpy(this->row_map, that.row_map, J_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.diag_ind != nullptr)
    {
        if(this->diag_ind == nullptr)
        {
            THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->diag_ind), I_size * that.m));
        }
        THROW_IF_HIP_ERROR(
            hipMemcpy(this->diag_ind, that.diag_ind, I_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.transposed_perm != nullptr)
    {
        if(this->transposed_perm == nullptr)
        {
            THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_perm), I_size * that.nnz));
        }
        THROW_IF_HIP_ERROR(hipMemcpy(this->transposed_perm,
                                     that.transposed_perm,
                                     I_size * that.nnz,
                                     hipMemcpyDeviceToDevice));
    }

    if(that.transposed_row_ptr != nullptr)
    {
        if(this->transposed_row_ptr == nullptr)
        {
            THROW_IF_HIP_ERROR(
                rocsparse_hipMalloc(&(this->transposed_row_ptr), I_size * (that.m + 1)));
        }
        THROW_IF_HIP_ERROR(hipMemcpy(this->transposed_row_ptr,
                                     that.transposed_row_ptr,
                                     I_size * (that.m + 1),
                                     hipMemcpyDeviceToDevice));
    }

    if(that.transposed_col_ind != nullptr)
    {
        if(this->transposed_col_ind == nullptr)
        {
            THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_col_ind), J_size * that.nnz));
        }
        THROW_IF_HIP_ERROR(hipMemcpy(this->transposed_col_ind,
                                     that.transposed_col_ind,
                                     J_size * that.nnz,
                                     hipMemcpyDeviceToDevice));
    }

    this->max_nnz      = that.max_nnz;
    this->m            = that.m;
    this->nnz          = that.nnz;
    this->index_type_I = that.index_type_I;
    this->index_type_J = that.index_type_J;

    // Not owned by the info struct. Just pointers to externally allocated memory
    this->descr   = that.descr;
    this->row_ptr = that.row_ptr;
    this->col_ind = that.col_ind;
    return *this;
}

void rocsparse::trm_info_t::copy(rocsparse::trm_info_t* __restrict__* const p_dest,
                                 const rocsparse::trm_info_t* const __restrict__ that)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(p_dest[0] == nullptr)
    {
        p_dest[0] = new rocsparse::trm_info_t(that[0]);
    }
    else
    {
        if(p_dest[0] != that)
        {
            p_dest[0][0] = that[0];
        }
    }
}
