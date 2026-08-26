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

#include "rocsparse_trm_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_utility.hpp"

namespace
{
    // Duplicates a cached transpose, including the arrays it owns, so that the
    // clone can outlive the original.
    rocsparse_spmat_descr clone_transposed_matrix(rocsparse_const_spmat_descr that)
    {
        if(that == nullptr)
        {
            return nullptr;
        }

        // Only the ELL layout is ever cached here; anything else would need its
        // own array inventory to be duplicated correctly.
        if(that->format != rocsparse_format_ell)
        {
            THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }

        const size_t count = static_cast<size_t>(that->rows) * static_cast<size_t>(that->ell_width);
        const size_t col_bytes = count * rocsparse::indextype_sizeof(that->col_type);
        const size_t val_bytes = count * rocsparse::datatype_sizeof(that->data_type);

        void* col_data{nullptr};
        void* val_data{nullptr};
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&col_data, col_bytes));
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&val_data, val_bytes));
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(
            col_data, that->const_col_data, col_bytes, hipMemcpyDeviceToDevice));
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(
            val_data, that->const_val_data, val_bytes, hipMemcpyDeviceToDevice));

        rocsparse_spmat_descr clone{nullptr};
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_ell_descr(&clone,
                                                            that->rows,
                                                            that->cols,
                                                            col_data,
                                                            val_data,
                                                            that->ell_width,
                                                            that->col_type,
                                                            that->idx_base,
                                                            that->data_type));

        clone->descr->fill_mode = that->descr->fill_mode;
        clone->descr->diag_type = that->descr->diag_type;

        return clone;
    }
}

rocsparse::trm_info_t::~trm_info_t()
{
    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    WARNING_IF_HIP_ERROR(rocsparse_hipDeviceSynchronize());

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

    this->clear_transposed_matrix();
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

rocsparse_const_spmat_descr rocsparse::trm_info_t::get_transposed_matrix() const
{
    return this->transposed_matrix;
}

rocsparse_spmat_descr rocsparse::trm_info_t::get_transposed_matrix()
{
    return this->transposed_matrix;
}

void rocsparse::trm_info_t::set_transposed_matrix(rocsparse_spmat_descr value)
{
    if(this->transposed_matrix == value)
    {
        return;
    }
    this->clear_transposed_matrix();
    this->transposed_matrix = value;
}

void rocsparse::trm_info_t::clear_transposed_matrix()
{
    if(this->transposed_matrix == nullptr)
    {
        return;
    }

    // rocsparse_destroy_spmat_descr releases the descriptor and its nested
    // objects but never the matrix arrays, which this info struct owns.
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->transposed_matrix->col_data));
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->transposed_matrix->val_data));
    WARNING_IF_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(this->transposed_matrix));

    this->transposed_matrix = nullptr;
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
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(
            this->row_map, that.row_map, J_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.diag_ind != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->diag_ind), I_size * that.m));
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(
            this->diag_ind, that.diag_ind, I_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.transposed_perm != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_perm), I_size * that.nnz));
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(this->transposed_perm,
                                               that.transposed_perm,
                                               I_size * that.nnz,
                                               hipMemcpyDeviceToDevice));
    }

    if(that.transposed_row_ptr != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_row_ptr), I_size * (that.m + 1)));
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(this->transposed_row_ptr,
                                               that.transposed_row_ptr,
                                               I_size * (that.m + 1),
                                               hipMemcpyDeviceToDevice));
    }

    if(that.transposed_col_ind != nullptr)
    {
        THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_col_ind), J_size * that.nnz));

        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(this->transposed_col_ind,
                                               that.transposed_col_ind,
                                               J_size * that.nnz,
                                               hipMemcpyDeviceToDevice));
    }

    this->transposed_matrix = clone_transposed_matrix(that.transposed_matrix);
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
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(
            this->row_map, that.row_map, J_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.diag_ind != nullptr)
    {
        if(this->diag_ind == nullptr)
        {
            THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->diag_ind), I_size * that.m));
        }
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(
            this->diag_ind, that.diag_ind, I_size * that.m, hipMemcpyDeviceToDevice));
    }

    if(that.transposed_perm != nullptr)
    {
        if(this->transposed_perm == nullptr)
        {
            THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&(this->transposed_perm), I_size * that.nnz));
        }
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(this->transposed_perm,
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
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(this->transposed_row_ptr,
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
        THROW_IF_HIP_ERROR(rocsparse_hipMemcpy(this->transposed_col_ind,
                                               that.transposed_col_ind,
                                               J_size * that.nnz,
                                               hipMemcpyDeviceToDevice));
    }

    this->set_transposed_matrix(clone_transposed_matrix(that.transposed_matrix));

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
