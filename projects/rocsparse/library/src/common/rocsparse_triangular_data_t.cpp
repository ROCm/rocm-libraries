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

#include "rocsparse_triangular_data_t.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_utility.hpp"

void rocsparse::triangular_data_t::copy(const rocsparse::triangular_data_t* that)
{
    if(that != nullptr && that != this)
    {
        for(int i = 0; i < 4; ++i)
        {
            const rocsparse::triangular_info_t* src = that->m_data[i];
            if(src == nullptr)
            {
                continue;
            }

            int j = 0;
            while(j < i && that->m_data[j] != src)
            {
                ++j;
            }

            if(j < i)
            {
                // Preserve the source aliasing.
                if(this->m_data[i] != this->m_data[j])
                {
                    this->release(i);
                    this->m_data[i] = this->m_data[j];
                }
            }
            else
            {
                // Do not overwrite a triangular_info_t still referenced by another slot.
                for(int k = 0; k < 4; ++k)
                {
                    if(k != i && this->m_data[k] == this->m_data[i])
                    {
                        this->m_data[i] = nullptr;
                        break;
                    }
                }
                rocsparse::triangular_info_t::copy(&this->m_data[i], src);
                this->rewire_storage(this->m_data[i], src, that);
            }
        }
    }
}

void rocsparse::triangular_data_t::set_storage(
    rocsparse::pivot_symbolic_info_t*               pivot_info,
    const rocsparse::idnvec_descr_t*                diag_indices,
    const rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern)
{
    this->m_pivot_info                  = pivot_info;
    this->m_diag_indices                = diag_indices;
    this->m_transposed_sparsity_pattern = transposed_sparsity_pattern;
}

void rocsparse::triangular_data_t::attach_storage(
    rocsparse::triangular_info_t*                   trm_info,
    const rocsparse::idnvec_descr_t*                diag_indices,
    const rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern)
{
    if(this->m_diag_indices == nullptr || this->m_transposed_sparsity_pattern == nullptr
       || diag_indices != this->m_diag_indices
       || transposed_sparsity_pattern != this->m_transposed_sparsity_pattern)
    {
        delete trm_info;
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    trm_info->set_diag_ind(this->m_diag_indices);
    trm_info->set_transposed(this->m_transposed_sparsity_pattern);
}

void rocsparse::triangular_data_t::rewire_storage(rocsparse::triangular_info_t*       dst,
                                                  const rocsparse::triangular_info_t* src,
                                                  const triangular_data_t*            that)
{
    if(this->m_diag_indices == nullptr || that->m_diag_indices == nullptr)
    {
        return;
    }

    if(src->get_diag_ind() == that->m_diag_indices)
    {
        dst->set_diag_ind(this->m_diag_indices);
    }

    const rocsparse::transposed_sparsity_pattern_t* src_transposed
        = that->m_transposed_sparsity_pattern;
    if(src->get_transposed() == src_transposed)
    {
        const rocsparse::transposed_sparsity_pattern_t* dst_transposed
            = this->m_transposed_sparsity_pattern;
        dst->set_transposed(dst_transposed);
    }
}

void rocsparse::triangular_data_t::release(int index)
{
    rocsparse::triangular_info_t* trm_info = this->m_data[index];
    this->m_data[index]                    = nullptr;
    if(trm_info != nullptr)
    {
        for(int i = 0; i < 4; ++i)
        {
            if(this->m_data[i] == trm_info)
            {
                return;
            }
        }
        delete trm_info;
    }
}

void rocsparse::triangular_data_t::clear()
{
    for(int i = 0; i < 4; ++i)
    {
        this->release(i);
    }
}

rocsparse::triangular_data_t::~triangular_data_t()
{
    this->clear();
}

rocsparse::triangular_info_t* rocsparse::triangular_data_t::get(rocsparse_operation operation,
                                                                rocsparse_fill_mode fill_mode)
{
    const int idx = rocsparse::triangular_data_t::storage_index(operation, fill_mode);
    return this->m_data[idx];
}

const rocsparse::triangular_info_t*
    rocsparse::triangular_data_t::get(rocsparse_operation operation,
                                      rocsparse_fill_mode fill_mode) const
{
    const auto idx = rocsparse::triangular_data_t::storage_index(operation, fill_mode);
    return this->m_data[idx];
}

void rocsparse::triangular_data_t::set(rocsparse_operation           operation,
                                       rocsparse_fill_mode           fill_mode,
                                       rocsparse::triangular_info_t* trm_info)
{
    const int idx = rocsparse::triangular_data_t::storage_index(operation, fill_mode);
    if(this->m_data[idx] != trm_info)
    {
        this->release(idx);
        this->m_data[idx] = trm_info;
    }
}

rocsparse_indextype rocsparse::triangular_data_t::get_indextype_J() const
{
    for(int i = 0; i < 4; ++i)
    {
        if(this->m_data[i] != nullptr)
            return this->m_data[i]->get_row_map()->get_indextype();
    }
    return rocsparse_indextype_i32;
}

rocsparse::triangular_info_t* rocsparse::triangular_data_t::create(
    rocsparse_handle                          handle,
    rocsparse_operation                       operation,
    rocsparse_const_spmat_descr               A,
    rocsparse::idnvec_descr_t*                diag_indices,
    rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern,
    size_t                                    buffer_size,
    void*                                     buffer)
{
    rocsparse::triangular_info_t* trm_info = new rocsparse::triangular_info_t();
    this->attach_storage(trm_info, diag_indices, transposed_sparsity_pattern);

    switch(A->format)
    {
    case rocsparse_format_ell:
    {
        THROW_IF_ROCSPARSE_ERROR(rocsparse::gellsv_analysis(
            handle, operation, A, trm_info, this->m_pivot_info, buffer_size, buffer));
        break;
    }
    case rocsparse_format_csr:
    case rocsparse_format_bsr:
    {
        // The analysis of a BSR matrix only levels its block sparsity pattern.
        THROW_IF_ROCSPARSE_ERROR(rocsparse::gtrm_analysis(handle,
                                                          operation,
                                                          A,
                                                          trm_info,
                                                          this->m_pivot_info,
                                                          diag_indices,
                                                          transposed_sparsity_pattern,
                                                          buffer_size,
                                                          buffer));
        break;
    }
    default:
    {
        delete trm_info;
        THROW_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    }

    return trm_info;
}

rocsparse_status rocsparse::triangular_data_t::recreate(
    rocsparse_handle                          handle,
    rocsparse_operation                       operation,
    rocsparse_fill_mode                       fill_mode,
    rocsparse_const_spmat_descr               A,
    rocsparse::idnvec_descr_t*                diag_indices,
    rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern,
    size_t                                    buffer_size,
    void*                                     buffer)
{
    const int index = this->storage_index(operation, fill_mode);
    this->release(index);
    this->m_data[index] = this->create(
        handle, operation, A, diag_indices, transposed_sparsity_pattern, buffer_size, buffer);
    return rocsparse_status_success;
}
