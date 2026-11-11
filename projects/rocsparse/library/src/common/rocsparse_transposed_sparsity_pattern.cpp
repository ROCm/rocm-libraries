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

#include "rocsparse_transposed_sparsity_pattern.hpp"

void rocsparse::transposed_sparsity_pattern_t::free()
{
    this->m_perm.free();
    this->m_row_ptr.free();
    this->m_col_ind.free();
}

void rocsparse::transposed_sparsity_pattern_t::copy(const transposed_sparsity_pattern_t& that)
{
    this->m_perm.copy(that.m_perm);
    this->m_row_ptr.copy(that.m_row_ptr);
    this->m_col_ind.copy(that.m_col_ind);
}

const rocsparse::idnvec_descr_t* rocsparse::transposed_sparsity_pattern_t::get_perm() const
{
    return &this->m_perm;
}

rocsparse::idnvec_descr_t* rocsparse::transposed_sparsity_pattern_t::get_perm()
{
    return &this->m_perm;
}

const rocsparse::idnvec_descr_t* rocsparse::transposed_sparsity_pattern_t::get_row_ptr() const
{
    return &this->m_row_ptr;
}

rocsparse::idnvec_descr_t* rocsparse::transposed_sparsity_pattern_t::get_row_ptr()
{
    return &this->m_row_ptr;
}

const rocsparse::idnvec_descr_t* rocsparse::transposed_sparsity_pattern_t::get_col_ind() const
{
    return &this->m_col_ind;
}

rocsparse::idnvec_descr_t* rocsparse::transposed_sparsity_pattern_t::get_col_ind()
{
    return &this->m_col_ind;
}
