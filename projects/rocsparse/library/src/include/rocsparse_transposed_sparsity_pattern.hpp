/*! \file */
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

#pragma once

#include "rocsparse_idnvec_descr.hpp"

namespace rocsparse
{
    // Sparsity pattern of the transpose of a CSR matrix, with the permutation into its values.
    struct transposed_sparsity_pattern_t
    {
    protected:
        rocsparse::idnvec_descr_t m_perm;
        rocsparse::idnvec_descr_t m_row_ptr;
        rocsparse::idnvec_descr_t m_col_ind;

    public:
        transposed_sparsity_pattern_t()                                     = default;
        ~transposed_sparsity_pattern_t()                                    = default;
        transposed_sparsity_pattern_t(const transposed_sparsity_pattern_t&) = delete;
        transposed_sparsity_pattern_t& operator=(const transposed_sparsity_pattern_t&) = delete;

        void free();
        void copy(const transposed_sparsity_pattern_t& that);

        const rocsparse::idnvec_descr_t* get_perm() const;
        rocsparse::idnvec_descr_t*       get_perm();
        const rocsparse::idnvec_descr_t* get_row_ptr() const;
        rocsparse::idnvec_descr_t*       get_row_ptr();
        const rocsparse::idnvec_descr_t* get_col_ind() const;
        rocsparse::idnvec_descr_t*       get_col_ind();
    };
}
