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

#pragma once

#include "rocsparse_idnvec_descr.hpp"
#include "rocsparse_mat_descr.hpp"
#include "rocsparse_transposed_sparsity_pattern.hpp"

namespace rocsparse
{

    /********************************************************************************
   * \brief triangular_info_t is a structure holding the rocsparse bsrsv, csrsv,
   * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
   * csrilu0_analysis and csric0_analysis.
   *******************************************************************************/
    struct triangular_info_t
    {
    protected:
        // maximum non-zero entries per row
        int64_t max_nnz{0};

        // row permutation
        rocsparse::idnvec_descr_t row_map;
        // position of the diagonal entry of each row
        const rocsparse::idnvec_descr_t*                diag_ind{};
        const rocsparse::transposed_sparsity_pattern_t* transposed{};

        // some data to verify correct execution
        int64_t                     m{0};
        int64_t                     nnz{0};
        const _rocsparse_mat_descr* descr{nullptr};

    public:
        triangular_info_t(const triangular_info_t&);
        triangular_info_t();
        ~triangular_info_t();

        triangular_info_t& operator=(const triangular_info_t&);
        int64_t            get_max_nnz() const;
        void               set_max_nnz(int64_t);

        int64_t get_m() const;
        void    set_m(int64_t);

        const rocsparse::idnvec_descr_t* get_row_map() const;
        rocsparse::idnvec_descr_t*       get_row_map();

        const rocsparse::idnvec_descr_t* get_diag_ind() const;
        void                             set_diag_ind(const rocsparse::idnvec_descr_t*);

        const rocsparse::transposed_sparsity_pattern_t* get_transposed() const;
        void set_transposed(const rocsparse::transposed_sparsity_pattern_t*);

        const _rocsparse_mat_descr* get_descr() const;
        void                        set_descr(const _rocsparse_mat_descr*);

        int64_t get_nnz() const;
        void    set_nnz(int64_t);

        static void destroy(triangular_info_t*);
        static void copy(triangular_info_t* __restrict__*, const triangular_info_t* __restrict__);
    };

}
