/*! \file */
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

#pragma once

#include "rocsparse_mat_descr.hpp"

namespace rocsparse
{

    /********************************************************************************
   * \brief trm_info_t is a structure holding the rocsparse bsrsv, csrsv,
   * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
   * csrilu0_analysis and csric0_analysis.
   *******************************************************************************/
    struct trm_info_t
    {
    protected:
        // maximum non-zero entries per row
        int64_t max_nnz{0};

        // device array to hold row permutation
        void* row_map{nullptr};
        // device array to hold pointer to diagonal entry
        void* diag_ind{nullptr};
        // device pointers to hold transposed data
        void* transposed_perm{nullptr};
        void* transposed_row_ptr{nullptr};
        void* transposed_col_ind{nullptr};

        // some data to verify correct execution
        int64_t                     m{0};
        int64_t                     nnz{0};
        const _rocsparse_mat_descr* descr{nullptr};
        const void*                 row_ptr{nullptr};
        const void*                 col_ind{nullptr};
        rocsparse_indextype         index_type_I{(rocsparse_indextype)-1};
        rocsparse_indextype         index_type_J{(rocsparse_indextype)-1};

    public:
        trm_info_t(const trm_info_t&);
        trm_info_t();
        ~trm_info_t();

        trm_info_t& operator=(const trm_info_t&);
        int64_t     get_max_nnz() const;
        void        set_max_nnz(int64_t);

        int64_t get_m() const;
        void    set_m(int64_t);

        const void* get_row_map() const;
        void*       get_row_map();
        void**      get_ref_row_map();
        const void* get_diag_ind() const;
        void*       get_diag_ind();
        void**      get_ref_diag_ind();

        const void* get_transposed_perm() const;
        void*       get_transposed_perm();
        void**      get_ref_transposed_perm();
        const void* get_transposed_row_ptr() const;
        void*       get_transposed_row_ptr();
        void**      get_ref_transposed_row_ptr();

        const void* get_transposed_col_ind() const;
        void*       get_transposed_col_ind();
        void**      get_ref_transposed_col_ind();

        const void* get_row_ptr();
        void        set_row_ptr(const void*);

        const void* get_col_ind();
        void        set_col_ind(const void*);

        const _rocsparse_mat_descr* get_descr() const;
        void                        set_descr(const _rocsparse_mat_descr*);
        rocsparse_indextype         get_offset_indextype() const;
        rocsparse_indextype         get_index_indextype() const;
        void                        set_offset_indextype(rocsparse_indextype);
        void                        set_index_indextype(rocsparse_indextype);

        int64_t get_nnz() const;
        void    set_nnz(int64_t);

        static void destroy(trm_info_t*);
        static void copy(trm_info_t* __restrict__*, const trm_info_t* __restrict__);
    };

}
