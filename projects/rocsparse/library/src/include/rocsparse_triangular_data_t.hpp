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
#include "rocsparse-types.h"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_idnvec_descr.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_pivot_info_t.hpp"
#include "rocsparse_transposed_sparsity_pattern.hpp"
#include "rocsparse_triangular_info_t.hpp"
#include <memory>

namespace rocsparse
{
    rocsparse_status gtrm_analysis(rocsparse_handle                          handle,
                                   rocsparse_operation                       operation,
                                   rocsparse_const_spmat_descr               A,
                                   rocsparse::triangular_info_t*             info,
                                   rocsparse::pivot_symbolic_info_t*         pivot_info,
                                   rocsparse::idnvec_descr_t*                diag_ind,
                                   rocsparse::transposed_sparsity_pattern_t* transposed,
                                   size_t                                    buffer_size,
                                   void*                                     buffer);

    rocsparse_status gellsv_analysis(rocsparse_handle                  handle,
                                     rocsparse_operation               operation,
                                     rocsparse_const_spmat_descr       A,
                                     rocsparse::triangular_info_t*     info,
                                     rocsparse::pivot_symbolic_info_t* pivot_info,
                                     size_t                            buffer_size,
                                     void*                             buffer);
}

namespace rocsparse
{

    struct triangular_data_t
    {
    protected:
        rocsparse::triangular_info_t* m_data[4]{};

        // Owned by the mat_info.
        rocsparse::pivot_symbolic_info_t* m_pivot_info{};

        // Owned by the mat_info and shared by every slot; mutable access is passed in by the mat_info.
        const rocsparse::idnvec_descr_t*                m_diag_indices{};
        const rocsparse::transposed_sparsity_pattern_t* m_transposed_sparsity_pattern{};

        // Wires a new triangular_info_t to the storage, which must be the one of the mat_info.
        void attach_storage(
            rocsparse::triangular_info_t*                   trm_info,
            const rocsparse::idnvec_descr_t*                diag_indices,
            const rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern);

        // Points a copied triangular_info_t at this storage instead of the storage of that.
        void       rewire_storage(rocsparse::triangular_info_t*       dst,
                                  const rocsparse::triangular_info_t* src,
                                  const triangular_data_t*            that);
        static int storage_index(rocsparse_operation operation, rocsparse_fill_mode fill_mode)
        {
            return ((operation != rocsparse_operation_none) ? 1 : 0) * 2 + fill_mode;
        }

        // Slots may alias the same triangular_info_t; delete it only when no other slot refers to it.
        void release(int index);

    public:
        triangular_data_t() = default;
        ~triangular_data_t();
        triangular_data_t(const triangular_data_t&) = delete;
        triangular_data_t& operator=(const triangular_data_t&) = delete;

        void copy(const triangular_data_t* that);
        void clear();

        void set_storage(
            rocsparse::pivot_symbolic_info_t*               pivot_info,
            const rocsparse::idnvec_descr_t*                diag_indices,
            const rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern);

        rocsparse::pivot_symbolic_info_t* get_pivot_info()
        {
            return this->m_pivot_info;
        }

        const rocsparse::pivot_symbolic_info_t* get_pivot_info() const
        {
            return this->m_pivot_info;
        }

        rocsparse::triangular_info_t* get(rocsparse_operation operation,
                                          rocsparse_fill_mode fill_mode);

        const rocsparse::triangular_info_t* get(rocsparse_operation operation,
                                                rocsparse_fill_mode fill_mode) const;

        void set(rocsparse_operation           operation,
                 rocsparse_fill_mode           fill_mode,
                 rocsparse::triangular_info_t* trm_info);

        rocsparse_indextype get_indextype_J() const;

        rocsparse::triangular_info_t*
            create(rocsparse_handle                          handle,
                   rocsparse_operation                       operation,
                   rocsparse_const_spmat_descr               A,
                   rocsparse::idnvec_descr_t*                diag_indices,
                   rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern,
                   size_t                                    buffer_size,
                   void*                                     buffer);

        rocsparse_status
            recreate(rocsparse_handle                          handle,
                     rocsparse_operation                       operation,
                     rocsparse_fill_mode                       fill_mode,
                     rocsparse_const_spmat_descr               A,
                     rocsparse::idnvec_descr_t*                diag_indices,
                     rocsparse::transposed_sparsity_pattern_t* transposed_sparsity_pattern,
                     size_t                                    buffer_size,
                     void*                                     buffer);
    };

}
