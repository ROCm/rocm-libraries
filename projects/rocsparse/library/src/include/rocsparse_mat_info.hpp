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

#include "rocsparse_bsric0_info.hpp"
#include "rocsparse_bsrilu0_info.hpp"
#include "rocsparse_bsrmv_info.hpp"
#include "rocsparse_bsrsm_info.hpp"
#include "rocsparse_bsrsv_info.hpp"
#include "rocsparse_coomv_info.hpp"
#include "rocsparse_csrgemm_info.hpp"
#include "rocsparse_csric0_info.hpp"
#include "rocsparse_csrildlt0_info.hpp"
#include "rocsparse_csrilu0_info.hpp"
#include "rocsparse_csritsv_info.hpp"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_csrsm_info.hpp"
#include "rocsparse_csrsv_info.hpp"
#include "rocsparse_ellsv_info.hpp"
#include "rocsparse_numeric_boost.hpp"
#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_sorted_coo2csr_info.hpp"
#include "rocsparse_triangular_data_t.hpp"
#include "rocsparse_triangular_info_t.hpp"

#include <memory>

/********************************************************************************
 * \brief rocsparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_mat_info().
 *******************************************************************************/

struct _rocsparse_mat_info
{
protected:
    rocsparse_csrmv_info csrmv_info{};
    rocsparse_bsrmv_info bsrmv_info{};
    rocsparse_coomv_info coomv_info{};

    rocsparse::sorted_coo2csr_info_t* m_sorted_coo2csr_info{};

    // Single triangular data instance; the triangular infos only hold a const pointer to it.
    rocsparse::triangular_data_t m_triangular_data;

    // Exact numeric singularity shared by every triangular info.
    rocsparse::pivot_numeric_info_t* m_singularity_numeric_exact{};

    // Near numeric singularity shared by the csrilu0, csric0 and csrildlt0 infos.
    rocsparse::pivot_numeric_info_t* m_singularity_numeric_near{};

    // Shared by every triangular_info_t of m_triangular_data, which refer to them as const.
    rocsparse::idnvec_descr_t*                m_diag_ind{};
    rocsparse::transposed_sparsity_pattern_t* m_transposed{};

    // Pivot of the triangular data, which holds a mutable pointer to it.
    rocsparse::pivot_symbolic_info_t* m_pivot_info{};

    std::shared_ptr<_rocsparse_csrsv_info>     m_csrsv_info;
    std::shared_ptr<_rocsparse_csrsm_info>     m_csrsm_info;
    std::shared_ptr<_rocsparse_csrilu0_info>   m_csrilu0_info;
    std::shared_ptr<_rocsparse_csric0_info>    m_csric0_info;
    std::shared_ptr<_rocsparse_csrildlt0_info> m_csrildlt0_info;
    std::shared_ptr<_rocsparse_bsrsv_info>     m_bsrsv_info;
    std::shared_ptr<_rocsparse_bsrsm_info>     m_bsrsm_info;
    std::shared_ptr<_rocsparse_bsrilu0_info>   m_bsrilu0_info;
    std::shared_ptr<_rocsparse_bsric0_info>    m_bsric0_info;
    std::shared_ptr<_rocsparse_ellsv_info>     m_ellsv_info;

    void release_trm_data_if_unused();

    rocsparse::numeric_boost m_boost{};

public:
    // Recreates a slot of the triangular data, handing it the storage owned by this mat_info.

    rocsparse_status recreate_triangular_info(rocsparse_handle            handle,
                                              rocsparse_operation         operation,
                                              rocsparse_fill_mode         fill_mode,
                                              rocsparse_const_spmat_descr A,
                                              size_t                      buffer_size,
                                              void*                       buffer)
    {
        return this->m_triangular_data.recreate(handle,
                                                operation,
                                                fill_mode,
                                                A,
                                                this->m_diag_ind,
                                                this->m_transposed,
                                                buffer_size,
                                                buffer);
    }

    // Triangular infos only see the triangular data as const; mutable access goes through here.
    const rocsparse::triangular_data_t* get_triangular_data() const;
    rocsparse::triangular_data_t*       get_triangular_data();

    rocsparse::pivot_numeric_info_t* get_singularity_numeric_exact();

    // Mutable access to transposed storage owned by this mat_info, nullptr otherwise.
    rocsparse::transposed_sparsity_pattern_t*
        get_transposed_sparsity_pattern(const rocsparse::transposed_sparsity_pattern_t* transposed);
    rocsparse::pivot_numeric_info_t* get_singularity_numeric_near();

    void duplicate_trdata(const rocsparse_mat_info src, hipStream_t stream);
    const rocsparse::numeric_boost*            get_boost() const;
    rocsparse::numeric_boost*                  get_boost();
    std::shared_ptr<_rocsparse_csrsv_info>     get_shared_csrsv_info();
    std::shared_ptr<_rocsparse_csrsm_info>     get_shared_csrsm_info();
    std::shared_ptr<_rocsparse_csrilu0_info>   get_shared_csrilu0_info();
    std::shared_ptr<_rocsparse_csric0_info>    get_shared_csric0_info();
    std::shared_ptr<_rocsparse_csrildlt0_info> get_shared_csrildlt0_info();
    std::shared_ptr<_rocsparse_bsrsv_info>     get_shared_bsrsv_info();
    std::shared_ptr<_rocsparse_bsrsm_info>     get_shared_bsrsm_info();
    std::shared_ptr<_rocsparse_bsrilu0_info>   get_shared_bsrilu0_info();
    std::shared_ptr<_rocsparse_bsric0_info>    get_shared_bsric0_info();
    std::shared_ptr<_rocsparse_ellsv_info>     get_shared_ellsv_info();
    void                                       clear_csrsv_info();
    void                                       clear_csrsm_info();
    void                                       clear_csrilu0_info();
    void                                       clear_csric0_info();
    void                                       clear_csrildlt0_info();
    void                                       clear_bsrsv_info();
    void                                       clear_bsrsm_info();
    void                                       clear_bsrilu0_info();
    void                                       clear_bsric0_info();
    void                                       clear_ellsv_info();

    rocsparse_bsric0_info         get_bsric0_info();
    rocsparse::triangular_info_t* get_bsric0_info(rocsparse_operation operation,
                                                  rocsparse_fill_mode fill_mode);
    void                          set_bsric0_info(rocsparse_operation           operation,
                                                  rocsparse_fill_mode           fill_mode,
                                                  rocsparse::triangular_info_t* trm_info);

    rocsparse_bsrilu0_info        get_bsrilu0_info();
    rocsparse::triangular_info_t* get_bsrilu0_info(rocsparse_operation operation,
                                                   rocsparse_fill_mode fill_mode);
    void                          set_bsrilu0_info(rocsparse_operation           operation,
                                                   rocsparse_fill_mode           fill_mode,
                                                   rocsparse::triangular_info_t* trm_info);

    rocsparse_csric0_info         get_csric0_info();
    rocsparse::triangular_info_t* get_csric0_info(rocsparse_operation operation,
                                                  rocsparse_fill_mode fill_mode);
    void                          set_csric0_info(rocsparse_operation           operation,
                                                  rocsparse_fill_mode           fill_mode,
                                                  rocsparse::triangular_info_t* trm_info);

    rocsparse_csrildlt0_info      get_csrildlt0_info();
    rocsparse::triangular_info_t* get_csrildlt0_info(rocsparse_operation operation,
                                                     rocsparse_fill_mode fill_mode);
    void                          set_csrildlt0_info(rocsparse_operation           operation,
                                                     rocsparse_fill_mode           fill_mode,
                                                     rocsparse::triangular_info_t* trm_info);

    rocsparse_csrilu0_info        get_csrilu0_info();
    rocsparse::triangular_info_t* get_csrilu0_info(rocsparse_operation operation,
                                                   rocsparse_fill_mode fill_mode);
    void                          set_csrilu0_info(rocsparse_operation           operation,
                                                   rocsparse_fill_mode           fill_mode,
                                                   rocsparse::triangular_info_t* trm_info);

    rocsparse_csrsv_info          get_csrsv_info();
    rocsparse::triangular_info_t* get_csrsv_info(rocsparse_operation operation,
                                                 rocsparse_fill_mode fill_mode);
    void                          set_csrsv_info(rocsparse_operation           operation,
                                                 rocsparse_fill_mode           fill_mode,
                                                 rocsparse::triangular_info_t* trm_info);

    rocsparse_csrsm_info          get_csrsm_info();
    rocsparse::triangular_info_t* get_csrsm_info(rocsparse_operation operation,
                                                 rocsparse_fill_mode fill_mode);
    void                          set_csrsm_info(rocsparse_operation           operation,
                                                 rocsparse_fill_mode           fill_mode,
                                                 rocsparse::triangular_info_t* trm_info);

    rocsparse_bsrsv_info          get_bsrsv_info();
    rocsparse::triangular_info_t* get_bsrsv_info(rocsparse_operation operation,
                                                 rocsparse_fill_mode fill_mode);
    void                          set_bsrsv_info(rocsparse_operation           operation,
                                                 rocsparse_fill_mode           fill_mode,
                                                 rocsparse::triangular_info_t* trm_info);

    rocsparse_bsrsm_info          get_bsrsm_info();
    rocsparse::triangular_info_t* get_bsrsm_info(rocsparse_operation operation,
                                                 rocsparse_fill_mode fill_mode);
    void                          set_bsrsm_info(rocsparse_operation           operation,
                                                 rocsparse_fill_mode           fill_mode,
                                                 rocsparse::triangular_info_t* trm_info);

    rocsparse_ellsv_info          get_ellsv_info();
    rocsparse::triangular_info_t* get_ellsv_info(rocsparse_operation operation,
                                                 rocsparse_fill_mode fill_mode);
    void                          set_ellsv_info(rocsparse_operation           operation,
                                                 rocsparse_fill_mode           fill_mode,
                                                 rocsparse::triangular_info_t* trm_info);

    rocsparse_csrgemm_info csrgemm_info{};
    rocsparse_csritsv_info csritsv_info{};

    _rocsparse_mat_info();
    ~_rocsparse_mat_info();

    void                 set_csrmv_info(rocsparse_csrmv_info value);
    rocsparse_csrmv_info get_csrmv_info();

    void                 set_bsrmv_info(rocsparse_bsrmv_info value);
    rocsparse_bsrmv_info get_bsrmv_info();

    void                 set_coomv_info(rocsparse_coomv_info value);
    rocsparse_coomv_info get_coomv_info();

    void set_sorted_coo2csr_info(rocsparse::sorted_coo2csr_info_t* value);
    rocsparse::sorted_coo2csr_info_t* get_sorted_coo2csr_info();
};
