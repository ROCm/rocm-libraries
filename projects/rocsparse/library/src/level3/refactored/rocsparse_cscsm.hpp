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

#include "rocsparse_control.hpp"
#include "rocsparse_csrsm_info.hpp"

#include <cstdlib>

namespace rocsparse
{
    // The limited-memory version of the CSC sptrsm and spsm (BUILD_WITH_SPTRSM_CSC) does not build
    // the transposed matrix: once an unknown is computed, its column of the matrix updates the
    // unknowns that depend on it, and the partial sums are accumulated in B. It applies to a CSC
    // matrix without transpose; its analysis, the one of the CSC sptrsv, is stored in a trm_info_t
    // of csrsm_info under the key (operation_none, fill_mode):
    //   diag_ind           position of the diagonal entry of each column
    //   transposed_col_ind number of strict entries of each row
    //   row_map            column of each task, in elimination order
    //   transposed_perm    first entry of each task
    //   max_nnz            number of tasks
    // The other configurations use the transposed CSR solve.
    bool cscsm_use_limited_memory(rocsparse_operation         op_A,
                                  rocsparse_operation         op_B,
                                  rocsparse_const_dnvec_descr alpha,
                                  rocsparse_const_spmat_descr A,
                                  rocsparse_const_dnmat_descr B,
                                  bool                        limited_memory);

    static inline bool is_limited_memory(rocsparse_sptrsm_alg alg)
    {
#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
        return alg == rocsparse_sptrsm_alg_limited_memory;
#else
        return false;
#endif
    }

    static inline bool is_limited_memory(rocsparse_spsm_alg alg)
    {
#if defined(ROCSPARSE_WITH_SPTRSM_CSC)
        return alg == rocsparse_spsm_alg_limited_memory;
#else
        return false;
#endif
    }

    // Right-hand sides per panel: power of two >= nrhs, at most ROCSPARSE_CSCSM_PANEL (default 16,
    // at most 128; see ROCSPARSE_CSCSM_Q for the lanes). The (task, panel) pairs share one ticket
    // per batch instance.
    inline int64_t cscsm_panel_width(int64_t nrhs)
    {
        const char* env = std::getenv("ROCSPARSE_CSCSM_PANEL");
        int64_t     cap = (env != nullptr) ? std::atoll(env) : 16;
        cap             = (cap < 1) ? 1 : ((cap > 128) ? 128 : cap);
        int64_t p       = 1;
        while(p < nrhs && p < cap)
        {
            p *= 2;
        }
        return p;
    }

    size_t cscsm_push_analysis_buffer_size(rocsparse_handle handle, rocsparse_const_spmat_descr A);
    size_t cscsm_push_solve_buffer_size(int64_t                     nrhs,
                                        rocsparse_const_spmat_descr A,
                                        rocsparse_const_dnmat_descr B);

    rocsparse_status cscsm_analysis_buffer_size(rocsparse_handle            handle,
                                                const int64_t               nrhs,
                                                rocsparse_operation         op_A,
                                                rocsparse_operation         op_B,
                                                rocsparse_const_dnvec_descr alpha,
                                                rocsparse_const_spmat_descr A,
                                                rocsparse_const_dnmat_descr X,
                                                size_t*                     p_buffer_size_in_bytes,
                                                rocsparse_error*            p_error,
                                                bool                        limited_memory);

    rocsparse_status cscsm_solve_buffer_size(rocsparse_handle            handle,
                                             const int64_t               nrhs,
                                             rocsparse_operation         op_A,
                                             rocsparse_operation         op_B,
                                             rocsparse_const_dnvec_descr alpha,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_dnmat_descr X,
                                             size_t*                     p_buffer_size_in_bytes,
                                             rocsparse_error*            p_error,
                                             bool                        limited_memory);

    rocsparse_status cscsm_buffer_size(rocsparse_handle            handle,
                                       int64_t                     nrhs,
                                       rocsparse_operation         trans_A,
                                       rocsparse_operation         trans_B,
                                       rocsparse_const_dnvec_descr alpha,
                                       rocsparse_const_spmat_descr A,
                                       rocsparse_const_dnmat_descr B,
                                       size_t*                     p_buffer_size,
                                       rocsparse_error*            p_error,
                                       bool                        limited_memory);

    rocsparse_status cscsm_analysis(rocsparse_handle            handle,
                                    int64_t                     nrhs,
                                    rocsparse_operation         trans_A,
                                    rocsparse_operation         trans_B,
                                    rocsparse_const_dnvec_descr alpha,
                                    rocsparse_const_spmat_descr A,
                                    rocsparse_const_dnmat_descr B,
                                    rocsparse_analysis_policy   analysis,
                                    rocsparse_csrsm_info*       p_csrsm_info,
                                    size_t                      buffer_size_in_bytes,
                                    void*                       buffer,
                                    rocsparse_error*            p_error,
                                    bool                        limited_memory);

    rocsparse_status cscsm_compute(rocsparse_handle            handle,
                                   int64_t                     nrhs,
                                   rocsparse_operation         trans_A,
                                   rocsparse_operation         trans_B,
                                   rocsparse_const_dnvec_descr alpha,
                                   rocsparse_const_spmat_descr A,
                                   rocsparse_dnmat_descr       B,
                                   rocsparse_csrsm_info        csrsm_info,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer,
                                   rocsparse_error*            p_error,
                                   bool                        limited_memory,
                                   bool                        conj);

}
