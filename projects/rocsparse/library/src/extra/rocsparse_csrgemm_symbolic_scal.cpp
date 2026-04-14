/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrgemm_symbolic_scal.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // Copy an array
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_symbolic_copy(I size,
                               const J* __restrict__ in,
                               J* __restrict__ out,
                               rocsparse_index_base idx_base_in,
                               rocsparse_index_base idx_base_out)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(idx >= size)
        {
            return;
        }

        out[idx] = in[idx] - idx_base_in + idx_base_out;
    }
}

rocsparse_status rocsparse::csrgemm_symbolic_scal_quickreturn(rocsparse_handle          handle,
                                                              int64_t                   m,
                                                              int64_t                   n,
                                                              const rocsparse_mat_descr descr_D,
                                                              int64_t                   nnz_D,
                                                              const void* csr_row_ptr_D,
                                                              const void* csr_col_ind_D,
                                                              const rocsparse_mat_descr descr_C,
                                                              int64_t                   nnz_C,
                                                              const void* csr_row_ptr_C,
                                                              void*       csr_col_ind_C,
                                                              const rocsparse_mat_info info_C,
                                                              void*                    temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(false == mul && true == add)
    {
        if(m == 0 || n == 0 || nnz_C == 0 || nnz_D == 0)
        {
            return rocsparse_status_success;
        }
        else
        {
            return rocsparse_status_continue;
        }
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_symbolic_scal_core(rocsparse_handle          handle,
                                                       J                         m,
                                                       J                         n,
                                                       const rocsparse_mat_descr descr_D,
                                                       I                         nnz_D,
                                                       const I*                  csr_row_ptr_D,
                                                       const J*                  csr_col_ind_D,
                                                       const rocsparse_mat_descr descr_C,
                                                       I                         nnz_C,
                                                       const I*                  csr_row_ptr_C,
                                                       J*                        csr_col_ind_C,
                                                       const rocsparse_mat_info  info_C,
                                                       void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(false == mul && true == add)
    {
#define CSRGEMM_DIM 1024

        dim3 csrgemm_blocks((nnz_D - 1) / CSRGEMM_DIM + 1);
        dim3 csrgemm_threads(CSRGEMM_DIM);

        // Copy column entries, if D != C
        if(csr_col_ind_C != csr_col_ind_D)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_symbolic_copy<CSRGEMM_DIM>),
                                               csrgemm_blocks,
                                               csrgemm_threads,
                                               0,
                                               handle->stream,
                                               nnz_D,
                                               csr_col_ind_D,
                                               csr_col_ind_C,
                                               descr_D->base,
                                               descr_C->base);
        }

#undef CSRGEMM_DIM

        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }
}

#define INSTANTIATE(I, J)                                            \
    template rocsparse_status rocsparse::csrgemm_symbolic_scal_core( \
        rocsparse_handle          handle,                            \
        J                         m,                                 \
        J                         n,                                 \
        const rocsparse_mat_descr descr_D,                           \
        I                         nnz_D,                             \
        const I*                  csr_row_ptr_D,                     \
        const J*                  csr_col_ind_D,                     \
        const rocsparse_mat_descr descr_C,                           \
        I                         nnz_C,                             \
        const I*                  csr_row_ptr_C,                     \
        J*                        csr_col_ind_C,                     \
        const rocsparse_mat_info  info_C,                            \
        void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);

#undef INSTANTIATE
