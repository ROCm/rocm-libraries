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

#include "rocsparse-complex-types.h"
#include "rocsparse-types.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csrildlt0_info.hpp"
#include "rocsparse_floating_data_t.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    typedef rocsparse_status (*csrildlt0_kernel_launch_t)(rocsparse_handle         handle,
                                                          rocsparse_csrildlt0_info csrildlt0_info,
                                                          rocsparse_spmat_descr    A,
                                                          void*                    diag,
                                                          size_t                   buffer_size,
                                                          void*                    buffer);

    // Gather the real diagonal D out of the factor.
    //
    // After the ILDLT(0) factorization, each D_i is stored in-place on the (implicit unit)
    // diagonal of the L factor, i.e. at csr_val[csr_diag_ind[i]]. This kernel copies that
    // diagonal into the user-provided dense vector diag[]. A missing diagonal entry
    // (csr_diag_ind[i] < 0) corresponds to a zero pivot and is reported as 0.
    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrildlt0_copy_diag_kernel(J m,
                                    const T* __restrict__ csr_val,
                                    int64_t csr_val_stride,
                                    const I* __restrict__ csr_diag_ind,
                                    floating_data_t<T>* __restrict__ diag,
                                    int64_t diag_stride)
    {
        const J row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        if(row >= m)
        {
            return;
        }

        const auto batch_index = hipBlockIdx_y;
        const I    diag_pos    = csr_diag_ind[row];

        diag[batch_index * diag_stride + row]
            = (diag_pos >= 0) ? rocsparse::real(csr_val[batch_index * csr_val_stride + diag_pos])
                              : static_cast<floating_data_t<T>>(0);
    }

    // Copy the real diagonal D into the optional user-provided vector once the factorization
    // has completed. Does nothing when diag is null. Runs on handle->stream, after the
    // factorization kernel, so csr_val already holds the final diagonal.
    template <typename T, typename I, typename J>
    rocsparse_status csrildlt0_copy_diag(rocsparse_handle         handle,
                                         rocsparse_csrildlt0_info csrildlt0_info,
                                         rocsparse_spmat_descr    A,
                                         void*                    diag)
    {
        if(diag == nullptr)
        {
            return rocsparse_status_success;
        }

        auto trm_info = csrildlt0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

        static constexpr uint32_t BLOCKSIZE = 256;
        const dim3                blocks((A->rows - 1) / BLOCKSIZE + 1, A->batch_count);
        const dim3                threads(BLOCKSIZE);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrildlt0_copy_diag_kernel<BLOCKSIZE, T, I, J>),
            blocks,
            threads,
            0,
            handle->stream,
            static_cast<J>(A->rows),
            reinterpret_cast<const T*>(A->val_data),
            A->batch_stride,
            reinterpret_cast<const I*>(trm_info->get_diag_ind()),
            reinterpret_cast<floating_data_t<T>*>(diag),
            static_cast<int64_t>(A->rows));

        return rocsparse_status_success;
    }
}
