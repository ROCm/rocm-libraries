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

#include "rocsparse_csrilu0_kernel_launch.hpp"
#include "rocsparse_csrilu0_kernel_binsearch.hpp"
#include "rocsparse_csrilu0_kernel_hash.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // A zero batch stride collapses the factorization to instance 0, which is the
    // only one that can write a numeric pivot. Copy its positions to the others.
    template <typename J>
    ROCSPARSE_KERNEL(256)
    void csrilu0_broadcast_pivot_kernel(int64_t n, int64_t stride, J* exact, J* near)
    {
        const int64_t step = static_cast<int64_t>(hipGridDim_x) * hipBlockDim_x;
        for(int64_t i = 1 + static_cast<int64_t>(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;
            i < n;
            i += step)
        {
            exact[i * stride] = exact[0];
            near[i * stride]  = near[0];
        }
    }

    template <typename J>
    static rocsparse_status csrilu0_broadcast_pivot(rocsparse_handle       handle,
                                                    rocsparse::position_t* exact,
                                                    rocsparse::position_t* near,
                                                    int64_t                n)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrilu0_broadcast_pivot_kernel<J>),
                                           dim3(rocsparse::get_grid_size_x(handle, n, 256)),
                                           dim3(256),
                                           0,
                                           handle->stream,
                                           n,
                                           exact->get_stride(),
                                           reinterpret_cast<J*>(exact->get_position()),
                                           reinterpret_cast<J*>(near->get_position()));
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrilu0_kernel_launch(rocsparse_handle          handle, // 0
                                                  rocsparse_csrilu0_info    csrilu0_info, // 1
                                                  rocsparse_spmat_descr     A, // 2
                                                  rocsparse::numeric_boost* boost,
                                                  size_t                    buffer_size, // 7
                                                  void*                     buffer) // 8
{

    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, csrilu0_info);
    ROCSPARSE_CHECKARG_POINTER(2, A);

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(8, buffer_size, buffer);

    ROCSPARSE_CHECKARG(2, A, (A->descr == nullptr), rocsparse_status_invalid_pointer);

    ROCSPARSE_CHECKARG(
        2, A, (A->descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG(2,
                       A,
                       (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    auto trm_info = csrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

    // Max nnz per row
    const int64_t max_nnz = trm_info->get_max_nnz();

    const bool sleep
        = (rocsparse::handle_get_arch_name(handle) == rocsparse_arch_names::gfx908 && //
           handle->asic_rev < 2);

    rocsparse::csrilu0_kernel_launch_t launch{};

    if(sleep || //
       ((handle->wavefront_size == 32) && (max_nnz >= 512)) || //
       ((handle->wavefront_size == 64) && (max_nnz >= 1024)))
    {
        launch = rocsparse::find_csrilu0_kernel_binsearch_launch(handle, csrilu0_info, A);
    }
    else
    {
        launch = rocsparse::find_csrilu0_kernel_hash_launch(handle, csrilu0_info, A);
    }

    const int64_t A_batch_count = (A->batch_stride == 0) ? 1 : A->batch_count;
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(reinterpret_cast<char*>(buffer) + 256,
                                                 0,
                                                 sizeof(int32_t) * A->rows * A_batch_count,
                                                 handle->stream));

    RETURN_IF_ROCSPARSE_ERROR(launch(handle, csrilu0_info, A, boost, buffer_size, buffer));

    // Every instance aliases instance 0, so they all share its pivot.
    if(A->batch_stride == 0 && A->batch_count > 1)
    {
        auto exact = csrilu0_info->get_singularity_numeric_exact();
        auto near  = csrilu0_info->get_singularity_numeric_near();
        if(A->col_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrilu0_broadcast_pivot<int32_t>(handle, exact, near, A->batch_count));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrilu0_broadcast_pivot<int64_t>(handle, exact, near, A->batch_count));
        }
    }
    return rocsparse_status_success;
}
