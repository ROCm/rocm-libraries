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
#include "rocsparse_utility.hpp"

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
        = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908 && //
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
    return rocsparse_status_success;
}
