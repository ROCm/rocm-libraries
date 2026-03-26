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

#include "rocsparse_csric0_kernel_launch.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_csric0_kernel_binsearch.hpp"
#include "rocsparse_csric0_kernel_hash.hpp"

rocsparse_status rocsparse::csric0_kernel_launch(rocsparse_handle      handle,
                                                 rocsparse_csric0_info csric0_info,
                                                 rocsparse_spmat_descr A,
                                                 size_t                buffer_size,
                                                 void*                 buffer)
{

    const bool sleep = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908
                        && handle->asic_rev < 2);

    auto trm_info = csric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);

    rocsparse::csric0_kernel_launch_t launch{};

    if(sleep || (trm_info->get_max_nnz() > 512))
    {
        launch = rocsparse::find_csric0_kernel_binsearch_launch(handle, csric0_info, A);
    }
    else
    {

        launch = rocsparse::find_csric0_kernel_hash_launch(handle, csric0_info, A);
    }

    RETURN_IF_HIP_ERROR(hipMemsetAsync(reinterpret_cast<char*>(buffer) + 256,
                                       0,
                                       sizeof(int32_t) * A->rows * A->batch_count,
                                       handle->stream));

    RETURN_IF_ROCSPARSE_ERROR(launch(handle, csric0_info, A, buffer_size, buffer));

    return rocsparse_status_success;
}
