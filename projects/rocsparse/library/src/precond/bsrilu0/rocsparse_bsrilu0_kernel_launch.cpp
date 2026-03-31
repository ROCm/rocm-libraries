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

#include "rocsparse_bsrilu0_kernel_launch.hpp"
#include "rocsparse_bsrilu0_kernel_2_8.hpp"
#include "rocsparse_bsrilu0_kernel_33_64.hpp"
#include "rocsparse_bsrilu0_kernel_9_32.hpp"
#include "rocsparse_bsrilu0_kernel_general.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::bsrilu0_kernel_launch(rocsparse_handle          handle,
                                                  rocsparse_bsrilu0_info    bsrilu0_info,
                                                  rocsparse_spmat_descr     A,
                                                  rocsparse::numeric_boost* boost,
                                                  size_t                    buffer_size,
                                                  void*                     buffer)
{
    rocsparse::bsrilu0_kernel_launch_t launch        = nullptr;
    const std::string                  gcn_arch_name = rocsparse::handle_get_arch_name(handle);

    const bool sleep     = (gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2);
    const auto block_dim = A->block_dim;

    if(sleep || (handle->wavefront_size == 32) || (block_dim > 64)
       || ((block_dim > 32) && (A->data_type == rocsparse_datatype_f64_c)))
    {
        launch = find_bsrilu0_kernel_general_launch(handle, bsrilu0_info, A);
    }
    else if(block_dim <= 8)
    {
        launch = find_bsrilu0_kernel_2_8_launch(handle, bsrilu0_info, A);
    }
    else if(block_dim <= 32)
    {
        launch = find_bsrilu0_kernel_9_32_launch(handle, bsrilu0_info, A);
    }
    else
    {
        launch = find_bsrilu0_kernel_33_64_launch(handle, bsrilu0_info, A);
    }

    //
    // Set done array to zero.
    //
    RETURN_IF_HIP_ERROR(hipMemsetAsync(reinterpret_cast<char*>(buffer) + 256,
                                       0,
                                       sizeof(int32_t) * A->rows * A->batch_count,
                                       handle->stream));

    //
    // Launch gpu kernel.
    //
    RETURN_IF_ROCSPARSE_ERROR(launch(handle, bsrilu0_info, A, boost, buffer_size, buffer));
    return rocsparse_status_success;
}
