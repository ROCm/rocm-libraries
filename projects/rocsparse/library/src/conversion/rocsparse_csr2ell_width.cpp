/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "csr2ell_device.h"
#include "internal/conversion/rocsparse_csr2ell.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2ell.hpp"
#include "rocsparse_utility.hpp"

template <typename J>
rocsparse_status rocsparse::csr2ell_width_quickreturn(rocsparse_handle          handle,
                                                      int64_t                   m,
                                                      const rocsparse_mat_descr csr_descr,
                                                      const void*               csr_row_ptr,
                                                      const rocsparse_mat_descr ell_descr,
                                                      J*                        ell_width)
{
    ROCSPARSE_ROUTINE_TRACE;

    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(ell_width, 0, sizeof(J), stream));
        }
        else
        {
            *ell_width = 0;
        }
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

namespace rocsparse
{
    template <typename J>
    static rocsparse_status csr2ell_width_checkarg(rocsparse_handle          handle, //0
                                                   int64_t                   m, //1
                                                   const rocsparse_mat_descr csr_descr, //2
                                                   const void*               csr_row_ptr, //3
                                                   const rocsparse_mat_descr ell_descr, //4
                                                   J*                        ell_width) //5
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);

        ROCSPARSE_CHECKARG_POINTER(2, csr_descr);
        ROCSPARSE_CHECKARG(2,
                           csr_descr,
                           (csr_descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(2,
                           csr_descr,
                           (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(3, m, csr_row_ptr);

        ROCSPARSE_CHECKARG_POINTER(4, ell_descr);
        ROCSPARSE_CHECKARG(4,
                           ell_descr,
                           (ell_descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(4,
                           ell_descr,
                           (ell_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_POINTER(5, ell_width);

        const rocsparse_status status = rocsparse::csr2ell_width_quickreturn(
            handle, m, csr_descr, csr_row_ptr, ell_descr, ell_width);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csr2ell_width_core(rocsparse_handle          handle,
                                               J                         m,
                                               const rocsparse_mat_descr csr_descr,
                                               const I*                  csr_row_ptr,
                                               const rocsparse_mat_descr ell_descr,
                                               J*                        ell_width)
{
    ROCSPARSE_ROUTINE_TRACE;

    hipStream_t stream = handle->stream;

#define CSR2ELL_DIM 256
    // Workspace size
    J nblocks = CSR2ELL_DIM;

    // Get workspace from handle device buffer
    J* workspace = reinterpret_cast<J*>(handle->buffer);

    // Get workspace from handle device buffer
    dim3 csr2ell_blocks(nblocks);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    // Compute maximum nnz per row
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((ell_width_kernel_part1<CSR2ELL_DIM>),
                                       csr2ell_blocks,
                                       csr2ell_threads,
                                       0,
                                       stream,
                                       m,
                                       csr_row_ptr,
                                       workspace);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((ell_width_kernel_part2<CSR2ELL_DIM>),
                                       dim3(1),
                                       csr2ell_threads,
                                       0,
                                       stream,
                                       nblocks,
                                       workspace);

    // Copy ELL width back to host, if handle says so
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(ell_width, workspace, sizeof(J), hipMemcpyDeviceToDevice, stream));
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(ell_width, workspace, sizeof(J), hipMemcpyDeviceToHost, stream));
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(J)                                              \
    template rocsparse_status rocsparse::csr2ell_width_quickreturn( \
        rocsparse_handle          handle,                           \
        int64_t                   m,                                \
        const rocsparse_mat_descr csr_descr,                        \
        const void*               csr_row_ptr,                      \
        const rocsparse_mat_descr ell_descr,                        \
        J*                        ell_width)

INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE

#define INSTANTIATE(I, J)                                                                          \
    template rocsparse_status rocsparse::csr2ell_width_core(rocsparse_handle          handle,      \
                                                            J                         m,           \
                                                            const rocsparse_mat_descr csr_descr,   \
                                                            const I*                  csr_row_ptr, \
                                                            const rocsparse_mat_descr ell_descr,   \
                                                            J*                        ell_width)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

namespace rocsparse
{
    template <typename... P>
    rocsparse_status csr2ell_width_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_csr2ell_width", p...);
        const rocsparse_status status = rocsparse::csr2ell_width_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_width_core(p...));
        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_csr2ell_width(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    const rocsparse_mat_descr csr_descr,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    const rocsparse_mat_descr ell_descr,
                                                    rocsparse_int*            ell_width)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csr2ell_width_impl(handle, m, csr_descr, csr_row_ptr, ell_descr, ell_width));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
