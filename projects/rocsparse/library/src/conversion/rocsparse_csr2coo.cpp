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
#include "rocsparse_utility.hpp"

#include "internal/conversion/rocsparse_csr2coo.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csr2coo.hpp"

#include "csr2coo_device.h"

namespace
{
    // Blocks needed to cover m rows at (BLOCKSIZE / WF_SIZE) rows each, clamped to the
    // device grid.x limit. csr2coo_kernel grid-strides over the block rows the clamp
    // drops.
    //
    // The int64_t cast on WF_SIZE * m is the pre-existing, deliberate one: it is the
    // reference idiom of this epic and the arithmetic below it was never the problem.
    // What was missing (AISPARSE-685) is that the correctly computed 64-bit result was
    // assigned straight into a dim3, narrowing it back to unsigned int with nothing
    // behind it. Eight call sites differ only in WF_SIZE, so they share this.
    //
    // Deliberately local: AISPARSE-696 (PR #11512) adds rocsparse::ceil_div() and
    // rocsparse::get_grid_size() to rocsparse_common.h, but it has not merged and
    // rocsparse_common.h/.hpp are a live conflict zone (AISPARSE-677/678/696). Once
    // #11512 lands the body below is the one-line
    //   return rocsparse::get_grid_size(
    //       rocsparse::ceil_div(static_cast<int64_t>(WF_SIZE) * m, BLOCKSIZE),
    //       handle->properties.maxGridSize[0]);
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE>
    int64_t csr2coo_grid_size_x(rocsparse_handle handle, int64_t m)
    {
        return rocsparse::min((static_cast<int64_t>(WF_SIZE) * m - 1) / BLOCKSIZE + 1,
                              static_cast<int64_t>(handle->properties.maxGridSize[0]));
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csr2coo_core(rocsparse_handle     handle,
                                         const I*             csr_row_ptr_begin,
                                         const I*             csr_row_ptr_end,
                                         I                    nnz,
                                         J                    m,
                                         J*                   coo_row_ind,
                                         rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

    I nnz_per_row = nnz / m;

#define CSR2COO_DIM 256
    if(nnz_per_row < 4)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 2>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 2>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
    else if(nnz_per_row < 8)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 4>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 4>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
    else if(nnz_per_row < 16)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 8>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 8>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
    else if(nnz_per_row < 32)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 16>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 16>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
    else if(nnz_per_row < 64)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 32>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 32>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
    else if(nnz_per_row < 128)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 64>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 64>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
    else if(nnz_per_row < 256)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 128>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 128>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
    else
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2coo_kernel<CSR2COO_DIM, 256>),
                                           dim3(csr2coo_grid_size_x<CSR2COO_DIM, 256>(handle, m)),
                                           dim3(CSR2COO_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_begin,
                                           csr_row_ptr_end,
                                           coo_row_ind,
                                           idx_base);
    }
#undef CSR2COO_DIM
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csr2coo_template(rocsparse_handle     handle,
                                             const I*             csr_row_ptr,
                                             I                    nnz,
                                             J                    m,
                                             J*                   coo_row_ind,
                                             rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(nnz == 0 || m == 0)
    {
        return rocsparse_status_success;
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2coo_core(
        handle, csr_row_ptr, csr_row_ptr + 1, nnz, m, coo_row_ind, idx_base));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::csr2coo_impl(rocsparse_handle     handle,
                                         const I*             csr_row_ptr,
                                         I                    nnz,
                                         J                    m,
                                         J*                   coo_row_ind,
                                         rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging TODO bench logging
    rocsparse::log_trace(handle,
                         "rocsparse_csr2coo",
                         (const void*&)csr_row_ptr,
                         nnz,
                         m,
                         (const void*&)coo_row_ind,
                         idx_base);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(5, idx_base);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);
    ROCSPARSE_CHECKARG_SIZE(3, m);
    ROCSPARSE_CHECKARG_ARRAY(1, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csr2coo_template(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base));
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE)                                                                 \
    template rocsparse_status rocsparse::csr2coo_template<ITYPE, JTYPE>(                          \
        rocsparse_handle     handle,                                                              \
        const ITYPE*         csr_row_ptr,                                                         \
        ITYPE                nnz,                                                                 \
        JTYPE                m,                                                                   \
        JTYPE*               coo_row_ind,                                                         \
        rocsparse_index_base idx_base);                                                           \
    template rocsparse_status rocsparse::csr2coo_impl<ITYPE, JTYPE>(rocsparse_handle handle,      \
                                                                    const ITYPE*     csr_row_ptr, \
                                                                    ITYPE            nnz,         \
                                                                    JTYPE            m,           \
                                                                    JTYPE*           coo_row_ind, \
                                                                    rocsparse_index_base idx_base)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_csr2coo(rocsparse_handle     handle,
                                              const rocsparse_int* csr_row_ptr,
                                              rocsparse_int        nnz,
                                              rocsparse_int        m,
                                              rocsparse_int*       coo_row_ind,
                                              rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csr2coo_impl(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
