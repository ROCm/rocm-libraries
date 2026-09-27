/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

// rocsparse::bsrsm_copy_scale, rocsparse::bsrsm_{lower,upper}_large_kernel (the
// __global__ wrappers carrying the grid-stride loops) and the grid sizing helpers
// live in these two headers so clients/unittests can launch the kernels with a
// deliberately undersized grid.
#include "bsrsm_device.h"
#include "bsrsm_device_large.h"
#include "rocsparse_bsrsm.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include <limits>

namespace rocsparse
{
    // wfsize * nnzb was a signed 32 bit product that overflowed at 33,554,431
    // non-zero blocks with a 64 wide wavefront, so cast before the multiply. The
    // clamp is a one line substitution for
    //     rocsparse::get_grid_size(num_blocks, rocsparse::max_grid_size_x)
    // once PR #11512 (AISPARSE-696) lands. rocsparse::bsr_gather grid-strides over
    // whatever the clamp leaves behind; that kernel is shared with bsrsv, so this
    // is deliberately the same expression AISPARSE-656 uses there (PR #11118).
#define LAUNCH_BSRSM_GTHR_DIM(bsize, wfsize, dim)                                      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                \
        (rocsparse::bsr_gather<wfsize, bsize / wfsize, dim>),                          \
        dim3(rocsparse::min((static_cast<int64_t>(wfsize) * nnzb - 1) / bsize + 1,     \
                            static_cast<int64_t>(handle->properties.maxGridSize[0]))), \
        dim3(wfsize, bsize / wfsize),                                                  \
        0,                                                                             \
        stream,                                                                        \
        dir,                                                                           \
        nnzb,                                                                          \
        (const rocsparse_int*)trm_info->get_transposed_perm(),                         \
        bsr_val,                                                                       \
        bsrt_val,                                                                      \
        block_dim)

#define LAUNCH_BSRSM_GTHR(bsize, wfsize, dim) \
    if(dim <= 2)                              \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 4, 2);   \
    }                                         \
    else if(dim <= 4)                         \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else if(wfsize == 32)                     \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else                                      \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 64, 8);  \
    }

    template <typename T>
    rocsparse_status bsrsm_solve_template_large(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_X,
                                                rocsparse_int             mb,
                                                rocsparse_int             nrhs,
                                                rocsparse_int             nnzb,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                const T*                  B,
                                                int64_t                   ldb,
                                                T*                        X,
                                                int64_t                   ldx,
                                                void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

// One thread block per (RHS panel, block row) pair, flattened onto grid.x.
// ((nrhs - 1) / NCOL + 1) * mb was a signed 32 bit product, so the count is now
// formed in 64 bit and clamped to the device grid limit before it is narrowed;
// the kernels grid-stride over whatever is left over.
//
// bsrsm_solve_grid_size returns at most maxGridSize[0], except when mb alone
// exceeds it: then it returns mb so the launch fails with
// hipErrorInvalidConfiguration instead of running a grid too small to hold one
// RHS panel, which the kernels cannot stride. Saturate the narrowing so that
// stays true if mb also exceeds UINT32_MAX, rather than wrapping into a legal
// looking grid.
#define LAUNCH_LARGE_KERNEL(K_, M_, S_)                                                        \
    const int64_t bsrsm_grid_x                                                                 \
        = rocsparse::bsrsm_solve_grid_size(mb, nrhs, NCOL, handle->properties.maxGridSize[0]); \
    dim3 bsrsm_blocks(static_cast<uint32_t>(rocsparse::min(                                    \
        bsrsm_grid_x, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))));           \
    dim3 bsrsm_threads(NCOL* M_);                                                              \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((K_<NCOL * M_, NCOL, S_>),                              \
                                       bsrsm_blocks,                                           \
                                       bsrsm_threads,                                          \
                                       0,                                                      \
                                       stream,                                                 \
                                       mb,                                                     \
                                       nrhs,                                                   \
                                       local_bsr_row_ptr,                                      \
                                       local_bsr_col_ind,                                      \
                                       local_bsr_val,                                          \
                                       block_dim,                                              \
                                       Xt,                                                     \
                                       ldimX,                                                  \
                                       done_array,                                             \
                                       (const rocsparse_int*)trm_info->get_row_map(),          \
                                       (rocsparse_int*)info->get_bsrsm_info()->get_position(), \
                                       descr->base,                                            \
                                       descr->diag_type,                                       \
                                       dir);

        hipStream_t stream = handle->stream;

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);

        ptr += 256;

        // 16 columns per block seem to work very well
        static constexpr uint32_t NCOL = 16;

        const int narrays = (nrhs - 1) / NCOL + 1;

        // Number of scalar rows. mb * block_dim was a signed 32 bit product at
        // every use below (the bsrsm_copy_scale grid and row count, and the three
        // dense_transpose calls, whose m parameter is a template one that forwards
        // to an int64_t overload and so truncated before it widened). Form it once
        // in 64 bit (AISPARSE-670).
        const int64_t m_rows = static_cast<int64_t>(mb) * block_dim;

        // done_array
        int* done_array = reinterpret_cast<int*>(ptr);
        ptr += ((sizeof(int) * size_t(mb) * narrays - 1) / 256 + 1) * 256;

        // Temporary array to store transpose of X
        T* Xt = X;
        if(trans_X == rocsparse_operation_none)
        {
            Xt = reinterpret_cast<T*>(ptr);
            ptr += ((sizeof(T) * size_t(mb) * block_dim * nrhs - 1) / 256 + 1) * 256;
        }

        // Initialize buffers
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemsetAsync(done_array, 0, sizeof(int) * mb * narrays, stream));

        auto bsrsm_info = info->get_bsrsm_info();
        auto trm_info   = info->get_bsrsm_info(trans_A, descr->fill_mode);

        // If diag type is unit, re-initialize zero pivot to remove structural zeros
        if(descr->diag_type == rocsparse_diag_type_unit)
        {
            static const rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(bsrsm_info->get_position(),
                                                         &max,
                                                         sizeof(rocsparse_int),
                                                         hipMemcpyHostToDevice,
                                                         stream));
        }

        rocsparse_fill_mode fill_mode = descr->fill_mode;

        // Transpose X if X is not transposed yet to improve performance
        int64_t ldimX = ldx;
        if(trans_X == rocsparse_operation_none)
        {
            // Leading dimension for transposed X
            ldimX = nrhs;

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(
                    handle, m_rows, static_cast<int64_t>(nrhs), alpha, B, ldb, Xt, ldimX));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(
                    handle, m_rows, static_cast<int64_t>(nrhs), *alpha, B, ldb, Xt, ldimX));
            }
        }
        else
        {
            // Copy B into X and scale it with alpha.
            //
            // The grid is built from the 64 bit m_rows and clamped to the device
            // limit -- another one line substitution for rocsparse::get_grid_size
            // once PR #11512 (AISPARSE-696) lands -- and bsrsm_copy_scale
            // grid-strides over the rows the clamp left behind. No panel rounding
            // is needed: that kernel has no cross-block dependencies, so any
            // stride covers all rows.
            static constexpr uint32_t COPY_BLOCKSIZE = 1024;

            const int64_t copy_blocks
                = rocsparse::min((m_rows - 1) / COPY_BLOCKSIZE + 1,
                                 static_cast<int64_t>(handle->properties.maxGridSize[0]));

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrsm_copy_scale<COPY_BLOCKSIZE>),
                                               dim3(static_cast<uint32_t>(copy_blocks)),
                                               dim3(COPY_BLOCKSIZE),
                                               0,
                                               stream,
                                               m_rows,
                                               nrhs,
                                               ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                                               B,
                                               ldb,
                                               X,
                                               ldx,
                                               handle->pointer_mode == rocsparse_pointer_mode_host);
        }

        // Pointers to differentiate between transpose mode
        const rocsparse_int* local_bsr_row_ptr = bsr_row_ptr;
        const rocsparse_int* local_bsr_col_ind = bsr_col_ind;
        const T*             local_bsr_val     = bsr_val;

        // When computing transposed triangular solve, we first need to update the
        // transposed matrix values
        if(trans_A == rocsparse_operation_transpose)
        {
            T* bsrt_val = reinterpret_cast<T*>(ptr);

            LAUNCH_BSRSM_GTHR(256, 64, block_dim);

            local_bsr_row_ptr = (const rocsparse_int*)trm_info->get_transposed_row_ptr();
            local_bsr_col_ind = (const rocsparse_int*)trm_info->get_transposed_col_ind();
            local_bsr_val     = (const T*)bsrt_val;

            switch(fill_mode)
            {
            case rocsparse_fill_mode_lower:
                fill_mode = rocsparse_fill_mode_upper;
                break;
            case rocsparse_fill_mode_upper:
                fill_mode = rocsparse_fill_mode_lower;
                break;
            }
        }

        // Determine gcn_arch and ASIC revision
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        const int         asicRev       = handle->asic_rev;
        const int         wfSize        = handle->wavefront_size;

        // gfx908 A0/1
        if(gcn_arch_name == rocsparse_arch_names::gfx908 && asicRev < 2)
        {
            switch(fill_mode)
            {
            case rocsparse_fill_mode_upper:
            {
                LAUNCH_LARGE_KERNEL(rocsparse::bsrsm_upper_large_kernel, 16, true);
                break;
            }
            case rocsparse_fill_mode_lower:
            {
                LAUNCH_LARGE_KERNEL(rocsparse::bsrsm_lower_large_kernel, 16, true);
                break;
            }
            }
        }
        else
        {
            // Select tuned kernel

            uint32_t nbsr = rocsparse::max(4U, fnp2(block_dim));

            while(nbsr > wfSize)
            {
                nbsr >>= 1;
            }

            switch(nbsr)
            {
#define DEFINE_CASE(i)                                               \
    case i:                                                          \
    {                                                                \
        switch(fill_mode)                                            \
        {                                                            \
        case rocsparse_fill_mode_upper:                              \
        {                                                            \
            LAUNCH_LARGE_KERNEL(bsrsm_upper_large_kernel, i, false); \
            break;                                                   \
        }                                                            \
        case rocsparse_fill_mode_lower:                              \
        {                                                            \
            LAUNCH_LARGE_KERNEL(bsrsm_lower_large_kernel, i, false); \
            break;                                                   \
        }                                                            \
        }                                                            \
        break;                                                       \
    }

                DEFINE_CASE(4);
                DEFINE_CASE(8);
                DEFINE_CASE(16);
                DEFINE_CASE(32);
                DEFINE_CASE(64);
#undef DEFINE_CASE
            }
        }
#undef LAUNCH_LARGE_KERNEL

        // Transpose X back if X was not initially transposed
        if(trans_X == rocsparse_operation_none)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose_back(
                handle, m_rows, static_cast<int64_t>(nrhs), Xt, ldimX, X, ldx));
        }

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(T)                                               \
    template rocsparse_status rocsparse::bsrsm_solve_template_large( \
        rocsparse_handle          handle,                            \
        rocsparse_direction       dir,                               \
        rocsparse_operation       trans_A,                           \
        rocsparse_operation       trans_X,                           \
        rocsparse_int             mb,                                \
        rocsparse_int             nrhs,                              \
        rocsparse_int             nnzb,                              \
        const T*                  alpha,                             \
        const rocsparse_mat_descr descr,                             \
        const T*                  bsr_val,                           \
        const rocsparse_int*      bsr_row_ptr,                       \
        const rocsparse_int*      bsr_col_ind,                       \
        rocsparse_int             block_dim,                         \
        rocsparse_mat_info        info,                              \
        const T*                  B,                                 \
        int64_t                   ldb,                               \
        T*                        X,                                 \
        int64_t                   ldx,                               \
        void*                     temp_buffer)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
