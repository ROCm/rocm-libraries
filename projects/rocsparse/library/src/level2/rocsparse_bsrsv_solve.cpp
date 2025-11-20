/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_bsrsv.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_bsrsv.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "bsrsv_device.h"

namespace rocsparse
{
#define LAUNCH_BSRSV_GTHR_DIM(bsize, wfsize, dim)                                            \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsr_gather<wfsize, bsize / wfsize, dim>), \
                                       dim3((wfsize * nnzb - 1) / bsize + 1),                \
                                       dim3(wfsize, bsize / wfsize),                         \
                                       0,                                                    \
                                       stream,                                               \
                                       dir,                                                  \
                                       nnzb,                                                 \
                                       (rocsparse_int*)trm_info->get_transposed_perm(),      \
                                       bsr_val,                                              \
                                       bsrt_val,                                             \
                                       block_dim)

#define LAUNCH_BSRSV_GTHR(bsize, wfsize, dim) \
    if(dim <= 2)                              \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 4, 2);   \
    }                                         \
    else if(dim <= 4)                         \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else if(wfsize == 32)                     \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else                                      \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 64, 8);  \
    }

#define LAUNCH_BSRSV_SHARED(fill, ptr, bsize, wfsize, dim, arch, asic) \
    if(fill == rocsparse_fill_mode_lower)                              \
    {                                                                  \
        if(arch == rocpsarse_arch_names::gfx908 && asic < 2)           \
        {                                                              \
            LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, true);       \
        }                                                              \
        else                                                           \
        {                                                              \
            LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, false);      \
        }                                                              \
    }                                                                  \
    else                                                               \
    {                                                                  \
        if(arch == rocpsarse_arch_names::gfx908 && asic < 2)           \
        {                                                              \
            LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, true);       \
        }                                                              \
        else                                                           \
        {                                                              \
            LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, false);      \
        }                                                              \
    }

#define LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, arch)           \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (bsrsv_lower_shared<bsize, wfsize, dim, arch>),               \
        dim3((wfsize * mb - 1) / bsize + 1),                          \
        dim3(bsize),                                                  \
        0,                                                            \
        stream,                                                       \
        mb,                                                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        local_bsr_row_ptr,                                            \
        local_bsr_col_ind,                                            \
        local_bsr_val,                                                \
        block_dim,                                                    \
        x,                                                            \
        y,                                                            \
        done_array,                                                   \
        (rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)bsrsv_info->get_zero_pivot(),                 \
        descr->base,                                                  \
        descr->diag_type,                                             \
        dir,                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, arch)           \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (bsrsv_upper_shared<bsize, wfsize, dim, arch>),               \
        dim3((wfsize * mb - 1) / bsize + 1),                          \
        dim3(bsize),                                                  \
        0,                                                            \
        stream,                                                       \
        mb,                                                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        local_bsr_row_ptr,                                            \
        local_bsr_col_ind,                                            \
        local_bsr_val,                                                \
        block_dim,                                                    \
        x,                                                            \
        y,                                                            \
        done_array,                                                   \
        (rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)bsrsv_info->get_zero_pivot(),                 \
        descr->base,                                                  \
        descr->diag_type,                                             \
        dir,                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_BSRSV_GENERAL(fill, ptr, bsize, wfsize, arch, asic) \
    if(fill == rocsparse_fill_mode_lower)                          \
    {                                                              \
        if(arch == rocpsarse_arch_names::gfx908 && asic < 2)       \
        {                                                          \
            LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, true);       \
        }                                                          \
        else                                                       \
        {                                                          \
            LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, false);      \
        }                                                          \
    }                                                              \
    else                                                           \
    {                                                              \
        if(arch == rocpsarse_arch_names::gfx908 && asic < 2)       \
        {                                                          \
            LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, true);       \
        }                                                          \
        else                                                       \
        {                                                          \
            LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, false);      \
        }                                                          \
    }

#define LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, arch)               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (bsrsv_lower_general<bsize, wfsize, arch>),                   \
        dim3((wfsize * mb - 1) / bsize + 1),                          \
        dim3(bsize),                                                  \
        0,                                                            \
        stream,                                                       \
        mb,                                                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        local_bsr_row_ptr,                                            \
        local_bsr_col_ind,                                            \
        local_bsr_val,                                                \
        block_dim,                                                    \
        x,                                                            \
        y,                                                            \
        done_array,                                                   \
        (rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)bsrsv_info->get_zero_pivot(),                 \
        descr->base,                                                  \
        descr->diag_type,                                             \
        dir,                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, arch)               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (bsrsv_upper_general<bsize, wfsize, arch>),                   \
        dim3((wfsize * mb - 1) / bsize + 1),                          \
        dim3(bsize),                                                  \
        0,                                                            \
        stream,                                                       \
        mb,                                                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        local_bsr_row_ptr,                                            \
        local_bsr_col_ind,                                            \
        local_bsr_val,                                                \
        block_dim,                                                    \
        x,                                                            \
        y,                                                            \
        done_array,                                                   \
        (rocsparse_int*)trm_info->get_row_map(),                      \
        (rocsparse_int*)bsrsv_info->get_zero_pivot(),                 \
        descr->base,                                                  \
        descr->diag_type,                                             \
        dir,                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, rocsparse_int BSRDIM, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_lower_shared(rocsparse_int mb,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const rocsparse_int* __restrict__ bsr_row_ptr,
                            const rocsparse_int* __restrict__ bsr_col_ind,
                            const T* __restrict__ bsr_val,
                            rocsparse_int block_dim,
                            const T* __restrict__ x,
                            T* __restrict__ y,
                            int* __restrict__ done_array,
                            rocsparse_int* __restrict__ map,
                            rocsparse_int* __restrict__ zero_pivot,
                            rocsparse_index_base idx_base,
                            rocsparse_diag_type  diag_type,
                            rocsparse_direction  dir,
                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::bsrsv_lower_shared_device<BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                               alpha,
                                                                               bsr_row_ptr,
                                                                               bsr_col_ind,
                                                                               bsr_val,
                                                                               block_dim,
                                                                               x,
                                                                               y,
                                                                               done_array,
                                                                               map,
                                                                               zero_pivot,
                                                                               idx_base,
                                                                               diag_type,
                                                                               dir);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, rocsparse_int BSRDIM, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_upper_shared(rocsparse_int mb,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const rocsparse_int* __restrict__ bsr_row_ptr,
                            const rocsparse_int* __restrict__ bsr_col_ind,
                            const T* __restrict__ bsr_val,
                            rocsparse_int block_dim,
                            const T* __restrict__ x,
                            T* __restrict__ y,
                            int* __restrict__ done_array,
                            rocsparse_int* __restrict__ map,
                            rocsparse_int* __restrict__ zero_pivot,
                            rocsparse_index_base idx_base,
                            rocsparse_diag_type  diag_type,
                            rocsparse_direction  dir,
                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::bsrsv_upper_shared_device<BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                               alpha,
                                                                               bsr_row_ptr,
                                                                               bsr_col_ind,
                                                                               bsr_val,
                                                                               block_dim,
                                                                               x,
                                                                               y,
                                                                               done_array,
                                                                               map,
                                                                               zero_pivot,
                                                                               idx_base,
                                                                               diag_type,
                                                                               dir);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_lower_general(rocsparse_int mb,
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                             const rocsparse_int* __restrict__ bsr_row_ptr,
                             const rocsparse_int* __restrict__ bsr_col_ind,
                             const T* __restrict__ bsr_val,
                             rocsparse_int block_dim,
                             const T* __restrict__ x,
                             T* __restrict__ y,
                             int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int* __restrict__ zero_pivot,
                             rocsparse_index_base idx_base,
                             rocsparse_diag_type  diag_type,
                             rocsparse_direction  dir,
                             bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::bsrsv_lower_general_device<BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                                        alpha,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        bsr_val,
                                                                        block_dim,
                                                                        x,
                                                                        y,
                                                                        done_array,
                                                                        map,
                                                                        zero_pivot,
                                                                        idx_base,
                                                                        diag_type,
                                                                        dir);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_upper_general(rocsparse_int mb,
                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                             const rocsparse_int* __restrict__ bsr_row_ptr,
                             const rocsparse_int* __restrict__ bsr_col_ind,
                             const T* __restrict__ bsr_val,
                             rocsparse_int block_dim,
                             const T* __restrict__ x,
                             T* __restrict__ y,
                             int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int* __restrict__ zero_pivot,
                             rocsparse_index_base idx_base,
                             rocsparse_diag_type  diag_type,
                             rocsparse_direction  dir,
                             bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::bsrsv_upper_general_device<BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                                        alpha,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        bsr_val,
                                                                        block_dim,
                                                                        x,
                                                                        y,
                                                                        done_array,
                                                                        map,
                                                                        zero_pivot,
                                                                        idx_base,
                                                                        diag_type,
                                                                        dir);
    }

    template <typename T>
    rocsparse_status bsrsv_solve_dispatch(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          rocsparse_int             mb,
                                          rocsparse_int             nnzb,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             block_dim,
                                          rocsparse_mat_info        info,
                                          const T*                  x,
                                          T*                        y,
                                          rocsparse_solve_policy    policy,
                                          void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);

        ptr += 256;

        // done array
        int* done_array = reinterpret_cast<int*>(ptr);
        ptr += ((sizeof(int) * mb - 1) / 256 + 1) * 256;

        // Initialize buffers
        RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * mb, stream));

        auto                   bsrsv_info = info->get_bsrsv_info();
        rocsparse::trm_info_t* trm_info   = info->get_bsrsv_info(trans, descr->fill_mode);
        if(trm_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }

        // If diag type is unit, re-initialize zero pivot to remove structural zeros
        if(descr->diag_type == rocsparse_diag_type_unit)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_async(
                reinterpret_cast<rocsparse_int*>(bsrsv_info->get_zero_pivot()),
                std::numeric_limits<rocsparse_int>::max(),
                stream));
        }

        // Pointers to differentiate between transpose mode
        const rocsparse_int* local_bsr_row_ptr = bsr_row_ptr;
        const rocsparse_int* local_bsr_col_ind = bsr_col_ind;
        const T*             local_bsr_val     = bsr_val;

        rocsparse_fill_mode fill_mode = descr->fill_mode;

        // When computing transposed triangular solve, we first need to update the
        // transposed matrix values
        if(trans == rocsparse_operation_transpose)
        {
            T* bsrt_val = reinterpret_cast<T*>(ptr);

            // Gather transposed values
            LAUNCH_BSRSV_GTHR(256, 64, block_dim);

            local_bsr_row_ptr = (rocsparse_int*)trm_info->get_transposed_row_ptr();
            local_bsr_col_ind = (rocsparse_int*)trm_info->get_transposed_col_ind();
            local_bsr_val     = (T*)bsrt_val;

            fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                                 : rocsparse_fill_mode_lower;
        }

        // Determine gcn_arch and ASIC revision
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        const int         asicRev       = handle->asic_rev;

        if(handle->wavefront_size == 64)
        {
            if(block_dim <= 8)
            {
                // Launch shared memory based kernel for small BSR block dimensions
                LAUNCH_BSRSV_SHARED(
                    fill_mode, handle->pointer_mode, 128, 64, 8, gcn_arch_name, asicRev);
            }
            else if(block_dim <= 16)
            {
                // Launch shared memory based kernel for small BSR block dimensions
                LAUNCH_BSRSV_SHARED(
                    fill_mode, handle->pointer_mode, 128, 64, 16, gcn_arch_name, asicRev);
            }
            else if(block_dim <= 32)
            {
                // Launch shared memory based kernel for small BSR block dimensions
                LAUNCH_BSRSV_SHARED(
                    fill_mode, handle->pointer_mode, 128, 64, 32, gcn_arch_name, asicRev);
            }
            else
            {
                // Launch general algorithm for large BSR block dimensions (> 32x32)
                LAUNCH_BSRSV_GENERAL(
                    fill_mode, handle->pointer_mode, 128, 64, gcn_arch_name, asicRev);
            }
        }
        else
        {
            //
            // This is wavefront 32, let's exclude it.
            //
            // LCOV_EXCL_START;

            // Launch general algorithm
            LAUNCH_BSRSV_GENERAL(fill_mode, handle->pointer_mode, 128, 32, gcn_arch_name, asicRev);

            // LCOV_EXCL_STOP;
        }

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::bsrsv_solve_template(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans,
                                                 rocsparse_int             mb,
                                                 rocsparse_int             nnzb,
                                                 const T*                  alpha_device_host,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  bsr_val,
                                                 const rocsparse_int*      bsr_row_ptr,
                                                 const rocsparse_int*      bsr_col_ind,
                                                 rocsparse_int             block_dim,
                                                 rocsparse_mat_info        info,
                                                 const T*                  x,
                                                 T*                        y,
                                                 rocsparse_solve_policy    policy,
                                                 void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG_POINTER(11, info);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbsrsv"),
                         dir,
                         trans,
                         mb,
                         nnzb,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                         (const void*&)descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         block_dim,
                         (const void*&)info,
                         (const void*&)x,
                         (const void*&)y,
                         policy,
                         (const void*&)temp_buffer);

    // Check direction
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_ENUM(2, trans);
    ROCSPARSE_CHECKARG_ENUM(14, policy);

    // Check operation type
    ROCSPARSE_CHECKARG(
        2,
        trans,
        (trans != rocsparse_operation_none && trans != rocsparse_operation_transpose),
        rocsparse_status_not_implemented);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        6, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(3, mb);
    ROCSPARSE_CHECKARG_SIZE(4, nnzb);
    ROCSPARSE_CHECKARG_SIZE(10, block_dim);
    ROCSPARSE_CHECKARG(10, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(8, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(12, mb, x);
    ROCSPARSE_CHECKARG_ARRAY(13, mb, y);

    // Quick return if possible
    if(mb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(5, alpha_device_host);
    ROCSPARSE_CHECKARG_POINTER(15, temp_buffer);

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_col_ind);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsv_solve_dispatch(handle,
                                                              dir,
                                                              trans,
                                                              mb,
                                                              nnzb,
                                                              alpha_device_host,
                                                              descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              info,
                                                              x,
                                                              y,
                                                              policy,
                                                              temp_buffer));

    return rocsparse_status_success;
}

// bsrsv_solve
#define C_IMPL(NAME, TYPE)                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,           \
                                     rocsparse_direction       dir,              \
                                     rocsparse_operation       trans,            \
                                     rocsparse_int             mb,               \
                                     rocsparse_int             nnzb,             \
                                     const TYPE*               alpha,            \
                                     const rocsparse_mat_descr descr,            \
                                     const TYPE*               bsr_val,          \
                                     const rocsparse_int*      bsr_row_ptr,      \
                                     const rocsparse_int*      bsr_col_ind,      \
                                     rocsparse_int             block_dim,        \
                                     rocsparse_mat_info        info,             \
                                     const TYPE*               x,                \
                                     TYPE*                     y,                \
                                     rocsparse_solve_policy    policy,           \
                                     void*                     temp_buffer)      \
    try                                                                          \
    {                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                 \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsv_solve_template(handle,        \
                                                                  dir,           \
                                                                  trans,         \
                                                                  mb,            \
                                                                  nnzb,          \
                                                                  alpha,         \
                                                                  descr,         \
                                                                  bsr_val,       \
                                                                  bsr_row_ptr,   \
                                                                  bsr_col_ind,   \
                                                                  block_dim,     \
                                                                  info,          \
                                                                  x,             \
                                                                  y,             \
                                                                  policy,        \
                                                                  temp_buffer)); \
        return rocsparse_status_success;                                         \
    }                                                                            \
    catch(...)                                                                   \
    {                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                            \
    }

C_IMPL(rocsparse_sbsrsv_solve, float);
C_IMPL(rocsparse_dbsrsv_solve, double);
C_IMPL(rocsparse_cbsrsv_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsv_solve, rocsparse_double_complex);
#undef C_IMPL
