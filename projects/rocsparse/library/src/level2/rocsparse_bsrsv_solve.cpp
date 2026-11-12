/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "bsrsv_device_large.h"
#include "rocsparse_grid.hpp"

#include <limits>
#include <type_traits>

namespace rocsparse
{
    // wfsize * nnzb passes INT32_MAX from 33,554,432 non-zero blocks with a 64
    // wide wavefront, so the grid is formed in 64 bit and clamped.
    // rocsparse::bsr_gather grid-strides over whatever the clamp leaves.
    // block_dim is passed as I, the row-pointer index type.
#define LAUNCH_BSRSV_GTHR_DIM(bsize, wfsize, dim)                                          \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                    \
        (rocsparse::bsr_gather<wfsize, bsize / wfsize, dim>),                              \
        dim3(rocsparse::get_grid_size_x(                                                   \
            handle,                                                                        \
            (static_cast<int64_t>(wfsize) * static_cast<int64_t>(nnzb) - 1) / (bsize) + 1, \
            static_cast<int64_t>(wfsize))),                                                \
        dim3(wfsize, bsize / wfsize),                                                      \
        0,                                                                                 \
        stream,                                                                            \
        dir,                                                                               \
        nnzb,                                                                              \
        (const I*)trm_info->get_transposed_perm(),                                         \
        bsr_val,                                                                           \
        bsrt_val,                                                                          \
        static_cast<I>(block_dim))

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
    LAUNCH_BSRSV_PREPARE(bsize, wfsize);                               \
    switch(fill)                                                       \
    {                                                                  \
    case rocsparse_fill_mode_lower:                                    \
    {                                                                  \
        if(arch == rocsparse_arch_names::gfx908 && asic < 2)           \
        {                                                              \
            LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, true);       \
        }                                                              \
        else                                                           \
        {                                                              \
            LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, false);      \
        }                                                              \
        break;                                                         \
    }                                                                  \
    case rocsparse_fill_mode_upper:                                    \
    {                                                                  \
        if(arch == rocsparse_arch_names::gfx908 && asic < 2)           \
        {                                                              \
            LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, true);       \
        }                                                              \
        else                                                           \
        {                                                              \
            LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, false);      \
        }                                                              \
        break;                                                         \
    }                                                                  \
    }

    // Grid in 64 bit, then the narrow kernel when the legacy 32-bit launch is
    // still exact and the wide kernel otherwise. bsrsv_legacy_indices is a
    // constexpr bool in the caller, so the narrow kernel is only instantiated
    // for rocsparse_int indices.
#define LAUNCH_BSRSV_PREPARE(bsize, wfsize)                                                       \
    const int64_t bsrsv_num_blocks = rocsparse::ceil_div(                                         \
        static_cast<int64_t>(mb) * static_cast<int64_t>(wfsize), static_cast<int64_t>(bsize));    \
    const uint32_t bsrsv_grid                                                                     \
        = rocsparse::get_grid_size_x(handle, bsrsv_num_blocks, static_cast<int64_t>(bsize));      \
    const bool bsrsv_wide = !bsrsv_legacy_indices                                                 \
                            || rocsparse::bsrsv_solve_needs_wide(static_cast<int64_t>(mb),        \
                                                                 static_cast<int64_t>(block_dim), \
                                                                 static_cast<int64_t>(nnzb),      \
                                                                 bsrsv_num_blocks,                \
                                                                 static_cast<int64_t>(bsrsv_grid))

#define LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, arch)                       \
    do                                                                            \
    {                                                                             \
        if constexpr(bsrsv_legacy_indices)                                        \
        {                                                                         \
            if(!bsrsv_wide)                                                       \
            {                                                                     \
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                    (bsrsv_lower_shared<bsize, wfsize, dim, arch>),               \
                    dim3(bsrsv_grid),                                             \
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
                    zero_pivot,                                                   \
                    descr->base,                                                  \
                    descr->diag_type,                                             \
                    dir,                                                          \
                    handle->pointer_mode == rocsparse_pointer_mode_host);         \
                break;                                                            \
            }                                                                     \
        }                                                                         \
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                       \
            (bsrsv_lower_shared_wide<bsize, wfsize, dim, arch, T, I, J>),         \
            dim3(bsrsv_grid),                                                     \
            dim3(bsize),                                                          \
            0,                                                                    \
            stream,                                                               \
            mb,                                                                   \
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),         \
            local_bsr_row_ptr,                                                    \
            local_bsr_col_ind,                                                    \
            local_bsr_val,                                                        \
            block_dim,                                                            \
            x,                                                                    \
            y,                                                                    \
            done_array,                                                           \
            (const J*)trm_info->get_row_map(),                                    \
            zero_pivot,                                                           \
            descr->base,                                                          \
            descr->diag_type,                                                     \
            dir,                                                                  \
            handle->pointer_mode == rocsparse_pointer_mode_host);                 \
    } while(0)

#define LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, arch)                       \
    do                                                                            \
    {                                                                             \
        if constexpr(bsrsv_legacy_indices)                                        \
        {                                                                         \
            if(!bsrsv_wide)                                                       \
            {                                                                     \
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                    (bsrsv_upper_shared<bsize, wfsize, dim, arch>),               \
                    dim3(bsrsv_grid),                                             \
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
                    zero_pivot,                                                   \
                    descr->base,                                                  \
                    descr->diag_type,                                             \
                    dir,                                                          \
                    handle->pointer_mode == rocsparse_pointer_mode_host);         \
                break;                                                            \
            }                                                                     \
        }                                                                         \
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                       \
            (bsrsv_upper_shared_wide<bsize, wfsize, dim, arch, T, I, J>),         \
            dim3(bsrsv_grid),                                                     \
            dim3(bsize),                                                          \
            0,                                                                    \
            stream,                                                               \
            mb,                                                                   \
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),         \
            local_bsr_row_ptr,                                                    \
            local_bsr_col_ind,                                                    \
            local_bsr_val,                                                        \
            block_dim,                                                            \
            x,                                                                    \
            y,                                                                    \
            done_array,                                                           \
            (const J*)trm_info->get_row_map(),                                    \
            zero_pivot,                                                           \
            descr->base,                                                          \
            descr->diag_type,                                                     \
            dir,                                                                  \
            handle->pointer_mode == rocsparse_pointer_mode_host);                 \
    } while(0)

#define LAUNCH_BSRSV_GENERAL(fill, ptr, bsize, wfsize, arch, asic) \
    LAUNCH_BSRSV_PREPARE(bsize, wfsize);                           \
    switch(fill)                                                   \
    {                                                              \
    case rocsparse_fill_mode_lower:                                \
    {                                                              \
        if(arch == rocsparse_arch_names::gfx908 && asic < 2)       \
        {                                                          \
            LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, true);       \
        }                                                          \
        else                                                       \
        {                                                          \
            LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, false);      \
        }                                                          \
        break;                                                     \
    }                                                              \
    case rocsparse_fill_mode_upper:                                \
    {                                                              \
        if(arch == rocsparse_arch_names::gfx908 && asic < 2)       \
        {                                                          \
            LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, true);       \
        }                                                          \
        else                                                       \
        {                                                          \
            LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, false);      \
        }                                                          \
        break;                                                     \
    }                                                              \
    }

#define LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, arch)                           \
    do                                                                            \
    {                                                                             \
        if constexpr(bsrsv_legacy_indices)                                        \
        {                                                                         \
            if(!bsrsv_wide)                                                       \
            {                                                                     \
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                    (bsrsv_lower_general<bsize, wfsize, arch>),                   \
                    dim3(bsrsv_grid),                                             \
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
                    zero_pivot,                                                   \
                    descr->base,                                                  \
                    descr->diag_type,                                             \
                    dir,                                                          \
                    handle->pointer_mode == rocsparse_pointer_mode_host);         \
                break;                                                            \
            }                                                                     \
        }                                                                         \
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                       \
            (bsrsv_lower_general_wide<bsize, wfsize, arch, T, I, J>),             \
            dim3(bsrsv_grid),                                                     \
            dim3(bsize),                                                          \
            0,                                                                    \
            stream,                                                               \
            mb,                                                                   \
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),         \
            local_bsr_row_ptr,                                                    \
            local_bsr_col_ind,                                                    \
            local_bsr_val,                                                        \
            block_dim,                                                            \
            x,                                                                    \
            y,                                                                    \
            done_array,                                                           \
            (const J*)trm_info->get_row_map(),                                    \
            zero_pivot,                                                           \
            descr->base,                                                          \
            descr->diag_type,                                                     \
            dir,                                                                  \
            handle->pointer_mode == rocsparse_pointer_mode_host);                 \
    } while(0)

#define LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, arch)                           \
    do                                                                            \
    {                                                                             \
        if constexpr(bsrsv_legacy_indices)                                        \
        {                                                                         \
            if(!bsrsv_wide)                                                       \
            {                                                                     \
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                    (bsrsv_upper_general<bsize, wfsize, arch>),                   \
                    dim3(bsrsv_grid),                                             \
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
                    zero_pivot,                                                   \
                    descr->base,                                                  \
                    descr->diag_type,                                             \
                    dir,                                                          \
                    handle->pointer_mode == rocsparse_pointer_mode_host);         \
                break;                                                            \
            }                                                                     \
        }                                                                         \
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                       \
            (bsrsv_upper_general_wide<bsize, wfsize, arch, T, I, J>),             \
            dim3(bsrsv_grid),                                                     \
            dim3(bsize),                                                          \
            0,                                                                    \
            stream,                                                               \
            mb,                                                                   \
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),         \
            local_bsr_row_ptr,                                                    \
            local_bsr_col_ind,                                                    \
            local_bsr_val,                                                        \
            block_dim,                                                            \
            x,                                                                    \
            y,                                                                    \
            done_array,                                                           \
            (const J*)trm_info->get_row_map(),                                    \
            zero_pivot,                                                           \
            descr->base,                                                          \
            descr->diag_type,                                                     \
            dir,                                                                  \
            handle->pointer_mode == rocsparse_pointer_mode_host);                 \
    } while(0)

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, rocsparse_int BSRDIM, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrsv_lower_shared(rocsparse_int mb,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const rocsparse_int* __restrict__ bsr_row_ptr,
                            const rocsparse_int* __restrict__ bsr_col_ind,
                            const T* __restrict__ bsr_val,
                            rocsparse_int block_dim,
                            const T* __restrict__ x,
                            T* y,
                            int* __restrict__ done_array,
                            rocsparse_int* __restrict__ map,
                            rocsparse_int*       zero_pivot,
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
                            T* y,
                            int* __restrict__ done_array,
                            rocsparse_int* __restrict__ map,
                            rocsparse_int*       zero_pivot,
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
                             T* y,
                             int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int*       zero_pivot,
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
                             T* y,
                             int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int*       zero_pivot,
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

    // Solve core. The analysis object and the zero-pivot pointer are arguments,
    // so a caller that owns the analysis (the legacy mat_info path, and later
    // rocsparse_sptrsv) does not have to read them back out of rocsparse_mat_info.
    template <typename T, typename I, typename J>
    rocsparse_status bsrsv_solve_core(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_operation       trans,
                                      J                         mb,
                                      I                         nnzb,
                                      const T*                  alpha_device_host,
                                      const rocsparse_mat_descr descr,
                                      const T*                  bsr_val,
                                      const I*                  bsr_row_ptr,
                                      const J*                  bsr_col_ind,
                                      rocsparse_int             block_dim,
                                      rocsparse::trm_info_t*    trm_info,
                                      rocsparse_indextype       zero_pivot_indextype,
                                      J*                        zero_pivot,
                                      const T*                  x,
                                      T*                        y,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        constexpr bool bsrsv_legacy_indices
            = std::is_same<I, rocsparse_int>::value && std::is_same<J, rocsparse_int>::value;

        // Validate before any device memory is touched. A future rocsparse_sptrsv
        // caller reaches this core with other index types, so the analysis must have
        // been built with the same row-pointer and column-index types.
        if(trm_info == nullptr || temp_buffer == nullptr || zero_pivot == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }

        if(!trm_info->has_indextypes(rocsparse::get_indextype<I>(), rocsparse::get_indextype<J>()))
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }

        // The pivot buffer is typed by the analysis that wrote it. A reuse of
        // another fill mode can leave a different width in the shared pivot.
        if(zero_pivot_indextype != rocsparse::get_indextype<J>())
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }

        // Stream
        hipStream_t stream = handle->stream;

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);

        ptr += 256;

        // done array. mb is the real index width, so the byte count is 64 bit.
        int* done_array = reinterpret_cast<int*>(ptr);
        ptr += ((sizeof(int) * static_cast<size_t>(mb) - 1) / 256 + 1) * 256;

        // Initialize buffers
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemsetAsync(done_array, 0, sizeof(int) * static_cast<size_t>(mb), stream));

        // If diag type is unit, re-initialize zero pivot to remove structural zeros
        if(descr->diag_type == rocsparse_diag_type_unit)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_async(
                static_cast<int64_t>(1), zero_pivot, std::numeric_limits<J>::max(), stream));
        }

        // Pointers to differentiate between transpose mode
        const I* local_bsr_row_ptr = bsr_row_ptr;
        const J* local_bsr_col_ind = bsr_col_ind;
        const T* local_bsr_val     = bsr_val;

        rocsparse_fill_mode fill_mode = descr->fill_mode;

        // When computing transposed triangular solve, we first need to update the
        // transposed matrix values
        if(trans == rocsparse_operation_transpose)
        {
            T* bsrt_val = reinterpret_cast<T*>(ptr);

            // Gather transposed values
            LAUNCH_BSRSV_GTHR(256, 64, block_dim);

            local_bsr_row_ptr = (const I*)trm_info->get_transposed_row_ptr();
            local_bsr_col_ind = (const J*)trm_info->get_transposed_col_ind();
            local_bsr_val     = (T*)bsrt_val;

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

#define ROCSPARSE_BSRSV_CORE_INST(T, I, J)                                                    \
    template rocsparse_status rocsparse::bsrsv_solve_core<T, I, J>(rocsparse_handle,          \
                                                                   rocsparse_direction,       \
                                                                   rocsparse_operation,       \
                                                                   J,                         \
                                                                   I,                         \
                                                                   const T*,                  \
                                                                   const rocsparse_mat_descr, \
                                                                   const T*,                  \
                                                                   const I*,                  \
                                                                   const J*,                  \
                                                                   rocsparse_int,             \
                                                                   rocsparse::trm_info_t*,    \
                                                                   rocsparse_indextype,       \
                                                                   J*,                        \
                                                                   const T*,                  \
                                                                   T*,                        \
                                                                   rocsparse_solve_policy,    \
                                                                   void*)

#define ROCSPARSE_BSRSV_CORE_INST_T(I, J)                     \
    ROCSPARSE_BSRSV_CORE_INST(float, I, J);                   \
    ROCSPARSE_BSRSV_CORE_INST(double, I, J);                  \
    ROCSPARSE_BSRSV_CORE_INST(rocsparse_float_complex, I, J); \
    ROCSPARSE_BSRSV_CORE_INST(rocsparse_double_complex, I, J)

ROCSPARSE_BSRSV_CORE_INST_T(int32_t, int32_t);
ROCSPARSE_BSRSV_CORE_INST_T(int64_t, int32_t);
ROCSPARSE_BSRSV_CORE_INST_T(int64_t, int64_t);

#undef ROCSPARSE_BSRSV_CORE_INST_T
#undef ROCSPARSE_BSRSV_CORE_INST

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

    auto                   bsrsv_info = info->get_bsrsv_info();
    rocsparse::trm_info_t* trm_info   = info->get_bsrsv_info(trans, descr->fill_mode);
    if(trm_info == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::bsrsv_solve_core<T, rocsparse_int, rocsparse_int>(
        handle,
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
        trm_info,
        bsrsv_info->get_indextype(),
        reinterpret_cast<rocsparse_int*>(bsrsv_info->get_position()),
        x,
        y,
        policy,
        temp_buffer)));

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
