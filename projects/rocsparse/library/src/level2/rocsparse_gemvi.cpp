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

#include "internal/level2/rocsparse_gemvi.h"
#include "rocsparse_common.h"
#include "rocsparse_gemvi.hpp"

#include "gemvi_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t UNROLL, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemvi_kernel_part1(I m,
                            I n,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const T* __restrict__ A,
                            int64_t lda,
                            I       nnz,
                            const T* __restrict__ x_val,
                            const I* __restrict__ x_ind,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                            T* __restrict__ y,
                            T* __restrict__ workspace,
                            rocsparse_index_base idx_base,
                            bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::gemvi_device_part1<BLOCKSIZE, WFSIZE, UNROLL>(
                m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, workspace, idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemvi_kernel_part2(I   m,
                            I   n,
                            int grid_y,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                            const T* __restrict__ workspace,
                            T*   y,
                            bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::gemvi_device_part2<BLOCKSIZE, WFSIZE>(
                m, n, grid_y, alpha, beta, workspace, y);
        }
    }

    // part1 launch configuration. gemvi_buffer_size_template must size the
    // workspace from the same values that gemvi_dispatch launches with.
    constexpr uint32_t gemvi_part1_blocksize = 256;
    constexpr uint32_t gemvi_part1_unroll    = 8;

    template <uint32_t WFSIZE>
    inline int gemvi_part1_grid_x(int m)
    {
        return (m - 1) / WFSIZE + 1;
    }

    // Number of part1 blocks the device can hold concurrently, taken from the
    // real occupancy of the kernel.
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t UNROLL, typename T, typename I>
    inline int gemvi_part1_resident_blocks(const hipDeviceProp_t& prop)
    {
        int blocks_per_cu = 0;
        if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
               &blocks_per_cu, gemvi_kernel_part1<BLOCKSIZE, WFSIZE, UNROLL, I, T>, BLOCKSIZE, 0)
               != hipSuccess
           || blocks_per_cu < 1)
        {
            // LCOV_EXCL_START
            // Fall back to the device's resident thread capacity.
            blocks_per_cu = 1;
            if(prop.maxThreadsPerMultiProcessor > 0)
            {
                blocks_per_cu = rocsparse::max(
                    prop.maxThreadsPerMultiProcessor / static_cast<int>(BLOCKSIZE), 1);
            }
            // LCOV_EXCL_STOP
        }

        return rocsparse::max(prop.multiProcessorCount * blocks_per_cu, 1);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t UNROLL, typename T, typename I>
    inline int gemvi_part1_grid_y(rocsparse_handle handle, I m, I nnz)
    {
        constexpr int max_grid_y = 256;

        // Sparse vector entries one wavefront worker consumes per unrolled step
        // of part1, i.e. the granularity at which the split is useful at all.
        constexpr int step = (BLOCKSIZE / WFSIZE) * UNROLL;

        // Below this much work per worker the part2 reduction costs more than
        // the idle hardware the split recovers (measured on gfx1201).
        constexpr int min_split_steps = 16;

        if(nnz < min_split_steps * step)
        {
            return 1;
        }

        const int grid_x = gemvi_part1_grid_x<WFSIZE>(m);
        const int resident
            = gemvi_part1_resident_blocks<BLOCKSIZE, WFSIZE, UNROLL, T, I>(handle->properties);

        if(grid_x >= resident)
        {
            return 1;
        }

        // Grow only until the device is covered, and never so far that a worker
        // is left with less than one unrolled step of the sparse vector.
        const int ny_occ  = (resident + grid_x - 1) / grid_x;
        const int ny_work = rocsparse::min(rocsparse::max(nnz / step, 1), max_grid_y);

        return rocsparse::max(1, rocsparse::min(ny_work, ny_occ));
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t UNROLL, typename I, typename T>
    rocsparse_status gemvi_kernel_dispatch(rocsparse_handle     handle,
                                           I                    m,
                                           I                    n,
                                           const T*             alpha_device_host,
                                           const T*             A,
                                           int64_t              lda,
                                           I                    nnz,
                                           const T*             x_val,
                                           const I*             x_ind,
                                           const T*             beta_device_host,
                                           T*                   y,
                                           rocsparse_index_base idx_base,
                                           T*                   workspace)
    {
        const int grid_x = gemvi_part1_grid_x<WFSIZE>(m);
        const int grid_y = gemvi_part1_grid_y<BLOCKSIZE, WFSIZE, UNROLL, T>(handle, m, nnz);

        dim3 grid(grid_x, grid_y, 1);
        dim3 blocks(BLOCKSIZE, 1, 1);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (gemvi_kernel_part1<BLOCKSIZE, WFSIZE, UNROLL>),
            grid,
            blocks,
            0,
            handle->stream,
            m,
            n,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
            A,
            lda,
            nnz,
            x_val,
            x_ind,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y,
            workspace,
            idx_base,
            handle->pointer_mode == rocsparse_pointer_mode_host);

        if(grid_y > 1)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (gemvi_kernel_part2<BLOCKSIZE, WFSIZE>),
                dim3(grid_x),
                dim3(BLOCKSIZE),
                0,
                handle->stream,
                m,
                n,
                grid_y,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                workspace,
                y,
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }

        return rocsparse_status_success;
    }

    template <uint32_t WFSIZE, typename I, typename T>
    rocsparse_status gemvi_dispatch_by_wavefront(rocsparse_handle     handle,
                                                 I                    m,
                                                 I                    n,
                                                 const T*             alpha_device_host,
                                                 const T*             A,
                                                 int64_t              lda,
                                                 I                    nnz,
                                                 const T*             x_val,
                                                 const I*             x_ind,
                                                 const T*             beta_device_host,
                                                 T*                   y,
                                                 rocsparse_index_base idx_base,
                                                 T*                   workspace)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(nnz < (gemvi_part1_blocksize / WFSIZE) * gemvi_part1_unroll)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (gemvi_kernel_dispatch<WFSIZE, WFSIZE, gemvi_part1_unroll>(handle,
                                                                           m,
                                                                           n,
                                                                           alpha_device_host,
                                                                           A,
                                                                           lda,
                                                                           nnz,
                                                                           x_val,
                                                                           x_ind,
                                                                           beta_device_host,
                                                                           y,
                                                                           idx_base,
                                                                           workspace)));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (gemvi_kernel_dispatch<gemvi_part1_blocksize, WFSIZE, gemvi_part1_unroll>(
                    handle,
                    m,
                    n,
                    alpha_device_host,
                    A,
                    lda,
                    nnz,
                    x_val,
                    x_ind,
                    beta_device_host,
                    y,
                    idx_base,
                    workspace)));
        }

        return rocsparse_status_success;
    }

    template <typename I, typename T>
    rocsparse_status gemvi_dispatch(rocsparse_handle     handle,
                                    rocsparse_operation  trans,
                                    I                    m,
                                    I                    n,
                                    const T*             alpha_device_host,
                                    const T*             A,
                                    int64_t              lda,
                                    I                    nnz,
                                    const T*             x_val,
                                    const I*             x_ind,
                                    const T*             beta_device_host,
                                    T*                   y,
                                    rocsparse_index_base idx_base,
                                    void*                temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // If nnz is zero, only compute beta * y
        if(nnz == 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, m, beta_device_host, y));

            return rocsparse_status_success;
        }

        T* workspace = reinterpret_cast<T*>(temp_buffer);

        if(trans == rocsparse_operation_none)
        {
            if(handle->wavefront_size == 32)
            {
                RETURN_IF_ROCSPARSE_ERROR((gemvi_dispatch_by_wavefront<32>(handle,
                                                                           m,
                                                                           n,
                                                                           alpha_device_host,
                                                                           A,
                                                                           lda,
                                                                           nnz,
                                                                           x_val,
                                                                           x_ind,
                                                                           beta_device_host,
                                                                           y,
                                                                           idx_base,
                                                                           workspace)));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR((gemvi_dispatch_by_wavefront<64>(handle,
                                                                           m,
                                                                           n,
                                                                           alpha_device_host,
                                                                           A,
                                                                           lda,
                                                                           nnz,
                                                                           x_val,
                                                                           x_ind,
                                                                           beta_device_host,
                                                                           y,
                                                                           idx_base,
                                                                           workspace)));
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        return rocsparse_status_success;
    }

    template <typename T, typename I>
    rocsparse_status gemvi_buffer_size_template(rocsparse_handle    handle, //0
                                                rocsparse_operation trans, //1
                                                I                   m, //2
                                                I                   n, //3
                                                I                   nnz, //4
                                                size_t*             buffer_size) //5
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        // Check operation mode
        ROCSPARSE_CHECKARG_ENUM(1, trans);

        // Check sizes
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, n);
        ROCSPARSE_CHECKARG_SIZE(4, nnz);

        // nnz of sparse vector cannot exceed its size
        ROCSPARSE_CHECKARG(4, nnz, (nnz > n), rocsparse_status_invalid_size);

        const int grid_x = (handle->wavefront_size == 32) ? gemvi_part1_grid_x<32>(m)
                                                          : gemvi_part1_grid_x<64>(m);
        const int grid_y
            = (handle->wavefront_size == 32)
                  ? gemvi_part1_grid_y<gemvi_part1_blocksize, 32, gemvi_part1_unroll, T>(
                        handle, m, nnz)
                  : gemvi_part1_grid_y<gemvi_part1_blocksize, 64, gemvi_part1_unroll, T>(
                        handle, m, nnz);

        if(grid_y > 1)
        {
            *buffer_size = sizeof(T) * handle->wavefront_size * grid_y * grid_x;
        }
        else
        {
            *buffer_size = 0;
        }

        return rocsparse_status_success;
    }

    template <typename I, typename T>
    rocsparse_status gemvi_template(rocsparse_handle     handle, //0
                                    rocsparse_operation  trans, //1
                                    I                    m, //2
                                    I                    n, //3
                                    const T*             alpha_device_host, //4
                                    const T*             A, //5
                                    int64_t              lda, //6
                                    I                    nnz, //7
                                    const T*             x_val, //8
                                    const I*             x_ind, //9
                                    const T*             beta_device_host, //10
                                    T*                   y, //11
                                    rocsparse_index_base idx_base, //12
                                    void*                temp_buffer) //13
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xgemvi"),
                             trans,
                             m,
                             n,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                             (const void*&)A,
                             lda,
                             nnz,
                             (const void*&)x_val,
                             (const void*&)x_ind,
                             LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                             (const void*&)y,
                             idx_base,
                             (const void*&)temp_buffer);

        // Check operation mode
        ROCSPARSE_CHECKARG_ENUM(1, trans);

        // Check index base
        ROCSPARSE_CHECKARG_ENUM(12, idx_base);

        // Check sizes
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, n);
        ROCSPARSE_CHECKARG_SIZE(7, nnz);

        // nnz of sparse vector cannot exceed its size
        ROCSPARSE_CHECKARG(7, nnz, (nnz > n), rocsparse_status_invalid_size);

        // Check leading dimension
        ROCSPARSE_CHECKARG(6,
                           lda,
                           ((lda < m) && (trans == rocsparse_operation_none)),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(6,
                           lda,
                           ((lda < n) && (trans != rocsparse_operation_none)),
                           rocsparse_status_invalid_size);

        // Quick return if possible
        if(m == 0)
        {
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(4, alpha_device_host);

        // Check invalid pointers
        if(m > 0 && n > 0 && nnz > 0)
        {
            ROCSPARSE_CHECKARG_POINTER(5, A);
            ROCSPARSE_CHECKARG_POINTER(8, x_val);
            ROCSPARSE_CHECKARG_POINTER(9, x_ind);
            // Allow temp_buffer to be nullptr
        }

        ROCSPARSE_CHECKARG_POINTER(10, beta_device_host);
        ROCSPARSE_CHECKARG_POINTER(11, y);

        // Quick return if there is no work to do - alpha can be (valid) nullptr!
        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            if(alpha_device_host == nullptr && *beta_device_host == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }

            if(alpha_device_host != nullptr)
            {
                if(*alpha_device_host == static_cast<T>(0)
                   && *beta_device_host == static_cast<T>(1))
                {
                    return rocsparse_status_success;
                }
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemvi_dispatch(handle,
                                                            trans,
                                                            m,
                                                            n,
                                                            alpha_device_host,
                                                            A,
                                                            lda,
                                                            nnz,
                                                            x_val,
                                                            x_ind,
                                                            beta_device_host,
                                                            y,
                                                            idx_base,
                                                            temp_buffer));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

// Definition of the C-implementation

// rocsparse_xgemvi_buffer_size
#define CAPI_IMPL(name_, type_)                                                                   \
    rocsparse_status name_(rocsparse_handle    handle,                                            \
                           rocsparse_operation trans,                                             \
                           rocsparse_int       m,                                                 \
                           rocsparse_int       n,                                                 \
                           rocsparse_int       nnz,                                               \
                           size_t*             buffer_size)                                       \
    try                                                                                           \
    {                                                                                             \
        ROCSPARSE_ROUTINE_TRACE;                                                                  \
        RETURN_IF_ROCSPARSE_ERROR(                                                                \
            rocsparse::gemvi_buffer_size_template<type_>(handle, trans, m, n, nnz, buffer_size)); \
        return rocsparse_status_success;                                                          \
    }                                                                                             \
    catch(...)                                                                                    \
    {                                                                                             \
        RETURN_ROCSPARSE_EXCEPTION();                                                             \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi_buffer_size, float);
CAPI_IMPL(rocsparse_dgemvi_buffer_size, double);
CAPI_IMPL(rocsparse_cgemvi_buffer_size, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi_buffer_size, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL

// rocsparse_xgemvi
#define CAPI_IMPL(name_, type_)                                                \
    rocsparse_status name_(rocsparse_handle     handle,                        \
                           rocsparse_operation  trans,                         \
                           rocsparse_int        m,                             \
                           rocsparse_int        n,                             \
                           const type_*         alpha,                         \
                           const type_*         A,                             \
                           rocsparse_int        lda,                           \
                           rocsparse_int        nnz,                           \
                           const type_*         x_val,                         \
                           const rocsparse_int* x_ind,                         \
                           const type_*         beta,                          \
                           type_*               y,                             \
                           rocsparse_index_base idx_base,                      \
                           void*                temp_buffer)                   \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            ROCSPARSE_ROUTINE_TRACE;                                           \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemvi_template(handle,        \
                                                                trans,         \
                                                                m,             \
                                                                n,             \
                                                                alpha,         \
                                                                A,             \
                                                                lda,           \
                                                                nnz,           \
                                                                x_val,         \
                                                                x_ind,         \
                                                                beta,          \
                                                                y,             \
                                                                idx_base,      \
                                                                temp_buffer)); \
            return rocsparse_status_success;                                   \
        }                                                                      \
        catch(...)                                                             \
        {                                                                      \
            RETURN_ROCSPARSE_EXCEPTION();                                      \
        }                                                                      \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi, float);
CAPI_IMPL(rocsparse_dgemvi, double);
CAPI_IMPL(rocsparse_cgemvi, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL
}
