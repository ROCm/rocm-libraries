/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_gtsv_no_pivot_strided_batch.hpp"
#include "internal/precond/rocsparse_gtsv.h"

#include "gtsv_nopivot_strided_batch_device.h"
#include "gtsv_nopivot_strided_batch_large_device.h"
#include "gtsv_nopivot_strided_batch_medium_device.h"

#include <map>

// LCOV_EXCL_START
static constexpr int determine_spike_solver_blocksize()
{
    return 256;
}
// LCOV_EXCL_STOP

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_STAGE1(T, block_size, stride, iter) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_pow2_stage1_kernel<block_size>),    \
        dim3(((m - 1) / block_size + 1), batch_count, 1),                              \
        dim3(block_size, 1, 1),                                                        \
        0,                                                                             \
        handle->stream,                                                                \
        stride,                                                                        \
        m,                                                                             \
        batch_count,                                                                   \
        ((iter == 0) ? batch_stride : m),                                              \
        ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),                          \
        ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),                           \
        ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),                          \
        ((iter == 0) ? x : (((iter & 1) == 0) ? drhs0 : drhs1)),                       \
        (((iter & 1) == 0) ? da1 : da0),                                               \
        (((iter & 1) == 0) ? db1 : db0),                                               \
        (((iter & 1) == 0) ? dc1 : dc0),                                               \
        (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_STAGE2(T, block_size, iter)      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                            \
        (rocsparse::gtsv_nopivot_strided_batch_cr_pow2_stage2_kernel<block_size>), \
        dim3(subsystem_count, batch_count, 1),                                     \
        dim3(block_size),                                                          \
        0,                                                                         \
        handle->stream,                                                            \
        m,                                                                         \
        batch_count,                                                               \
        batch_stride,                                                              \
        (((iter & 1) != 0) ? da1 : da0),                                           \
        (((iter & 1) != 0) ? db1 : db0),                                           \
        (((iter & 1) != 0) ? dc1 : dc0),                                           \
        (((iter & 1) != 0) ? drhs1 : drhs0),                                       \
        x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE1(T, block_size, stride, iter) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                           \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_stage1_kernel<block_size>),    \
        dim3(((m - 1) / block_size + 1), batch_count, 1),                         \
        dim3(block_size),                                                         \
        0,                                                                        \
        handle->stream,                                                           \
        stride,                                                                   \
        m,                                                                        \
        batch_count,                                                              \
        ((iter == 0) ? batch_stride : m),                                         \
        ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),                     \
        ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),                      \
        ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),                     \
        ((iter == 0) ? x : (((iter & 1) == 0) ? drhs0 : drhs1)),                  \
        (((iter & 1) == 0) ? da1 : da0),                                          \
        (((iter & 1) == 0) ? db1 : db0),                                          \
        (((iter & 1) == 0) ? dc1 : dc0),                                          \
        (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE2(T, block_size, iter)      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                        \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_stage2_kernel<block_size>), \
        dim3(subsystem_count, batch_count, 1),                                 \
        dim3(block_size),                                                      \
        0,                                                                     \
        handle->stream,                                                        \
        m,                                                                     \
        batch_count,                                                           \
        batch_stride,                                                          \
        (((iter & 1) != 0) ? da1 : da0),                                       \
        (((iter & 1) != 0) ? db1 : db0),                                       \
        (((iter & 1) != 0) ? dc1 : dc0),                                       \
        (((iter & 1) != 0) ? drhs1 : drhs0),                                   \
        x);

template <typename T>
rocsparse_status
    rocsparse::gtsv_no_pivot_strided_batch_buffer_size_template(rocsparse_handle handle,
                                                                rocsparse_int    m,
                                                                const T*         dl,
                                                                const T*         d,
                                                                const T*         du,
                                                                const T*         x,
                                                                rocsparse_int    batch_count,
                                                                rocsparse_int    batch_stride,
                                                                size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(
        handle,
        rocsparse::replaceX<T>("rocsparse_Xgtsv_no_pivot_strided_batch_buffer_size"),
        m,
        (const void*&)dl,
        (const void*&)d,
        (const void*&)du,
        (const void*&)x,
        batch_count,
        batch_stride,
        (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(7, batch_stride, (batch_stride < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(6, batch_count);

    ROCSPARSE_CHECKARG_ARRAY(2, batch_count, dl);
    ROCSPARSE_CHECKARG_ARRAY(3, batch_count, d);
    ROCSPARSE_CHECKARG_ARRAY(4, batch_count, du);
    ROCSPARSE_CHECKARG_ARRAY(5, batch_count, x);
    ROCSPARSE_CHECKARG_POINTER(8, buffer_size);

    // Quick return if possible
    if(batch_count == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    if(m <= 512)
    {
        *buffer_size = 0;
    }
    else if(m <= 131072) //2^17
    {
        *buffer_size = 0;

        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // dl_modified
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // d_modified
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // du_modified
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // B_modified

        constexpr int BLOCKSIZE  = determine_spike_solver_blocksize();
        const int     nblocks    = ((m - 1) / BLOCKSIZE + 1);
        const int     num_spikes = 2 * nblocks;

        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // dl_spike
        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // d_spike
        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // du_spike
        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // B_spike
    }
    else
    {
        *buffer_size = 0;

        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // da0
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // da1
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // db0
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // db1
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // dc0
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // dc1
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // drhs0
        *buffer_size += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256; // drhs1
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_cramer_rule_kernel(rocsparse_handle handle,
                                               rocsparse_int    n,
                                               rocsparse_int    stride,
                                               int64_t          ldb,
                                               const T*         dl,
                                               const T*         d,
                                               const T*         du,
                                               T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_2x2_kernel<BLOCKSIZE>),
                                           dim3((n - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           n,
                                           stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_3(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_3x3_kernel<BLOCKSIZE>),
                                           dim3((n - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           n,
                                           stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_4(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_4x4_kernel<BLOCKSIZE>),
                                           dim3((n - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           n,
                                           stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_5(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_5x5_kernel<BLOCKSIZE>),
                                           dim3((n - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           n,
                                           stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_6(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_6x6_kernel<BLOCKSIZE>),
                                           dim3((n - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           n,
                                           stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_7(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_7x7_kernel<BLOCKSIZE>),
                                           dim3((n - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           n,
                                           stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t M, typename T>
    rocsparse_status launch_thomas_kernel_m(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_thomas_kernel<BLOCKSIZE, M>),
                                           dim3((n - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           n,
                                           stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    // Wavefront PCR kernels: m <= 8, 16, 32 (WF_SIZE varies)
    template <uint32_t WF_SIZE, typename T>
    rocsparse_status launch_pcr_wavefront_kernel(rocsparse_handle handle,
                                                 rocsparse_int    m,
                                                 rocsparse_int    n,
                                                 int64_t          batch_stride,
                                                 int64_t          ldb,
                                                 const T*         dl,
                                                 const T*         d,
                                                 const T*         du,
                                                 T*               B)
    {
        constexpr uint32_t BLOCKSIZE = 256;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_batch_pcr_wavefront_kernel<BLOCKSIZE, WF_SIZE>),
            dim3((n - 1) / (BLOCKSIZE / WF_SIZE) + 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
            batch_stride,
            ldb,
            dl,
            d,
            du,
            B);
        return rocsparse_status_success;
    }

    // Shared-memory PCR kernels: m <= 64, 128, 256 (BLOCKSIZE varies)
    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_pcr_shared_kernel(rocsparse_handle handle,
                                              rocsparse_int    m,
                                              rocsparse_int    n,
                                              int64_t          batch_stride,
                                              int64_t          ldb,
                                              const T*         dl,
                                              const T*         d,
                                              const T*         du,
                                              T*               B)
    {
        constexpr uint32_t WF_SIZE = 32;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_batch_pcr_shared_kernel<BLOCKSIZE, WF_SIZE>),
            dim3(n),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
            batch_stride,
            ldb,
            dl,
            d,
            du,
            B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t HALF_BLOCKSIZE, typename T>
    rocsparse_status launch_crpcr_pow2_shared_kernel(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     rocsparse_int    n,
                                                     int64_t          batch_stride,
                                                     int64_t          ldb,
                                                     const T*         dl,
                                                     const T*         d,
                                                     const T*         du,
                                                     T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_batch_crpcr_shared_kernel<BLOCKSIZE, HALF_BLOCKSIZE>),
            dim3(n),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
            batch_stride,
            ldb,
            dl,
            d,
            du,
            B);
        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status gtsv_no_pivot_strided_batch_small_template(rocsparse_handle handle,
                                                                rocsparse_int    m,
                                                                const T*         dl,
                                                                const T*         d,
                                                                const T*         du,
                                                                T*               x,
                                                                rocsparse_int    batch_count,
                                                                rocsparse_int    batch_stride,
                                                                void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m <= 1024, "This function is designed for m <= 1024.");

        using thomas_kernel_func_ptr = rocsparse_status (*)(rocsparse_handle handle,
                                                            rocsparse_int    n,
                                                            rocsparse_int    stride,
                                                            int64_t          ldb,
                                                            const T*         dl,
                                                            const T*         d,
                                                            const T*         du,
                                                            T*               B);

        // Kernel dispatch table for thomas solver
        static const std::map<int, thomas_kernel_func_ptr> s_thomas_kernel_dispatch
            = {{2, launch_cramer_rule_kernel<256>},
               {3, launch_thomas_kernel_3<256>},
               {4, launch_thomas_kernel_4<256>},
               {5, launch_thomas_kernel_5<256>},
               {6, launch_thomas_kernel_6<256>},
               {7, launch_thomas_kernel_7<256>},
               {8, launch_thomas_kernel_m<256, 8>},
               {9, launch_thomas_kernel_m<256, 9>},
               {10, launch_thomas_kernel_m<256, 10>},
               {11, launch_thomas_kernel_m<256, 11>},
               {12, launch_thomas_kernel_m<256, 12>},
               {13, launch_thomas_kernel_m<256, 13>},
               {14, launch_thomas_kernel_m<256, 14>},
               {15, launch_thomas_kernel_m<256, 15>},
               {16, launch_thomas_kernel_m<256, 16>}};

        if(m <= 16)
        {
            auto it = s_thomas_kernel_dispatch.find(m);

            if(it != s_thomas_kernel_dispatch.end())
            {
                return it->second(handle, batch_count, batch_stride, batch_stride, dl, d, du, x);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        using pcr_kernel_func_ptr = rocsparse_status (*)(rocsparse_handle handle,
                                                         rocsparse_int    m,
                                                         rocsparse_int    n,
                                                         int64_t          batch_stride,
                                                         int64_t          ldb,
                                                         const T*         dl,
                                                         const T*         d,
                                                         const T*         du,
                                                         T*               B);

        // Kernel dispatch table for PCR solver
        static const std::map<int, pcr_kernel_func_ptr> s_pcr_kernel_dispatch
            = {{32, launch_pcr_wavefront_kernel<32>},
               {64, launch_pcr_shared_kernel<64>},
               {128, launch_pcr_shared_kernel<128>},
               {256, launch_pcr_shared_kernel<256>},
               {512, launch_crpcr_pow2_shared_kernel<256, 128>},
               {1024, launch_crpcr_pow2_shared_kernel<512, 256>}};

        if(m <= 1024)
        {
            auto it = s_pcr_kernel_dispatch.lower_bound(m);

            if(it != s_pcr_kernel_dispatch.end())
            {
                return it->second(handle, m, batch_count, batch_stride, batch_stride, dl, d, du, x);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_backward_substitution_kernel(rocsparse_handle handle,
                                                         rocsparse_int    m,
                                                         rocsparse_int    n,
                                                         int64_t          ldb,
                                                         int              num_spikes,
                                                         const T*         dl_modified,
                                                         const T*         d_modified,
                                                         const T*         du_modified,
                                                         const T*         B_modified,
                                                         const T*         B_spike,
                                                         T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_strided_batch_pcr_tiled_backward_kernel<BLOCKSIZE>),
            dim3((m - 1) / BLOCKSIZE + 1, n, 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
            ldb,
            num_spikes,
            dl_modified,
            d_modified,
            du_modified,
            B_modified,
            B_spike,
            B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_forward_elimination_kernel(rocsparse_handle handle,
                                                       rocsparse_int    m,
                                                       rocsparse_int    n,
                                                       int64_t          stride,
                                                       int64_t          ldb,
                                                       rocsparse_int    num_spikes,
                                                       const T*         dl,
                                                       const T*         d,
                                                       const T*         du,
                                                       const T*         B,
                                                       T*               dl_modified,
                                                       T*               d_modified,
                                                       T*               du_modified,
                                                       T*               B_modified,
                                                       T*               dl_spike,
                                                       T*               d_spike,
                                                       T*               du_spike,
                                                       T*               B_spike)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_strided_batch_pcr_tiled_forward_kernel<BLOCKSIZE>),
            dim3((m - 1) / BLOCKSIZE + 1, n, 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
            stride,
            ldb,
            num_spikes,
            dl,
            d,
            du,
            B,
            dl_modified,
            d_modified,
            du_modified,
            B_modified,
            dl_spike,
            d_spike,
            du_spike,
            B_spike);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_spike_solver_kernel(rocsparse_handle handle,
                                                rocsparse_int    num_spikes,
                                                rocsparse_int    n,
                                                const T*         dl_spike,
                                                const T*         d_spike,
                                                const T*         du_spike,
                                                T*               B_spike)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_strided_batch_spike_solver_pcr_kernel<BLOCKSIZE>),
            dim3(n),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            num_spikes,
            n,
            dl_spike,
            d_spike,
            du_spike,
            B_spike);
        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status gtsv_no_pivot_strided_batch_medium_template(rocsparse_handle handle,
                                                                 rocsparse_int    m,
                                                                 const T*         dl,
                                                                 const T*         d,
                                                                 const T*         du,
                                                                 T*               x,
                                                                 rocsparse_int    batch_count,
                                                                 rocsparse_int    batch_stride,
                                                                 void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m > 1024 && m <= 131072,
                              "This function is designed for m > 1024 and m <= 131072.");

        char* ptr         = reinterpret_cast<char*>(temp_buffer);
        T*    dl_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256;
        T* d_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256;
        T* du_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256;
        T* B_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(m) * batch_count - 1) / 256 + 1) * 256;

        constexpr int BLOCKSIZE  = determine_spike_solver_blocksize();
        const int     nblocks    = ((m - 1) / BLOCKSIZE + 1);
        const int     num_spikes = 2 * nblocks;

        T* dl_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;
        T* d_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;
        T* du_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;
        T* B_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;

        RETURN_IF_ROCSPARSE_ERROR((launch_forward_elimination_kernel<BLOCKSIZE>(handle,
                                                                                m,
                                                                                batch_count,
                                                                                batch_stride,
                                                                                batch_stride,
                                                                                num_spikes,
                                                                                dl,
                                                                                d,
                                                                                du,
                                                                                x,
                                                                                dl_modified,
                                                                                d_modified,
                                                                                du_modified,
                                                                                B_modified,
                                                                                dl_spike,
                                                                                d_spike,
                                                                                du_spike,
                                                                                B_spike)));

        // Define function pointer type for kernel dispatch
        using KernelFuncPtr = rocsparse_status (*)(rocsparse_handle handle,
                                                   rocsparse_int    num_spikes,
                                                   rocsparse_int    n,
                                                   const T*         dl_spike,
                                                   const T*         d_spike,
                                                   const T*         du_spike,
                                                   T*               B_spike);

        // Kernel dispatch table for spike solver
        static const std::map<int, KernelFuncPtr> s_kernel_dispatch
            = {{4, launch_spike_solver_kernel<4, T>},
               {8, launch_spike_solver_kernel<8, T>},
               {16, launch_spike_solver_kernel<16, T>},
               {32, launch_spike_solver_kernel<32, T>},
               {64, launch_spike_solver_kernel<64, T>},
               {128, launch_spike_solver_kernel<128, T>},
               {256, launch_spike_solver_kernel<256, T>},
               {512, launch_spike_solver_kernel<512, T>},
               {1024, launch_spike_solver_kernel<1024, T>}

            };

        if(num_spikes <= 1024)
        {
            auto it = s_kernel_dispatch.lower_bound(num_spikes);

            if(it != s_kernel_dispatch.end())
            {
                RETURN_IF_ROCSPARSE_ERROR(it->second(
                    handle, num_spikes, batch_count, dl_spike, d_spike, du_spike, B_spike));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        RETURN_IF_ROCSPARSE_ERROR((launch_backward_substitution_kernel<BLOCKSIZE>(handle,
                                                                                  m,
                                                                                  batch_count,
                                                                                  batch_stride,
                                                                                  num_spikes,
                                                                                  dl_modified,
                                                                                  d_modified,
                                                                                  du_modified,
                                                                                  B_modified,
                                                                                  B_spike,
                                                                                  x)));

        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status gtsv_no_pivot_strided_batch_large_template(rocsparse_handle handle,
                                                                rocsparse_int    m,
                                                                const T*         dl,
                                                                const T*         d,
                                                                const T*         du,
                                                                T*               x,
                                                                rocsparse_int    batch_count,
                                                                rocsparse_int    batch_stride,
                                                                void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m > 512, "This function is designed for m > 512.");

        char* ptr = reinterpret_cast<char*>(temp_buffer);
        T*    da0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* da1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* db0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* db1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* dc0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* dc1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* drhs0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* drhs1 = reinterpret_cast<T*>(ptr);
        // ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            da0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            da1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            db0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            db1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            dc0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            dc1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            drhs0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            drhs1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));

        // Run special algorithm if m is power of 2
        if((m & (m - 1)) == 0)
        {
            // Stage1: Break large tridiagonal system into multiple smaller systems
            // using parallel cyclic reduction so that each sub system is of size 512.
            rocsparse_int iter = static_cast<rocsparse_int>(rocsparse::log2(m))
                                 - static_cast<rocsparse_int>(rocsparse::log2(512));

            rocsparse_int stride = 1;
            for(rocsparse_int i = 0; i < iter; i++)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_STAGE1(T, 256, stride, i);

                stride *= 2;
            }

            // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
            rocsparse_int subsystem_count = 1 << iter;

            LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_STAGE2(T, 256, iter);
        }
        else
        {
            // Stage1: Break large tridiagonal system into multiple smaller systems
            // using parallel cyclic reduction so that each sub system is of size 512 or less.
            rocsparse_int iter = static_cast<rocsparse_int>(rocsparse::log2(m))
                                 - static_cast<rocsparse_int>(rocsparse::log2(512)) + 1;

            rocsparse_int stride = 1;
            for(rocsparse_int i = 0; i < iter; i++)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE1(T, 256, stride, i);

                stride *= 2;
            }

            // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
            rocsparse_int subsystem_count = 1 << iter;

            LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE2(T, 512, iter);
        }

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::gtsv_no_pivot_strided_batch_template(rocsparse_handle handle,
                                                                 rocsparse_int    m,
                                                                 const T*         dl,
                                                                 const T*         d,
                                                                 const T*         du,
                                                                 T*               x,
                                                                 rocsparse_int    batch_count,
                                                                 rocsparse_int    batch_stride,
                                                                 void*            temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgtsv_no_pivot_strided_batch"),
                         m,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)x,
                         batch_count,
                         batch_stride,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(7, batch_stride, (batch_stride < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(6, batch_count);

    ROCSPARSE_CHECKARG_ARRAY(2, batch_count, dl);
    ROCSPARSE_CHECKARG_ARRAY(3, batch_count, d);
    ROCSPARSE_CHECKARG_ARRAY(4, batch_count, du);
    ROCSPARSE_CHECKARG_ARRAY(5, batch_count, x);
    ROCSPARSE_CHECKARG(
        8, temp_buffer, (m > 1024 && temp_buffer == nullptr), rocsparse_status_invalid_pointer);

    if(batch_count == 0)
    {
        return rocsparse_status_success;
    }

    // If m is small we can solve the systems entirely in shared memory
    if(m <= 1024)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_small_template(
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer));
        return rocsparse_status_success;
    }
    else if(m <= 131072)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_medium_template(
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_large_template(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer));
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                                  \
                                     rocsparse_int    m,                                       \
                                     const TYPE*      dl,                                      \
                                     const TYPE*      d,                                       \
                                     const TYPE*      du,                                      \
                                     const TYPE*      x,                                       \
                                     rocsparse_int    batch_count,                             \
                                     rocsparse_int    batch_stride,                            \
                                     size_t*          buffer_size)                             \
    try                                                                                        \
    {                                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                                               \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_buffer_size_template( \
            handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size));                 \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                                          \
    }

C_IMPL(rocsparse_sgtsv_no_pivot_strided_batch_buffer_size, float);
C_IMPL(rocsparse_dgtsv_no_pivot_strided_batch_buffer_size, double);
C_IMPL(rocsparse_cgtsv_no_pivot_strided_batch_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot_strided_batch_buffer_size, rocsparse_double_complex);

#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                      \
                                     rocsparse_int    m,                           \
                                     const TYPE*      dl,                          \
                                     const TYPE*      d,                           \
                                     const TYPE*      du,                          \
                                     TYPE*            x,                           \
                                     rocsparse_int    batch_count,                 \
                                     rocsparse_int    batch_stride,                \
                                     void*            temp_buffer)                 \
    try                                                                            \
    {                                                                              \
        ROCSPARSE_ROUTINE_TRACE;                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_template( \
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer));     \
        return rocsparse_status_success;                                           \
    }                                                                              \
    catch(...)                                                                     \
    {                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                              \
    }

C_IMPL(rocsparse_sgtsv_no_pivot_strided_batch, float);
C_IMPL(rocsparse_dgtsv_no_pivot_strided_batch, double);
C_IMPL(rocsparse_cgtsv_no_pivot_strided_batch, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot_strided_batch, rocsparse_double_complex);

#undef C_IMPL
