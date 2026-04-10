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

#include "rocsparse_gtsv_no_pivot_strided_batch.hpp"
#include "internal/precond/rocsparse_gtsv.h"

#include "gtsv_nopivot_strided_batch_device.h"
#include "gtsv_nopivot_strided_batch_medium_device.h"
#include "gtsv_nopivot_thomas_device.h"

#include <map>

namespace rocsparse
{
    // LCOV_EXCL_START
    static constexpr int determine_spike_solver_blocksize()
    {
        return 256;
    }
    static constexpr int determine_max_recursion_levels()
    {
        return 4;
    }
    // LCOV_EXCL_STOP
}

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

    if(m <= 1024)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    *buffer_size = 0;

    constexpr int BLOCKSIZE            = determine_spike_solver_blocksize();
    constexpr int MAX_RECURSION_LEVELS = determine_max_recursion_levels();

    int64_t current_m = m;
    for(int level = 0; level < MAX_RECURSION_LEVELS; level++)
    {
        if(current_m <= 1024)
            break;

        *buffer_size
            += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256; // dl_modified
        *buffer_size
            += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256; // d_modified
        *buffer_size
            += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256; // du_modified
        *buffer_size
            += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256; // B_modified

        const int nblocks    = ((current_m - 1) / BLOCKSIZE + 1);
        const int num_spikes = 2 * nblocks;

        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // dl_spike
        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // d_spike
        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // du_spike
        *buffer_size
            += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256; // B_spike

        current_m = num_spikes;
    }

    if(current_m > 1024)
    {
        *buffer_size = 0;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    struct gtsv_no_pivot_buffer_data
    {
        static constexpr int MAX_RECURSION_LEVELS = determine_max_recursion_levels();

        T* dl_modified[MAX_RECURSION_LEVELS]{};
        T* d_modified[MAX_RECURSION_LEVELS]{};
        T* du_modified[MAX_RECURSION_LEVELS]{};
        T* B_modified[MAX_RECURSION_LEVELS]{};

        T* dl_spike[MAX_RECURSION_LEVELS]{};
        T* d_spike[MAX_RECURSION_LEVELS]{};
        T* du_spike[MAX_RECURSION_LEVELS]{};
        T* B_spike[MAX_RECURSION_LEVELS]{};
    };

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_cramer_rule_kernel(rocsparse_handle handle,
                                               rocsparse_int    batch_count,
                                               int64_t          batch_stride,
                                               int64_t          ldb,
                                               const T*         dl,
                                               const T*         d,
                                               const T*         du,
                                               T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_2x2_kernel<BLOCKSIZE>),
                                           dim3((batch_count - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           batch_count,
                                           batch_stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_3(rocsparse_handle handle,
                                            rocsparse_int    batch_count,
                                            int64_t          batch_stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_3x3_kernel<BLOCKSIZE>),
                                           dim3((batch_count - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           batch_count,
                                           batch_stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_4(rocsparse_handle handle,
                                            rocsparse_int    batch_count,
                                            int64_t          batch_stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_4x4_kernel<BLOCKSIZE>),
                                           dim3((batch_count - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           batch_count,
                                           batch_stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_5(rocsparse_handle handle,
                                            rocsparse_int    batch_count,
                                            int64_t          batch_stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_5x5_kernel<BLOCKSIZE>),
                                           dim3((batch_count - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           batch_count,
                                           batch_stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_6(rocsparse_handle handle,
                                            rocsparse_int    batch_count,
                                            int64_t          batch_stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_6x6_kernel<BLOCKSIZE>),
                                           dim3((batch_count - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           batch_count,
                                           batch_stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_thomas_kernel_7(rocsparse_handle handle,
                                            rocsparse_int    batch_count,
                                            int64_t          batch_stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_7x7_kernel<BLOCKSIZE>),
                                           dim3((batch_count - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           batch_count,
                                           batch_stride,
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t M, typename T>
    rocsparse_status launch_thomas_kernel_m(rocsparse_handle handle,
                                            rocsparse_int    batch_count,
                                            int64_t          batch_stride,
                                            int64_t          ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_thomas_kernel<BLOCKSIZE, M>),
                                           dim3((batch_count - 1) / BLOCKSIZE + 1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           batch_count,
                                           batch_stride,
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
                                                 rocsparse_int    batch_count,
                                                 int64_t          batch_stride,
                                                 int64_t          ldb,
                                                 const T*         dl,
                                                 const T*         d,
                                                 const T*         du,
                                                 T*               B)
    {
        constexpr uint32_t BLOCKSIZE = 256;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_strided_batch_pcr_wavefront_kernel<BLOCKSIZE, WF_SIZE>),
            dim3((batch_count - 1) / (BLOCKSIZE / WF_SIZE) + 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            batch_count,
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
                                              rocsparse_int    batch_count,
                                              int64_t          batch_stride,
                                              int64_t          ldb,
                                              const T*         dl,
                                              const T*         d,
                                              const T*         du,
                                              T*               B)
    {
        constexpr uint32_t WF_SIZE = 32;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_strided_batch_pcr_shared_kernel<BLOCKSIZE, WF_SIZE>),
            dim3(batch_count),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            batch_count,
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
                                                     rocsparse_int    batch_count,
                                                     int64_t          batch_stride,
                                                     int64_t          ldb,
                                                     const T*         dl,
                                                     const T*         d,
                                                     const T*         du,
                                                     T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_strided_batch_crpcr_shared_kernel<BLOCKSIZE, HALF_BLOCKSIZE>),
            dim3(batch_count),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            batch_count,
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
                                                                rocsparse_int    batch_stride)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m <= 1024, "This function is designed for m <= 1024.");

        using thomas_kernel_func_ptr = rocsparse_status (*)(rocsparse_handle handle,
                                                            rocsparse_int    batch_count,
                                                            int64_t          batch_stride,
                                                            int64_t          ldb,
                                                            const T*         dl,
                                                            const T*         d,
                                                            const T*         du,
                                                            T*               x);

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
                                                         rocsparse_int    batch_count,
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
                                                         rocsparse_int    batch_count,
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
            dim3((m - 1) / BLOCKSIZE + 1, batch_count, 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            batch_count,
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
                                                       rocsparse_int    batch_count,
                                                       int64_t          batch_stride,
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
            dim3((m - 1) / BLOCKSIZE + 1, batch_count, 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            batch_count,
            batch_stride,
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
                                                rocsparse_int    batch_count,
                                                const T*         dl_spike,
                                                const T*         d_spike,
                                                const T*         du_spike,
                                                T*               B_spike)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_strided_batch_spike_solver_pcr_kernel<BLOCKSIZE>),
            dim3(batch_count),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            num_spikes,
            batch_count,
            dl_spike,
            d_spike,
            du_spike,
            B_spike);
        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status
        gtsv_no_pivot_strided_batch_medium_template(rocsparse_handle              handle,
                                                    rocsparse_int                 m,
                                                    const T*                      dl,
                                                    const T*                      d,
                                                    const T*                      du,
                                                    T*                            x,
                                                    rocsparse_int                 batch_count,
                                                    rocsparse_int                 batch_stride,
                                                    gtsv_no_pivot_buffer_data<T>* buffer_data,
                                                    int                           level)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m > 1024, "This function is designed for m > 1024.");

        constexpr int BLOCKSIZE  = determine_spike_solver_blocksize();
        const int     nblocks    = ((m - 1) / BLOCKSIZE + 1);
        const int     num_spikes = 2 * nblocks;

        RETURN_IF_ROCSPARSE_ERROR(
            (launch_forward_elimination_kernel<BLOCKSIZE>(handle,
                                                          m,
                                                          batch_count,
                                                          batch_stride,
                                                          batch_stride,
                                                          num_spikes,
                                                          dl,
                                                          d,
                                                          du,
                                                          x,
                                                          buffer_data->dl_modified[level],
                                                          buffer_data->d_modified[level],
                                                          buffer_data->du_modified[level],
                                                          buffer_data->B_modified[level],
                                                          buffer_data->dl_spike[level],
                                                          buffer_data->d_spike[level],
                                                          buffer_data->du_spike[level],
                                                          buffer_data->B_spike[level])));

        // Define function pointer type for kernel dispatch
        using KernelFuncPtr = rocsparse_status (*)(rocsparse_handle handle,
                                                   rocsparse_int    num_spikes,
                                                   rocsparse_int    batch_count,
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
                RETURN_IF_ROCSPARSE_ERROR(it->second(handle,
                                                     num_spikes,
                                                     batch_count,
                                                     buffer_data->dl_spike[level],
                                                     buffer_data->d_spike[level],
                                                     buffer_data->du_spike[level],
                                                     buffer_data->B_spike[level]));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else
        {
            if(level + 1 >= determine_max_recursion_levels())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }

            RETURN_IF_ROCSPARSE_ERROR(
                gtsv_no_pivot_strided_batch_template_dispatch(handle,
                                                              num_spikes,
                                                              buffer_data->dl_spike[level],
                                                              buffer_data->d_spike[level],
                                                              buffer_data->du_spike[level],
                                                              buffer_data->B_spike[level],
                                                              batch_count,
                                                              num_spikes,
                                                              buffer_data,
                                                              level + 1));
        }

        RETURN_IF_ROCSPARSE_ERROR(
            (launch_backward_substitution_kernel<BLOCKSIZE>(handle,
                                                            m,
                                                            batch_count,
                                                            batch_stride,
                                                            num_spikes,
                                                            buffer_data->dl_modified[level],
                                                            buffer_data->d_modified[level],
                                                            buffer_data->du_modified[level],
                                                            buffer_data->B_modified[level],
                                                            buffer_data->B_spike[level],
                                                            x)));

        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status
        gtsv_no_pivot_strided_batch_template_dispatch(rocsparse_handle              handle,
                                                      rocsparse_int                 m,
                                                      const T*                      dl,
                                                      const T*                      d,
                                                      const T*                      du,
                                                      T*                            x,
                                                      rocsparse_int                 batch_count,
                                                      rocsparse_int                 batch_stride,
                                                      gtsv_no_pivot_buffer_data<T>* buffer_data,
                                                      int                           level)
    {
        // If m is small we can solve the systems entirely in shared memory
        if(m <= 1024)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_small_template(
                handle, m, dl, d, du, x, batch_count, batch_stride));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_medium_template(
                handle, m, dl, d, du, x, batch_count, batch_stride, buffer_data, level));
            return rocsparse_status_success;
        }
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

    if(m <= 1024)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_small_template(
            handle, m, dl, d, du, x, batch_count, batch_stride));
        return rocsparse_status_success;
    }

    gtsv_no_pivot_buffer_data<T> buffer_data;

    char* ptr = reinterpret_cast<char*>(temp_buffer);

    int64_t current_m = m;
    size_t  offset    = 0;
    for(int level = 0; level < determine_max_recursion_levels(); level++)
    {
        if(current_m <= 1024)
            break;

        buffer_data.dl_modified[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256;
        buffer_data.d_modified[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256;
        buffer_data.du_modified[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256;
        buffer_data.B_modified[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(current_m) * batch_count - 1) / 256 + 1) * 256;

        constexpr int BLOCKSIZE  = determine_spike_solver_blocksize();
        const int     nblocks    = ((current_m - 1) / BLOCKSIZE + 1);
        const int     num_spikes = 2 * nblocks;

        buffer_data.dl_spike[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;
        buffer_data.d_spike[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;
        buffer_data.du_spike[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;
        buffer_data.B_spike[level] = reinterpret_cast<T*>(ptr + offset);
        offset += ((sizeof(T) * int64_t(num_spikes) * batch_count - 1) / 256 + 1) * 256;

        current_m = num_spikes;
    }

    if(current_m > 1024)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    RETURN_IF_ROCSPARSE_ERROR(gtsv_no_pivot_strided_batch_template_dispatch(
        handle, m, dl, d, du, x, batch_count, batch_stride, &buffer_data, 0));
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
