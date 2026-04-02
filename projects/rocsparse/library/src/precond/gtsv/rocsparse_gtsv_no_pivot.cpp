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

#include "rocsparse_gtsv_no_pivot.hpp"
#include "internal/precond/rocsparse_gtsv.h"

#include "gtsv_nopivot_device.h"
#include "gtsv_nopivot_large_device.h"
#include "gtsv_nopivot_medium_device.h"

#include <map>

// LCOV_EXCL_START
static constexpr int determine_spike_solver_blocksize()
{
    return 256;
}
// LCOV_EXCL_STOP

template <typename T>
rocsparse_status rocsparse::gtsv_no_pivot_buffer_size_template(rocsparse_handle handle,
                                                               rocsparse_int    m,
                                                               rocsparse_int    n,
                                                               const T*         dl,
                                                               const T*         d,
                                                               const T*         du,
                                                               const T*         B,
                                                               rocsparse_int    ldb,
                                                               size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgtsv_no_pivot_buffer_size"),
                         m,
                         n,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)B,
                         ldb,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG(7,
                       ldb,
                       (ldb < rocsparse::max(static_cast<rocsparse_int>(1), m)),
                       rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(3, n, dl);
    ROCSPARSE_CHECKARG_ARRAY(4, n, d);
    ROCSPARSE_CHECKARG_ARRAY(5, n, du);
    ROCSPARSE_CHECKARG_ARRAY(6, n, B);
    ROCSPARSE_CHECKARG_POINTER(8, buffer_size);

    if(n == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    if(m <= 1024)
    {
        *buffer_size = 0;
    }
    else if(m <= 131072) //2^17
    {
        *buffer_size = 0;

        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // dl_modified
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // d_modified
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // du_modified
        *buffer_size += ((sizeof(T) * int64_t(m) * n - 1) / 256 + 1) * 256; // B_modified

        constexpr int BLOCKSIZE  = determine_spike_solver_blocksize();
        const int     nblocks    = ((m - 1) / BLOCKSIZE + 1);
        const int     num_spikes = 2 * nblocks;

        *buffer_size += ((sizeof(T) * num_spikes - 1) / 256 + 1) * 256; // dl_spike
        *buffer_size += ((sizeof(T) * num_spikes - 1) / 256 + 1) * 256; // d_spike
        *buffer_size += ((sizeof(T) * num_spikes - 1) / 256 + 1) * 256; // du_spike
        *buffer_size += ((sizeof(T) * int64_t(num_spikes) * n - 1) / 256 + 1) * 256; // B_spike
    }
    else
    {
        *buffer_size = 0;

        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // da0
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // da1
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // db0
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // db1
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // dc0
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256; // dc1
        *buffer_size += ((sizeof(T) * int64_t(m) * n - 1) / 256 + 1) * 256; // drhs0
        *buffer_size += ((sizeof(T) * int64_t(m) * n - 1) / 256 + 1) * 256; // drhs1
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    rocsparse_status launch_cramer_rule_kernel(rocsparse_handle handle,
                                               rocsparse_int    n,
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
                                           ldb,
                                           dl,
                                           d,
                                           du,
                                           B);
        return rocsparse_status_success;
    }

    // LCOV_EXCL_START
    template <typename T>
    static constexpr uint32_t determine_num_rhs()
    {
        if constexpr(std::is_same<T, float>())
        {
            return 4;
        }
        else if constexpr(std::is_same<T, double>() || std::is_same<T, rocsparse_float_complex>())
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }
    // LCOV_EXCL_STOP

    // Wavefront PCR kernels: m <= 8, 16, 32 (WF_SIZE varies)
    template <uint32_t WF_SIZE, typename T>
    rocsparse_status launch_pcr_wavefront_kernel(rocsparse_handle handle,
                                                 rocsparse_int    m,
                                                 rocsparse_int    n,
                                                 int64_t          ldb,
                                                 const T*         dl,
                                                 const T*         d,
                                                 const T*         du,
                                                 T*               B)
    {
        constexpr uint32_t NUM_RHS   = 8;
        constexpr uint32_t BLOCKSIZE = 256;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_pcr_wavefront_kernel<BLOCKSIZE, WF_SIZE, NUM_RHS>),
            dim3((n - 1) / (BLOCKSIZE / (WF_SIZE / NUM_RHS)) + 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
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
                                              int64_t          ldb,
                                              const T*         dl,
                                              const T*         d,
                                              const T*         du,
                                              T*               B)
    {
        constexpr uint32_t NUM_RHS = 8;
        constexpr uint32_t WF_SIZE = 32;
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_no_pivot_pcr_shared_kernel<BLOCKSIZE, WF_SIZE, NUM_RHS>),
            dim3((n - 1) / NUM_RHS + 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
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
                                                     int64_t          ldb,
                                                     const T*         dl,
                                                     const T*         d,
                                                     const T*         du,
                                                     T*               B)
    {
        constexpr uint32_t NUM_RHS = determine_num_rhs<T>();
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_nopivot_crpcr_shared_kernel<BLOCKSIZE, HALF_BLOCKSIZE, NUM_RHS>),
            dim3((n - 1) / NUM_RHS + 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
            ldb,
            dl,
            d,
            du,
            B);
        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status gtsv_no_pivot_small_template(rocsparse_handle handle,
                                                  rocsparse_int    m,
                                                  rocsparse_int    n,
                                                  const T*         dl,
                                                  const T*         d,
                                                  const T*         du,
                                                  T*               B,
                                                  rocsparse_int    ldb,
                                                  void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m <= 1024, "This function is designed for m <= 1024.");

        using thomas_kernel_func_ptr = rocsparse_status (*)(rocsparse_handle handle,
                                                            rocsparse_int    n,
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
                return it->second(handle, n, ldb, dl, d, du, B);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        using pcr_kernel_func_ptr = rocsparse_status (*)(rocsparse_handle handle,
                                                         rocsparse_int    m,
                                                         rocsparse_int    n,
                                                         int64_t          ldb,
                                                         const T*         dl,
                                                         const T*         d,
                                                         const T*         du,
                                                         T*               B);

        // Kernel dispatch table for PCR solver
        static const std::map<int, pcr_kernel_func_ptr> s_pcr_kernel_dispatch
            = {{8, launch_pcr_wavefront_kernel<8>},
               {16, launch_pcr_wavefront_kernel<16>},
               {32, launch_pcr_wavefront_kernel<32>},
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
                return it->second(handle, m, n, ldb, dl, d, du, B);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
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
            (rocsparse::gtsv_no_pivot_pcr_tiled_backward_kernel<BLOCKSIZE, NUM_RHS>),
            dim3((m - 1) / BLOCKSIZE + 1, (n - 1) / NUM_RHS + 1, 1),
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

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    rocsparse_status launch_forward_elimination_kernel(rocsparse_handle handle,
                                                       rocsparse_int    m,
                                                       rocsparse_int    n,
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
            (rocsparse::gtsv_no_pivot_pcr_tiled_forward_kernel<BLOCKSIZE, NUM_RHS>),
            dim3((m - 1) / BLOCKSIZE + 1, (n - 1) / NUM_RHS + 1, 1),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            m,
            n,
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

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    rocsparse_status launch_spike_solver_kernel(rocsparse_handle handle,
                                                rocsparse_int    num_spikes,
                                                rocsparse_int    n,
                                                const T*         dl_spike,
                                                const T*         d_spike,
                                                const T*         du_spike,
                                                T*               B_spike)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gtsv_no_pivot_spike_solver_pcr_kernel<BLOCKSIZE, NUM_RHS>),
            dim3((n - 1) / NUM_RHS + 1),
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
    rocsparse_status gtsv_no_pivot_medium_template(rocsparse_handle handle,
                                                   rocsparse_int    m,
                                                   rocsparse_int    n,
                                                   const T*         dl,
                                                   const T*         d,
                                                   const T*         du,
                                                   T*               B,
                                                   rocsparse_int    ldb,
                                                   void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m > 1024 && m <= 131072,
                              "This function is designed for m > 1024 and m <= 131072.");

        char* ptr         = reinterpret_cast<char*>(temp_buffer);
        T*    dl_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* d_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* du_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* B_modified = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(m) * n - 1) / 256 + 1) * 256;

        constexpr int BLOCKSIZE  = determine_spike_solver_blocksize();
        const int     nblocks    = ((m - 1) / BLOCKSIZE + 1);
        const int     num_spikes = 2 * nblocks;

        T* dl_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * num_spikes - 1) / 256 + 1) * 256;
        T* d_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * num_spikes - 1) / 256 + 1) * 256;
        T* du_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * num_spikes - 1) / 256 + 1) * 256;
        T* B_spike = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(num_spikes) * n - 1) / 256 + 1) * 256;

        constexpr int NUM_RHS = 8;

        RETURN_IF_ROCSPARSE_ERROR(
            (launch_forward_elimination_kernel<BLOCKSIZE, NUM_RHS>(handle,
                                                                   m,
                                                                   n,
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
            = {{4, launch_spike_solver_kernel<4, NUM_RHS, T>},
               {8, launch_spike_solver_kernel<8, NUM_RHS, T>},
               {16, launch_spike_solver_kernel<16, NUM_RHS, T>},
               {32, launch_spike_solver_kernel<32, NUM_RHS, T>},
               {64, launch_spike_solver_kernel<64, NUM_RHS, T>},
               {128, launch_spike_solver_kernel<128, NUM_RHS, T>},
               {256, launch_spike_solver_kernel<256, NUM_RHS, T>},
               {512, launch_spike_solver_kernel<512, 4, T>},
               {1024, launch_spike_solver_kernel<1024, 1, T>}

            };

        if(num_spikes <= 1024)
        {
            auto it = s_kernel_dispatch.lower_bound(num_spikes);

            if(it != s_kernel_dispatch.end())
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    it->second(handle, num_spikes, n, dl_spike, d_spike, du_spike, B_spike));
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

        RETURN_IF_ROCSPARSE_ERROR(
            (launch_backward_substitution_kernel<BLOCKSIZE, NUM_RHS>(handle,
                                                                     m,
                                                                     n,
                                                                     ldb,
                                                                     num_spikes,
                                                                     dl_modified,
                                                                     d_modified,
                                                                     du_modified,
                                                                     B_modified,
                                                                     B_spike,
                                                                     B)));

        return rocsparse_status_success;
    }

#define LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1(T, block_size, stride, iter) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                  \
        (rocsparse::gtsv_nopivot_pcr_pow2_stage1_kernel<block_size>),    \
        dim3(((m - 1) / block_size + 1), 1, 1),                          \
        dim3(block_size, 1, 1),                                          \
        0,                                                               \
        handle->stream,                                                  \
        stride,                                                          \
        m,                                                               \
        n,                                                               \
        ((iter == 0) ? ldb : m),                                         \
        ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),            \
        ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),             \
        ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),            \
        ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)),         \
        (((iter & 1) == 0) ? da1 : da0),                                 \
        (((iter & 1) == 0) ? db1 : db0),                                 \
        (((iter & 1) == 0) ? dc1 : dc0),                                 \
        (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_PCR_STAGE1(T, block_size, stride, iter)                             \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_pcr_stage1_kernel<block_size>), \
                                       dim3(((m - 1) / block_size + 1), 1, 1),                  \
                                       dim3(block_size),                                        \
                                       0,                                                       \
                                       handle->stream,                                          \
                                       stride,                                                  \
                                       m,                                                       \
                                       n,                                                       \
                                       ((iter == 0) ? ldb : m),                                 \
                                       ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),    \
                                       ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),     \
                                       ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),    \
                                       ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)), \
                                       (((iter & 1) == 0) ? da1 : da0),                         \
                                       (((iter & 1) == 0) ? db1 : db0),                         \
                                       (((iter & 1) == 0) ? dc1 : dc0),                         \
                                       (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_THOMAS_POW2_STAGE2(T, block_size, system_size, iter)      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                               \
        (rocsparse::gtsv_nopivot_thomas_pow2_stage2_kernel<block_size, system_size>), \
        dim3(((subsystem_count - 1) / block_size + 1), n, 1),                         \
        dim3(block_size),                                                             \
        0,                                                                            \
        handle->stream,                                                               \
        stride,                                                                       \
        m,                                                                            \
        n,                                                                            \
        ldb,                                                                          \
        (((iter & 1) != 0) ? da1 : da0),                                              \
        (((iter & 1) != 0) ? db1 : db0),                                              \
        (((iter & 1) != 0) ? dc1 : dc0),                                              \
        (((iter & 1) != 0) ? drhs1 : drhs0),                                          \
        (((iter & 1) != 0) ? da0 : da1),                                              \
        (((iter & 1) != 0) ? db0 : db1),                                              \
        (((iter & 1) != 0) ? dc0 : dc1),                                              \
        (((iter & 1) != 0) ? drhs0 : drhs1),                                          \
        B);

#define LAUNCH_GTSV_NOPIVOT_THOMAS_STAGE2(T, block_size, iter)                                     \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_thomas_stage2_kernel<block_size>), \
                                       dim3(((subsystem_count - 1) / block_size + 1), n, 1),       \
                                       dim3(block_size),                                           \
                                       0,                                                          \
                                       handle->stream,                                             \
                                       stride,                                                     \
                                       m,                                                          \
                                       n,                                                          \
                                       ldb,                                                        \
                                       (((iter & 1) != 0) ? da1 : da0),                            \
                                       (((iter & 1) != 0) ? db1 : db0),                            \
                                       (((iter & 1) != 0) ? dc1 : dc0),                            \
                                       (((iter & 1) != 0) ? drhs1 : drhs0),                        \
                                       (((iter & 1) != 0) ? da0 : da1),                            \
                                       (((iter & 1) != 0) ? db0 : db1),                            \
                                       (((iter & 1) != 0) ? dc0 : dc1),                            \
                                       (((iter & 1) != 0) ? drhs0 : drhs1),                        \
                                       B);

    template <typename T>
    rocsparse_status gtsv_no_pivot_large_template(rocsparse_handle handle,
                                                  rocsparse_int    m,
                                                  rocsparse_int    n,
                                                  const T*         dl,
                                                  const T*         d,
                                                  const T*         du,
                                                  T*               B,
                                                  rocsparse_int    ldb,
                                                  void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m > 65536, "This function is designed for m > 65536.");

        char* ptr = reinterpret_cast<char*>(temp_buffer);
        T*    da0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* da1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* db0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* db1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* dc0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* dc1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        T* drhs0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * int64_t(m) * n - 1) / 256 + 1) * 256;
        T* drhs1 = reinterpret_cast<T*>(ptr);
        // ptr += ((sizeof(T) * int64_t(m) * n - 1) / 256 + 1) * 256;

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
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1(T, 256, stride, i);

                stride *= 2;
            }

            rocsparse_int subsystem_count = stride;

            // Stage2: Solve the many systems from stage1 in parallel using p-thread thomas algorithm.
            LAUNCH_GTSV_NOPIVOT_THOMAS_POW2_STAGE2(T, 256, 512, iter);
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
                LAUNCH_GTSV_NOPIVOT_PCR_STAGE1(T, 256, stride, i);

                stride *= 2;
            }

            // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
            rocsparse_int subsystem_count = 1 << iter;

            LAUNCH_GTSV_NOPIVOT_THOMAS_STAGE2(T, 256, iter);
        }

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::gtsv_no_pivot_template(rocsparse_handle handle,
                                                   rocsparse_int    m,
                                                   rocsparse_int    n,
                                                   const T*         dl,
                                                   const T*         d,
                                                   const T*         du,
                                                   T*               B,
                                                   rocsparse_int    ldb,
                                                   void*            temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgtsv_no_pivot"),
                         m,
                         n,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)B,
                         ldb,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG(7,
                       ldb,
                       (ldb < rocsparse::max(static_cast<rocsparse_int>(1), m)),
                       rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(3, n, dl);
    ROCSPARSE_CHECKARG_ARRAY(4, n, d);
    ROCSPARSE_CHECKARG_ARRAY(5, n, du);
    ROCSPARSE_CHECKARG_ARRAY(6, n, B);
    ROCSPARSE_CHECKARG(
        8, temp_buffer, (m > 1024 && temp_buffer == nullptr), rocsparse_status_invalid_pointer);

    // Quick return if possible
    if(n == 0)
    {
        return rocsparse_status_success;
    }

    // If m is small we can solve the systems entirely in shared memory
    if(m <= 1024)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gtsv_no_pivot_small_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
        return rocsparse_status_success;
    }
    else if(m <= 131072) //2^17
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gtsv_no_pivot_medium_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gtsv_no_pivot_large_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                    \
                                     rocsparse_int    m,                         \
                                     rocsparse_int    n,                         \
                                     const TYPE*      dl,                        \
                                     const TYPE*      d,                         \
                                     const TYPE*      du,                        \
                                     const TYPE*      B,                         \
                                     rocsparse_int    ldb,                       \
                                     size_t*          buffer_size)               \
    try                                                                          \
    {                                                                            \
        ROCSPARSE_ROUTINE_TRACE;                                                 \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_buffer_size_template( \
            handle, m, n, dl, d, du, B, ldb, buffer_size));                      \
        return rocsparse_status_success;                                         \
    }                                                                            \
    catch(...)                                                                   \
    {                                                                            \
        RETURN_ROCSPARSE_EXCEPTION();                                            \
    }

C_IMPL(rocsparse_sgtsv_no_pivot_buffer_size, float);
C_IMPL(rocsparse_dgtsv_no_pivot_buffer_size, double);
C_IMPL(rocsparse_cgtsv_no_pivot_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot_buffer_size, rocsparse_double_complex);

#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                                 \
                                     rocsparse_int    m,                                      \
                                     rocsparse_int    n,                                      \
                                     const TYPE*      dl,                                     \
                                     const TYPE*      d,                                      \
                                     const TYPE*      du,                                     \
                                     TYPE*            B,                                      \
                                     rocsparse_int    ldb,                                    \
                                     void*            temp_buffer)                            \
    try                                                                                       \
    {                                                                                         \
        ROCSPARSE_ROUTINE_TRACE;                                                              \
        RETURN_IF_ROCSPARSE_ERROR(                                                            \
            rocsparse::gtsv_no_pivot_template(handle, m, n, dl, d, du, B, ldb, temp_buffer)); \
        return rocsparse_status_success;                                                      \
    }                                                                                         \
    catch(...)                                                                                \
    {                                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                                         \
    }

C_IMPL(rocsparse_sgtsv_no_pivot, float);
C_IMPL(rocsparse_dgtsv_no_pivot, double);
C_IMPL(rocsparse_cgtsv_no_pivot, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot, rocsparse_double_complex);

#undef C_IMPL
