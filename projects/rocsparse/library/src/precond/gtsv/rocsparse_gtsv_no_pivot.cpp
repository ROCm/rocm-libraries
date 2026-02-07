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

#include "rocsparse_gtsv_no_pivot.hpp"
#include "internal/precond/rocsparse_gtsv.h"

#include "gtsv_nopivot_device.h"

#include <map>

#define LAUNCH_GTSV_NOPIVOT_CR_POW2_SHARED(block_size)               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                              \
        (rocsparse::gtsv_nopivot_cr_pow2_shared_kernel<block_size>), \
        dim3(n),                                                     \
        dim3(block_size),                                            \
        0,                                                           \
        handle->stream,                                              \
        m,                                                           \
        n,                                                           \
        ldb,                                                         \
        dl,                                                          \
        d,                                                           \
        du,                                                          \
        B);

#define LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(block_size)               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (rocsparse::gtsv_nopivot_pcr_pow2_shared_kernel<block_size>), \
        dim3(n),                                                      \
        dim3(block_size),                                             \
        0,                                                            \
        handle->stream,                                               \
        m,                                                            \
        n,                                                            \
        ldb,                                                          \
        dl,                                                           \
        d,                                                            \
        du,                                                           \
        B);



















#define LAUNCH_GTSV_NOPIVOT_5x5_THOMAS(BLOCKSIZE)                                       \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_5x5_kernel<BLOCKSIZE>), \
                                       dim3((n - 1) / BLOCKSIZE + 1),                   \
                                       dim3(BLOCKSIZE),                                 \
                                       0,                                               \
                                       handle->stream,                                  \
                                       n,                                               \
                                       ldb,                                             \
                                       dl,                                              \
                                       d,                                               \
                                       du,                                              \
                                       B);

#define LAUNCH_GTSV_NOPIVOT_6x6_THOMAS(BLOCKSIZE)                                       \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_6x6_kernel<BLOCKSIZE>), \
                                       dim3((n - 1) / BLOCKSIZE + 1),                   \
                                       dim3(BLOCKSIZE),                                 \
                                       0,                                               \
                                       handle->stream,                                  \
                                       n,                                               \
                                       ldb,                                             \
                                       dl,                                              \
                                       d,                                               \
                                       du,                                              \
                                       B);

#define LAUNCH_GTSV_NOPIVOT_7x7_THOMAS(BLOCKSIZE)                                       \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_7x7_kernel<BLOCKSIZE>), \
                                       dim3((n - 1) / BLOCKSIZE + 1),                   \
                                       dim3(BLOCKSIZE),                                 \
                                       0,                                               \
                                       handle->stream,                                  \
                                       n,                                               \
                                       ldb,                                             \
                                       dl,                                              \
                                       d,                                               \
                                       du,                                              \
                                       B);

#define LAUNCH_GTSV_NOPIVOT_THOMAS(BLOCKSIZE, M)                                              \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_thomas_kernel<BLOCKSIZE, M>), \
                                       dim3((n - 1) / BLOCKSIZE + 1),                         \
                                       dim3(BLOCKSIZE),                                       \
                                       0,                                                     \
                                       handle->stream,                                        \
                                       n,                                                     \
                                       ldb,                                                   \
                                       dl,                                                    \
                                       d,                                                     \
                                       du,                                                    \
                                       B);

















#define LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(block_size, pcr_size)               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                           \
        (rocsparse::gtsv_nopivot_crpcr_pow2_shared_kernel<block_size, pcr_size>), \
        dim3(n),                                                                  \
        dim3(block_size),                                                         \
        0,                                                                        \
        handle->stream,                                                           \
        m,                                                                        \
        n,                                                                        \
        ldb,                                                                      \
        dl,                                                                       \
        d,                                                                        \
        du,                                                                       \
        B);

#define LAUNCH_GTSV_NOPIVOT_PCR_SHARED(block_size)                                              \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_pcr_shared_kernel<block_size>), \
                                       dim3(n),                                                 \
                                       dim3(block_size),                                        \
                                       0,                                                       \
                                       handle->stream,                                          \
                                       m,                                                       \
                                       n,                                                       \
                                       ldb,                                                     \
                                       dl,                                                      \
                                       d,                                                       \
                                       du,                                                      \
                                       B);

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

    if(m <= 512)
    {
        *buffer_size = 0;
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
        *buffer_size += ((sizeof(T) * m * n - 1) / 256 + 1) * 256; // drhs0
        *buffer_size += ((sizeof(T) * m * n - 1) / 256 + 1) * 256; // drhs1
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status launch_cramer_rule_kernel(rocsparse_handle handle,
                                               rocsparse_int    n,
                                               rocsparse_int    ldb,
                                               const T*         dl,
                                               const T*         d,
                                               const T*         du,
                                               T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_2x2_kernel<256>),
                                           dim3((n - 1) / 256 + 1),
                                           dim3(256),
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

    template <typename T>
    rocsparse_status launch_thomas_kernel_3(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_3x3_kernel<256>),
                                           dim3((n - 1) / 256 + 1),
                                           dim3(256),
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

    template <typename T>
    rocsparse_status launch_thomas_kernel_4(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_4x4_kernel<256>),
                                           dim3((n - 1) / 256 + 1),
                                           dim3(256),
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

    template <uint32_t M, typename T>
    rocsparse_status launch_thomas_kernel_m(rocsparse_handle handle,
                                            rocsparse_int    n,
                                            rocsparse_int    ldb,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_thomas_kernel<256, M>),
                                           dim3((n - 1) / 256 + 1),
                                           dim3(256),
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

        rocsparse_host_assert(m <= 512, "This function is designed for m <= 512.");

        // Define function pointer type for kernel dispatch
        using KernelFuncPtr = rocsparse_status (*)(
            rocsparse_handle, rocsparse_int, rocsparse_int, const T*, const T*, const T*, T*);

        // Kernel dispatch table for small matrix sizes
        static const std::map<int, KernelFuncPtr> s_kernel_dispatch = {
            {2, launch_cramer_rule_kernel<T>},   {3, launch_thomas_kernel_3<T>},
            {4, launch_thomas_kernel_4<T>},      {5, launch_thomas_kernel_m<5, T>},
            {6, launch_thomas_kernel_m<6, T>},   {7, launch_thomas_kernel_m<7, T>},
            {8, launch_thomas_kernel_m<8, T>},   {9, launch_thomas_kernel_m<9, T>},
            {10, launch_thomas_kernel_m<10, T>}, {11, launch_thomas_kernel_m<11, T>},
            {12, launch_thomas_kernel_m<12, T>}, {13, launch_thomas_kernel_m<13, T>},
            {14, launch_thomas_kernel_m<14, T>}, {14, launch_thomas_kernel_m<15, T>},
            {16, launch_thomas_kernel_m<16, T>}, {17, launch_thomas_kernel_m<17, T>},
            {18, launch_thomas_kernel_m<18, T>}, {19, launch_thomas_kernel_m<19, T>},
            {20, launch_thomas_kernel_m<20, T>}, {21, launch_thomas_kernel_m<21, T>},
            {22, launch_thomas_kernel_m<22, T>}, {23, launch_thomas_kernel_m<23, T>},
            {24, launch_thomas_kernel_m<24, T>}, {25, launch_thomas_kernel_m<25, T>},
            {26, launch_thomas_kernel_m<26, T>}, {27, launch_thomas_kernel_m<27, T>},
            {28, launch_thomas_kernel_m<28, T>}, {29, launch_thomas_kernel_m<29, T>},
            {30, launch_thomas_kernel_m<30, T>}, {31, launch_thomas_kernel_m<31, T>},
            {32, launch_thomas_kernel_m<32, T>},

            // {33, launch_thomas_kernel_m<33, T>}, {34, launch_thomas_kernel_m<34, T>},
            // {35, launch_thomas_kernel_m<35, T>}, {36, launch_thomas_kernel_m<36, T>},
            // {37, launch_thomas_kernel_m<37, T>}, {38, launch_thomas_kernel_m<38, T>},
            // {39, launch_thomas_kernel_m<39, T>}, {40, launch_thomas_kernel_m<40, T>},
            // {41, launch_thomas_kernel_m<41, T>}, {42, launch_thomas_kernel_m<42, T>},
            // {43, launch_thomas_kernel_m<43, T>}, {44, launch_thomas_kernel_m<44, T>},
            // {45, launch_thomas_kernel_m<45, T>}, {46, launch_thomas_kernel_m<46, T>},
            // {47, launch_thomas_kernel_m<47, T>}, {48, launch_thomas_kernel_m<48, T>},
            // {49, launch_thomas_kernel_m<49, T>}, {50, launch_thomas_kernel_m<50, T>},
            // {51, launch_thomas_kernel_m<51, T>}, {52, launch_thomas_kernel_m<52, T>},
            // {53, launch_thomas_kernel_m<53, T>}, {54, launch_thomas_kernel_m<54, T>},
            // {55, launch_thomas_kernel_m<55, T>}, {56, launch_thomas_kernel_m<56, T>},
            // {57, launch_thomas_kernel_m<57, T>}, {58, launch_thomas_kernel_m<58, T>},
            // {59, launch_thomas_kernel_m<59, T>}, {60, launch_thomas_kernel_m<60, T>},
            // {61, launch_thomas_kernel_m<61, T>}, {62, launch_thomas_kernel_m<62, T>},
            // {63, launch_thomas_kernel_m<63, T>}, {64, launch_thomas_kernel_m<64, T>},
        };

        // if(m == 32 && n == 32)
        // {
        //     constexpr uint32_t WF_SIZE = 32;
        //     constexpr uint32_t NUM_ELEMENTS = 32;
        //     constexpr uint32_t SUB_GROUP_SIZE = 32;
        //     std::vector<T> htemp(WF_SIZE * NUM_ELEMENTS, 0);
        //     for(int i = 0; i < WF_SIZE * NUM_ELEMENTS; i++)
        //     {
        //         htemp[i] = i;
        //     }

        //     std::cout << "Before transpose htemp" << std::endl;
        //     for(int i = 0; i < NUM_ELEMENTS; i++)
        //     {
        //         for(int j = 0; j < WF_SIZE; j++)
        //         {
        //             std::cout << htemp[WF_SIZE * i + j] << " ";
        //         }
        //         std::cout << "" << std::endl;
        //     }
        //     std::cout << "" << std::endl;

        //     T* dtemp = nullptr;
        //     RETURN_IF_HIP_ERROR(hipMalloc((void**)&dtemp, sizeof(T) * WF_SIZE * NUM_ELEMENTS));
        //     RETURN_IF_HIP_ERROR(
        //         hipMemcpy(dtemp, htemp.data(), sizeof(T) * WF_SIZE * NUM_ELEMENTS, hipMemcpyHostToDevice));

        //     RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        //         (rocsparse::gtsv_nopivot_thomas_transpose_kernel<32, WF_SIZE, NUM_ELEMENTS, SUB_GROUP_SIZE>),
        //         dim3((n - 1) / 32 + 1),
        //         dim3(32),
        //         0,
        //         handle->stream,
        //         m,
        //         n,
        //         ldb,
        //         dl,
        //         d,
        //         du,
        //         B,
        //         dtemp);

        //     RETURN_IF_HIP_ERROR(
        //         hipMemcpy(htemp.data(), dtemp, sizeof(T) * WF_SIZE * NUM_ELEMENTS, hipMemcpyDeviceToHost));

        //     std::cout << "After transpose htemp" << std::endl;
        //     for(int i = 0; i < NUM_ELEMENTS; i++)
        //     {
        //         for(int j = 0; j < WF_SIZE; j++)
        //         {
        //             std::cout << htemp[WF_SIZE * i + j] << " ";
        //         }
        //         std::cout << "" << std::endl;
        //     }
        //     std::cout << "" << std::endl;

        //     RETURN_IF_HIP_ERROR(hipFree(dtemp));

        //     return rocsparse_status_success;
        // }


        // if(m == 32)
        // {
        //     constexpr uint32_t WF_SIZE = 32;
        //     constexpr uint32_t NUM_ELEMENTS = 32;
        //     constexpr uint32_t SUB_GROUP_SIZE = 32;
        //     T* dtemp = nullptr;
        //     RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        //         (rocsparse::gtsv_nopivot_thomas_transpose_kernel<256, WF_SIZE, NUM_ELEMENTS, SUB_GROUP_SIZE>),
        //         dim3((n - 1) / 256 + 1),
        //         dim3(256),
        //         0,
        //         handle->stream,
        //         m,
        //         n,
        //         ldb,
        //         dl,
        //         d,
        //         du,
        //         B,
        //         dtemp);
        //     return rocsparse_status_success;
        // }
        if(m == 8)
        {
            //std::cout << "AAAA" << std::endl;
            constexpr uint32_t WF_SIZE = 32;
            constexpr uint32_t TILE_X = 8;
            T* dtemp = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::thomas_shared_transpose_kernel<256, WF_SIZE, TILE_X>),
                dim3((n - 1) / 256 + 1),
                dim3(256),
                0,
                handle->stream,
                m,
                n,
                ldb,
                dl,
                d,
                du,
                B,
                dtemp);
            return rocsparse_status_success;
        }




        // Thomas algorithm good up to m=64
        if(m <= 32)
        {
            auto it = s_kernel_dispatch.find(m);

            if(it != s_kernel_dispatch.end())
            {
                return it->second(handle, n, ldb, dl, d, du, B);
            }
            else
            {
                // Handle error: m not in dispatch table
            }
        }

        // Run special algorithm if m is power of 2
        if((m & (m - 1)) == 0)
        {
            if(m == 64)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(64);
            }
            else if(m == 128)
            {
                LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(64, 64);
            }
            else if(m == 256)
            {
                LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(128, 64);
            }
            else if(m == 512)
            {
                LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(256, 64);
            }
        }
        else
        {
            if(m <= 64)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(64);
            }
            else if(m <= 128)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(128);
            }
            else if(m <= 256)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(256);
            }
            else if(m <= 512)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(512);
            }
        }

        return rocsparse_status_success;
    }

#define LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1_N(T, block_size, stride, iter) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                    \
        (rocsparse::gtsv_nopivot_pcr_pow2_stage1_n_kernel<block_size>),    \
        dim3(((m - 1) / block_size + 1), n, 1),                            \
        dim3(block_size, 1, 1),                                            \
        0,                                                                 \
        handle->stream,                                                    \
        stride,                                                            \
        m,                                                                 \
        n,                                                                 \
        ((iter == 0) ? ldb : m),                                           \
        ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),              \
        ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),               \
        ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),              \
        ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)),           \
        (((iter & 1) == 0) ? da1 : da0),                                   \
        (((iter & 1) == 0) ? db1 : db0),                                   \
        (((iter & 1) == 0) ? dc1 : dc0),                                   \
        (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_PCR_STAGE1_N(T, block_size, stride, iter)                             \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_pcr_stage1_n_kernel<block_size>), \
                                       dim3(((m - 1) / block_size + 1), n, 1),                    \
                                       dim3(block_size),                                          \
                                       0,                                                         \
                                       handle->stream,                                            \
                                       stride,                                                    \
                                       m,                                                         \
                                       n,                                                         \
                                       ((iter == 0) ? ldb : m),                                   \
                                       ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),      \
                                       ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),       \
                                       ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),      \
                                       ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)),   \
                                       (((iter & 1) == 0) ? da1 : da0),                           \
                                       (((iter & 1) == 0) ? db1 : db0),                           \
                                       (((iter & 1) == 0) ? dc1 : dc0),                           \
                                       (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_CR_POW2_STAGE2(T, block_size, iter)      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                              \
        (rocsparse::gtsv_nopivot_cr_pow2_stage2_kernel<block_size>), \
        dim3(subsystem_count, n, 1),                                 \
        dim3(block_size),                                            \
        0,                                                           \
        handle->stream,                                              \
        m,                                                           \
        n,                                                           \
        ldb,                                                         \
        (((iter & 1) != 0) ? da1 : da0),                             \
        (((iter & 1) != 0) ? db1 : db0),                             \
        (((iter & 1) != 0) ? dc1 : dc0),                             \
        (((iter & 1) != 0) ? drhs1 : drhs0),                         \
        B);

#define LAUNCH_GTSV_NOPIVOT_PCR_STAGE2(T, block_size, iter)                                     \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gtsv_nopivot_pcr_stage2_kernel<block_size>), \
                                       dim3(subsystem_count, n, 1),                             \
                                       dim3(block_size),                                        \
                                       0,                                                       \
                                       handle->stream,                                          \
                                       m,                                                       \
                                       n,                                                       \
                                       ldb,                                                     \
                                       (((iter & 1) != 0) ? da1 : da0),                         \
                                       (((iter & 1) != 0) ? db1 : db0),                         \
                                       (((iter & 1) != 0) ? dc1 : dc0),                         \
                                       (((iter & 1) != 0) ? drhs1 : drhs0),                     \
                                       B);

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

        rocsparse_host_assert(m > 512 && m <= 65536,
                              "This function is designed for m > 512 and m <= 65536.");

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
        ptr += ((sizeof(T) * m * n - 1) / 256 + 1) * 256;
        T* drhs1 = reinterpret_cast<T*>(ptr);
        // ptr += ((sizeof(T) * m * n - 1) / 256 + 1) * 256;

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
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1_N(T, 256, stride, i);

                stride *= 2;
            }

            // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
            rocsparse_int subsystem_count = 1 << iter;

            LAUNCH_GTSV_NOPIVOT_CR_POW2_STAGE2(T, 256, iter);
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
                LAUNCH_GTSV_NOPIVOT_PCR_STAGE1_N(T, 256, stride, i);

                stride *= 2;
            }

            // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
            rocsparse_int subsystem_count = 1 << iter;

            LAUNCH_GTSV_NOPIVOT_PCR_STAGE2(T, 512, iter);
        }

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
        ptr += ((sizeof(T) * m * n - 1) / 256 + 1) * 256;
        T* drhs1 = reinterpret_cast<T*>(ptr);
        // ptr += ((sizeof(T) * m * n - 1) / 256 + 1) * 256;

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
        8, temp_buffer, (m > 512 && temp_buffer == nullptr), rocsparse_status_invalid_pointer);

    // Quick return if possible
    if(n == 0)
    {
        return rocsparse_status_success;
    }

    // If m is small we can solve the systems entirely in shared memory
    if(m <= 512)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gtsv_no_pivot_small_template(handle, m, n, dl, d, du, B, ldb, temp_buffer));
        return rocsparse_status_success;
    }
    else if(m <= 65536)
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
