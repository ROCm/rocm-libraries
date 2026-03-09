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
#include "gtsv_no_pivot_large_device.h"

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
        B, (T*)nullptr, (T*)nullptr, (T*)nullptr, (T*)nullptr);

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

    template<typename T>
    constexpr uint32_t determine_num_rhs()
    {
        if constexpr(std::is_same<T, float>())
        {
            return 4;
        }
        else if(std::is_same<T, double>() || std::is_same<T, rocsparse_float_complex>())
        {
            return 2;
        }
        else 
        {
            return 1;
        }
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


        if(m == 8 && n == 1)
        {
            constexpr int BLOCKSIZE = 4;
            int nblocks = (m - 1) / BLOCKSIZE + 1;
            int num_spikes = 2 * nblocks;

            std::cout << "nblocks: " << nblocks << " num_spikes: " << num_spikes << " n: " << n << std::endl;

            std::vector<T> hdl_modified(m);
            std::vector<T> hd_modified(m);
            std::vector<T> hdu_modified(m);
            std::vector<T> hB_modified(m * n);

            std::vector<T> hspike_lower(2 * nblocks);
            std::vector<T> hspike_main(2 * nblocks);
            std::vector<T> hspike_upper(2 * nblocks);
            std::vector<T> hspike_B(2 * nblocks);

            T* dl_modified = nullptr;
            T* d_modified = nullptr;
            T* du_modified = nullptr;
            T* dB_modified = nullptr;
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&dl_modified, sizeof(T) * m));
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&d_modified, sizeof(T) * m));
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&du_modified, sizeof(T) * m));
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&dB_modified, sizeof(T) * m * n));

            T* dspike_lower = nullptr;
            T* dspike_main = nullptr;
            T* dspike_upper = nullptr;
            T* dspike_B = nullptr;
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&dspike_lower, sizeof(T) * 2 * nblocks));
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&dspike_main, sizeof(T) * 2 * nblocks));
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&dspike_upper, sizeof(T) * 2 * nblocks));
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&dspike_B, sizeof(T) * 2 * nblocks));


            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::pcr_tiled_forward_kernel<BLOCKSIZE>),
                dim3(nblocks),
                dim3(BLOCKSIZE),
                0,
                handle->stream,
                m,
                n,
                ldb,
                dl,
                d,
                du,
                B,
                dl_modified,
                d_modified,
                du_modified,
                dB_modified,
                dspike_lower,
                dspike_main,
                dspike_upper,
                dspike_B);

            RETURN_IF_HIP_ERROR(hipMemcpy(hdl_modified.data(), dl_modified, sizeof(T) * m, hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(hd_modified.data(), d_modified, sizeof(T) * m, hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(hdu_modified.data(), du_modified, sizeof(T) * m, hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(hB_modified.data(), dB_modified, sizeof(T) * m * n, hipMemcpyDeviceToHost));

            RETURN_IF_HIP_ERROR(hipMemcpy(hspike_lower.data(), dspike_lower, sizeof(T) * 2 * nblocks, hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(hspike_main.data(), dspike_main, sizeof(T) * 2 * nblocks, hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(hspike_upper.data(), dspike_upper, sizeof(T) * 2 * nblocks, hipMemcpyDeviceToHost));
            RETURN_IF_HIP_ERROR(hipMemcpy(hspike_B.data(), dspike_B, sizeof(T) * 2 * nblocks, hipMemcpyDeviceToHost));

            std::cout << "hdl_modified" << std::endl;
            for(size_t i = 0; i < hdl_modified.size(); i++)
            {
                std::cout << hdl_modified[i] << " "; 
            }
            std::cout << "" << std::endl;

            std::cout << "hd_modified" << std::endl;
            for(size_t i = 0; i < hd_modified.size(); i++)
            {
                std::cout << hd_modified[i] << " "; 
            }
            std::cout << "" << std::endl;

            std::cout << "hdu_modified" << std::endl;
            for(size_t i = 0; i < hdu_modified.size(); i++)
            {
                std::cout << hdu_modified[i] << " "; 
            }
            std::cout << "" << std::endl;

            std::cout << "hB_modified" << std::endl;
            for(size_t i = 0; i < hB_modified.size(); i++)
            {
                std::cout << hB_modified[i] << " "; 
            }
            std::cout << "" << std::endl;

            std::cout << "hspike_lower" << std::endl;
            for(size_t i = 0; i < hspike_lower.size(); i++)
            {
                std::cout << hspike_lower[i] << " "; 
            }
            std::cout << "" << std::endl;

            std::cout << "hspike_main" << std::endl;
            for(size_t i = 0; i < hspike_main.size(); i++)
            {
                std::cout << hspike_main[i] << " "; 
            }
            std::cout << "" << std::endl;

            std::cout << "hspike_upper" << std::endl;
            for(size_t i = 0; i < hspike_upper.size(); i++)
            {
                std::cout << hspike_upper[i] << " "; 
            }
            std::cout << "" << std::endl;

            std::cout << "hspike_B" << std::endl;
            for(size_t i = 0; i < hspike_B.size(); i++)
            {
                std::cout << hspike_B[i] << " "; 
            }
            std::cout << "" << std::endl;

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::spike_solver_pcr_kernel<BLOCKSIZE>),
                dim3(1),
                dim3(BLOCKSIZE),
                0,
                handle->stream,
                num_spikes,
                dspike_lower,
                dspike_main,
                dspike_upper,
                dspike_B);

            RETURN_IF_HIP_ERROR(hipMemcpy(hspike_B.data(), dspike_B, sizeof(T) * 2 * nblocks, hipMemcpyDeviceToHost));

            std::cout << "hspike_B" << std::endl;
            for(size_t i = 0; i < hspike_B.size(); i++)
            {
                std::cout << hspike_B[i] << " "; 
            }
            std::cout << "" << std::endl;

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::backward_sweep_kernel<BLOCKSIZE>),
                dim3(nblocks),
                dim3(BLOCKSIZE),
                0,
                handle->stream,
                m,
                dl_modified,
                d_modified,
                du_modified,
                dB_modified,
                dspike_B,
                B);

            std::vector<T> hB(m * n);
            RETURN_IF_HIP_ERROR(hipMemcpy(hB.data(), B, sizeof(T) * m * n, hipMemcpyDeviceToHost));

            std::cout << "hB" << std::endl;
            for(size_t i = 0; i < hB.size(); i++)
            {
                std::cout << hB[i] << " "; 
            }
            std::cout << "" << std::endl;

            RETURN_IF_HIP_ERROR(hipFree(dl_modified));
            RETURN_IF_HIP_ERROR(hipFree(d_modified));
            RETURN_IF_HIP_ERROR(hipFree(du_modified));
            RETURN_IF_HIP_ERROR(hipFree(dB_modified));

            RETURN_IF_HIP_ERROR(hipFree(dspike_lower));
            RETURN_IF_HIP_ERROR(hipFree(dspike_main));
            RETURN_IF_HIP_ERROR(hipFree(dspike_upper));
            RETURN_IF_HIP_ERROR(hipFree(dspike_B));
            return rocsparse_status_success;
        }



















        if(m == 8)
        {
            constexpr uint32_t WF_SIZE = 32;
            constexpr uint32_t TILE_X = 8;
            constexpr uint32_t TILE_Y = 4;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::thomas_shared_transpose_kernel1<256, WF_SIZE, 8, TILE_X, TILE_Y>),
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
                B);
            return rocsparse_status_success;
        }
        else if(m == 16)
        {
            constexpr uint32_t WF_SIZE = 32;
            constexpr uint32_t TILE_X = 16;
            constexpr uint32_t TILE_Y = 2;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::thomas_shared_transpose_kernel1<128, WF_SIZE, 16, TILE_X, TILE_Y>),
                dim3((n - 1) / 128 + 1),
                dim3(128),
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
        else if(m == 32)
        {
            constexpr uint32_t WF_SIZE = 32;
            constexpr uint32_t TILE_X = 32;
            constexpr uint32_t TILE_Y = 1;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::thomas_shared_transpose_kernel1<64, WF_SIZE, 32, TILE_X, TILE_Y>),
                dim3((n - 1) / 64 + 1),
                dim3(64),
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
        else if(m == 64)
        {
            constexpr uint32_t WF_SIZE = 32;
            constexpr uint32_t TILE_X = 32;
            constexpr uint32_t TILE_Y = 1;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::thomas_shared_transpose_kernel2<64, WF_SIZE, 64, TILE_X, TILE_Y>),
                dim3((n - 1) / 64 + 1),
                dim3(64),
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
        else if(m == 96)
        {
            constexpr uint32_t WF_SIZE = 32;
            constexpr uint32_t TILE_X = 32;
            constexpr uint32_t TILE_Y = 1;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::thomas_shared_transpose_kernel2<64, WF_SIZE, 96, TILE_X, TILE_Y>),
                dim3((n - 1) / 64 + 1),
                dim3(64),
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
        else if(m == 128)
        {
            constexpr uint32_t WF_SIZE = 32;
            constexpr uint32_t TILE_X = 32;
            constexpr uint32_t TILE_Y = 1;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::thomas_shared_transpose_kernel2<64, WF_SIZE, 128, TILE_X, TILE_Y>),
                dim3((n - 1) / 64 + 1),
                dim3(64),
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

        if(m <= 8)
        {
            constexpr int NUM_RHS = 8;
            constexpr int WF_SIZE = 8;
            constexpr int BLOCKSIZE = 256;
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_B = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_nopivot_pcr_wavefront_kernel_32<BLOCKSIZE, WF_SIZE, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_B);
            return rocsparse_status_success;
        }
        else if(m <= 16)
        {
            constexpr int NUM_RHS = 8;
            constexpr int WF_SIZE = 16;
            constexpr int BLOCKSIZE = 256;
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_B = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_nopivot_pcr_wavefront_kernel_32<BLOCKSIZE, WF_SIZE, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_B);
            return rocsparse_status_success;
        }
        else if(m <= 32)
        {
            constexpr int NUM_RHS = 8;
            constexpr int WF_SIZE = 32;
            constexpr int BLOCKSIZE = 256;
        
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_B = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_nopivot_pcr_wavefront_kernel_32<BLOCKSIZE, WF_SIZE, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_B);
            return rocsparse_status_success;
        }
        else if(m <= 64)
        {
            constexpr int NUM_RHS = 8;
            constexpr int WF_SIZE = 32;
            constexpr int BLOCKSIZE = 64;
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_B = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_no_pivot_pcr_shared_kernel2<BLOCKSIZE, WF_SIZE, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_B);
            return rocsparse_status_success;
        }
        else if(m <= 128)
        {
            constexpr int NUM_RHS = 8;
            constexpr int WF_SIZE = 32;
            constexpr int BLOCKSIZE = 128;
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_B = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_no_pivot_pcr_shared_kernel2<BLOCKSIZE, WF_SIZE, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_B);
            return rocsparse_status_success;
        }
        else if(m <= 256)
        {
            constexpr int NUM_RHS = 8;
            constexpr int WF_SIZE = 32;
            constexpr int BLOCKSIZE = 256;
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_B = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_no_pivot_pcr_shared_kernel2<BLOCKSIZE, WF_SIZE, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_B);
            return rocsparse_status_success;
        }
        else if(m <= 512)
        {
            constexpr int BLOCKSIZE = 256;
            constexpr int NUM_RHS = determine_num_rhs<T>();
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_d = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_nopivot_crpcr_pow2_shared_kernel2<BLOCKSIZE, 128, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_d);

            return rocsparse_status_success;
        }
        else if(m <= 1024)
        {
            constexpr int BLOCKSIZE = 512;
            constexpr int NUM_RHS = determine_num_rhs<T>();
            T* dtemp_a = nullptr;
            T* dtemp_b = nullptr;
            T* dtemp_c = nullptr;
            T* dtemp_d = nullptr;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::gtsv_nopivot_crpcr_pow2_shared_kernel2<BLOCKSIZE, 256, NUM_RHS>),
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
                B,
                dtemp_a,
                dtemp_b,
                dtemp_c,
                dtemp_d);
            return rocsparse_status_success;
        }












        // // Define function pointer type for kernel dispatch
        // using KernelFuncPtr = rocsparse_status (*)(
        //     rocsparse_handle, rocsparse_int, rocsparse_int, const T*, const T*, const T*, T*);

        // // Kernel dispatch table for small matrix sizes
        // static const std::map<int, KernelFuncPtr> s_kernel_dispatch = {
        //     {2, launch_cramer_rule_kernel<T>},   {3, launch_thomas_kernel_3<T>},
        //     {4, launch_thomas_kernel_4<T>},      {5, launch_thomas_kernel_m<5, T>},
        //     {6, launch_thomas_kernel_m<6, T>},   {7, launch_thomas_kernel_m<7, T>},
        //     {8, launch_thomas_kernel_m<8, T>},   {9, launch_thomas_kernel_m<9, T>},
        //     {10, launch_thomas_kernel_m<10, T>}, {11, launch_thomas_kernel_m<11, T>},
        //     {12, launch_thomas_kernel_m<12, T>}, {13, launch_thomas_kernel_m<13, T>},
        //     {14, launch_thomas_kernel_m<14, T>}, {15, launch_thomas_kernel_m<15, T>},
        //     {16, launch_thomas_kernel_m<16, T>}, {17, launch_thomas_kernel_m<17, T>},
        //     {18, launch_thomas_kernel_m<18, T>}, {19, launch_thomas_kernel_m<19, T>},
        //     {20, launch_thomas_kernel_m<20, T>}, {21, launch_thomas_kernel_m<21, T>},
        //     {22, launch_thomas_kernel_m<22, T>}, {23, launch_thomas_kernel_m<23, T>},
        //     {24, launch_thomas_kernel_m<24, T>}, {25, launch_thomas_kernel_m<25, T>},
        //     {26, launch_thomas_kernel_m<26, T>}, {27, launch_thomas_kernel_m<27, T>},
        //     {28, launch_thomas_kernel_m<28, T>}, {29, launch_thomas_kernel_m<29, T>},
        //     {30, launch_thomas_kernel_m<30, T>}, {31, launch_thomas_kernel_m<31, T>},

        //     {32, launch_thomas_kernel_m<32, T>}, {33, launch_thomas_kernel_m<33, T>},
        //     {34, launch_thomas_kernel_m<34, T>}, {35, launch_thomas_kernel_m<35, T>},
        //     {36, launch_thomas_kernel_m<36, T>}, {37, launch_thomas_kernel_m<37, T>},
        //     {38, launch_thomas_kernel_m<38, T>}, {39, launch_thomas_kernel_m<39, T>},
        //     {40, launch_thomas_kernel_m<40, T>}, {41, launch_thomas_kernel_m<41, T>},
        //     {42, launch_thomas_kernel_m<42, T>}, {43, launch_thomas_kernel_m<43, T>},
        //     {44, launch_thomas_kernel_m<44, T>}, {45, launch_thomas_kernel_m<45, T>},
        //     {46, launch_thomas_kernel_m<46, T>}, {47, launch_thomas_kernel_m<47, T>},
        //     {48, launch_thomas_kernel_m<48, T>}, {49, launch_thomas_kernel_m<49, T>},
        //     {50, launch_thomas_kernel_m<50, T>}, {51, launch_thomas_kernel_m<51, T>},

        //     {52, launch_thomas_kernel_m<52, T>}, {53, launch_thomas_kernel_m<53, T>},
        //     {54, launch_thomas_kernel_m<54, T>}, {55, launch_thomas_kernel_m<55, T>},
        //     {56, launch_thomas_kernel_m<56, T>}, {57, launch_thomas_kernel_m<57, T>},
        //     {58, launch_thomas_kernel_m<58, T>}, {59, launch_thomas_kernel_m<59, T>},
        //     {60, launch_thomas_kernel_m<60, T>}, {61, launch_thomas_kernel_m<61, T>},
        //     {62, launch_thomas_kernel_m<62, T>}, {63, launch_thomas_kernel_m<63, T>},
        //     {64, launch_thomas_kernel_m<64, T>}, {65, launch_thomas_kernel_m<65, T>},
        //     {66, launch_thomas_kernel_m<66, T>}, {67, launch_thomas_kernel_m<67, T>},
        //     {68, launch_thomas_kernel_m<68, T>}, {69, launch_thomas_kernel_m<69, T>},
        //     {70, launch_thomas_kernel_m<70, T>}, {71, launch_thomas_kernel_m<71, T>},

        //     {72, launch_thomas_kernel_m<72, T>}, {73, launch_thomas_kernel_m<73, T>},
        //     {74, launch_thomas_kernel_m<74, T>}, {75, launch_thomas_kernel_m<75, T>},
        //     {76, launch_thomas_kernel_m<76, T>}, {77, launch_thomas_kernel_m<77, T>},
        //     {78, launch_thomas_kernel_m<78, T>}, {79, launch_thomas_kernel_m<79, T>},
        //     {80, launch_thomas_kernel_m<80, T>}, {81, launch_thomas_kernel_m<81, T>},
        //     {82, launch_thomas_kernel_m<82, T>}, {83, launch_thomas_kernel_m<83, T>},
        //     {84, launch_thomas_kernel_m<84, T>}, {85, launch_thomas_kernel_m<85, T>},
        //     {86, launch_thomas_kernel_m<86, T>}, {87, launch_thomas_kernel_m<87, T>},
        //     {88, launch_thomas_kernel_m<88, T>}, {89, launch_thomas_kernel_m<89, T>},
        //     {90, launch_thomas_kernel_m<90, T>}, {91, launch_thomas_kernel_m<91, T>},

        //     {92, launch_thomas_kernel_m<92, T>}, {93, launch_thomas_kernel_m<93, T>},
        //     {94, launch_thomas_kernel_m<94, T>}, {95, launch_thomas_kernel_m<95, T>},
        //     {96, launch_thomas_kernel_m<96, T>}, {97, launch_thomas_kernel_m<97, T>},
        //     {98, launch_thomas_kernel_m<98, T>}, {99, launch_thomas_kernel_m<99, T>},
        //     {100, launch_thomas_kernel_m<100, T>}, {101, launch_thomas_kernel_m<101, T>},
        //     {102, launch_thomas_kernel_m<102, T>}, {103, launch_thomas_kernel_m<103, T>},
        //     {104, launch_thomas_kernel_m<104, T>}, {105, launch_thomas_kernel_m<105, T>},
        //     {106, launch_thomas_kernel_m<106, T>}, {107, launch_thomas_kernel_m<107, T>},
        //     {108, launch_thomas_kernel_m<108, T>}, {109, launch_thomas_kernel_m<109, T>},
        //     {110, launch_thomas_kernel_m<110, T>}, {111, launch_thomas_kernel_m<111, T>},

        //     {112, launch_thomas_kernel_m<112, T>}, {113, launch_thomas_kernel_m<113, T>},
        //     {114, launch_thomas_kernel_m<114, T>}, {115, launch_thomas_kernel_m<115, T>},
        //     {116, launch_thomas_kernel_m<116, T>}, {117, launch_thomas_kernel_m<117, T>},
        //     {118, launch_thomas_kernel_m<118, T>}, {119, launch_thomas_kernel_m<119, T>},
        //     {120, launch_thomas_kernel_m<120, T>}, {121, launch_thomas_kernel_m<121, T>},
        //     {122, launch_thomas_kernel_m<122, T>}, {123, launch_thomas_kernel_m<123, T>},
        //     {124, launch_thomas_kernel_m<124, T>}, {125, launch_thomas_kernel_m<125, T>},
        //     {126, launch_thomas_kernel_m<126, T>}, {127, launch_thomas_kernel_m<127, T>},
        //     {128, launch_thomas_kernel_m<128, T>}, {129, launch_thomas_kernel_m<129, T>},
        //     {130, launch_thomas_kernel_m<130, T>}, {131, launch_thomas_kernel_m<131, T>},
        //     {132, launch_thomas_kernel_m<132, T>}, {133, launch_thomas_kernel_m<133, T>},
        //     {134, launch_thomas_kernel_m<134, T>}, {135, launch_thomas_kernel_m<135, T>},
        //     {136, launch_thomas_kernel_m<136, T>}, {137, launch_thomas_kernel_m<137, T>},
        //     {138, launch_thomas_kernel_m<138, T>}, {139, launch_thomas_kernel_m<139, T>},
        //     {140, launch_thomas_kernel_m<140, T>}, {141, launch_thomas_kernel_m<141, T>},
        //     {142, launch_thomas_kernel_m<142, T>}, {143, launch_thomas_kernel_m<143, T>},
        //     {144, launch_thomas_kernel_m<144, T>}, {145, launch_thomas_kernel_m<145, T>},
        //     {146, launch_thomas_kernel_m<146, T>}, {147, launch_thomas_kernel_m<147, T>},
        //     {148, launch_thomas_kernel_m<148, T>}, {149, launch_thomas_kernel_m<149, T>},
        //     {150, launch_thomas_kernel_m<150, T>}, {151, launch_thomas_kernel_m<151, T>},
        //     {152, launch_thomas_kernel_m<152, T>}, {153, launch_thomas_kernel_m<153, T>},
        //     {154, launch_thomas_kernel_m<154, T>}, {155, launch_thomas_kernel_m<155, T>},
        //     {156, launch_thomas_kernel_m<156, T>}, {157, launch_thomas_kernel_m<157, T>},
        //     {158, launch_thomas_kernel_m<158, T>}, {159, launch_thomas_kernel_m<159, T>},
        //     {160, launch_thomas_kernel_m<160, T>}, {161, launch_thomas_kernel_m<161, T>},
        //     {162, launch_thomas_kernel_m<162, T>}, {163, launch_thomas_kernel_m<163, T>},
        //     {164, launch_thomas_kernel_m<164, T>}, {165, launch_thomas_kernel_m<165, T>},
        //     {166, launch_thomas_kernel_m<166, T>}, {167, launch_thomas_kernel_m<167, T>},
        //     {168, launch_thomas_kernel_m<168, T>}, {169, launch_thomas_kernel_m<169, T>},
        //     {170, launch_thomas_kernel_m<170, T>}, {171, launch_thomas_kernel_m<171, T>},
        //     {172, launch_thomas_kernel_m<172, T>}, {173, launch_thomas_kernel_m<173, T>},
        //     {174, launch_thomas_kernel_m<174, T>}, {175, launch_thomas_kernel_m<175, T>},
        //     {176, launch_thomas_kernel_m<176, T>}, {177, launch_thomas_kernel_m<177, T>},
        //     {178, launch_thomas_kernel_m<178, T>}, {179, launch_thomas_kernel_m<179, T>},
        //     {180, launch_thomas_kernel_m<180, T>}, {181, launch_thomas_kernel_m<181, T>},
        //     {182, launch_thomas_kernel_m<182, T>}, {183, launch_thomas_kernel_m<183, T>},
        //     {184, launch_thomas_kernel_m<184, T>}, {185, launch_thomas_kernel_m<185, T>},
        //     {186, launch_thomas_kernel_m<186, T>}, {187, launch_thomas_kernel_m<187, T>},
        //     {188, launch_thomas_kernel_m<188, T>}, {189, launch_thomas_kernel_m<189, T>},
        //     {190, launch_thomas_kernel_m<190, T>}, {191, launch_thomas_kernel_m<191, T>},
        //     {192, launch_thomas_kernel_m<192, T>}, {193, launch_thomas_kernel_m<193, T>},
        //     {194, launch_thomas_kernel_m<194, T>}, {195, launch_thomas_kernel_m<195, T>},
        //     {196, launch_thomas_kernel_m<196, T>}, {197, launch_thomas_kernel_m<197, T>},
        //     {198, launch_thomas_kernel_m<198, T>}, {199, launch_thomas_kernel_m<199, T>},
        //     {200, launch_thomas_kernel_m<200, T>}, {201, launch_thomas_kernel_m<201, T>},
        //     {202, launch_thomas_kernel_m<202, T>}, {203, launch_thomas_kernel_m<203, T>},
        //     {204, launch_thomas_kernel_m<204, T>}, {205, launch_thomas_kernel_m<205, T>},
        //     {206, launch_thomas_kernel_m<206, T>}, {207, launch_thomas_kernel_m<207, T>},
        //     {208, launch_thomas_kernel_m<208, T>}, {209, launch_thomas_kernel_m<209, T>},
        //     {210, launch_thomas_kernel_m<210, T>}, {211, launch_thomas_kernel_m<211, T>},
        //     {212, launch_thomas_kernel_m<212, T>}, {213, launch_thomas_kernel_m<213, T>},
        //     {214, launch_thomas_kernel_m<214, T>}, {215, launch_thomas_kernel_m<215, T>},
        //     {216, launch_thomas_kernel_m<216, T>}, {217, launch_thomas_kernel_m<217, T>},
        //     {218, launch_thomas_kernel_m<218, T>}, {219, launch_thomas_kernel_m<219, T>},
        //     {220, launch_thomas_kernel_m<220, T>}, {221, launch_thomas_kernel_m<221, T>},
        //     {222, launch_thomas_kernel_m<222, T>}, {223, launch_thomas_kernel_m<223, T>},
        //     {224, launch_thomas_kernel_m<224, T>}, {225, launch_thomas_kernel_m<225, T>},
        //     {226, launch_thomas_kernel_m<226, T>}, {227, launch_thomas_kernel_m<227, T>},
        //     {228, launch_thomas_kernel_m<228, T>}, {229, launch_thomas_kernel_m<229, T>},
        //     {230, launch_thomas_kernel_m<230, T>}, {231, launch_thomas_kernel_m<231, T>},
        //     {232, launch_thomas_kernel_m<232, T>}, {233, launch_thomas_kernel_m<233, T>},
        //     {234, launch_thomas_kernel_m<234, T>}, {235, launch_thomas_kernel_m<235, T>},
        //     {236, launch_thomas_kernel_m<236, T>}, {237, launch_thomas_kernel_m<237, T>},
        //     {238, launch_thomas_kernel_m<238, T>}, {239, launch_thomas_kernel_m<239, T>},
        //     {240, launch_thomas_kernel_m<240, T>}, {241, launch_thomas_kernel_m<241, T>},
        //     {242, launch_thomas_kernel_m<242, T>}, {243, launch_thomas_kernel_m<243, T>},
        //     {244, launch_thomas_kernel_m<244, T>}, {245, launch_thomas_kernel_m<245, T>},
        //     {246, launch_thomas_kernel_m<246, T>}, {247, launch_thomas_kernel_m<247, T>},
        //     {248, launch_thomas_kernel_m<248, T>}, {249, launch_thomas_kernel_m<249, T>},
        //     {250, launch_thomas_kernel_m<250, T>}, {251, launch_thomas_kernel_m<251, T>},
        //     {252, launch_thomas_kernel_m<252, T>}, {253, launch_thomas_kernel_m<253, T>},
        //     {254, launch_thomas_kernel_m<254, T>}, {255, launch_thomas_kernel_m<255, T>},
        //     {256, launch_thomas_kernel_m<256, T>}
        // };

        // // Thomas algorithm good up to m=93 after which perf drops (dispatch table)
        // if(m <= 256)
        // {
        //     auto it = s_kernel_dispatch.find(m);

        //     if(it != s_kernel_dispatch.end())
        //     {
        //         return it->second(handle, n, ldb, dl, d, du, B);
        //     }
        //     else
        //     {
        //         // Handle error: m not in dispatch table
        //         return rocsparse_status_not_implemented;
        //     }
        // }
        // return rocsparse_status_not_implemented;

        // Run special algorithm if m is power of 2
        if((m & (m - 1)) == 0)
        {
            if(m == 2)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(2);
            }
            else if(m == 4)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(4);
            }
            else if(m == 8)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(8);
            }
            else if(m == 16)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(16);
            }
            else if(m == 32)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(32);
            }
            else if(m == 64)
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
            if(m <= 4)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(4);
            }
            else if(m <= 8)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(8);
            }
            else if(m <= 16)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(16);
            }
            else if(m <= 32)
            {
                LAUNCH_GTSV_NOPIVOT_PCR_SHARED(32);
            }
            else if(m <= 64)
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
    //if(m <= 512)
    if(m <= 1024)
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
