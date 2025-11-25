/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{
    // Perform dense matrix transposition
    template <uint32_t DIMX, uint32_t DIMY, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dense_transpose_device(
        I m, I n, T alpha, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb)
    {
        const uint32_t lid = threadIdx.x & (DIMX - 1);
        const uint32_t wid = threadIdx.x / DIMX;

        const I row_A = blockIdx.x * DIMX + lid;
        const I row_B = blockIdx.x * DIMX + wid;

        __shared__ T sdata[DIMX][DIMX];

        for(I j = 0; j < n; j += DIMX)
        {
            __syncthreads();

            const I col_A = j + wid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(row_A < m && col_A + k < n)
                {
                    sdata[wid + k][lid] = A[row_A + lda * (col_A + k)];
                }
            }

            __syncthreads();

            const I col_B = j + lid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(col_B < n && row_B + k < m)
                {
                    B[col_B + ldb * (row_B + k)] = alpha * sdata[lid][wid + k];
                }
            }
        }
    }

    // For host scalars
    template <typename T>
    static __forceinline__ __device__ T load_scalar_device_host(T x)
    {
        return x;
    }

    // For device scalars
    template <typename T>
    static __forceinline__ __device__ T load_scalar_device_host(const T* xp)
    {
        return *xp;
    }

    template <uint32_t DIM_X, uint32_t DIM_Y, typename I, typename T, typename U>
    ROCSPARSE_KERNEL(DIM_X* DIM_Y)
    void dense_transpose_strided_batched_kernel(I        m,
                                                I        n,
                                                U        alpha_device_host,
                                                int64_t  alpha_stride,
                                                const T* A,
                                                int64_t  lda,
                                                int64_t  A_stride,
                                                T*       B,
                                                int64_t  ldb,
                                                int64_t  B_stride,
                                                bool     is_host)
    {
        const auto i     = hipBlockIdx_y;
        const auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        rocsparse::dense_transpose_device<DIM_X, DIM_Y>(m,
                                                        n,
                                                        (is_host) ? alpha
                                                                  : alpha + i * alpha_stride,
                                                        A + i * A_stride,
                                                        lda,
                                                        B + i * B_stride,
                                                        ldb);
    }

    template <typename I, typename T>
    static rocsparse_status launch(rocsparse_handle       handle,
                                   rocsparse_pointer_mode mode,
                                   int64_t                batch_count,
                                   int64_t                m,
                                   int64_t                n,
                                   const void*            alpha_device_host,
                                   int64_t                alpha_stride,
                                   const void*            A,
                                   int64_t                lda,
                                   int64_t                A_stride,
                                   void*                  B,
                                   int64_t                ldb,
                                   int64_t                B_stride)
    {
        if(mode == rocsparse_pointer_mode_host)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::dense_transpose_strided_batched_kernel<32, 8, I, T>),
                dim3((m - 1) / 32 + 1, batch_count),
                dim3(32 * 8),
                0,
                handle->stream,
                static_cast<I>(m),
                static_cast<I>(n),
                (alpha_device_host == nullptr) ? static_cast<T>(1)
                                               : *reinterpret_cast<const T*>(alpha_device_host),
                0,
                reinterpret_cast<const T*>(A),
                lda,
                A_stride,
                reinterpret_cast<T*>(B),
                ldb,
                B_stride,
                (mode == rocsparse_pointer_mode_host));
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::dense_transpose_strided_batched_kernel<32, 8>),
                dim3((m - 1) / 32 + 1, batch_count),
                dim3(32 * 8),
                0,
                handle->stream,
                static_cast<I>(m),
                static_cast<I>(n),
                reinterpret_cast<const T*>(alpha_device_host),
                alpha_stride,
                reinterpret_cast<const T*>(A),
                lda,
                A_stride,
                reinterpret_cast<T*>(B),
                ldb,
                B_stride,
                (mode == rocsparse_pointer_mode_host));
        }
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*launch_t)(rocsparse_handle       handle,
                                         rocsparse_pointer_mode mode,
                                         int64_t                batch_count,
                                         int64_t                m,
                                         int64_t                n,
                                         const void*            alpha_device_host,
                                         int64_t                alpha_stride,
                                         const void*            A,
                                         int64_t                lda,
                                         int64_t                A_stride,
                                         void*                  B,
                                         int64_t                ldb,
                                         int64_t                B_stride);
    template <typename I>
    static launch_t find_T(rocsparse_datatype T_datatype)
    {
        switch(T_datatype)
        {
        case rocsparse_datatype_f32_r:
            return launch<I, float>;
        case rocsparse_datatype_f64_r:
            return launch<I, double>;
        case rocsparse_datatype_f32_c:
            return launch<I, rocsparse_float_complex>;
        case rocsparse_datatype_f64_c:
            return launch<I, rocsparse_double_complex>;
        default:
            return nullptr;
        }
    }

    static launch_t find(rocsparse_indextype I_indextype, rocsparse_datatype T_datatype)
    {
        switch(I_indextype)
        {
        case rocsparse_indextype_i32:
            return find_T<int32_t>(T_datatype);
        case rocsparse_indextype_i64:
            return find_T<int64_t>(T_datatype);
        default:
            return nullptr;
        }
    }

    rocsparse_status dense_transpose_strided_batched(rocsparse_handle       handle,
                                                     rocsparse_pointer_mode mode,
                                                     int64_t                batch_count,
                                                     int64_t                m,
                                                     int64_t                n,
                                                     rocsparse_datatype     alpha_datatype,
                                                     const void*            alpha_device_host,
                                                     int64_t                alpha_stride,
                                                     rocsparse_datatype     A_datatype,
                                                     const void*            A,
                                                     int64_t                lda,
                                                     int64_t                A_stride,
                                                     rocsparse_datatype     B_datatype,
                                                     void*                  B,
                                                     int64_t                ldb,
                                                     int64_t                B_stride)
    {
        const rocsparse_indextype I_indextype
            = (m <= std::numeric_limits<int32_t>::max() && n <= std::numeric_limits<int32_t>::max())
                  ? rocsparse_indextype_i32
                  : rocsparse_indextype_i64;

        auto f = find(I_indextype, A_datatype);
        if(f == nullptr)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   "dense_transpose find failed");
        }
        RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                    mode,
                                    batch_count,
                                    m,
                                    n,
                                    alpha_device_host,
                                    alpha_stride,
                                    A,
                                    lda,
                                    A_stride,
                                    B,
                                    ldb,
                                    B_stride));
        return rocsparse_status_success;
    }

    rocsparse_status dense_transpose(rocsparse_handle   handle,
                                     int64_t            m,
                                     int64_t            n,
                                     rocsparse_datatype A_datatype,
                                     const void*        A,
                                     int64_t            lda,
                                     rocsparse_datatype B_datatype,
                                     void*              B,
                                     int64_t            ldb)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_transpose(handle,
                                                             rocsparse_pointer_mode_host,
                                                             m,
                                                             n,
                                                             A_datatype,
                                                             nullptr,
                                                             A_datatype,
                                                             A,
                                                             lda,
                                                             B_datatype,
                                                             B,
                                                             ldb));
        return rocsparse_status_success;
    }

    rocsparse_status dense_transpose(rocsparse_handle       handle,
                                     rocsparse_pointer_mode mode,
                                     int64_t                m,
                                     int64_t                n,
                                     rocsparse_datatype     alpha_datatype,
                                     const void*            alpha,
                                     rocsparse_datatype     A_datatype,
                                     const void*            A,
                                     int64_t                lda,
                                     rocsparse_datatype     B_datatype,
                                     void*                  B,
                                     int64_t                ldb)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::dense_transpose_strided_batched(handle,
                                                       mode,
                                                       static_cast<int64_t>(1),
                                                       m,
                                                       n,
                                                       //
                                                       alpha_datatype,
                                                       alpha,
                                                       static_cast<int64_t>(0),
                                                       //
                                                       A_datatype,
                                                       A,
                                                       lda,
                                                       static_cast<int64_t>(0),
                                                       //
                                                       B_datatype,
                                                       B,
                                                       ldb,
                                                       static_cast<int64_t>(0)));
        return rocsparse_status_success;
    }

}
