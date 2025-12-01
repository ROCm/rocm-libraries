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

#include "rocsparse_dense_transpose_back.hpp"
#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{

    // Perform dense matrix back transposition
    template <uint32_t DIMX, uint32_t DIMY, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dense_transpose_back_device(
        I m, I n, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb)
    {
        const int lid = hipThreadIdx_x & (DIMX - 1);
        const int wid = hipThreadIdx_x / DIMX;

        const I row_A = hipBlockIdx_x * DIMX + wid;
        const I row_B = hipBlockIdx_x * DIMX + lid;

        __shared__ T sdata[DIMX][DIMX];

        for(I j = 0; j < n; j += DIMX)
        {
            __syncthreads();

            const I col_A = j + lid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(col_A < n && row_A + k < m)
                {
                    sdata[wid + k][lid] = A[col_A + lda * (row_A + k)];
                }
            }

            __syncthreads();

            const I col_B = j + wid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(row_B < m && col_B + k < n)
                {
                    B[row_B + ldb * (col_B + k)] = sdata[lid][wid + k];
                }
            }
        }
    }

    template <uint32_t DIM_X, uint32_t DIM_Y, typename I, typename T>
    ROCSPARSE_KERNEL(DIM_X* DIM_Y)
    void dense_transpose_back_kernel(
        I m, I n, const T* A, int64_t lda, int64_t A_stride, T* B, int64_t ldb, int64_t B_stride)
    {
        const auto i = hipBlockIdx_y;
        rocsparse::dense_transpose_back_device<DIM_X, DIM_Y>(
            m, n, A + i * A_stride, lda, B + i * B_stride, ldb);
    }

    template <typename I, typename T>
    static rocsparse_status launch(rocsparse_handle handle,
                                   int64_t          batch_count,
                                   int64_t          m,
                                   int64_t          n,
                                   const void*      A,
                                   int64_t          lda,
                                   int64_t          A_stride,
                                   void*            B,
                                   int64_t          ldb,
                                   int64_t          B_stride)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dense_transpose_back_kernel<32, 8, I, T>),
                                           dim3((static_cast<I>(m) - 1) / 32 + 1, batch_count),
                                           dim3(32 * 8),
                                           0,
                                           handle->stream,
                                           static_cast<I>(m),
                                           static_cast<I>(n),
                                           reinterpret_cast<const T*>(A),
                                           lda,
                                           A_stride,
                                           reinterpret_cast<T*>(B),
                                           ldb,
                                           B_stride);
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*launch_t)(rocsparse_handle,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         const void*,
                                         int64_t,
                                         int64_t,
                                         void*,
                                         int64_t,
                                         int64_t);

    template <typename... P>
    static launch_t find_B(rocsparse_datatype T_datatype)
    {
        switch(T_datatype)
        {
        case rocsparse_datatype_f32_r:
            return launch<P..., float>;
        case rocsparse_datatype_f64_r:
            return launch<P..., double>;
        case rocsparse_datatype_f32_c:
            return launch<P..., rocsparse_float_complex>;
        case rocsparse_datatype_f64_c:
            return launch<P..., rocsparse_double_complex>;
        default:
            return nullptr;
        }
    }

    static launch_t find(rocsparse_indextype I_indextype,
                         rocsparse_datatype  A_datatype,
                         rocsparse_datatype  B_datatype)
    {
        switch(I_indextype)
        {
        case rocsparse_indextype_i32:
            return find_B<int32_t>(B_datatype);
        case rocsparse_indextype_i64:
            return find_B<int64_t>(B_datatype);
        default:
            return nullptr;
        }
    }
}

rocsparse_status rocsparse::dense_transpose_back_strided_batched(rocsparse_handle   handle,
                                                                 int64_t            batch_count,
                                                                 int64_t            m,
                                                                 int64_t            n,
                                                                 rocsparse_datatype A_datatype,
                                                                 const void*        A,
                                                                 int64_t            lda,
                                                                 int64_t            A_stride,
                                                                 rocsparse_datatype B_datatype,
                                                                 void*              B,
                                                                 int64_t            ldb,
                                                                 int64_t            B_stride)
{
    const rocsparse_indextype I_indextype
        = (m <= std::numeric_limits<int32_t>::max() && n <= std::numeric_limits<int32_t>::max())
              ? rocsparse_indextype_i32
              : rocsparse_indextype_i64;

    auto launch_kernel = find(I_indextype, A_datatype, B_datatype);
    if(launch_kernel == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "dense_transpose_back find failed");
    }
    RETURN_IF_ROCSPARSE_ERROR(
        launch_kernel(handle, batch_count, m, n, A, lda, A_stride, B, ldb, B_stride));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::dense_transpose_back(rocsparse_handle   handle,
                                                 int64_t            m,
                                                 int64_t            n,
                                                 rocsparse_datatype A_datatype,
                                                 const void*        A,
                                                 int64_t            lda,
                                                 rocsparse_datatype B_datatype,
                                                 void*              B,
                                                 int64_t            ldb)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::dense_transpose_back_strided_batched(handle,
                                                        static_cast<int64_t>(1),
                                                        m,
                                                        n,
                                                        A_datatype,
                                                        A,
                                                        lda,
                                                        static_cast<int64_t>(0),
                                                        B_datatype,
                                                        B,
                                                        ldb,
                                                        static_cast<int64_t>(0)));
    return rocsparse_status_success;
}
