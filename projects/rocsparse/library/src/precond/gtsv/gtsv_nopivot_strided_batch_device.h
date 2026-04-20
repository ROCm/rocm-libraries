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

#pragma once

#include "rocsparse_common.hpp"

namespace rocsparse
{
    // Parallel cyclic reduction algorithm
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_wavefront_kernel(rocsparse_int m,
                                                         rocsparse_int batch_count,
                                                         int64_t       batch_stride,
                                                         int64_t       ldb,
                                                         const T* __restrict__ dl,
                                                         const T* __restrict__ d,
                                                         const T* __restrict__ du,
                                                         T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        const int iter   = rocsparse::log2_pow2<WF_SIZE / 2>::value;
        int       stride = 1;

        const int batch = (BLOCKSIZE / WF_SIZE) * bid + wid;

        T a = ((lid < m && batch < batch_count) && lid != 0) ? dl[batch_stride * batch + lid]
                                                             : static_cast<T>(0);
        T b = (lid < m && batch < batch_count) ? d[batch_stride * batch + lid] : static_cast<T>(1);
        T c = ((lid < m && batch < batch_count) && lid != (m - 1)) ? du[batch_stride * batch + lid]
                                                                   : static_cast<T>(0);
        T x = (lid < m && batch < batch_count) ? B[ldb * batch + lid] : static_cast<T>(0);

        for(int it = 0; it < iter; it++)
        {
            const int right = lid + stride;
            const int left  = lid - stride;

            T a_left = shfl_up(a, stride, WF_SIZE);
            T b_left = shfl_up(b, stride, WF_SIZE);
            T c_left = shfl_up(c, stride, WF_SIZE);

            if(left < 0)
            {
                a_left = static_cast<T>(0);
                b_left = static_cast<T>(0);
                c_left = static_cast<T>(0);
            }

            T a_right = shfl_down(a, stride, WF_SIZE);
            T b_right = shfl_down(b, stride, WF_SIZE);
            T c_right = shfl_down(c, stride, WF_SIZE);

            if(right > (WF_SIZE - 1))
            {
                a_right = static_cast<T>(0);
                b_right = static_cast<T>(0);
                c_right = static_cast<T>(0);
            }

            const T k1 = (left >= 0) ? a / b_left : static_cast<T>(0);
            const T k2 = (right <= WF_SIZE - 1) ? c / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;

            a = a_new;
            b = b_new;
            c = c_new;

            T x_left = shfl_up(x, stride, WF_SIZE);
            if(left < 0)
            {
                x_left = static_cast<T>(0);
            }
            T x_right = shfl_down(x, stride, WF_SIZE);
            if(right > (WF_SIZE - 1))
            {
                x_right = static_cast<T>(0);
            }

            const T x_new = x - x_left * k1 - x_right * k2;

            x = x_new;

            stride <<= 1; //stride *= 2;
        }

        // Solve 2x2 systems (j = lid + stride)
        // bi ci
        // aj bj
        //
        // det = bi * bj - aj * ci
        const T aj = shfl_down(a, stride, WF_SIZE);
        const T bj = shfl_down(b, stride, WF_SIZE);

        const T det = static_cast<T>(1) / (b * bj - aj * c);

        const T xj = shfl_down(x, stride, WF_SIZE);

        if(lid < WF_SIZE / 2) // same as lid < stride
        {
            if(lid < m && batch < batch_count)
            {
                B[ldb * batch + lid] = (bj * x - c * xj) * det;
            }
            if((lid + stride) < m && batch < batch_count)
            {
                B[ldb * batch + lid + stride] = (xj * b - x * aj) * det;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_shared_kernel(rocsparse_int m,
                                                      rocsparse_int batch_count,
                                                      int64_t       batch_stride,
                                                      int64_t       ldb,
                                                      const T* __restrict__ dl,
                                                      const T* __restrict__ d,
                                                      const T* __restrict__ du,
                                                      T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        const int iter_BLOCKSIZE = rocsparse::log2_pow2<BLOCKSIZE / 2>::value;
        const int iter_WF_SIZE   = rocsparse::log2_pow2<WF_SIZE / 2>::value;
        const int iter           = iter_BLOCKSIZE - iter_WF_SIZE;

        int stride = 1; // rename to cr_stride

        T a = ((tid < m) && tid != 0) ? dl[batch_stride * bid + tid] : static_cast<T>(0);
        T b = (tid < m) ? d[batch_stride * bid + tid] : static_cast<T>(1);
        T c = ((tid < m) && tid != (m - 1)) ? du[batch_stride * bid + tid] : static_cast<T>(0);
        T x = (tid < m) ? B[ldb * bid + tid] : static_cast<T>(0);

        // Parallel cyclic reduction shared memory
        __shared__ T a_shared[BLOCKSIZE];
        __shared__ T b_shared[BLOCKSIZE];
        __shared__ T c_shared[BLOCKSIZE];
        __shared__ T x_shared[BLOCKSIZE];

        // Fill parallel cyclic reduction shared memory
        a_shared[tid] = a;
        b_shared[tid] = b;
        c_shared[tid] = c;
        x_shared[tid] = x;
        __syncthreads();

        for(int j = 0; j < iter; j++)
        {
            const int right = tid + stride;
            const int left  = tid - stride;

            const T a_left = (left >= 0) ? a_shared[left] : static_cast<T>(0);
            const T b_left = (left >= 0) ? b_shared[left] : static_cast<T>(0);
            const T c_left = (left >= 0) ? c_shared[left] : static_cast<T>(0);

            const T a_right = (right < BLOCKSIZE) ? a_shared[right] : static_cast<T>(0);
            const T b_right = (right < BLOCKSIZE) ? b_shared[right] : static_cast<T>(0);
            const T c_right = (right < BLOCKSIZE) ? c_shared[right] : static_cast<T>(0);

            const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
            const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;

            __syncthreads();
            a_shared[tid] = a_new;
            b_shared[tid] = b_new;
            c_shared[tid] = c_new;

            a = a_new;
            b = b_new;
            c = c_new;

            const T x_left  = (left >= 0) ? x_shared[left] : static_cast<T>(0);
            const T x_right = (right < BLOCKSIZE) ? x_shared[right] : static_cast<T>(0);

            const T x_new = x - x_left * k1 - x_right * k2;

            __syncthreads();
            x_shared[tid] = x_new;

            x = x_new;
            __syncthreads();

            stride *= 2;
        }

        a = a_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        b = b_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        c = c_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        x = x_shared[(BLOCKSIZE / WF_SIZE) * lid + wid];
        __syncthreads();

        int stride2 = 1;
        for(int it = 0; it < iter_WF_SIZE; it++)
        {
            const int right = lid + stride2;
            const int left  = lid - stride2;

            T a_left = shfl_up(a, stride2, WF_SIZE);
            T b_left = shfl_up(b, stride2, WF_SIZE);
            T c_left = shfl_up(c, stride2, WF_SIZE);

            if(left < 0)
            {
                a_left = static_cast<T>(0);
                b_left = static_cast<T>(0);
                c_left = static_cast<T>(0);
            }

            T a_right = shfl_down(a, stride2, WF_SIZE);
            T b_right = shfl_down(b, stride2, WF_SIZE);
            T c_right = shfl_down(c, stride2, WF_SIZE);

            if(right > (WF_SIZE - 1))
            {
                a_right = static_cast<T>(0);
                b_right = static_cast<T>(0);
                c_right = static_cast<T>(0);
            }

            const T k1 = (b_left != static_cast<T>(0)) ? a / b_left : static_cast<T>(0);
            const T k2 = (b_right != static_cast<T>(0)) ? c / b_right : static_cast<T>(0);

            const T a_new = -a_left * k1;
            const T b_new = b - c_left * k1 - a_right * k2;
            const T c_new = -c_right * k2;

            a = a_new;
            b = b_new;
            c = c_new;

            T x_left  = shfl_up(x, stride2, WF_SIZE);
            T x_right = shfl_down(x, stride2, WF_SIZE);

            if(left < 0)
            {
                x_left = static_cast<T>(0);
            }

            if(right > (WF_SIZE - 1))
            {
                x_right = static_cast<T>(0);
            }

            const T x_new = x - x_left * k1 - x_right * k2;
            x             = x_new;

            stride2 <<= 1; //stride2 *= 2;
        }

        // Solve 2x2 systems (j = lid + stride2)
        // bi ci
        // aj bj
        //
        // det = bi * bj - aj * ci
        const T aj = shfl_down(a, stride2, WF_SIZE);
        const T bj = shfl_down(b, stride2, WF_SIZE);

        const T det = static_cast<T>(1) / (b * bj - aj * c);

        const T xj = shfl_down(x, stride2, WF_SIZE);

        if(lid < WF_SIZE / 2) // same as lid < stride2
        {
            if(((BLOCKSIZE / WF_SIZE) * lid + wid) < m)
            {
                B[ldb * bid + (BLOCKSIZE / WF_SIZE) * lid + wid] = (bj * x - c * xj) * det;
            }
            if(((BLOCKSIZE / WF_SIZE) * (lid + stride2) + wid) < m)
            {
                B[ldb * bid + (BLOCKSIZE / WF_SIZE) * (lid + stride2) + wid]
                    = (xj * b - x * aj) * det;
            }
        }
    }

    // Combined Parallel cyclic reduction and cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_crpcr_shared_kernel(rocsparse_int m,
                                                        rocsparse_int batch_count,
                                                        int64_t       batch_stride,
                                                        int64_t       ldb,
                                                        const T* __restrict__ dl,
                                                        const T* __restrict__ d,
                                                        const T* __restrict__ du,
                                                        T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;

        const int tot_iter = rocsparse::log2_pow2<(2 * BLOCKSIZE) / 2>::value;
        const int pcr_iter = rocsparse::log2_pow2<PCR_SIZE / 2>::value;
        const int cr_iter  = tot_iter - pcr_iter;

        int stride         = 1;
        int active_threads = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];

        // Fill cyclic reduction shared memory
        sa[tid] = (tid < m && tid != 0) ? dl[batch_stride * bid + tid] : static_cast<T>(0);
        sa[tid + BLOCKSIZE]
            = (tid + BLOCKSIZE < m) ? dl[batch_stride * bid + tid + BLOCKSIZE] : static_cast<T>(0);
        sb[tid] = (tid < m) ? d[batch_stride * bid + tid] : static_cast<T>(1);
        sb[tid + BLOCKSIZE]
            = (tid + BLOCKSIZE < m) ? d[batch_stride * bid + tid + BLOCKSIZE] : static_cast<T>(1);
        sc[tid]             = (tid < m) ? du[batch_stride * bid + tid] : static_cast<T>(0);
        sc[tid + BLOCKSIZE] = (tid + BLOCKSIZE < m && (tid + BLOCKSIZE) != (m - 1))
                                  ? du[batch_stride * bid + tid + BLOCKSIZE]
                                  : static_cast<T>(0);
        srhs[tid]           = (tid < m) ? B[ldb * bid + tid] : static_cast<T>(0);
        srhs[tid + BLOCKSIZE]
            = (tid + BLOCKSIZE < m) ? B[ldb * bid + tid + BLOCKSIZE] : static_cast<T>(0);

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(int j = 0; j < cr_iter; j++)
        {
            stride *= 2;

            if(tid < active_threads)
            {
                const int index = stride * tid + stride - 1;
                int       left  = index - stride / 2;
                int       right = index + stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index] = sb[index] - sc[left] * k1 - sa[right] * k2;
                sa[index] = -sa[left] * k1;
                sc[index] = -sc[right] * k2;

                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
            }

            active_threads /= 2;

            __syncthreads();
        }

        // Parallel cyclic reduction
        const int index      = stride * tid + stride - 1;
        int       pcr_stride = stride;

        for(int j = 0; j < pcr_iter; j++)
        {
            T ta;
            T tb;
            T tc;
            T trhs;

            if(tid < PCR_SIZE)
            {
                rocsparse_int right = index + pcr_stride;
                if(right >= 2 * BLOCKSIZE)
                    right = 2 * BLOCKSIZE - 1;

                rocsparse_int left = index - pcr_stride;
                if(left < 0)
                    left = 0;

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                tb = sb[index] - sc[left] * k1 - sa[right] * k2;
                ta = -sa[left] * k1;
                tc = -sc[right] * k2;

                trhs = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
            }

            __syncthreads();
            if(tid < PCR_SIZE)
            {
                sb[index]   = tb;
                sa[index]   = ta;
                sc[index]   = tc;
                srhs[index] = trhs;
            }
            pcr_stride *= 2;
            __syncthreads();
        }

        if(tid < PCR_SIZE / 2)
        {
            const int index = stride * tid + stride - 1;

            // Solve 2x2 systems
            const int i   = index;
            const int j   = index + pcr_stride;
            const T   det = static_cast<T>(1) / (sb[j] * sb[i] - sc[i] * sa[j]);

            const T rhs_i = srhs[i];
            const T rhs_j = srhs[j];

            srhs[i] = (sb[j] * rhs_i - sc[i] * rhs_j) * det;
            srhs[j] = (rhs_j * sb[i] - rhs_i * sa[j]) * det;
        }

        // Backward substitution using cyclic reduction
        active_threads = PCR_SIZE;
        for(int j = 0; j < cr_iter; j++)
        {
            __syncthreads();

            if(tid < active_threads)
            {
                const int index = stride * tid + stride / 2 - 1;
                const int left  = index - stride / 2;
                const int right = index + stride / 2;

                const T rhs_left  = (left >= 0) ? srhs[left] : static_cast<T>(0);
                const T rhs_right = (right < m) ? srhs[right] : static_cast<T>(0);

                srhs[index]
                    = (srhs[index] - sa[index] * rhs_left - sc[index] * rhs_right) / sb[index];
            }

            stride /= 2;
            active_threads *= 2;
        }

        __syncthreads();

        if(tid < m)
        {
            B[ldb * bid + tid] = srhs[tid];
        }
        if(tid + BLOCKSIZE < m)
        {
            B[ldb * bid + tid + BLOCKSIZE] = srhs[tid + BLOCKSIZE];
        }
    }
}
