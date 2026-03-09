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

#pragma once

#include "rocsparse_common.hpp"

namespace rocsparse
{
    // Parallel cyclic reduction algorithm
    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t M, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_single_wavefront_kernel(rocsparse_int m,
                                                  rocsparse_int batch_count,
                                                  rocsparse_int batch_stride,
                                                  const T* __restrict__ dl,
                                                  const T* __restrict__ d,
                                                  const T* __restrict__ du,
                                                  T* __restrict__ x,
                                                  T* __restrict__ temp_a,
                                                  T* __restrict__ temp_b,
                                                  T* __restrict__ temp_c,
                                                  T* __restrict__ temp_x,
                                                  T* __restrict__ temp_cprime,
                                                  T* __restrict__ temp_xprime)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        int iter   = static_cast<int>(rocsparse::log2(M / 2));
        int stride = 1;

        T a[M / WF_SIZE];
        T b[M / WF_SIZE];
        T c[M / WF_SIZE];
        T xt[M / WF_SIZE];

        for(int i = 0; i < M / WF_SIZE; i++)
        {
            a[i] = dl[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            b[i] = d[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            c[i] = du[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            xt[i] = x[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
        }

        for(int it = 0; it < iter; it++)
        {
            for(int i = 0; i < M / WF_SIZE; i++)
            {
                const int right = lid + stride;
                const int left = lid - stride;

                T a_left = shfl_up(a[i], stride, WF_SIZE);
                T b_left = shfl_up(b[i], stride, WF_SIZE);
                T c_left = shfl_up(c[i], stride, WF_SIZE);
                T xt_left = shfl_up(xt[i], stride, WF_SIZE);

                if(left < 0)
                {
                    a_left = static_cast<T>(0);
                    b_left = static_cast<T>(0);
                    c_left = static_cast<T>(0);
                    xt_left = static_cast<T>(0);
                }

                T a_right = shfl_down(a[i], stride, WF_SIZE);
                T b_right = shfl_down(b[i], stride, WF_SIZE);
                T c_right = shfl_down(c[i], stride, WF_SIZE);
                T xt_right = shfl_down(xt[i], stride, WF_SIZE);

                if(right > (WF_SIZE - 1))
                {
                    a_right = static_cast<T>(0);
                    b_right = static_cast<T>(0);
                    c_right = static_cast<T>(0);
                    xt_right = static_cast<T>(0);
                }

                const T k1 = (left >= 0) ? a[i] / b_left : static_cast<T>(0);
                const T k2 = (right <= WF_SIZE - 1) ? c[i] / b_right : static_cast<T>(0);

                const T a_new = -a_left * k1;
                const T b_new = b[i] - c_left * k1 - a_right * k2;
                const T c_new = -c_right * k2;
                const T xt_new = xt[i] - xt_left * k1 - xt_right * k2;

                a[i] = a_new;
                b[i] = b_new;
                c[i] = c_new;
                xt[i] = xt_new;

                //temp_a[lid] = a_left;
                //temp_b[lid] = b_left;
                //temp_c[lid] = c_left;
                //temp_x[lid] = xt_left;
            }

            stride <<= 1; //stride *= 2;
        }

        for(int i = 0; i < M / WF_SIZE; i++)
        {
            // Solve 2x2 systems (j = lid + stride)
            // bi ci
            // aj bj
            // 
            // det = bi * bj - aj * ci
            T aj = shfl_down(a[i], stride, WF_SIZE);
            T bj = shfl_down(b[i], stride, WF_SIZE);
            T xtj = shfl_down(xt[i], stride, WF_SIZE);

            if(lid < WF_SIZE / 2) // same as lid < stride
            {
                T   det = b[i] * bj - aj * c[i];
                det     = static_cast<T>(1) / det;

                x[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid] = (bj * xt[i] - c[i] * xtj) * det;
                x[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid + stride] = (xtj * b[i] - xt[i] * aj) * det;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t M, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_wavefront_kernel(rocsparse_int m,
                                            rocsparse_int batch_count,
                                            rocsparse_int batch_stride,
                                            const T* __restrict__ dl,
                                            const T* __restrict__ d,
                                            const T* __restrict__ du,
                                            T* __restrict__ x,
                                            T* __restrict__ temp_a,
                                            T* __restrict__ temp_b,
                                            T* __restrict__ temp_c,
                                            T* __restrict__ temp_x,
                                            T* __restrict__ temp_cprime,
                                            T* __restrict__ temp_xprime)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        int iter   = static_cast<int>(rocsparse::log2(M / 2));
        int iter2  = static_cast<int>(rocsparse::log2(WF_SIZE));
        int stride = 1;

        T a[M / WF_SIZE];
        T b[M / WF_SIZE];
        T c[M / WF_SIZE];
        T xt[M / WF_SIZE];

        T a2[M / WF_SIZE];
        T b2[M / WF_SIZE];
        T c2[M / WF_SIZE];
        T xt2[M / WF_SIZE];

        for(int i = 0; i < M / WF_SIZE; i++)
        {
            a[i] = dl[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            b[i] = d[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            c[i] = du[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
            xt[i] = x[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid];
        }

        for(int it = 0; it < iter2; it++)
        {
            for(int i = 0; i < M / WF_SIZE; i++)
            {
                const int right = lid + stride;
                const int left = lid - stride;

                const T a_left_patch = (i > 0) ? shfl(a[i - 1], WF_SIZE + left, WF_SIZE) : static_cast<T>(0);
                const T b_left_patch = (i > 0) ? shfl(b[i - 1], WF_SIZE + left, WF_SIZE) : static_cast<T>(0);
                const T c_left_patch = (i > 0) ? shfl(c[i - 1], WF_SIZE + left, WF_SIZE) : static_cast<T>(0);
                const T xt_left_patch = (i > 0) ? shfl(xt[i - 1], WF_SIZE + left, WF_SIZE) : static_cast<T>(0);

                const T a_right_patch = (i < (M / WF_SIZE - 1)) ? shfl(a[i + 1], right - WF_SIZE, WF_SIZE) : static_cast<T>(0);
                const T b_right_patch = (i < (M / WF_SIZE - 1)) ? shfl(b[i + 1], right - WF_SIZE, WF_SIZE) : static_cast<T>(0);
                const T c_right_patch = (i < (M / WF_SIZE - 1)) ? shfl(c[i + 1], right - WF_SIZE, WF_SIZE) : static_cast<T>(0);
                const T xt_right_patch = (i < (M / WF_SIZE - 1)) ? shfl(xt[i + 1], right - WF_SIZE, WF_SIZE) : static_cast<T>(0);

                T a_left = shfl_up(a[i], stride, WF_SIZE);
                T b_left = shfl_up(b[i], stride, WF_SIZE);
                T c_left = shfl_up(c[i], stride, WF_SIZE);
                T xt_left = shfl_up(xt[i], stride, WF_SIZE);

                if(left < 0)
                {
                    a_left = a_left_patch;
                    b_left = b_left_patch;
                    c_left = c_left_patch;
                    xt_left = xt_left_patch;
                }

                T a_right = shfl_down(a[i], stride, WF_SIZE);
                T b_right = shfl_down(b[i], stride, WF_SIZE);
                T c_right = shfl_down(c[i], stride, WF_SIZE);
                T xt_right = shfl_down(xt[i], stride, WF_SIZE);

                if(right > (WF_SIZE - 1))
                {
                    a_right = a_right_patch;
                    b_right = b_right_patch;
                    c_right = c_right_patch;
                    xt_right = xt_right_patch;
                }

                const T k1 = (b_left != static_cast<T>(0)) ? a[i] / b_left : static_cast<T>(0);
                const T k2 = (b_right != static_cast<T>(0)) ? c[i] / b_right : static_cast<T>(0);

                const T a_new = -a_left * k1;
                const T b_new = b[i] - c_left * k1 - a_right * k2;
                const T c_new = -c_right * k2;
                const T xt_new = xt[i] - xt_left * k1 - xt_right * k2;

                a2[i] = a_new;
                b2[i] = b_new;
                c2[i] = c_new;
                xt2[i] = xt_new;

                // temp_a[WF_SIZE * i + lid] = a_new;
                // temp_b[WF_SIZE * i + lid] = b_new;
                // temp_c[WF_SIZE * i + lid] = c_new;
                // temp_x[WF_SIZE * i + lid] = xt_new;
            }

            for(int i = 0; i < M / WF_SIZE; i++)
            {
                a[i] = a2[i];
                b[i] = b2[i];
                c[i] = c2[i];
                xt[i] = xt2[i];
            }

            stride <<= 1; //stride *= 2;
        }

        // stride now equals M / 2
        // assert(stride == M / 2);
        //assert(stride == 2 * WF_SIZE);


        // for(int i = 0; i < M / WF_SIZE; i++)
        // {
        //     // temp_a[WF_SIZE * i + lid] = iter2;
        //     // temp_b[WF_SIZE * i + lid] = stride;
        //     temp_a[WF_SIZE * i + lid] = a[i];
        //     temp_b[WF_SIZE * i + lid] = b[i];
        //     temp_c[WF_SIZE * i + lid] = c[i];
        //     temp_x[WF_SIZE * i + lid] = xt[i];
        // }

        // if(lid == 0)
        // {
        //     for(int i = 0; i < WF_SIZE; i++)
        //     {
        //         temp_x[WF_SIZE * i + lid] = a[i];
        //     }
        // }

        // Finish with thomas algorithm
        T c_prime[M / WF_SIZE];
        T x_prime[M / WF_SIZE];

        // Forward sweep
        c_prime[0] = c[0] / b[0];
        for(int i = 1; i < M / WF_SIZE - 1; i++)
        {
            T num       = c[i];
            T denom     = b[i] - a[i] * c_prime[i - 1];
            c_prime[i] = num / denom;
        }

        x_prime[0] = xt[0] / b[0];
        for(int i = 1; i < M / WF_SIZE; i++)
        {
            T num      = xt[i] - a[i] * x_prime[i - 1];
            T denom    = b[i] - a[i] * c_prime[i - 1];
            x_prime[i] = num / denom;
        }

        // Backward sweep
        xt[(M / WF_SIZE) - 1] = x_prime[(M / WF_SIZE) - 1];
        for(int i = (M / WF_SIZE) - 2; i >= 0; i--)
        {
            xt[i] = x_prime[i] - c_prime[i] * xt[i + 1];
        }

        // Write results to output
        x[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * ((M / WF_SIZE) - 1) + lid] = xt[(M / WF_SIZE) - 1];//x_prime[(M / WF_SIZE) - 1];
        for(int i = (M / WF_SIZE) - 2; i >= 0; i--)
        {
            x[batch_stride * ((BLOCKSIZE / WF_SIZE) * bid + wid) + WF_SIZE * i + lid] = xt[i];
        }
    }
















































    // Cyclic reduction + parallel cyclic reduction algorithms based on paper
    // "Fast Tridiagonal Solvers on the GPU" by Yao Zhang, Jonathan Cohen, and John Owens
    //
    // Matrix has form:
    //
    // [ b0 c0 0  0  0  0  0  0 ]
    // [ a1 b1 c1 0  0  0  0  0 ]
    // [ 0  a2 b2 c2 0  0  0  0 ]
    // [ 0  0  a3 b3 c3 0  0  0 ]
    // [ 0  0  0  a4 b4 c4 0  0 ]
    // [ 0  0  0  0  a5 b5 c5 0 ]
    // [ 0  0  0  0  0  a6 b6 c6]
    // [ 0  0  0  0  0  0  a7 b7]

    // Cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_cr_pow2_shared_kernel(rocsparse_int m,
                                                          rocsparse_int batch_count,
                                                          rocsparse_int batch_stride,
                                                          const T* __restrict__ dl,
                                                          const T* __restrict__ d,
                                                          const T* __restrict__ du,
                                                          T* __restrict__ x)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;
        rocsparse_int gid = tid + batch_stride * bid;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE));
        rocsparse_int stride = 1;
        rocsparse_int i      = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];
        __shared__ T sx[2 * BLOCKSIZE];

        sa[tid]               = dl[gid];
        sa[tid + BLOCKSIZE]   = dl[gid + BLOCKSIZE];
        sb[tid]               = d[gid];
        sb[tid + BLOCKSIZE]   = d[gid + BLOCKSIZE];
        sc[tid]               = du[gid];
        sc[tid + BLOCKSIZE]   = du[gid + BLOCKSIZE];
        srhs[tid]             = x[gid];
        srhs[tid + BLOCKSIZE] = x[gid + BLOCKSIZE];

        // The first entry of the lower diagonal and the last entry of the upper
        // diagonal should be treated as zero
        if(tid == 0)
        {
            sa[tid] = 0;
        }
        if(tid == (BLOCKSIZE - 1))
        {
            sc[tid + BLOCKSIZE] = 0;
        }

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(rocsparse_int j = 0; j < iter; j++)
        {
            stride <<= 1; //stride *= 2;

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index]   = sb[index] - sc[left] * k1 - sa[right] * k2;
                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
                sa[index]   = -sa[left] * k1;
                sc[index]   = -sc[right] * k2;
            }

            i >>= 1; //i /= 2;

            __syncthreads();
        }

        if(tid == 0)
        {
            // Solve 2x2 system
            rocsparse_int i   = stride - 1;
            rocsparse_int j   = 2 * stride - 1;
            T             det = sb[j] * sb[i] - sc[i] * sa[j];
            det               = static_cast<T>(1) / det;

            sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
            sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
        }

        // Backward substitution using cyclic reduction
        i = 2;
        for(rocsparse_int j = 0; j < iter; j++)
        {
            __syncthreads();

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride / 2 - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(left < 0)
                {
                    sx[index] = (srhs[index] - sc[index] * sx[right]) / sb[index];
                }
                else
                {
                    sx[index]
                        = (srhs[index] - sa[index] * sx[left] - sc[index] * sx[right]) / sb[index];
                }
            }

            stride >>= 1; //stride /= 2;
            i <<= 1; //i *= 2;
        }

        __syncthreads();

        x[gid]             = sx[tid];
        x[gid + BLOCKSIZE] = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_pow2_shared_kernel(rocsparse_int m,
                                                           rocsparse_int batch_count,
                                                           rocsparse_int batch_stride,
                                                           const T* __restrict__ dl,
                                                           const T* __restrict__ d,
                                                           const T* __restrict__ du,
                                                           T* __restrict__ x)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;
        rocsparse_int gid = tid + batch_stride * bid;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE + 1];
        __shared__ T sb[BLOCKSIZE + 1];
        __shared__ T sc[BLOCKSIZE + 1];
        __shared__ T srhs[BLOCKSIZE + 1];
        __shared__ T sx[BLOCKSIZE + 1];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = dl[gid];
        sb[tid]   = d[gid];
        sc[tid]   = du[gid];
        srhs[tid] = x[gid];

        // The first entry of the lower diagonal and the last entry of the upper
        // diagonal should be treated as zero
        if(tid == 0)
        {
            sa[tid] = static_cast<T>(0);
        }
        if(tid == (BLOCKSIZE - 1))
        {
            sc[tid] = static_cast<T>(0);
        }

        __syncthreads();

        for(rocsparse_int j = 0; j < iter; j++)
        {
            rocsparse_int right = tid + stride;
            if(right >= BLOCKSIZE)
                right = BLOCKSIZE - 1;

            rocsparse_int left = tid - stride;
            if(left < 0)
                left = 0;

            T k1 = sa[tid] / sb[left];
            T k2 = sc[tid] / sb[right];

            T tb   = sb[tid] - sc[left] * k1 - sa[right] * k2;
            T trhs = srhs[tid] - srhs[left] * k1 - srhs[right] * k2;
            T ta   = -sa[left] * k1;
            T tc   = -sc[right] * k2;

            __syncthreads();

            sb[tid]   = tb;
            srhs[tid] = trhs;
            sa[tid]   = ta;
            sc[tid]   = tc;

            stride <<= 1; //stride *= 2;

            __syncthreads();
        }

        if(tid < BLOCKSIZE / 2)
        {
            // Solve 2x2 systems
            rocsparse_int i   = tid;
            rocsparse_int j   = tid + stride;
            T             det = sb[j] * sb[i] - sc[i] * sa[j];
            det               = static_cast<T>(1) / det;

            sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
            sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
        }

        __syncthreads();

        x[gid] = sx[tid];
    }

    // Combined Parallel cyclic reduction and cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_crpcr_pow2_shared_kernel(rocsparse_int m,
                                                             rocsparse_int batch_count,
                                                             rocsparse_int batch_stride,
                                                             const T* __restrict__ dl,
                                                             const T* __restrict__ d,
                                                             const T* __restrict__ du,
                                                             T* __restrict__ x)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;
        rocsparse_int gid = tid + batch_stride * bid;

        rocsparse_int tot_iter = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE));
        rocsparse_int pcr_iter = static_cast<rocsparse_int>(rocsparse::log2(PCR_SIZE / 2));
        rocsparse_int cr_iter  = tot_iter - pcr_iter;
        rocsparse_int stride   = 1;
        rocsparse_int i        = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];
        __shared__ T sx[2 * BLOCKSIZE];

        // Parallel cyclic reduction shared memory
        __shared__ T spa[PCR_SIZE];
        __shared__ T spb[PCR_SIZE];
        __shared__ T spc[PCR_SIZE];
        __shared__ T sprhs[PCR_SIZE];
        __shared__ T spx[PCR_SIZE];

        // Fill cyclic reduction shared memory
        sa[tid]               = dl[gid];
        sa[tid + BLOCKSIZE]   = dl[gid + BLOCKSIZE];
        sb[tid]               = d[gid];
        sb[tid + BLOCKSIZE]   = d[gid + BLOCKSIZE];
        sc[tid]               = du[gid];
        sc[tid + BLOCKSIZE]   = du[gid + BLOCKSIZE];
        srhs[tid]             = x[gid];
        srhs[tid + BLOCKSIZE] = x[gid + BLOCKSIZE];

        // The first entry of the lower diagonal and the last entry of the upper
        // diagonal should be treated as zero
        if(tid == 0)
        {
            sa[tid] = static_cast<T>(0);
        }
        if(tid == (BLOCKSIZE - 1))
        {
            sc[tid + BLOCKSIZE] = static_cast<T>(0);
        }

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(rocsparse_int j = 0; j < cr_iter; j++)
        {
            stride <<= 1; //stride *= 2;

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index]   = sb[index] - sc[left] * k1 - sa[right] * k2;
                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
                sa[index]   = -sa[left] * k1;
                sc[index]   = -sc[right] * k2;
            }

            i >>= 1; //i /= 2;

            __syncthreads();
        }

        // Parallel cyclic reduction
        if(tid < PCR_SIZE)
        {
            spa[tid]   = sa[tid * stride + stride - 1];
            spb[tid]   = sb[tid * stride + stride - 1];
            spc[tid]   = sc[tid * stride + stride - 1];
            sprhs[tid] = srhs[tid * stride + stride - 1];
        }

        __syncthreads();

        rocsparse_int pcr_stride = 1;
        for(rocsparse_int j = 0; j < pcr_iter; j++)
        {
            T ta;
            T tb;
            T tc;
            T trhs;

            if(tid < PCR_SIZE)
            {
                rocsparse_int right = tid + pcr_stride;
                if(right >= PCR_SIZE)
                    right = PCR_SIZE - 1;

                rocsparse_int left = tid - pcr_stride;
                if(left < 0)
                    left = 0;

                T k1 = spa[tid] / spb[left];
                T k2 = spc[tid] / spb[right];

                tb   = spb[tid] - spc[left] * k1 - spa[right] * k2;
                trhs = sprhs[tid] - sprhs[left] * k1 - sprhs[right] * k2;
                ta   = -spa[left] * k1;
                tc   = -spc[right] * k2;
            }

            __syncthreads();

            if(tid < PCR_SIZE)
            {
                spb[tid]   = tb;
                sprhs[tid] = trhs;
                spa[tid]   = ta;
                spc[tid]   = tc;
            }

            pcr_stride <<= 1; //pcr_stride *= 2;

            __syncthreads();
        }

        if(tid < pcr_stride) // same as PCR_SIZE / 2
        {
            // Solve 2x2 systems
            rocsparse_int i   = tid;
            rocsparse_int j   = tid + pcr_stride;
            T             det = spb[j] * spb[i] - spc[i] * spa[j];
            det               = static_cast<T>(1) / det;

            spx[i] = (spb[j] * sprhs[i] - spc[i] * sprhs[j]) * det;
            spx[j] = (sprhs[j] * spb[i] - sprhs[i] * spa[j]) * det;
        }

        __syncthreads();

        if(tid < PCR_SIZE)
        {
            sx[tid * stride + stride - 1] = spx[tid];
        }

        // Backward substitution using cyclic reduction
        i = PCR_SIZE;
        for(rocsparse_int j = 0; j < cr_iter; j++)
        {
            __syncthreads();

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride / 2 - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(left < 0)
                {
                    sx[index] = (srhs[index] - sc[index] * sx[right]) / sb[index];
                }
                else
                {
                    sx[index]
                        = (srhs[index] - sa[index] * sx[left] - sc[index] * sx[right]) / sb[index];
                }
            }

            (stride >>= 1); //stride /= 2;
            i <<= 1; //i *= 2;
        }

        __syncthreads();

        x[gid]             = sx[tid];
        x[gid + BLOCKSIZE] = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_shared_kernel(rocsparse_int m,
                                                      rocsparse_int batch_count,
                                                      rocsparse_int batch_stride,
                                                      const T* __restrict__ dl,
                                                      const T* __restrict__ d,
                                                      const T* __restrict__ du,
                                                      T* __restrict__ x)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int bid = hipBlockIdx_x;
        rocsparse_int gid = tid + batch_stride * bid;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T srhs[BLOCKSIZE];
        __shared__ T sx[BLOCKSIZE];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = (tid < m) ? dl[gid] : static_cast<T>(0);
        sb[tid]   = (tid < m) ? d[gid] : static_cast<T>(0);
        sc[tid]   = (tid < m) ? du[gid] : static_cast<T>(0);
        srhs[tid] = (tid < m) ? x[gid] : static_cast<T>(0);

        // The first entry of the lower diagonal and the last entry of the upper
        // diagonal should be treated as zero
        if(tid == 0)
        {
            sa[tid] = static_cast<T>(0);
        }
        if(tid == (m - 1))
        {
            sc[tid] = static_cast<T>(0);
        }

        __syncthreads();

        for(rocsparse_int j = 0; j < iter; j++)
        {
            rocsparse_int right = tid + stride;
            if(right >= m)
                right = m - 1;

            rocsparse_int left = tid - stride;
            if(left < 0)
                left = 0;

            T k1 = sa[tid] / sb[left];
            T k2 = sc[tid] / sb[right];

            T tb   = sb[tid] - sc[left] * k1 - sa[right] * k2;
            T trhs = srhs[tid] - srhs[left] * k1 - srhs[right] * k2;
            T ta   = -sa[left] * k1;
            T tc   = -sc[right] * k2;

            __syncthreads();

            sb[tid]   = tb;
            srhs[tid] = trhs;
            sa[tid]   = ta;
            sc[tid]   = tc;

            stride <<= 1; //stride *= 2;

            __syncthreads();
        }

        if(tid < BLOCKSIZE / 2)
        {
            rocsparse_int i = tid;
            rocsparse_int j = tid + stride;

            if(j < m)
            {
                // Solve 2x2 systems
                T det = sb[j] * sb[i] - sc[i] * sa[j];
                det   = static_cast<T>(1) / det;

                sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
                sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
            }
            else
            {
                // Solve 1x1 systems
                sx[i] = srhs[i] / sb[i];
            }
        }

        __syncthreads();

        if(tid < m)
        {
            x[gid] = sx[tid];
        }
    }

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_pow2_stage1_kernel(rocsparse_int stride,
                                                           rocsparse_int m,
                                                           rocsparse_int batch_count,
                                                           rocsparse_int batch_stride,
                                                           const T* __restrict__ a0,
                                                           const T* __restrict__ b0,
                                                           const T* __restrict__ c0,
                                                           const T* __restrict__ rhs0,
                                                           T* __restrict__ a1,
                                                           T* __restrict__ b1,
                                                           T* __restrict__ c1,
                                                           T* __restrict__ rhs1)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;
        rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

        const T* a0_col   = a0 + batch_stride * bidy;
        T*       a1_col   = a1 + m * bidy;
        const T* b0_col   = b0 + batch_stride * bidy;
        T*       b1_col   = b1 + m * bidy;
        const T* c0_col   = c0 + batch_stride * bidy;
        T*       c1_col   = c1 + m * bidy;
        const T* rhs0_col = rhs0 + batch_stride * bidy;
        T*       rhs1_col = rhs1 + m * bidy;

        // rocsparse_int right = gid + stride;
        // if(right >= m)
        //     right = m - 1;

        // rocsparse_int left = gid - stride;
        // if(left < 0)
        //     left = 0;

        // T k1 = a0[gid] / b0[left];
        // T k2 = c0[gid] / b0[right];
        // T k3 = rhs0_col[right];
        // T k4 = rhs0_col[left];

        // b1[gid]       = b0[gid] - c0[left] * k1 - a0[right] * k2;
        // rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
        // k3            = -a0[left];
        // k4            = -c0[right];

        // a1[gid] = k3 * k1;
        // c1[gid] = k4 * k2;
        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = (gid > 0) ? a0_col[gid] / b0_col[left] : static_cast<T>(0);
        T k2 = (gid < (m - 1)) ? c0_col[gid] / b0_col[right] : static_cast<T>(0);
        T k3 = rhs0_col[right];
        T k4 = rhs0_col[left];

        b1_col[gid]   = b0_col[gid] - c0_col[left] * k1 - a0_col[right] * k2;
        rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
        k3            = (left > 0) ? -a0_col[left] : static_cast<T>(0);
        k4            = (right < (m - 1)) ? -c0_col[right] : static_cast<T>(0);

        a1_col[gid] = k3 * k1;
        c1_col[gid] = k4 * k2;
    }

    // Cyclic reduction algorithm using shared memory to solve multiple small matrices produced from
    // stage 1 above in parallel
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_cr_pow2_stage2_kernel(rocsparse_int m,
                                                          rocsparse_int batch_count,
                                                          rocsparse_int batch_stride,
                                                          const T* __restrict__ a,
                                                          const T* __restrict__ b,
                                                          const T* __restrict__ c,
                                                          const T* __restrict__ rhs,
                                                          T* __restrict__ x)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE));
        rocsparse_int stride = 1;
        rocsparse_int i      = BLOCKSIZE;

        // Cyclic reduction shared memory
        __shared__ T sa[2 * BLOCKSIZE];
        __shared__ T sb[2 * BLOCKSIZE];
        __shared__ T sc[2 * BLOCKSIZE];
        __shared__ T srhs[2 * BLOCKSIZE];
        __shared__ T sx[2 * BLOCKSIZE];

        // sa[tid]   = a[hipGridDim_x * tid + hipBlockIdx_x];
        // sb[tid]   = b[hipGridDim_x * tid + hipBlockIdx_x];
        // sc[tid]   = c[hipGridDim_x * tid + hipBlockIdx_x];
        // srhs[tid] = rhs[hipGridDim_x * tid + hipBlockIdx_x + m * hipBlockIdx_y];
        // sx[tid]   = static_cast<T>(0);

        // sa[tid + BLOCKSIZE] = a[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        // sb[tid + BLOCKSIZE] = b[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        // sc[tid + BLOCKSIZE] = c[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        // srhs[tid + BLOCKSIZE]
        //     = rhs[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + m * hipBlockIdx_y];
        // sx[tid + BLOCKSIZE] = static_cast<T>(0);
        sa[tid]   = a[hipGridDim_x * tid + hipBlockIdx_x + m * hipBlockIdx_y];
        sb[tid]   = b[hipGridDim_x * tid + hipBlockIdx_x + m * hipBlockIdx_y];
        sc[tid]   = c[hipGridDim_x * tid + hipBlockIdx_x + m * hipBlockIdx_y];
        srhs[tid] = rhs[hipGridDim_x * tid + hipBlockIdx_x + m * hipBlockIdx_y];
        sx[tid]   = static_cast<T>(0);

        sa[tid + BLOCKSIZE]
            = a[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + m * hipBlockIdx_y];
        sb[tid + BLOCKSIZE]
            = b[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + m * hipBlockIdx_y];
        sc[tid + BLOCKSIZE]
            = c[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + m * hipBlockIdx_y];
        srhs[tid + BLOCKSIZE]
            = rhs[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + m * hipBlockIdx_y];
        sx[tid + BLOCKSIZE] = static_cast<T>(0);

        __syncthreads();

        // Forward reduction using cyclic reduction
        for(rocsparse_int j = 0; j < iter; j++)
        {
            stride <<= 1; //stride *= 2;

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(right >= 2 * BLOCKSIZE)
                {
                    right = 2 * BLOCKSIZE - 1;
                }

                T k1 = sa[index] / sb[left];
                T k2 = sc[index] / sb[right];

                sb[index]   = sb[index] - sc[left] * k1 - sa[right] * k2;
                srhs[index] = srhs[index] - srhs[left] * k1 - srhs[right] * k2;
                sa[index]   = -sa[left] * k1;
                sc[index]   = -sc[right] * k2;
            }

            i >>= 1; //i /= 2;

            __syncthreads();
        }

        if(tid == 0)
        {
            // Solve 2x2 system
            rocsparse_int i   = stride - 1;
            rocsparse_int j   = 2 * stride - 1;
            T             det = sb[j] * sb[i] - sc[i] * sa[j];
            det               = static_cast<T>(1) / det;

            sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
            sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
        }

        // Backward substitution using cyclic reduction
        i = 2;
        for(rocsparse_int j = 0; j < iter; j++)
        {
            __syncthreads();

            if(tid < i)
            {
                rocsparse_int index = stride * tid + stride / 2 - 1;
                rocsparse_int left  = index - (stride >> 1); //stride / 2;
                rocsparse_int right = index + (stride >> 1); //stride / 2;

                if(left < 0)
                {
                    sx[index] = (srhs[index] - sc[index] * sx[right]) / sb[index];
                }
                else
                {
                    sx[index]
                        = (srhs[index] - sa[index] * sx[left] - sc[index] * sx[right]) / sb[index];
                }
            }

            (stride >>= 1); //stride /= 2;
            i <<= 1; //i *= 2;
        }

        __syncthreads();

        x[hipGridDim_x * tid + hipBlockIdx_x + batch_stride * hipBlockIdx_y] = sx[tid];
        x[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + batch_stride * hipBlockIdx_y]
            = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_stage1_kernel(rocsparse_int stride,
                                                      rocsparse_int m,
                                                      rocsparse_int batch_count,
                                                      rocsparse_int batch_stride,
                                                      const T* __restrict__ a0,
                                                      const T* __restrict__ b0,
                                                      const T* __restrict__ c0,
                                                      const T* __restrict__ rhs0,
                                                      T* __restrict__ a1,
                                                      T* __restrict__ b1,
                                                      T* __restrict__ c1,
                                                      T* __restrict__ rhs1)
    {
        rocsparse_int tidx = hipThreadIdx_x;
        rocsparse_int bidx = hipBlockIdx_x;
        rocsparse_int bidy = hipBlockIdx_y;
        rocsparse_int gid  = tidx + BLOCKSIZE * bidx; // + batch_stride * bidy;

        if(gid >= m)
        {
            return;
        }

        const T* a0_col   = a0 + batch_stride * bidy;
        T*       a1_col   = a1 + m * bidy;
        const T* b0_col   = b0 + batch_stride * bidy;
        T*       b1_col   = b1 + m * bidy;
        const T* c0_col   = c0 + batch_stride * bidy;
        T*       c1_col   = c1 + m * bidy;
        const T* rhs0_col = rhs0 + batch_stride * bidy;
        T*       rhs1_col = rhs1 + m * bidy;

        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = (gid > 0) ? a0_col[gid] / b0_col[left] : static_cast<T>(0);
        T k2 = (gid < (m - 1)) ? c0_col[gid] / b0_col[right] : static_cast<T>(0);
        T k3 = rhs0_col[right];
        T k4 = rhs0_col[left];

        b1_col[gid]   = b0_col[gid] - c0_col[left] * k1 - a0_col[right] * k2;
        rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
        k3            = (left > 0) ? -a0_col[left] : static_cast<T>(0);
        k4            = (right < (m - 1)) ? -c0_col[right] : static_cast<T>(0);

        a1_col[gid] = k3 * k1;
        c1_col[gid] = k4 * k2;
    }

    // Parallel cyclic reduction algorithm using shared memory to solve multiple small matrices produced from
    // stage 1 above in parallel
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_strided_batch_pcr_stage2_kernel(rocsparse_int m,
                                                      rocsparse_int batch_count,
                                                      rocsparse_int batch_stride,
                                                      const T* __restrict__ a,
                                                      const T* __restrict__ b,
                                                      const T* __restrict__ c,
                                                      const T* __restrict__ rhs,
                                                      T* __restrict__ x)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int gid = hipGridDim_x * tid + hipBlockIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T srhs[BLOCKSIZE];
        __shared__ T sx[BLOCKSIZE];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = (gid < m) ? a[gid + m * hipBlockIdx_y]
                              : a[m - hipGridDim_x + hipBlockIdx_x + m * hipBlockIdx_y];
        sb[tid]   = (gid < m) ? b[gid + m * hipBlockIdx_y]
                              : b[m - hipGridDim_x + hipBlockIdx_x + m * hipBlockIdx_y];
        sc[tid]   = (gid < m) ? c[gid + m * hipBlockIdx_y]
                              : c[m - hipGridDim_x + hipBlockIdx_x + m * hipBlockIdx_y];
        srhs[tid] = (gid < m) ? rhs[gid + m * hipBlockIdx_y]
                              : rhs[m - hipGridDim_x + hipBlockIdx_x + m * hipBlockIdx_y];

        __syncthreads();

        for(rocsparse_int j = 0; j < iter; j++)
        {
            rocsparse_int right = tid + stride;
            if(right >= BLOCKSIZE)
                right = BLOCKSIZE - 1;
            ;

            rocsparse_int left = tid - stride;
            if(left < 0)
                left = 0;

            T k1 = sa[tid] / sb[left];
            T k2 = sc[tid] / sb[right];

            T tb   = sb[tid] - sc[left] * k1 - sa[right] * k2;
            T trhs = srhs[tid] - srhs[left] * k1 - srhs[right] * k2;
            T ta   = -sa[left] * k1;
            T tc   = -sc[right] * k2;

            __syncthreads();

            sb[tid]   = tb;
            srhs[tid] = trhs;
            sa[tid]   = ta;
            sc[tid]   = tc;

            stride <<= 1; //stride *= 2;

            __syncthreads();
        }

        if(tid < BLOCKSIZE / 2)
        {
            rocsparse_int i = tid;
            rocsparse_int j = tid + stride;

            if(j < BLOCKSIZE)
            {
                // Solve 2x2 systems
                T det = sb[j] * sb[i] - sc[i] * sa[j];
                det   = static_cast<T>(1) / det;

                sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
                sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
            }
            else
            {
                // Solve 1x1 systems
                sx[i] = srhs[i] / sb[i];
            }
        }

        __syncthreads();

        if(gid < m)
        {
            x[gid + batch_stride * hipBlockIdx_y] = sx[tid];
        }
    }
}
