/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

namespace rocsparse
{
    // Small sized problems

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
    void gtsv_nopivot_cr_pow2_shared_kernel(rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int ldb,
                                            const T* __restrict__ dl,
                                            const T* __restrict__ d,
                                            const T* __restrict__ du,
                                            T* __restrict__ B)
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

        sa[tid]               = dl[tid];
        sa[tid + BLOCKSIZE]   = dl[tid + BLOCKSIZE];
        sb[tid]               = d[tid];
        sb[tid + BLOCKSIZE]   = d[tid + BLOCKSIZE];
        sc[tid]               = du[tid];
        sc[tid + BLOCKSIZE]   = du[tid + BLOCKSIZE];
        srhs[tid]             = B[tid + ldb * hipBlockIdx_x];
        srhs[tid + BLOCKSIZE] = B[tid + BLOCKSIZE + ldb * hipBlockIdx_x];

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

        B[tid + ldb * hipBlockIdx_x]             = sx[tid];
        B[tid + BLOCKSIZE + ldb * hipBlockIdx_x] = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_pow2_shared_kernel(rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int ldb,
                                             const T* __restrict__ dl,
                                             const T* __restrict__ d,
                                             const T* __restrict__ du,
                                             T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE + 1];
        __shared__ T sb[BLOCKSIZE + 1];
        __shared__ T sc[BLOCKSIZE + 1];
        __shared__ T srhs[BLOCKSIZE + 1];
        __shared__ T sx[BLOCKSIZE + 1];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = dl[tid];
        sb[tid]   = d[tid];
        sc[tid]   = du[tid];
        srhs[tid] = B[tid + ldb * hipBlockIdx_x];

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

        B[tid + ldb * hipBlockIdx_x] = sx[tid];
    }

    // Combined Parallel cyclic reduction and cyclic reduction algorithm using shared memory
    template <uint32_t BLOCKSIZE, uint32_t PCR_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_crpcr_pow2_shared_kernel(rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int ldb,
                                               const T* __restrict__ dl,
                                               const T* __restrict__ d,
                                               const T* __restrict__ du,
                                               T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

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
        sa[tid]               = dl[tid];
        sa[tid + BLOCKSIZE]   = dl[tid + BLOCKSIZE];
        sb[tid]               = d[tid];
        sb[tid + BLOCKSIZE]   = d[tid + BLOCKSIZE];
        sc[tid]               = du[tid];
        sc[tid + BLOCKSIZE]   = du[tid + BLOCKSIZE];
        srhs[tid]             = B[tid + ldb * hipBlockIdx_x];
        srhs[tid + BLOCKSIZE] = B[tid + BLOCKSIZE + ldb * hipBlockIdx_x];

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

        B[tid + ldb * hipBlockIdx_x]             = sx[tid];
        B[tid + BLOCKSIZE + ldb * hipBlockIdx_x] = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_shared_kernel(rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ dl,
                                        const T* __restrict__ d,
                                        const T* __restrict__ du,
                                        T* __restrict__ B)
    {
        rocsparse_int tid = hipThreadIdx_x;

        rocsparse_int iter   = static_cast<rocsparse_int>(rocsparse::log2(BLOCKSIZE / 2));
        rocsparse_int stride = 1;

        // Parallel cyclic reduction shared memory
        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T srhs[BLOCKSIZE];
        __shared__ T sx[BLOCKSIZE];

        // Fill parallel cyclic reduction shared memory
        sa[tid]   = (tid < m) ? dl[tid] : static_cast<T>(0);
        sb[tid]   = (tid < m) ? d[tid] : static_cast<T>(0);
        sc[tid]   = (tid < m) ? du[tid] : static_cast<T>(0);
        srhs[tid] = (tid < m) ? B[tid + ldb * hipBlockIdx_x] : static_cast<T>(0);

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
            B[tid + ldb * hipBlockIdx_x] = sx[tid];
        }
    }

    // Medium sized problems

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_pow2_stage1_n_kernel(rocsparse_int stride,
                                               rocsparse_int m,
                                               rocsparse_int n,
                                               rocsparse_int ldb,
                                               const T* __restrict__ a0,
                                               const T* __restrict__ b0,
                                               const T* __restrict__ c0,
                                               const T* __restrict__ rhs0,
                                               T* __restrict__ a1,
                                               T* __restrict__ b1,
                                               T* __restrict__ c1,
                                               T* __restrict__ rhs1)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int gid = tid + BLOCKSIZE * hipBlockIdx_x;

        const T* rhs0_col = rhs0 + ldb * hipBlockIdx_y;
        T*       rhs1_col = rhs1 + m * hipBlockIdx_y;

        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = a0[gid] / b0[left];
        T k2 = c0[gid] / b0[right];
        T k3 = -a0[left];
        T k4 = -c0[right];

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        k3            = rhs0_col[right];
        k4            = rhs0_col[left];
        rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_stage1_n_kernel(rocsparse_int stride,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int ldb,
                                          const T* __restrict__ a0,
                                          const T* __restrict__ b0,
                                          const T* __restrict__ c0,
                                          const T* __restrict__ rhs0,
                                          T* __restrict__ a1,
                                          T* __restrict__ b1,
                                          T* __restrict__ c1,
                                          T* __restrict__ rhs1)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int gid = tid + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= m)
        {
            return;
        }

        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = a0[gid] / b0[left];
        T k2 = c0[gid] / b0[right];
        T k3 = -a0[left];
        T k4 = -c0[right];

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        const T* rhs0_col = rhs0 + ldb * hipBlockIdx_y;
        T*       rhs1_col = rhs1 + m * hipBlockIdx_y;

        k3            = rhs0_col[right];
        k4            = rhs0_col[left];
        rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
    }

    // Cyclic reduction algorithm using shared memory to solve multiple small matrices produced from
    // stage 1 above in parallel
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_cr_pow2_stage2_kernel(rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int ldb,
                                            const T* __restrict__ a,
                                            const T* __restrict__ b,
                                            const T* __restrict__ c,
                                            const T* __restrict__ rhs,
                                            T* __restrict__ B)
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

        sa[tid]   = a[hipGridDim_x * tid + hipBlockIdx_x];
        sb[tid]   = b[hipGridDim_x * tid + hipBlockIdx_x];
        sc[tid]   = c[hipGridDim_x * tid + hipBlockIdx_x];
        srhs[tid] = rhs[hipGridDim_x * tid + hipBlockIdx_x + m * hipBlockIdx_y];
        sx[tid]   = static_cast<T>(0);

        sa[tid + BLOCKSIZE] = a[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        sb[tid + BLOCKSIZE] = b[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
        sc[tid + BLOCKSIZE] = c[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x];
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

        B[hipGridDim_x * tid + hipBlockIdx_x + ldb * hipBlockIdx_y] = sx[tid];
        B[hipGridDim_x * (tid + BLOCKSIZE) + hipBlockIdx_x + ldb * hipBlockIdx_y]
            = sx[tid + BLOCKSIZE];
    }

    // Parallel cyclic reduction algorithm using shared memory to solve multiple small matrices produced from
    // stage 1 above in parallel
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_stage2_kernel(rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        const T* __restrict__ c,
                                        const T* __restrict__ rhs,
                                        T* __restrict__ B)
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
        sa[tid]   = (gid < m) ? a[gid] : a[m - hipGridDim_x + hipBlockIdx_x];
        sb[tid]   = (gid < m) ? b[gid] : b[m - hipGridDim_x + hipBlockIdx_x];
        sc[tid]   = (gid < m) ? c[gid] : c[m - hipGridDim_x + hipBlockIdx_x];
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
            B[gid + ldb * hipBlockIdx_y] = sx[tid];
        }
    }

    // Large size problems

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_pow2_stage1_kernel(rocsparse_int stride,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int ldb,
                                             const T* __restrict__ a0,
                                             const T* __restrict__ b0,
                                             const T* __restrict__ c0,
                                             const T* __restrict__ rhs0,
                                             T* __restrict__ a1,
                                             T* __restrict__ b1,
                                             T* __restrict__ c1,
                                             T* __restrict__ rhs1)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = a0[gid] / b0[left];
        T k2 = c0[gid] / b0[right];
        T k3 = -a0[left];
        T k4 = -c0[right];

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        for(rocsparse_int i = 0; i < n; i++)
        {
            const T* rhs0_col = rhs0 + ldb * i;
            T*       rhs1_col = rhs1 + m * i;

            k3            = rhs0_col[right];
            k4            = rhs0_col[left];
            rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
        }
    }

    // Parallel cyclic reduction algorithm using global memory for partitioning large matrices into
    // multiple small ones that can be solved in parallel in stage 2
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_pcr_stage1_kernel(rocsparse_int stride,
                                        rocsparse_int m,
                                        rocsparse_int n,
                                        rocsparse_int ldb,
                                        const T* __restrict__ a0,
                                        const T* __restrict__ b0,
                                        const T* __restrict__ c0,
                                        const T* __restrict__ rhs0,
                                        T* __restrict__ a1,
                                        T* __restrict__ b1,
                                        T* __restrict__ c1,
                                        T* __restrict__ rhs1)
    {
        rocsparse_int tid = hipThreadIdx_x;
        rocsparse_int gid = tid + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= m)
        {
            return;
        }

        rocsparse_int right = gid + stride;
        if(right >= m)
            right = m - 1;

        rocsparse_int left = gid - stride;
        if(left < 0)
            left = 0;

        T k1 = a0[gid] / b0[left];
        T k2 = c0[gid] / b0[right];
        T k3 = -a0[left];
        T k4 = -c0[right];

        b1[gid] = b0[gid] - c0[left] * k1 - a0[right] * k2;
        a1[gid] = k3 * k1;
        c1[gid] = k4 * k2;

        for(rocsparse_int i = 0; i < n; i++)
        {
            const T* rhs0_col = rhs0 + ldb * i;
            T*       rhs1_col = rhs1 + m * i;

            k3            = rhs0_col[right];
            k4            = rhs0_col[left];
            rhs1_col[gid] = rhs0_col[gid] - k4 * k1 - k3 * k2;
        }
    }

    // See Nikolai Sakharnykh. Efficient tridiagonal solvers for adi methods and fluid simulation.
    // In NVIDIA GPU Technology Conference 2010, September 2010.
    template <uint32_t BLOCKSIZE, uint32_t SYSTEM_SIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_thomas_pow2_stage2_kernel(rocsparse_int stride,
                                                rocsparse_int m,
                                                rocsparse_int n,
                                                rocsparse_int ldb,
                                                const T* __restrict__ a0,
                                                const T* __restrict__ b0,
                                                const T* __restrict__ c0,
                                                const T* __restrict__ rhs0,
                                                T* __restrict__ a1,
                                                T* __restrict__ b1,
                                                T* __restrict__ c1,
                                                T* __restrict__ rhs1,
                                                T* __restrict__ B)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= stride)
        {
            return;
        }

        // Forward elimination
        c1[gid]                       = c0[gid] / b0[gid];
        rhs1[gid + m * hipBlockIdx_y] = rhs0[gid + m * hipBlockIdx_y] / b0[gid];

        for(rocsparse_int i = 1; i < SYSTEM_SIZE; i++)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int minus = stride * (i - 1) + gid;

            T k = static_cast<T>(1) / (b0[index] - c1[minus] * a0[index]);

            c1[index] = c0[index] * k;
            rhs1[index + m * hipBlockIdx_y]
                = (rhs0[index + m * hipBlockIdx_y] - rhs1[minus + m * hipBlockIdx_y] * a0[index])
                  * k;
        }

        // backward substitution
        B[stride * (SYSTEM_SIZE - 1) + gid + ldb * hipBlockIdx_y]
            = rhs1[stride * (SYSTEM_SIZE - 1) + gid + m * hipBlockIdx_y];

        for(rocsparse_int i = SYSTEM_SIZE - 2; i >= 0; i--)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int plus  = stride * (i + 1) + gid;

            B[index + ldb * hipBlockIdx_y]
                = rhs1[index + m * hipBlockIdx_y] - c1[index] * B[plus + ldb * hipBlockIdx_y];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_nopivot_thomas_stage2_kernel(rocsparse_int stride,
                                           rocsparse_int m,
                                           rocsparse_int n,
                                           rocsparse_int ldb,
                                           const T* __restrict__ a0,
                                           const T* __restrict__ b0,
                                           const T* __restrict__ c0,
                                           const T* __restrict__ rhs0,
                                           T* __restrict__ a1,
                                           T* __restrict__ b1,
                                           T* __restrict__ c1,
                                           T* __restrict__ rhs1,
                                           T* __restrict__ B)
    {
        rocsparse_int gid = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(gid >= stride)
        {
            return;
        }

        rocsparse_int system_size = (m - gid - 1) / stride + 1;

        // Forward elimination
        c1[gid]                       = c0[gid] / b0[gid];
        rhs1[gid + m * hipBlockIdx_y] = rhs0[gid + m * hipBlockIdx_y] / b0[gid];

        for(rocsparse_int i = 1; i < system_size; i++)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int minus = stride * (i - 1) + gid;

            T k = static_cast<T>(1) / (b0[index] - c1[minus] * a0[index]);

            c1[index] = c0[index] * k;
            rhs1[index + m * hipBlockIdx_y]
                = (rhs0[index + m * hipBlockIdx_y] - rhs1[minus + m * hipBlockIdx_y] * a0[index])
                  * k;
        }

        // backward substitution
        B[stride * (system_size - 1) + gid + ldb * hipBlockIdx_y]
            = rhs1[stride * (system_size - 1) + gid + m * hipBlockIdx_y];

        for(rocsparse_int i = system_size - 2; i >= 0; i--)
        {
            rocsparse_int index = stride * i + gid;
            rocsparse_int plus  = stride * (i + 1) + gid;

            B[index + ldb * hipBlockIdx_y]
                = rhs1[index + m * hipBlockIdx_y] - c1[index] * B[plus + ldb * hipBlockIdx_y];
        }
    }
}
