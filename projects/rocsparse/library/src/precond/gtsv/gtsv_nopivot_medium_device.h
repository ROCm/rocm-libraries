/*! \file */
/* ************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    // bidy is the block of NUM_RHS right-hand sides handled by this call. The kernel
    // wrapper drives it from a grid-stride loop so that a grid.y clamped to the
    // hardware maximum still covers every block. It is 64 bit because it now ranges
    // over the full block count instead of the clamped grid extent, and it scales the
    // m and num_spikes column offsets below (ldb is already 64 bit).
    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    ROCSPARSE_DEVICE_ILF void gtsv_no_pivot_pcr_tiled_forward_device(int64_t       bidy,
                                                                     rocsparse_int m,
                                                                     rocsparse_int n,
                                                                     int64_t       ldb,
                                                                     rocsparse_int num_spikes,
                                                                     const T* __restrict__ dl,
                                                                     const T* __restrict__ d,
                                                                     const T* __restrict__ du,
                                                                     const T* __restrict__ B,
                                                                     T* __restrict__ dl_modified,
                                                                     T* __restrict__ d_modified,
                                                                     T* __restrict__ du_modified,
                                                                     T* __restrict__ B_modified,
                                                                     T* __restrict__ dl_spike,
                                                                     T* __restrict__ d_spike,
                                                                     T* __restrict__ du_spike,
                                                                     T* __restrict__ B_spike)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;
        const rocsparse_int gid = bid * BLOCKSIZE + tid;

        T a = (gid < m && gid != 0) ? dl[gid] : static_cast<T>(0);
        T b = (gid < m) ? d[gid] : static_cast<T>(1);
        T c = (gid < m && gid != (m - 1)) ? du[gid] : static_cast<T>(0);

        T x[NUM_RHS];
        for(int rhs = 0; rhs < NUM_RHS; ++rhs)
        {
            x[rhs] = (gid < m && (NUM_RHS * bidy + rhs) < n) ? B[ldb * (NUM_RHS * bidy + rhs) + gid]
                                                             : static_cast<T>(0);
        }

        __shared__ T a_shared[BLOCKSIZE];
        __shared__ T b_shared[BLOCKSIZE];
        __shared__ T c_shared[BLOCKSIZE];
        __shared__ T x_shared[BLOCKSIZE * NUM_RHS];

        a_shared[tid] = a;
        b_shared[tid] = b;
        c_shared[tid] = c;
        for(int rhs = 0; rhs < NUM_RHS; ++rhs)
        {
            x_shared[BLOCKSIZE * rhs + tid] = x[rhs];
        }
        __syncthreads();

        // Perform PCR iterations
        for(int k = 1; k < BLOCKSIZE; k <<= 1)
        {
            const int left  = tid - k;
            const int right = tid + k;

            const T a_left = (left >= 0) ? a_shared[left] : static_cast<T>(0);
            const T b_left = (left >= 0) ? b_shared[left] : static_cast<T>(1);
            const T c_left = (left >= 0) ? c_shared[left] : static_cast<T>(0);

            const T a_right = (right < BLOCKSIZE) ? a_shared[right] : static_cast<T>(0);
            const T b_right = (right < BLOCKSIZE) ? b_shared[right] : static_cast<T>(1);
            const T c_right = (right < BLOCKSIZE) ? c_shared[right] : static_cast<T>(0);

            const T alpha = (left >= 0) ? -a / b_left : static_cast<T>(0);
            const T gamma = (right < BLOCKSIZE) ? -c / b_right : static_cast<T>(0);

            const T a_new = (left >= 0) ? alpha * a_left : a;
            const T b_new = b + alpha * c_left + gamma * a_right;
            const T c_new = (right < BLOCKSIZE) ? gamma * c_right : c;

            T d_new[NUM_RHS];
            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                const T d_left = (left >= 0) ? x_shared[left + rhs * BLOCKSIZE] : static_cast<T>(0);
                const T d_right
                    = (right < BLOCKSIZE) ? x_shared[right + rhs * BLOCKSIZE] : static_cast<T>(0);

                d_new[rhs] = x[rhs] + alpha * d_left + gamma * d_right;
            }

            __syncthreads();

            a_shared[tid] = a_new;
            b_shared[tid] = b_new;
            c_shared[tid] = c_new;

            a = a_new;
            b = b_new;
            c = c_new;

            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                x_shared[BLOCKSIZE * rhs + tid] = d_new[rhs];

                x[rhs] = d_new[rhs];
            }

            __syncthreads();
        }

        // Write modified coefficients back to Global Memory
        if(gid < m)
        {
            dl_modified[gid] = a_shared[tid];
            d_modified[gid]  = b_shared[tid];
            du_modified[gid] = c_shared[tid];
            for(int rhs = 0; rhs < NUM_RHS; ++rhs)
            {
                if((NUM_RHS * bidy + rhs) < n)
                {
                    B_modified[m * (NUM_RHS * bidy + rhs) + gid] = x[rhs];
                }
            }
        }

        // Write Interface Rows to the Global spike System
        // Consider an 8x8 system with tile size 4. This results in a 4x4 spike system.
        // If we wrote out the spike coefficients in order this would result in a system:
        //
        // |b0 0 c0 0| |x0| |d0|
        // |0 b3 c3 0| |x3|=|d3|
        // |0 a4 b4 0| |x4| |d4|
        // |0 a7 0 b7| |x7| |d7|
        //
        // This however is not tridiagonal. Instead if we swap column 1 and 2 (assuming
        // zero indexing) we get:
        //
        // |b0 c0 0 0| |x0| |d0|
        // |0 c3 b3 0| |x4|=|d3|
        // |0 b4 a4 0| |x3| |d4|
        // |0 0 a7 b7| |x7| |d7|
        //
        // Having the spike system also be tridiagonal allows us to solve it using tridiagonal
        // solvers. In general the solution vector ordering goes from (assuiming tile size is 4):
        //
        // x = [x0, x3, x4, x7, x8, x11, x12, x15, x16....]
        //
        // to:
        //
        // x = [x0, x4, x3, x8, x7, x12, x11, x16, x15....]
        //
        // Similarly the interior neighbouring columns of the spike system should be swapped.
        if(tid == 0 || tid == BLOCKSIZE - 1)
        {
            const int first = 0;
            const int last  = hipGridDim_x - 1;
            const int row   = (tid == 0) ? (2 * hipBlockIdx_x) : (2 * hipBlockIdx_x + 1);

            // Default to 0 to maintain sparse structure
            dl_spike[row] = 0;
            d_spike[row]  = 0;
            du_spike[row] = 0;

            // Interior / Boundary Logic
            if(hipBlockIdx_x == first && tid == 0) // Row 0
            {
                d_spike[0]  = b_shared[0];
                du_spike[0] = c_shared[0];
            }
            else if(hipBlockIdx_x == last && tid == BLOCKSIZE - 1) // Final Row (2*last + 1)
            {
                dl_spike[row] = a_shared[BLOCKSIZE - 1];
                d_spike[row]  = b_shared[BLOCKSIZE - 1];
            }
            else if(tid == 0) // Start of middle/last tiles (Row 2, 4, 6...)
            {
                dl_spike[row] = b_shared[0]; // From previous tile's neighbor
                d_spike[row]  = a_shared[0];

                if(hipBlockIdx_x != hipGridDim_x - 1)
                {
                    du_spike[row] = c_shared[0];
                }
            }
            else // End of middle/first tiles (Row 1, 3, 5...)
            {
                if(hipBlockIdx_x != 0)
                {
                    dl_spike[row] = a_shared[BLOCKSIZE - 1];
                }

                d_spike[row]  = c_shared[BLOCKSIZE - 1];
                du_spike[row] = b_shared[BLOCKSIZE - 1];
            }

            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                if((NUM_RHS * bidy + rhs) < n)
                {
                    B_spike[num_spikes * (NUM_RHS * bidy + rhs) + row]
                        = x_shared[tid + rhs * BLOCKSIZE];
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_no_pivot_pcr_tiled_forward_kernel(rocsparse_int m,
                                                rocsparse_int n,
                                                int64_t       ldb,
                                                rocsparse_int num_spikes,
                                                const T* __restrict__ dl,
                                                const T* __restrict__ d,
                                                const T* __restrict__ du,
                                                const T* __restrict__ B,
                                                T* __restrict__ dl_modified,
                                                T* __restrict__ d_modified,
                                                T* __restrict__ du_modified,
                                                T* __restrict__ B_modified,
                                                T* __restrict__ dl_spike,
                                                T* __restrict__ d_spike,
                                                T* __restrict__ du_spike,
                                                T* __restrict__ B_spike)
    {
        // grid.y indexes blocks of NUM_RHS right-hand sides and is clamped to the
        // hardware maximum, so grid-stride over those blocks: one step of the loop
        // advances the right-hand side base by NUM_RHS, matching the
        // NUM_RHS * bidy + rhs indexing inside. This mirrors the small path, which
        // puts the same count on grid.x. The bound depends only on n, NUM_RHS,
        // hipBlockIdx_y and hipGridDim_y, all block uniform, so every thread of a
        // block runs the same number of iterations and the barriers inside the device
        // function stay convergent.
        // Evaluated in 64 bit: NUM_RHS is an unsigned template parameter, so a plain
        // (n - 1) / NUM_RHS would promote n - 1 to unsigned.
        const int64_t nblocks_rhs = (static_cast<int64_t>(n) - 1) / NUM_RHS + 1;

        for(int64_t bidy = hipBlockIdx_y; bidy < nblocks_rhs; bidy += hipGridDim_y)
        {
            rocsparse::gtsv_no_pivot_pcr_tiled_forward_device<BLOCKSIZE, NUM_RHS>(bidy,
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

            // The shared PCR tiles are still being read when the device function
            // returns; synchronise before the next block of right-hand sides
            // overwrites them. The tridiagonal coefficients written to dl_modified,
            // d_modified, du_modified and the spike diagonals do not depend on the
            // right-hand side, so re-writing them on every iteration is redundant but
            // harmless, exactly as it already was across the blocks of grid.y.
            __syncthreads();
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_no_pivot_spike_solver_pcr_kernel(rocsparse_int num_spikes,
                                               rocsparse_int n,
                                               const T* __restrict__ dl_spike,
                                               const T* __restrict__ d_spike,
                                               const T* __restrict__ du_spike,
                                               T* __restrict__ B_spike)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int bid = hipBlockIdx_x;

        T a = (tid < num_spikes) ? dl_spike[tid] : static_cast<T>(0);
        T b = (tid < num_spikes) ? d_spike[tid] : static_cast<T>(1);
        T c = (tid < num_spikes) ? du_spike[tid] : static_cast<T>(0);
        T x[NUM_RHS];
        for(int rhs = 0; rhs < NUM_RHS; ++rhs)
        {
            x[rhs] = (tid < num_spikes && (NUM_RHS * bid + rhs) < n)
                         ? B_spike[num_spikes * (NUM_RHS * bid + rhs) + tid]
                         : static_cast<T>(0);
        }

        __shared__ T a_shared[BLOCKSIZE];
        __shared__ T b_shared[BLOCKSIZE];
        __shared__ T c_shared[BLOCKSIZE];
        __shared__ T x_shared[BLOCKSIZE * NUM_RHS];

        a_shared[tid] = a;
        b_shared[tid] = b;
        c_shared[tid] = c;
        for(int rhs = 0; rhs < NUM_RHS; ++rhs)
        {
            x_shared[BLOCKSIZE * rhs + tid] = x[rhs];
        }
        __syncthreads();

        // PCR Algorithm
        for(int h = 1; h < BLOCKSIZE; h *= 2)
        {
            const int left  = tid - h;
            const int right = tid + h;

            const T a_left = (left >= 0) ? a_shared[left] : static_cast<T>(0);
            const T b_left = (left >= 0) ? b_shared[left] : static_cast<T>(1);
            const T c_left = (left >= 0) ? c_shared[left] : static_cast<T>(0);

            const T a_right = (right < BLOCKSIZE) ? a_shared[right] : static_cast<T>(0);
            const T b_right = (right < BLOCKSIZE) ? b_shared[right] : static_cast<T>(1);
            const T c_right = (right < BLOCKSIZE) ? c_shared[right] : static_cast<T>(0);

            const T k1 = (left >= 0) ? a / b_left : static_cast<T>(0);
            const T k2 = (right < BLOCKSIZE) ? c / b_right : static_cast<T>(0);

            const T a_new = -k1 * a_left;
            const T b_new = b - k1 * c_left - k2 * a_right;
            const T c_new = -k2 * c_right;

            T d_new[NUM_RHS];
            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                const T d_left = (left >= 0) ? x_shared[left + rhs * BLOCKSIZE] : static_cast<T>(0);
                const T d_right
                    = (right < BLOCKSIZE) ? x_shared[right + rhs * BLOCKSIZE] : static_cast<T>(0);

                d_new[rhs] = x[rhs] - d_left * k1 - d_right * k2;
            }

            __syncthreads();

            a_shared[tid] = a_new;
            b_shared[tid] = b_new;
            c_shared[tid] = c_new;

            a = a_new;
            b = b_new;
            c = c_new;

            for(int rhs = 0; rhs < NUM_RHS; rhs++)
            {
                x_shared[tid + rhs * BLOCKSIZE] = d_new[rhs];

                x[rhs] = d_new[rhs];
            }

            __syncthreads();
        }

        // Final Solution
        if(tid < num_spikes)
        {
            for(int rhs = 0; rhs < NUM_RHS; ++rhs)
            {
                if((NUM_RHS * bid + rhs) < n)
                {
                    B_spike[num_spikes * (NUM_RHS * bid + rhs) + tid]
                        = x_shared[tid + rhs * BLOCKSIZE] / b_shared[tid];
                }
            }
        }
    }

    // bidy is the block of NUM_RHS right-hand sides handled by this call, supplied by
    // the kernel wrapper from a grid-stride loop, and 64 bit because it scales the m
    // and num_spikes column offsets over the full block count (ldb is already 64 bit).
    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    ROCSPARSE_DEVICE_ILF void
        gtsv_no_pivot_pcr_tiled_backward_device(int64_t       bidy,
                                                rocsparse_int m,
                                                rocsparse_int n,
                                                int64_t       ldb,
                                                int           num_spikes,
                                                const T* __restrict__ dl_modified,
                                                const T* __restrict__ d_modified,
                                                const T* __restrict__ du_modified,
                                                const T* __restrict__ B_modified,
                                                const T* __restrict__ B_spike,
                                                T* __restrict__ B)
    {
        const rocsparse_int tid = hipThreadIdx_x;
        const rocsparse_int gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const rocsparse_int N   = hipGridDim_x;

        if(gid >= m)
            return;

        // Load the modified coefficients from the Forward phase
        const T a_mod = dl_modified[gid];
        const T b_mod = d_modified[gid];
        const T c_mod = du_modified[gid];

        const int row_left  = (hipBlockIdx_x == 0) ? 0 : (2 * hipBlockIdx_x);
        const int row_right = (hipBlockIdx_x == N - 1) ? (2 * N - 1) : (2 * hipBlockIdx_x + 1);

        for(int rhs = 0; rhs < NUM_RHS; rhs++)
        {
            if((NUM_RHS * bidy + rhs) < n)
            {
                const T d_mod = B_modified[m * (NUM_RHS * bidy + rhs) + gid];

                // These are the solved x_interface values that affect this tile
                const T x_interface_left = B_spike[num_spikes * (NUM_RHS * bidy + rhs) + row_left];
                const T x_interface_right
                    = B_spike[num_spikes * (NUM_RHS * bidy + rhs) + row_right];

                const T x_final
                    = (d_mod - (a_mod * x_interface_left) - (c_mod * x_interface_right)) / b_mod;

                // Store result to global memory
                B[ldb * (NUM_RHS * bidy + rhs) + gid] = x_final;
            }
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t NUM_RHS, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gtsv_no_pivot_pcr_tiled_backward_kernel(rocsparse_int m,
                                                 rocsparse_int n,
                                                 int64_t       ldb,
                                                 int           num_spikes,
                                                 const T* __restrict__ dl_modified,
                                                 const T* __restrict__ d_modified,
                                                 const T* __restrict__ du_modified,
                                                 const T* __restrict__ B_modified,
                                                 const T* __restrict__ B_spike,
                                                 T* __restrict__ B)
    {
        // grid.y indexes blocks of NUM_RHS right-hand sides and is clamped to the
        // hardware maximum, so grid-stride over those blocks, stepping the right-hand
        // side base by NUM_RHS. The bound is block uniform and this kernel has no
        // shared state, so no extra barrier is needed.
        // Evaluated in 64 bit: NUM_RHS is an unsigned template parameter, so a plain
        // (n - 1) / NUM_RHS would promote n - 1 to unsigned.
        const int64_t nblocks_rhs = (static_cast<int64_t>(n) - 1) / NUM_RHS + 1;

        for(int64_t bidy = hipBlockIdx_y; bidy < nblocks_rhs; bidy += hipGridDim_y)
        {
            rocsparse::gtsv_no_pivot_pcr_tiled_backward_device<BLOCKSIZE, NUM_RHS>(bidy,
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
        }
    }
}