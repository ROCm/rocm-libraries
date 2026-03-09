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

#include <vector>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE) 
    void pcr_tiled_forward_kernel(rocsparse_int m,
                                  rocsparse_int n,
                                  rocsparse_int ldb,
                                  const T* __restrict__ dl,
                                  const T* __restrict__ d,
                                  const T* __restrict__ du,
                                  const T* __restrict__ B,
                                  T* __restrict__ dl_modified,
                                  T* __restrict__ d_modified,
                                  T* __restrict__ du_modified,
                                  T* __restrict__ B_modified,
                                  T* __restrict__ spike_lower,
                                  T* __restrict__ spike_main,
                                  T* __restrict__ spike_upper,
                                  T* __restrict__ spike_B)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        const int gid = bid * BLOCKSIZE + tid;

        // Shared memory for the tile's coefficients
        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T sd[BLOCKSIZE];

        // 1. Load data from Global to Shared Memory
        sa[tid] = (gid < m) ? dl[gid] : static_cast<T>(0);
        sb[tid] = (gid < m) ? d[gid] : static_cast<T>(1);
        sc[tid] = (gid < m) ? du[gid] : static_cast<T>(0);
        sd[tid] = (gid < m) ? B[gid] : static_cast<T>(0);
        __syncthreads();

        // 2. Perform Local PCR iterations (log2(BLOCKSIZE))
        for(int k = 1; k < BLOCKSIZE; k <<= 1)
        {
            const int left = tid - k;
            const int right = tid + k;

            const T a_i = sa[tid];
            const T b_i = sb[tid];
            const T c_i = sc[tid];
            const T d_i = sd[tid];

            const T a_left = (left >= 0) ? sa[left] : static_cast<T>(0);
            const T b_left = (left >= 0) ? sb[left] : static_cast<T>(1); 
            const T c_left = (left >= 0) ? sc[left] : static_cast<T>(0); 
            const T d_left = (left >= 0) ? sd[left] : static_cast<T>(0); 

            const T a_right = (right < BLOCKSIZE) ? sa[right] : static_cast<T>(0);
            const T b_right = (right < BLOCKSIZE) ? sb[right] : static_cast<T>(1);
            const T c_right = (right < BLOCKSIZE) ? sc[right] : static_cast<T>(0);
            const T d_right = (right < BLOCKSIZE) ? sd[right] : static_cast<T>(0);

            // Elimination math
            const T alpha = (left >= 0) ? -a_i / b_left : static_cast<T>(0);
            const T gamma = (right < BLOCKSIZE) ? -c_i / b_right : static_cast<T>(0);

            // If neighbors were out of tile, the alpha/gamma remains,
            // effectively preserving the dependency for the Global Glue phase.

            __syncthreads(); // Ensure all reads are done before writing

            sa[tid] = (left >= 0) ? alpha * a_left : a_i;
            sb[tid] = b_i + alpha * c_left + gamma * a_right;
            sc[tid] = (right < BLOCKSIZE) ? gamma * c_right : c_i;
            sd[tid] = d_i + alpha * d_left + gamma * d_right;

            __syncthreads(); // Sync for next iteration
        }

        if(gid < m)
        {
            dl_modified[gid] = sa[tid];
            d_modified[gid]  = sb[tid];
            du_modified[gid] = sc[tid];
            B_modified[gid]  = sd[tid];
        }

        // // 3. Write Interface Rows to the Global Glue System
        // // Each tile contributes its first and last rows
        if(tid == 0 || tid == BLOCKSIZE - 1)
        {
            // glue_idx: Tile 0 gives index 0,1; Tile 1 gives index 2,3...
            int spike_idx = bid * 2 + (tid == 0 ? 0 : 1);

            spike_lower[spike_idx] = sa[tid];
            spike_main[spike_idx]  = sb[tid];
            spike_upper[spike_idx] = sc[tid];
            spike_B[spike_idx]     = sd[tid];
        }


        if(tid == 0 || tid == BLOCKSIZE - 1)
        {
            if(tid == 0)
            {
                spike_main[2 * bid + 0] = sb[0];
            }

            if()
            {
                spike_main[bid + 1] = sc[1];
            }


            // glue_idx: Tile 0 gives index 0,1; Tile 1 gives index 2,3...
            //int spike_idx = bid * 2 + (tid == 0 ? 0 : 1);

            int spike_idx;
            if (tid == 0) {
                // These are the "Start" threads of each tile
                // Tile 0 -> index 0, Tile 1 -> index 1
                spike_idx = bid; 
            } else {
                // These are the "End" threads of each tile
                // Tile 0 -> index 2, Tile 1 -> index 3
                spike_idx = bid + 2; 
            }


            spike_lower[spike_idx] = sa[tid];
            spike_main[spike_idx]  = sb[tid];
            spike_upper[spike_idx] = sc[tid];
            spike_B[spike_idx]     = sd[tid];
        }

    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)  
    void spike_solver_pcr_kernel(int num_spikes, // e.g., 512
                                        const T* __restrict__ l_spike,
                                        const T* __restrict__ m_spike,
                                        const T* __restrict__ u_spike,
                                        T* __restrict__ B_spike)
    {
        const int tid = threadIdx.x;

        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T sd[BLOCKSIZE];

        // 1. Load the spike system into shared memory
        sa[tid] = (tid < num_spikes) ? l_spike[tid] : static_cast<T>(0);
        sb[tid] = (tid < num_spikes) ? m_spike[tid] : static_cast<T>(1);
        sc[tid] = (tid < num_spikes) ? u_spike[tid] : static_cast<T>(0);
        sd[tid] = (tid < num_spikes) ? B_spike[tid] : static_cast<T>(0);
        __syncthreads();

        // 2. PCR Algorithm
        for(int h = 1; h < BLOCKSIZE; h *= 2)
        {
            const int left  = tid - h;
            const int right = tid + h;

            const T a_left = (left >= 0) ? sa[left] : static_cast<T>(0);
            const T b_left = (left >= 0) ? sb[left] : static_cast<T>(1);
            const T c_left = (left >= 0) ? sc[left] : static_cast<T>(0);
            const T d_left = (left >= 0) ? sd[left] : static_cast<T>(0);

            const T a_right = (right < BLOCKSIZE) ? sa[right] : static_cast<T>(0);
            const T b_right = (right < BLOCKSIZE) ? sb[right] : static_cast<T>(1);
            const T c_right = (right < BLOCKSIZE) ? sc[right] : static_cast<T>(0);
            const T d_right = (right < BLOCKSIZE) ? sd[right] : static_cast<T>(0);

            const T a = sa[tid];
            const T b = sb[tid];
            const T c = sc[tid];
            const T d = sd[tid];

            const T k1 = (left >= 0) ? a / b_left : static_cast<T>(0);
            const T k2 = (right < BLOCKSIZE) ? c / b_right : static_cast<T>(0);

            __syncthreads(); // Wait for all threads to finish reading 'old' values

            // Update coefficients
            // If k1/k2 are 0 (out of bounds), the original values are preserved
            sb[tid] = b - k1 * c_left - k2 * a_right;
            sd[tid] = d - k1 * d_left - k2 * d_right;
            sa[tid] = -k1 * a_left;
            sc[tid] = -k2 * c_right;

            __syncthreads(); // Wait for all threads to write 'new' values
        }

        // 3. Final Solution
        // After log2(N) steps, the system is diagonalized: b_i * x_i = d_i
        if(tid < num_spikes)
        {
            B_spike[tid] = sd[tid] / sb[tid];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)  
    void backward_sweep_kernel(int m, 
                               const T* __restrict__ dl_modified,
                               const T* __restrict__ d_modified,
                               const T* __restrict__ du_modified,
                               const T* __restrict__ B_modified, 
                               const T* spike_res, 
                               T* X)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;
        int gid = bid * BLOCKSIZE + tid;

        __shared__ T sa[BLOCKSIZE];
        __shared__ T sb[BLOCKSIZE];
        __shared__ T sc[BLOCKSIZE];
        __shared__ T sd[BLOCKSIZE];

        sa[tid] = (gid < m) ? dl_modified[gid] : static_cast<T>(0);
        sb[tid] = (gid < m) ? d_modified[gid] : static_cast<T>(1);
        sc[tid] = (gid < m) ? du_modified[gid] : static_cast<T>(0);
        sd[tid] = (gid < m) ? B_modified[gid] : static_cast<T>(0);
        __syncthreads();

        const T sol_top    = spike_res[bid * 2];
        const T sol_bottom = spike_res[bid * 2 + 1];

        if (tid == 0)     sd[0] -= sa[0] * sol_top;
        if (tid == BLOCKSIZE - 1) sd[BLOCKSIZE - 1] -= sc[BLOCKSIZE - 1] * sol_bottom;
        __syncthreads();

        // 4. Sequential Thomas Algorithm (Forward Sweep)
        // We modify coefficients in shared memory to eliminate lower diagonal
        for (int i = 1; i < BLOCKSIZE; i++) 
        {
            T m_factor = sa[i] / sb[i - 1];
            sb[i] -= m_factor * sc[i - 1];
            sd[i] -= m_factor * sd[i - 1];
        }
        __syncthreads();

        // 5. Back Substitution
        // Result calculation
        T* sx = sd; // Reusing sd array for results
        sx[m - 1] /= sb[BLOCKSIZE - 1];
        for (int i = BLOCKSIZE - 2; i >= 0; i--) 
        {
            sx[i] = (sd[i] - sc[i] * sx[i + 1]) / sb[i];
        }
        __syncthreads();

        // 6. Write back to global memory
        if(gid < m)
        {
            X[gid] = sx[tid];
        }






        // if(gid >= m)
        //     return;

        // // Each internal node i needs its neighbors (i-stride) and (i+stride)
        // // In a tiled approach, the "neighbors" for the internal nodes
        // // were reduced to the spike values during the forward sweep.

        // T sol_top    = spike_res[bid * 2];
        // T sol_bottom = spike_res[bid * 2 + 1];

        // // Simple back-substitution using the local modified equations
        // // Note: In a production PCR, you'd store the intermediate
        // // forward sweep values to reconstruct x[gid] here.

        // // Example logic for intermediate rows:
        // // x[gid] = (d[gid] - l[gid]*sol_top - u[gid]*sol_bottom) / m[gid];
        // X[gid] = (d[gid] - low[gid] * sol_top - up[gid] * sol_bottom) / main[gid];
    }

}