/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_HIP_KERNELS_RSAMG_CSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_RSAMG_CSR_HPP_

#include "hip_atomics.hpp"
#include "hip_unordered_map.hpp"
#include "hip_unordered_set.hpp"
#include "hip_utils.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{
    __device__ float hash(uint64_t key)
    {
        key = (~key) + (key << 21); // key = (key << 21) - key - 1;
        key = key ^ (key >> 24);
        key = (key + (key << 3)) + (key << 8); // key * 265
        key = key ^ (key >> 14);
        key = (key + (key << 2)) + (key << 4); // key * 21
        key = key ^ (key >> 28);
        key = key + (key << 31);
        return static_cast<float>(key / (float)UINT64_MAX);
    }

    template <typename I>
    __global__ void kernel_set_omega(I nrow, int64_t global_row_offset, float* omega)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row < nrow)
        {
            omega[row] = hash(row + global_row_offset);
        }
    }

    // Determine strong influences
    template <bool         GLOBAL,
              unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              typename T,
              typename I,
              typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_strong_influences(I       nrow,
                                                  int64_t nnz,
                                                  const J* __restrict__ csr_row_ptr,
                                                  const I* __restrict__ csr_col_ind,
                                                  const T* __restrict__ csr_val,
                                                  const J* __restrict__ bnd_row_ptr,
                                                  const I* __restrict__ bnd_col_ind,
                                                  const T* __restrict__ bnd_val,
                                                  float eps,
                                                  float* __restrict__ omega,
                                                  bool* __restrict__ S)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this wavefront operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Determine minimum and maximum off-diagonal of the current row
        T min_a_ik = static_cast<T>(0);
        T max_a_ik = static_cast<T>(0);

        // Shared boolean that holds diagonal sign for each wavefront
        // where true means, the diagonal element is negative
        __shared__ bool sign[BLOCKSIZE / WFSIZE];

        J int_row_begin = csr_row_ptr[row];
        J int_row_end   = csr_row_ptr[row + 1];

        // Determine diagonal sign and min/max for interior
        for(J j = int_row_begin + lid; j < int_row_end; j += WFSIZE)
        {
            I col = csr_col_ind[j];
            T val = csr_val[j];

            if(col == row)
            {
                // Get diagonal entry sign
                sign[wid] = val < static_cast<T>(0);
            }
            else
            {
                // Get min / max entries
                min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                max_a_ik = (max_a_ik > val) ? max_a_ik : val;
            }
        }

        if(GLOBAL == true)
        {
            J gst_row_begin = bnd_row_ptr[row];
            J gst_row_end   = bnd_row_ptr[row + 1];

            // Determine diagonal sign and min/max for ghost
            for(J j = gst_row_begin + lid; j < gst_row_end; j += WFSIZE)
            {
                T val = bnd_val[j];

                min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                max_a_ik = (max_a_ik > val) ? max_a_ik : val;
            }
        }

        __threadfence_block();

        // Maximum or minimum, depending on the diagonal sign
        T cond = sign[wid] ? max_a_ik : min_a_ik;

        // Obtain extrema on all threads of the wavefront
        if(sign[wid])
        {
            wf_reduce_max<WFSIZE>(&cond);
        }
        else
        {
            wf_reduce_min<WFSIZE>(&cond);
        }

        // Threshold to check for strength of connection
        cond *= eps;

        // Fill S
        for(J j = int_row_begin + lid; j < int_row_end; j += WFSIZE)
        {
            I col = csr_col_ind[j];
            T val = csr_val[j];

            if(col != row && val < cond)
            {
                // col is strongly connected to row
                S[j] = true;

                // Increment omega, as it holds all strongly connected edges
                // of vertex col.
                // Additionally, omega holds a random number between 0 and 1 to
                // distinguish neighbor points with equal number of strong
                // connections.
                atomicAdd(&omega[col], 1.0f);
            }
        }

        if(GLOBAL == true)
        {
            J gst_row_begin = bnd_row_ptr[row];
            J gst_row_end   = bnd_row_ptr[row + 1];

            for(J j = gst_row_begin + lid; j < gst_row_end; j += WFSIZE)
            {
                I col = bnd_col_ind[j];
                T val = bnd_val[j];

                if(val < cond)
                {
                    // col is strongly connected to row
                    S[j + nnz] = true;

                    // Increment omega, as it holds all strongly connected edges
                    // of vertex col.
                    // Additionally, omega holds a random number between 0 and 1 to
                    // distinguish neighbor points with equal number of strong
                    // connections.
                    atomicAdd(&omega[col + nrow], 1.0f);
                }
            }
        }
    }

    // Mark all vertices that have not been assigned yet, as coarse
    template <typename I>
    __global__ void kernel_csr_rs_pmis_unassigned_to_coarse(I nrow,
                                                            const float* __restrict__ omega,
                                                            int* __restrict__ cf,
                                                            bool* __restrict__ workspace)
    {
        // Each thread processes a row
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // workspace keeps track, whether a vertex has been marked coarse
        // during the current iteration, or not.
        bool flag = false;

        // Check only undecided vertices
        if(cf[row] == 0)
        {
            // If this vertex has an edge, it might be a coarse one
            if(omega[row] >= 1.0f)
            {
                cf[row] = 1;

                // Keep in mind, that this vertex has been marked coarse in the
                // current iteration
                flag = true;
            }
            else
            {
                // This point does not influence any other points and thus is a
                // fine point
                cf[row] = 2;
            }
        }

        workspace[row] = flag;
    }

    // Correct previously marked vertices with respect to omega
    template <bool GLOBAL, unsigned int BLOCKSIZE, unsigned int WFSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_correct_coarse(I       nrow,
                                               int64_t nnz,
                                               const J* __restrict__ csr_row_ptr,
                                               const I* __restrict__ csr_col_ind,
                                               const J* __restrict__ gst_row_ptr,
                                               const I* __restrict__ gst_col_ind,
                                               const float* __restrict__ omega,
                                               const bool* __restrict__ S,
                                               int* __restrict__ cf,
                                               bool* __restrict__ workspace)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this wavefront operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // If this vertex has been marked coarse in the current iteration,
        // process it for further checks
        if(workspace[row])
        {
            J row_begin = csr_row_ptr[row];
            J row_end   = csr_row_ptr[row + 1];

            // Get the weight of the current row for comparison
            float omega_row = omega[row];

            // Loop over the full row to compare weights of other vertices that
            // have been marked coarse in the current iteration
            for(J j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Process only vertices that are strongly connected
                if(S[j])
                {
                    I col = csr_col_ind[j];

                    // If this vertex has been marked coarse in the current iteration,
                    // we need to check whether it is accepted as a coarse vertex or not.
                    if(workspace[col])
                    {
                        // Get the weight of the current vertex for comparison
                        float omega_col = omega[col];

                        if(omega_row > omega_col)
                        {
                            // The diagonal entry has more edges and will remain
                            // a coarse point, whereas this vertex gets reverted
                            // back to undecided, for further processing.
                            cf[col] = 0;
                        }
                        else if(omega_row < omega_col)
                        {
                            // The diagonal entry has fewer edges and gets
                            // reverted back to undecided for further processing,
                            // whereas this vertex stays
                            // a coarse one.
                            cf[row] = 0;
                        }
                    }
                }
            }

            if(GLOBAL)
            {
                row_begin = gst_row_ptr[row];
                row_end   = gst_row_ptr[row + 1];

                // Loop over the full boundary row to compare weights of other vertices that
                // have been marked coarse in the current iteration
                for(J j = row_begin + lid; j < row_end; j += WFSIZE)
                {
                    // Process only vertices that are strongly connected
                    if(S[j + nnz])
                    {
                        I col = gst_col_ind[j];

                        // If this vertex has been marked coarse in the current iteration,
                        // we need to check whether it is accepted as a coarse vertex or not.
                        if(workspace[col + nrow])
                        {
                            // Get the weight of the current ghost vertex for comparison
                            float omega_col = omega[col + nrow];

                            if(omega_row > omega_col)
                            {
                                // The entry has more edges and will remain
                                // a coarse point, whereas this boundary vertex gets reverted
                                // back to undecided, for further processing.
                                cf[col + nrow] = 0;
                            }
                            else if(omega_row < omega_col)
                            {
                                // The diagonal entry has fewer edges and gets
                                // reverted back to undecided for further processing,
                                // whereas this boundary vertex stays a coarse one.
                                cf[row] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Mark remaining edges of a coarse point to fine
    template <bool GLOBAL, unsigned int BLOCKSIZE, unsigned int WFSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_coarse_edges_to_fine(I       nrow,
                                                     int64_t nnz,
                                                     const J* __restrict__ csr_row_ptr,
                                                     const I* __restrict__ csr_col_ind,
                                                     const J* __restrict__ gst_row_ptr,
                                                     const I* __restrict__ gst_col_ind,
                                                     const bool* __restrict__ S,
                                                     int* __restrict__ cf)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this wavefront operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Process only undecided vertices
        if(cf[row] == 0)
        {
            J row_begin = csr_row_ptr[row];
            J row_end   = csr_row_ptr[row + 1];

            // Loop over all edges of this undecided vertex
            // and check, if there is a coarse point connected
            for(J j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Check, whether this edge is strongly connected to the vertex
                if(S[j])
                {
                    I col = csr_col_ind[j];

                    // If this edge is coarse, our vertex must be fine
                    if(cf[col] == 1)
                    {
                        cf[row] = 2;
                        return;
                    }
                }
            }

            if(GLOBAL)
            {
                row_begin = gst_row_ptr[row];
                row_end   = gst_row_ptr[row + 1];

                // Loop over all ghost edges of this undecided vertex
                // and check, if there is a coarse point connected
                for(J j = row_begin + lid; j < row_end; j += WFSIZE)
                {
                    // Check, whether this edge is strongly connected to the vertex
                    if(S[j + nnz])
                    {
                        I col = gst_col_ind[j];

                        // If this ghost edge is coarse, our vertex must be fine
                        if(cf[col + nrow] == 1)
                        {
                            cf[row] = 2;
                            return;
                        }
                    }
                }
            }
        }
    }

    // Check for undecided vertices
    template <unsigned int BLOCKSIZE, typename I>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_check_undecided(I nrow,
                                                const int* __restrict__ cf,
                                                bool* __restrict__ undecided)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        // Check whether current vertex is undecided
        if(cf[row] == 0)
        {
            *undecided = true;
        }
    }

    template <bool GLOBAL, unsigned int BLOCKSIZE, typename T, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_direct_interp_nnz(I       nrow,
                                             int64_t nnz,
                                             const J* __restrict__ int_csr_row_ptr,
                                             const I* __restrict__ int_csr_col_ind,
                                             const T* __restrict__ int_csr_val,
                                             const J* __restrict__ gst_csr_row_ptr,
                                             const I* __restrict__ gst_csr_col_ind,
                                             const T* __restrict__ gst_csr_val,
                                             const bool* __restrict__ S,
                                             const int* __restrict__ cf,
                                             T* __restrict__ Amin,
                                             T* __restrict__ Amax,
                                             J* __restrict__ int_row_nnz,
                                             J* __restrict__ gst_row_nnz,
                                             I* __restrict__ f2c)
    {
        // The row this thread operates on
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Coarse points generate a single entry
        if(cf[row] == 1)
        {
            // Set coarse flag
            f2c[row] = int_row_nnz[row] = 1;

            if(GLOBAL == true)
            {
                gst_row_nnz[row] = 0;
            }
        }
        else
        {
            // Set non-coarse flag
            f2c[row] = 0;

            I int_nnz = 0;

            T amin = static_cast<T>(0);
            T amax = static_cast<T>(0);

            J row_begin = int_csr_row_ptr[row];
            J row_end   = int_csr_row_ptr[row + 1];

            // Loop over the full row and determine minimum and maximum
            for(J j = row_begin; j < row_end; ++j)
            {
                // Process only vertices that are strongly connected
                if(S[j])
                {
                    I col = int_csr_col_ind[j];

                    // Process only coarse points
                    if(cf[col] == 1)
                    {
                        T val = int_csr_val[j];

                        amin = (amin < val) ? amin : val;
                        amax = (amax > val) ? amax : val;
                    }
                }
            }

            // Ghost part
            if(GLOBAL == true)
            {
                J gst_row_begin = gst_csr_row_ptr[row];
                J gst_row_end   = gst_csr_row_ptr[row + 1];

                // Loop over the ghost part of the row and determine minimum and maximum
                for(J j = gst_row_begin; j < gst_row_end; ++j)
                {
                    // Process only vertices that are strongly connected
                    if(S[j + nnz])
                    {
                        I col = gst_csr_col_ind[j];

                        // Process only coarse points
                        if(cf[col + nrow] == 1)
                        {
                            T val = gst_csr_val[j];

                            amin = (amin < val) ? amin : val;
                            amax = (amax > val) ? amax : val;
                        }
                    }
                }
            }

            Amin[row] = amin = amin * static_cast<T>(0.2);
            Amax[row] = amax = amax * static_cast<T>(0.2);

            // Loop over the full row to count eligible entries
            for(J j = row_begin; j < row_end; ++j)
            {
                // Process only vertices that are strongly connected
                if(S[j] == true)
                {
                    I col = int_csr_col_ind[j];

                    // Process only coarse points
                    if(cf[col] == 1)
                    {
                        T val = int_csr_val[j];

                        // If conditions are fulfilled, count up row nnz
                        if(val <= amin || val >= amax)
                        {
                            ++int_nnz;
                        }
                    }
                }
            }

            // Write row nnz back to global memory
            int_row_nnz[row] = int_nnz;

            // Ghost part
            if(GLOBAL == true)
            {
                I gst_nnz = 0;

                J gst_row_begin = gst_csr_row_ptr[row];
                J gst_row_end   = gst_csr_row_ptr[row + 1];

                // Loop over the full ghost row to count eligible entries
                for(J j = gst_row_begin; j < gst_row_end; ++j)
                {
                    // Process only vertices that are strongly connected
                    if(S[j + nnz] == true)
                    {
                        I col = gst_csr_col_ind[j];

                        // Process only coarse points
                        if(cf[col + nrow] == 1)
                        {
                            T val = gst_csr_val[j];

                            // If conditions are fulfilled, count up row nnz
                            if(val <= amin || val >= amax)
                            {
                                ++gst_nnz;
                            }
                        }
                    }
                }

                // Write row nnz back to global memory
                gst_row_nnz[row] = gst_nnz;
            }
        }
    }

    template <bool GLOBAL, unsigned int BLOCKSIZE, typename T, typename I, typename J, typename K>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_direct_interp_fill(I       nrow,
                                              int64_t nnz,
                                              const J* __restrict__ csr_row_ptr,
                                              const I* __restrict__ csr_col_ind,
                                              const T* __restrict__ csr_val,
                                              const J* __restrict__ gst_csr_row_ptr,
                                              const I* __restrict__ gst_csr_col_ind,
                                              const T* __restrict__ gst_csr_val,
                                              const J* __restrict__ prolong_csr_row_ptr,
                                              I* __restrict__ prolong_csr_col_ind,
                                              T* __restrict__ prolong_csr_val,
                                              const J* __restrict__ gst_prolong_csr_row_ptr,
                                              K* __restrict__ gst_prolong_csr_col_ind,
                                              T* __restrict__ gst_prolong_csr_val,
                                              const bool* __restrict__ S,
                                              const int* __restrict__ cf,
                                              const T* __restrict__ Amin,
                                              const T* __restrict__ Amax,
                                              const I* __restrict__ f2c,
                                              const K* __restrict__ l2g)
    {
        // The row this thread operates on
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // The row of P this thread operates on
        I row_P = prolong_csr_row_ptr[row];

        // If this is a coarse point, we can fill P and return
        if(cf[row] == 1)
        {
            prolong_csr_col_ind[row_P] = f2c[row];
            prolong_csr_val[row_P]     = static_cast<T>(1);

            return;
        }

        T diag  = static_cast<T>(0);
        T a_num = static_cast<T>(0), a_den = static_cast<T>(0);
        T b_num = static_cast<T>(0), b_den = static_cast<T>(0);
        T d_neg = static_cast<T>(0), d_pos = static_cast<T>(0);

        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Loop over the full row
        for(J j = row_begin; j < row_end; ++j)
        {
            I col = csr_col_ind[j];
            T val = csr_val[j];

            // Do not process the vertex itself
            if(col == row)
            {
                diag = val;
                continue;
            }

            if(val < static_cast<T>(0))
            {
                a_num += val;

                // Only process vertices that are strongly connected and coarse
                if(S[j] && cf[col] == 1)
                {
                    a_den += val;

                    if(val > Amin[row])
                    {
                        d_neg += val;
                    }
                }
            }
            else
            {
                b_num += val;

                // Only process vertices that are strongly connected and coarse
                if(S[j] && cf[col] == 1)
                {
                    b_den += val;

                    if(val < Amax[row])
                    {
                        d_pos += val;
                    }
                }
            }
        }

        // Ghost part
        if(GLOBAL == true)
        {
            J ghost_row_begin = gst_csr_row_ptr[row];
            J ghost_row_end   = gst_csr_row_ptr[row + 1];

            for(J j = ghost_row_begin; j < ghost_row_end; ++j)
            {
                I col = gst_csr_col_ind[j];
                T val = gst_csr_val[j];

                if(val < static_cast<T>(0))
                {
                    a_num += val;

                    if(S[j + nnz] && cf[col + nrow] == 1)
                    {
                        a_den += val;

                        if(val > Amin[row])
                        {
                            d_neg += val;
                        }
                    }
                }
                else
                {
                    b_num += val;

                    if(S[j + nnz] && cf[col + nrow] == 1)
                    {
                        b_den += val;

                        if(val < Amax[row])
                        {
                            d_pos += val;
                        }
                    }
                }
            }
        }

        T cf_neg = static_cast<T>(1);
        T cf_pos = static_cast<T>(1);

        if(abs(a_den - d_neg) > 1e-32)
        {
            cf_neg = a_den / (a_den - d_neg);
        }

        if(abs(b_den - d_pos) > 1e-32)
        {
            cf_pos = b_den / (b_den - d_pos);
        }

        if(b_num > static_cast<T>(0) && abs(b_den) < 1e-32)
        {
            diag += b_num;
        }

        T alpha = abs(a_den) > 1e-32 ? -cf_neg * a_num / (diag * a_den) : static_cast<T>(0);
        T beta  = abs(b_den) > 1e-32 ? -cf_pos * b_num / (diag * b_den) : static_cast<T>(0);

        // Loop over the full row to fill eligible entries
        for(J j = row_begin; j < row_end; ++j)
        {
            // Process only vertices that are strongly connected
            if(S[j] == true)
            {
                I col = csr_col_ind[j];
                T val = csr_val[j];

                // Process only coarse points
                if(cf[col] == 1)
                {
                    if(val > Amin[row] && val < Amax[row])
                    {
                        continue;
                    }

                    // Fill P
                    prolong_csr_col_ind[row_P] = f2c[col];
                    prolong_csr_val[row_P]     = (val < static_cast<T>(0) ? alpha : beta) * val;
                    ++row_P;
                }
            }
        }

        // Ghost part
        if(GLOBAL == true)
        {
            J ghost_row_P     = gst_prolong_csr_row_ptr[row];
            J ghost_row_begin = gst_csr_row_ptr[row];
            J ghost_row_end   = gst_csr_row_ptr[row + 1];

            for(J j = ghost_row_begin; j < ghost_row_end; ++j)
            {
                I col = gst_csr_col_ind[j];
                T val = gst_csr_val[j];

                if(S[j + nnz] && cf[col + nrow] == 1)
                {
                    if(val > Amin[row] && val < Amax[row])
                    {
                        continue;
                    }

                    // Fill ghost P
                    gst_prolong_csr_col_ind[ghost_row_P] = l2g[col];
                    gst_prolong_csr_val[ghost_row_P]
                        = (val < static_cast<T>(0) ? alpha : beta) * val;
                    ++ghost_row_P;
                }
            }
        }
    }

    template <bool GLOBAL, unsigned int BLOCKSIZE, unsigned int WFSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_extpi_interp_max(I       nrow,
                                            int64_t nnz,
                                            bool    FF1,
                                            const J* __restrict__ csr_row_ptr,
                                            const I* __restrict__ csr_col_ind,
                                            const J* __restrict__ gst_csr_row_ptr,
                                            const I* __restrict__ gst_csr_col_ind,
                                            const I* __restrict__ ext_csr_row_ptr,
                                            const bool* __restrict__ S,
                                            const int* __restrict__ cf,
                                            J* __restrict__ row_max)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Some helpers for readability
        constexpr int COARSE = 1;

        // Coarse points generate a single entry
        if(cf[row] == COARSE)
        {
            if(lid == 0)
            {
                // Set row nnz to one
                row_max[row] = 1;
            }

            return;
        }

        // Counter
        I row_nnz = 0;

        // Row entry and exit points
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Skip points that do not influence the current point
            if(S[j] == false)
            {
                continue;
            }

            // Get the column index
            I col_j = csr_col_ind[j];

            // Skip diagonal entries (i does not influence itself)
            if(col_j == row)
            {
                continue;
            }

            // Switch between coarse and fine points that influence the i-th point
            if(cf[col_j] == COARSE)
            {
                // This is a coarse point and thus contributes, count it
                ++row_nnz;
            }
            else
            {
                // This is a fine point, check for strongly connected coarse points

                // Row entry and exit of this fine point
                J row_begin_j = csr_row_ptr[col_j];
                J row_end_j   = csr_row_ptr[col_j + 1];

                // Loop over all columns of the fine point
                for(J k = row_begin_j; k < row_end_j; ++k)
                {
                    // Skip points that do not influence the fine point
                    if(S[k] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    I col_k = csr_col_ind[k];

                    // Skip diagonal entries (the fine point does not influence itself)
                    if(col_k == col_j)
                    {
                        continue;
                    }

                    // Check whether k is a coarse point
                    if(cf[col_k] == COARSE)
                    {
                        // This is a coarse point, it contributes, count it
                        ++row_nnz;

                        // Stop if FF interpolation is limited
                        if(FF1 == true)
                        {
                            break;
                        }
                    }
                }

                if(GLOBAL)
                {
                    // Row entry and exit of this fine point
                    row_begin_j = gst_csr_row_ptr[col_j];
                    row_end_j   = gst_csr_row_ptr[col_j + 1];

                    // Ghost iterate over the range of columns of B.
                    for(J k = row_begin_j; k < row_end_j; ++k)
                    {
                        // Skip points that do not influence the fine point
                        if(S[k + nnz] == false)
                        {
                            continue;
                        }

                        // Get the column index
                        I col_k = gst_csr_col_ind[k];

                        // Check whether k is a coarse point
                        if(cf[col_k + nrow] == COARSE)
                        {
                            ++row_nnz;
                        }
                    }
                }
            }
        }

        if(GLOBAL)
        {
            // Row entry and exit points
            row_begin = gst_csr_row_ptr[row];
            row_end   = gst_csr_row_ptr[row + 1];

            // Loop over all columns of the i-th row, whereas each lane processes a column
            for(J j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Skip points that do not influence the current point
                if(S[j + nnz] == false)
                {
                    continue;
                }

                // Get the column index
                I col_j = gst_csr_col_ind[j];

                // Switch between coarse and fine points that influence the i-th point
                if(cf[col_j + nrow] == COARSE)
                {
                    ++row_nnz;
                }
                else
                {
                    // This is a fine point, check for strongly connected coarse points

                    // Row entry and exit of this fine point
                    I row_begin_j = ext_csr_row_ptr[col_j];
                    I row_end_j   = ext_csr_row_ptr[col_j + 1];

                    row_nnz += row_end_j - row_begin_j;
                }
            }
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&row_nnz);

        if(lid == WFSIZE - 1)
        {
            // Write row nnz back to global memory
            row_max[row] = row_nnz;
        }
    }

    template <bool         GLOBAL,
              unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename I,
              typename J,
              typename K>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_extpi_interp_nnz(I       nrow,
                                            int64_t nnz,
                                            K       global_col_begin,
                                            K       global_col_end,
                                            bool    FF1,
                                            const J* __restrict__ csr_row_ptr,
                                            const I* __restrict__ csr_col_ind,
                                            const J* __restrict__ gst_csr_row_ptr,
                                            const I* __restrict__ gst_csr_col_ind,
                                            const I* __restrict__ ext_csr_row_ptr,
                                            const K* __restrict__ ext_csr_col_ind,
                                            const bool* __restrict__ S,
                                            const int* __restrict__ cf,
                                            const K* __restrict__ l2g,
                                            J* __restrict__ row_nnz,
                                            J* __restrict__ gst_row_nnz,
                                            I* __restrict__ state)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(WFSIZE == warpSize)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Some helpers for readability
        constexpr int COARSE = 1;

        // Coarse points generate a single entry
        if(cf[row] == COARSE)
        {
            if(lid == 0)
            {
                // Set this points state to coarse
                state[row] = 1;

                // Set row nnz
                row_nnz[row] = 1;

                if(GLOBAL)
                {
                    gst_row_nnz[row] = 0;
                }
            }

            return;
        }

        // Counter
        I int_nnz = 0;
        I gst_nnz = 0;

        // Shared memory for the unordered set
        __shared__ K sdata[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Each wavefront operates on its own set
        unordered_set<K, HASHSIZE, WFSIZE> set(sdata + wid * HASHSIZE);

        // Row entry and exit points
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Skip points that do not influence the current point
            if(S[j] == false)
            {
                continue;
            }

            // Get the column index
            I col_j = csr_col_ind[j];

            // Skip diagonal entries (i does not influence itself)
            if(col_j == row)
            {
                continue;
            }

            // Switch between coarse and fine points that influence the i-th point
            if(cf[col_j] == COARSE)
            {
                // This is a coarse point and thus contributes, count it for the row nnz
                // We need to use a set here, to discard duplicates.
                int_nnz += set.insert(col_j);
            }
            else
            {
                // This is a fine point, check for strongly connected coarse points

                // Row entry and exit of this fine point
                J row_begin_j = csr_row_ptr[col_j];
                J row_end_j   = csr_row_ptr[col_j + 1];

                // Loop over all columns of the fine point
                for(J k = row_begin_j; k < row_end_j; ++k)
                {
                    // Skip points that do not influence the fine point
                    if(S[k] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    I col_k = csr_col_ind[k];

                    // Skip diagonal entries (the fine point does not influence itself)
                    if(col_k == col_j)
                    {
                        continue;
                    }

                    // Check whether k is a coarse point
                    if(cf[col_k] == COARSE)
                    {
                        // This is a coarse point, it contributes, count it for the row nnz
                        // We need to use a set here, to discard duplicates.
                        int_nnz += set.insert(col_k);

                        // Stop if FF interpolation is limited
                        if(FF1 == true)
                        {
                            break;
                        }
                    }
                }

                if(GLOBAL)
                {
                    // Row entry and exit of this fine point
                    row_begin_j = gst_csr_row_ptr[col_j];
                    row_end_j   = gst_csr_row_ptr[col_j + 1];

                    // Ghost iterate over the range of columns of B.
                    for(J k = row_begin_j; k < row_end_j; ++k)
                    {
                        // Skip points that do not influence the fine point
                        if(S[k + nnz] == false)
                        {
                            continue;
                        }

                        // Check whether k is a coarse point
                        I col_k = gst_csr_col_ind[k];

                        // Check whether k is a coarse point
                        if(cf[col_k + nrow] == COARSE)
                        {
                            // Get (global) column index
                            K gcol_k = l2g[col_k] + global_col_end - global_col_begin;

                            // This is a coarse point, it contributes, count it for the row int_nnz
                            // We need to use a set here, to discard duplicates.
                            gst_nnz += set.insert(gcol_k);
                        }
                    }
                }
            }
        }

        if(GLOBAL)
        {
            // Row entry and exit points
            row_begin = gst_csr_row_ptr[row];
            row_end   = gst_csr_row_ptr[row + 1];

            // Loop over all columns of the i-th row, whereas each lane processes a column
            for(J j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Skip points that do not influence the current point
                if(S[j + nnz] == false)
                {
                    continue;
                }

                // Get the column index
                I col_j = gst_csr_col_ind[j];

                // Switch between coarse and fine points that influence the i-th point
                if(cf[col_j + nrow] == COARSE)
                {
                    // Get (global) column index
                    K gcol_j = l2g[col_j] + global_col_end - global_col_begin;

                    // This is a coarse point and thus contributes, count it for the row int_nnz
                    // We need to use a set here, to discard duplicates.
                    gst_nnz += set.insert(gcol_j);
                }
                else
                {
                    // This is a fine point, check for strongly connected coarse points

                    // Row entry and exit of this fine point
                    I row_begin_j = ext_csr_row_ptr[col_j];
                    I row_end_j   = ext_csr_row_ptr[col_j + 1];

                    // Loop over all columns of the fine point
                    for(I k = row_begin_j; k < row_end_j; ++k)
                    {
                        // Get the (global) column index
                        K gcol_k = ext_csr_col_ind[k];

                        // Differentiate between local and ghost column
                        if(gcol_k >= global_col_begin && gcol_k < global_col_end)
                        {
                            // Get (local) column index
                            I col_k = gcol_k - global_col_begin;

                            // This is a coarse point, it contributes, count it for the row nnz
                            // We need to use a set here, to discard duplicates.
                            int_nnz += set.insert(col_k);
                        }
                        else
                        {
                            // This is a coarse point, it contributes, count it for the row nnz
                            // We need to use a set here, to discard duplicates.
                            gst_nnz += set.insert(gcol_k + global_col_end - global_col_begin);
                        }
                    }
                }
            }
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&int_nnz);

        if(GLOBAL)
        {
            wf_reduce_sum<WFSIZE>(&gst_nnz);
        }

        if(lid == WFSIZE - 1)
        {
            // Write row nnz back to global memory
            row_nnz[row] = int_nnz;

            if(GLOBAL)
            {
                gst_row_nnz[row] = gst_nnz;
            }

            // Set this points state to fine
            state[row] = 0;
        }
    }

    template <bool         GLOBAL,
              unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename T,
              typename I,
              typename J,
              typename K>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_extpi_interp_fill(I       nrow,
                                             I       ncol,
                                             int64_t nnz,
                                             K       global_col_begin,
                                             K       global_col_end,
                                             bool    FF1,
                                             const J* __restrict__ csr_row_ptr,
                                             const I* __restrict__ csr_col_ind,
                                             const T* __restrict__ csr_val,
                                             const J* __restrict__ gst_csr_row_ptr,
                                             const I* __restrict__ gst_csr_col_ind,
                                             const T* __restrict__ gst_csr_val,
                                             const I* __restrict__ dummy_row_ptr,
                                             const K* __restrict__ dummy_col_ind,
                                             const I* __restrict__ ext_csr_row_ptr,
                                             const K* __restrict__ ext_csr_col_ind,
                                             const T* __restrict__ ext_csr_val,
                                             const K* __restrict__ l2g,
                                             const T* __restrict__ diag,
                                             const J* __restrict__ csr_row_ptr_P,
                                             I* __restrict__ csr_col_ind_P,
                                             T* __restrict__ csr_val_P,
                                             const J* __restrict__ gst_csr_row_ptr_P,
                                             K* __restrict__ gst_csr_col_ind_P,
                                             T* __restrict__ gst_csr_val_P,
                                             const bool* __restrict__ S,
                                             const int* __restrict__ cf,
                                             const I* __restrict__ f2c)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(WFSIZE == warpSize)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Some helpers for readability
        constexpr T zero = static_cast<T>(0);

        constexpr int COARSE = 1;
        constexpr int FINE   = 2;

        // Coarse points generate a single entry
        if(cf[row] == COARSE)
        {
            if(lid == 0)
            {
                // Get index into P
                J idx = csr_row_ptr_P[row];

                // Single entry in this row (coarse point)
                csr_col_ind_P[idx] = f2c[row];
                csr_val_P[idx]     = static_cast<T>(1);
            }

            return;
        }

        // Shared memory for the unordered map
        extern __shared__ char smem[];

        K* stable = reinterpret_cast<K*>(smem);
        T* sdata  = reinterpret_cast<T*>(stable + BLOCKSIZE / WFSIZE * HASHSIZE);

        // Unordered map
        unordered_map<K, T, HASHSIZE, WFSIZE> map(&stable[wid * HASHSIZE], &sdata[wid * HASHSIZE]);

        // Fill the map according to the nnz pattern of P
        // This is identical to the nnz per row kernel

        // Row entry and exit points
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J k = row_begin + lid; k < row_end; k += WFSIZE)
        {
            // Skip points that do not influence the current point
            if(S[k] == false)
            {
                continue;
            }

            // Get the column index
            I col_ik = csr_col_ind[k];

            // Skip diagonal entries (i does not influence itself)
            if(col_ik == row)
            {
                continue;
            }

            // Switch between coarse and fine points that influence the i-th point
            if(cf[col_ik] == COARSE)
            {
                // This is a coarse point and thus contributes
                map.insert(col_ik);
            }
            else
            {
                // This is a fine point, check for strongly connected coarse points

                // Row entry and exit of this fine point
                J row_begin_k = csr_row_ptr[col_ik];
                J row_end_k   = csr_row_ptr[col_ik + 1];

                // Loop over all columns of the fine point
                for(J l = row_begin_k; l < row_end_k; ++l)
                {
                    // Skip points that do not influence the fine point
                    if(S[l] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    I col_kl = csr_col_ind[l];

                    // Skip diagonal entries (the fine point does not influence itself)
                    if(col_kl == col_ik)
                    {
                        continue;
                    }

                    // Check whether l is a coarse point
                    if(cf[col_kl] == COARSE)
                    {
                        // This is a coarse point, it contributes
                        map.insert(col_kl);

                        // Stop if FF interpolation is limited
                        if(FF1 == true)
                        {
                            break;
                        }
                    }
                }

                if(GLOBAL)
                {
                    // Ghost iterate over the range of columns of B.
                    for(J l = gst_csr_row_ptr[col_ik]; l < gst_csr_row_ptr[col_ik + 1]; ++l)
                    {
                        // Skip points that do not influence the fine point
                        if(S[l + nnz] == false)
                        {
                            continue;
                        }

                        // Get the column index
                        I col_l = gst_csr_col_ind[l];

                        // Check whether l is a coarse point
                        if(cf[col_l + nrow] == COARSE)
                        {
                            // Global column shifted by local columns
                            K shifted_global_col = l2g[col_l] + global_col_end - global_col_begin;

                            // This is a coarse point, it contributes
                            map.insert(shifted_global_col);
                        }
                    }
                }
            }
        }

        if(GLOBAL)
        {
            // Row entry and exit points
            J gst_row_begin = gst_csr_row_ptr[row];
            J gst_row_end   = gst_csr_row_ptr[row + 1];

            // Loop over all ghost columns of the i-th row, whereas each lane processes a column
            for(J k = gst_row_begin + lid; k < gst_row_end; k += WFSIZE)
            {
                // Skip points that do not influence the current point
                if(S[k + nnz] == false)
                {
                    continue;
                }

                // Get the column index
                I col_ik = gst_csr_col_ind[k];

                // Switch between coarse and fine points that influence the i-th point
                if(cf[col_ik + nrow] == COARSE)
                {
                    // This is a coarse point and thus contributes
                    map.insert(l2g[col_ik] + global_col_end - global_col_begin);
                }
                else
                {
                    // This is a fine point, check for strongly connected coarse points

                    // Row entry and exit of this fine point
                    I ext_row_begin_k = dummy_row_ptr[col_ik];
                    I ext_row_end_k   = dummy_row_ptr[col_ik + 1];

                    // Loop over all columns of the fine point
                    for(I l = ext_row_begin_k; l < ext_row_end_k; ++l)
                    {
                        // Get the (global) column index
                        K gcol_kl = dummy_col_ind[l];

                        // Check whether this global id maps to the local process
                        if(gcol_kl >= global_col_begin && gcol_kl < global_col_end)
                        {
                            // Get the local column index
                            I col_kl = gcol_kl - global_col_begin;

                            // This is a coarse point, it contributes
                            map.insert(col_kl);
                        }
                        else
                        {
                            // This is a coarse point, it contributes
                            map.insert(gcol_kl + global_col_end - global_col_begin);
                        }
                    }
                }
            }
        }

        // Now, we need to do the numerical part

        // Diagonal entry of i-th row
        T val_ii = diag[row];

        // Sign of diagonal entry of i-th row
        bool pos_ii = val_ii >= zero;

        // Accumulators
        T sum_k = zero;
        T sum_n = zero;

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J k = row_begin + lid; k < row_end; k += WFSIZE)
        {
            // Get the column index
            I col_ik = csr_col_ind[k];

            // Skip diagonal entries (i does not influence itself)
            if(col_ik == row)
            {
                continue;
            }

            // Get the column value
            T val_ik = csr_val[k];

            // Check, whether the k-th entry of the row is a fine point and strongly
            // connected to the i-th point (e.g. k \in F^S_i)
            if(S[k] == true && cf[col_ik] == FINE)
            {
                // Accumulator for the sum over l
                T sum_l = zero;

                // Diagonal entry of k-th row
                T val_kk = diag[col_ik];

                // Store a_ki, if present
                T val_ki = zero;

                // Row entry and exit of this fine point
                J row_begin_k = csr_row_ptr[col_ik];
                J row_end_k   = csr_row_ptr[col_ik + 1];

                // Loop over all columns of the fine point
                for(J l = row_begin_k; l < row_end_k; ++l)
                {
                    // Get the column index
                    I col_kl = csr_col_ind[l];

                    // Get the column value
                    T val_kl = csr_val[l];

                    // Sign of a_kl
                    bool pos_kl = val_kl >= zero;

                    // Differentiate between diagonal and off-diagonal
                    if(col_kl == row)
                    {
                        // Column that matches the i-th row
                        // Since we sum up all l in C^hat_i and i, the diagonal need to
                        // be added to the sum over l, e.g. a^bar_kl
                        // a^bar contributes only, if the sign is different to the
                        // i-th row diagonal sign.
                        if(pos_ii != pos_kl)
                        {
                            sum_l += val_kl;
                        }

                        // If a_ki exists, keep it for later
                        val_ki = val_kl;
                    }
                    else if(cf[col_kl] == COARSE)
                    {
                        // Check if sign is different from i-th row diagonal
                        if(pos_ii != pos_kl)
                        {
                            // Entry contributes only, if it is a coarse point
                            // and part of C^hat (e.g. we need to check the map)
                            if(map.contains(col_kl))
                            {
                                sum_l += val_kl;
                            }
                        }
                    }
                }

                if(GLOBAL)
                {
                    // Row entry and exit of this fine point
                    J gst_row_begin_k = gst_csr_row_ptr[col_ik];
                    J gst_row_end_k   = gst_csr_row_ptr[col_ik + 1];

                    // Loop over all columns of the fine point
                    for(J l = gst_row_begin_k; l < gst_row_end_k; ++l)
                    {
                        // Get the column index
                        I col_kl = gst_csr_col_ind[l];

                        // Get the global column index
                        K gcol_kl = l2g[col_kl] + global_col_end - global_col_begin;

                        // Get the column value
                        T val_kl = gst_csr_val[l];

                        // Sign of a_kl
                        bool pos_kl = val_kl >= zero;

                        // Only coarse ghost parts contribute
                        if(cf[col_kl + nrow] == COARSE)
                        {
                            // Check if sign is different from i-th row diagonal
                            if(pos_ii != pos_kl)
                            {
                                // Entry contributes only, if it is a coarse point
                                // and part of C^hat (e.g. we need to check the map)
                                if(map.contains(gcol_kl))
                                {
                                    sum_l += val_kl;
                                }
                            }
                        }
                    }
                }

                // Update sum over l with a_ik
                sum_l = val_ik / sum_l;

                // Compute the sign of a_kk and a_ki, we need this for a_bar
                bool pos_kk = val_kk >= zero;
                bool pos_ki = val_ki >= zero;

                // Additionally, for eq19 we need to add all coarse points in row k,
                // if they have different sign than the diagonal a_kk
                for(J l = row_begin_k; l < row_end_k; ++l)
                {
                    // Get the column index
                    I col_kl = csr_col_ind[l];

                    // Only coarse points contribute
                    if(cf[col_kl] != COARSE)
                    {
                        continue;
                    }

                    // Get the column value
                    T val_kl = csr_val[l];

                    // Compute the sign of a_kl
                    bool pos_kl = val_kl >= zero;

                    // Check for different sign
                    if(pos_kk != pos_kl)
                    {
                        // Add to map, only if the element exists already
                        map.add(col_kl, val_kl * sum_l);
                    }
                }

                if(GLOBAL)
                {
                    // Row entry and exit of this fine point
                    row_begin_k = gst_csr_row_ptr[col_ik];
                    row_end_k   = gst_csr_row_ptr[col_ik + 1];

                    // Additionally, for eq19 we need to add all coarse points in row k,
                    // if they have different sign than the diagonal a_kk
                    for(J l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Get the column index
                        I col_kl = gst_csr_col_ind[l];

                        // Get the (global) column index
                        K gcol_kl = l2g[col_kl] + global_col_end - global_col_begin;

                        // Get the column value
                        T val_kl = gst_csr_val[l];

                        // Compute the sign of a_kl
                        bool pos_kl = val_kl >= zero;

                        // Check for different sign
                        if(pos_kk != pos_kl)
                        {
                            // Add to map, only if the element exists already
                            map.add(gcol_kl, val_kl * sum_l);
                        }
                    }
                }

                // If sign of a_ki and a_kk are different, a_ki contributes to the
                // sum over k in F^S_i
                if(pos_kk != pos_ki)
                {
                    sum_k += val_ki * sum_l;
                }
            }

            // Boolean, to flag whether a_ik is in C hat or not
            // (we can query the map for it)
            bool in_C_hat = false;

            // a_ik can only be in C^hat if it is coarse
            if(cf[col_ik] == COARSE)
            {
                // Append a_ik to the sum of eq19
                in_C_hat = map.add(col_ik, val_ik);
            }

            // If a_ik is not in C^hat and does not strongly influence i, it contributes
            // to sum_n
            if(in_C_hat == false && S[k] == false)
            {
                sum_n += val_ik;
            }
        }

        if(GLOBAL)
        {
            // ghost Iterate over the columns of A.
            for(J k = gst_csr_row_ptr[row] + lid; k < gst_csr_row_ptr[row + 1]; k += WFSIZE)
            {
                I col_ik = gst_csr_col_ind[k];
                T val_ik = gst_csr_val[k];

                if(S[k + nnz] && cf[col_ik + nrow] == FINE)
                {
                    T sum_l = zero;

                    I row_begin_k = ext_csr_row_ptr[col_ik];
                    I row_end_k   = ext_csr_row_ptr[col_ik + 1];

                    // Diagonal element // TODO outside of kernel!!
                    T val_kk = zero;
                    K grow_k = l2g[col_ik];

                    for(I l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Get the (global) column index
                        K gcol_kl = ext_csr_col_ind[l];

                        // Get the column value
                        T val_kl = ext_csr_val[l];

                        // Sign of a_kl
                        bool pos_kl = val_kl >= zero;

                        // Extract diagonal value
                        if(grow_k == gcol_kl)
                        {
                            val_kk = val_kl;
                        }

                        if(gcol_kl >= global_col_begin && gcol_kl < global_col_end)
                        {
                            // Get the (local) column index
                            I col_kl = gcol_kl - global_col_begin;

                            // Differentiate between diagonal and off-diagonal
                            if(col_kl == row)
                            {
                                // Check if sign is different from i-th row diagonal
                                if(pos_ii != pos_kl)
                                {
                                    sum_l += val_kl;
                                }
                            }
                            else
                            {
                                // Check if sign is different from i-th row diagonal
                                if(pos_ii != pos_kl)
                                {
                                    if(map.contains(col_kl))
                                    {
                                        sum_l += val_kl;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // Check if sign is different from i-th row diagonal
                            if(pos_ii != pos_kl)
                            {
                                if(map.contains(gcol_kl + global_col_end - global_col_begin))
                                {
                                    sum_l += val_kl;
                                }
                            }
                        }
                    }

                    T val_ki = (T)0;

                    // Load the kth inner sum.
                    sum_l = val_ik / sum_l;

                    // Aext
                    for(I l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Get the column index
                        K gcol_kl = ext_csr_col_ind[l];

                        // Get the column value
                        T val_kl = ext_csr_val[l];

                        // Sign of a_kl and a_kk
                        bool pos_kk = val_kk >= zero;
                        bool pos_kl = val_kl >= zero;

                        // Check for different sign
                        if(pos_kk != pos_kl)
                        {
                            if(gcol_kl >= global_col_begin && gcol_kl < global_col_end)
                            {
                                // Get the (local) column index
                                I col_kl = gcol_kl - global_col_begin;

                                // Differentiate between diagonal and off-diagonal
                                if(col_kl == row)
                                {
                                    // Extract diagonal value
                                    val_ki = val_kl;
                                }

                                // a_kl contributes, add it to the map but only if the
                                // key exists already
                                map.add(col_kl, val_kl * sum_l);
                            }
                            else
                            {
                                // a_kl contributes, add it to the map but only if the
                                // key exists already
                                map.add(gcol_kl + global_col_end - global_col_begin,
                                        val_kl * sum_l);
                            }
                        }
                    }

                    sum_n += val_ki * sum_l;
                }

                // Global column shifted by local columns
                K shifted_global_col = l2g[col_ik] + global_col_end - global_col_begin;

                // Boolean, to flag whether a_ik is in C hat or not
                // (we can query the map for it)
                bool in_C_hat = map.add(shifted_global_col, val_ik);

                // If a_ik is not in C^hat and does not strongly influence i, it contributes
                // to the sum
                if(in_C_hat == false && S[k + nnz] == false)
                {
                    sum_k += val_ik;
                }
            }
        }

        // Each lane accumulates the sums (over n and l)
        T a_ii_tilde = sum_n + sum_k;

        // Now, each lane of the wavefront should hold the global row sum
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            a_ii_tilde += hip_shfl_xor(a_ii_tilde, i);
        }

        // Precompute -1 / (a_ii_tilde + a_ii)
        a_ii_tilde = static_cast<T>(-1) / (a_ii_tilde + val_ii);

        // Access into P
        J aj = csr_row_ptr_P[row];

        // Finally, extract the numerical values from the map and fill P such
        // that the resulting matrix is sorted by columns
        for(unsigned int i = lid; i < HASHSIZE; i += WFSIZE)
        {
            // Get column from map to fill into C hat
            K col = map.get_key(i);

            // Skip, if table is empty
            if(col == map.empty_key())
            {
                continue;
            }

            // Differentiate between local and ghost part
            if(col < ncol || GLOBAL == false)
            {
                // Get index into P
                J idx = 0;

                // Hash table index counter
                unsigned int cnt = 0;

                // Go through the hash table, until we reach its end
                while(cnt < HASHSIZE)
                {
                    // We are searching for the right place in P to
                    // insert the i-th hash table entry.
                    // If the i-th hash table column entry is greater then the current one,
                    // we need to leave a slot to its left.
                    if(col > map.get_key(cnt))
                    {
                        ++idx;
                    }

                    // Process next hash table entry
                    ++cnt;
                }

                // Add hash table entry into P
                csr_col_ind_P[aj + idx] = f2c[col];
                csr_val_P[aj + idx]     = a_ii_tilde * map.get_val(i);
            }
            else
            {
                // Get index into P
                J idx = gst_csr_row_ptr_P[row];

                // Hash table index counter
                unsigned int cnt = 0;

                // Go through the hash table, until we reach its end
                while(cnt < HASHSIZE)
                {
                    // We are searching for the right place in P to
                    // insert the i-th hash table entry.
                    // If the i-th hash table column entry is greater then the current one,
                    // we need to leave a slot to its left.
                    K next_col = map.get_key(cnt);

                    if(col > next_col && next_col >= ncol)
                    {
                        ++idx;
                    }

                    // Process next hash table entry
                    ++cnt;
                }

                // Add hash table entry into P
                gst_csr_col_ind_P[idx] = col - global_col_end + global_col_begin;
                gst_csr_val_P[idx]     = a_ii_tilde * map.get_val(i);
            }
        }
    }

    // Extract all strongly connected, coarse boundary vertices
    template <typename I, typename J>
    __global__ void
        kernel_csr_rs_extpi_strong_coarse_boundary_rows_nnz(I       nrow,
                                                            int64_t nnz,
                                                            I       boundary_size,
                                                            const I* __restrict__ boundary_index,
                                                            const J* __restrict__ int_csr_row_ptr,
                                                            const I* __restrict__ int_csr_col_ind,
                                                            const J* __restrict__ gst_csr_row_ptr,
                                                            const I* __restrict__ gst_csr_col_ind,
                                                            const int* __restrict__ cf,
                                                            const bool* __restrict__ S,
                                                            I* __restrict__ row_nnz)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // Get boundary row
        I row = boundary_index[gid];

        // Counter
        I ext_nnz = 0;

        // Interior part
        J row_begin = int_csr_row_ptr[row];
        J row_end   = int_csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            // Check whether this vertex is strongly connected
            if(S[j] == false)
            {
                continue;
            }

            // Get column index
            I col = int_csr_col_ind[j];

            // Only coarse points contribute
            if(cf[col] != 2)
            {
                ++ext_nnz;
            }
        }

        // Ghost part
        row_begin = gst_csr_row_ptr[row];
        row_end   = gst_csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            // Check whether this vertex is strongly connected
            if(S[j + nnz] == false)
            {
                continue;
            }

            // Get column index
            I col = gst_csr_col_ind[j];

            // Only coarse points contribute
            if(cf[col + nrow] != 2)
            {
                ++ext_nnz;
            }
        }

        // Write total number of strongly connected coarse vertices to global memory
        row_nnz[gid] = ext_nnz;
    }

    template <typename I, typename J, typename K>
    __global__ void kernel_csr_rs_extpi_extract_strong_coarse_boundary_rows(
        I       nrow,
        int64_t nnz,
        K       global_col_begin,
        I       boundary_size,
        const I* __restrict__ boundary_index,
        const J* __restrict__ int_csr_row_ptr,
        const I* __restrict__ int_csr_col_ind,
        const J* __restrict__ gst_csr_row_ptr,
        const I* __restrict__ gst_csr_col_ind,
        const K* __restrict__ l2g,
        const I* __restrict__ cf,
        const bool* __restrict__ S,
        const I* __restrict__ ext_csr_row_ptr,
        K* __restrict__ ext_csr_col_ind)
    {
        I gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(gid >= boundary_size)
        {
            return;
        }

        // Get boundary row and index into A_ext
        I row = boundary_index[gid];
        I idx = ext_csr_row_ptr[gid];

        // Extract interior part
        J row_begin = int_csr_row_ptr[row];
        J row_end   = int_csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            // Check whether this vertex is strongly connected
            if(S[j] == false)
            {
                continue;
            }

            // Get column index
            I col = int_csr_col_ind[j];

            // Only coarse points contribute
            if(cf[col] != 2)
            {
                // Shift column by global column offset, to obtain the global column index
                ext_csr_col_ind[idx++] = col + global_col_begin;
            }
        }

        // Extract ghost part
        row_begin = gst_csr_row_ptr[row];
        row_end   = gst_csr_row_ptr[row + 1];

        for(J j = row_begin; j < row_end; ++j)
        {
            // Check whether this vertex is strongly connected
            if(S[j + nnz] == false)
            {
                continue;
            }

            // Get column index
            I col = gst_csr_col_ind[j];

            // Only coarse points contribute
            if(cf[col + nrow] != 2)
            {
                // Transform local ghost index into global column index
                ext_csr_col_ind[idx++] = l2g[col];
            }
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_RSAMG_CSR_HPP_
