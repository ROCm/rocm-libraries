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

#include "rocsparse_csric0_launch_kernel.hpp"
#include "rocsparse_common.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_DEVICE_ILF void csric0_hash_device(J m,
                                                 const I* __restrict__ csr_row_ptr,
                                                 const J* __restrict__ csr_col_ind,
                                                 T* __restrict__ csr_val,
                                                 const I* __restrict__ csr_diag_ind,
                                                 int32_t* __restrict__ done,
                                                 const J* __restrict__ map,
                                                 J* __restrict__ zero_pivot,
                                                 J* __restrict__ singular_pivot,
                                                 double               tol,
                                                 rocsparse_index_base idx_base)
    {
        const auto lid = hipThreadIdx_x & (WFSIZE - 1);
        const auto wid = hipThreadIdx_x / WFSIZE;

        __shared__ J stable[BLOCKSIZE * HASH];
        __shared__ I sdata[BLOCKSIZE * HASH];

        // Pointer to each wavefronts shared data
        J* table = &stable[wid * WFSIZE * HASH];
        I* data  = &sdata[wid * WFSIZE * HASH];

        // Initialize hash table with -1
        for(uint32_t j = lid; j < WFSIZE * HASH; j += WFSIZE)
        {
            table[j] = -1;
        }

        __threadfence_block();

        const auto idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= m)
        {
            return;
        }

        const auto tol_sq = tol * tol;

        // Current row this wavefront is working on
        J row = map[idx];

        // Diagonal entry point of the current row
        I row_diag = csr_diag_ind[row];

        // Row entry point
        I row_begin = csr_row_ptr[row] - idx_base;
        I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Row sum accumulator
        T sum = static_cast<T>(0);

        // Fill hash table
        // Loop over columns of current row and fill hash table with row dependencies
        // Each lane processes one entry
        for(I j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Insert key into hash table
            J key = csr_col_ind[j];
            // Compute hash
            int32_t hash = (key * 103) & (WFSIZE * HASH - 1);

            // Hash operation
            while(true)
            {
                if(table[hash] == key)
                {
                    // key is already inserted, done
                    break;
                }
                else if(rocsparse::atomic_cas(&table[hash], static_cast<J>(-1), key)
                        == static_cast<J>(-1))
                {
                    // inserted key into the table, done
                    data[hash] = j;
                    break;
                }
                else
                {
                    // collision, compute new hash
                    hash = (hash + 1) & (WFSIZE * HASH - 1);
                }
            }
        }

        __threadfence_block();

        // Loop over column of current row
        for(I j = row_begin; j < row_diag; ++j)
        {
            // Column index currently being processes
            J local_col = csr_col_ind[j] - idx_base;

            // Corresponding value
            T local_val = csr_val[j];

            // Beginning of the row that corresponds to local_col
            I local_begin = csr_row_ptr[local_col] - idx_base;

            // Diagonal entry point of row local_col
            I local_diag = csr_diag_ind[local_col];

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = row_diag - 1;
            }

            // Spin loop until dependency has been resolved
            while(!__hip_atomic_load(&done[local_col], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                ;

            // Make sure updated csr_val is visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load diagonal entry
            T diag_val = csr_val[local_diag];
            //
            if(diag_val == static_cast<T>(0))
            {
                break;
            };

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            for(I k = local_begin + lid; k < local_diag; k += WFSIZE)
            {
                // Get value from hash table
                J key = csr_col_ind[k];

                // Compute hash
                J hash = (key * 103) & (WFSIZE * HASH - 1);

                // Hash operation
                while(true)
                {
                    if(table[hash] == -1)
                    {
                        // No entry for the key, done
                        break;
                    }
                    else if(table[hash] == key)
                    {
                        // Entry found, do linear combination
                        I idx = data[hash];
                        local_sum
                            = rocsparse::fma(csr_val[k], rocsparse::conj(csr_val[idx]), local_sum);
                        break;
                    }
                    else
                    {
                        // Collision, compute new hash
                        hash = (hash + 1) & (WFSIZE * HASH - 1);
                    }
                }
            }

            // Accumulate row sum
            local_sum = rocsparse::wfreduce_sum<WFSIZE>(local_sum);

            // Last lane id computes the Cholesky factor and writes it to global memory
            if(lid == WFSIZE - 1)
            {
                local_val = (local_val - local_sum) * diag_val;
                sum       = rocsparse::fma(local_val, rocsparse::conj(local_val), sum);

                csr_val[j] = local_val;
            }
        }

        // Last lane processes the diagonal entry
        if(lid == WFSIZE - 1)
        {
            if((row_diag >= 0))
            {
                const T diag_val = csr_val[row_diag] - sum;

                // test for negative value and numerical small value
                if((rocsparse::real(diag_val) <= (tol_sq)) && (rocsparse::imag(diag_val) == 0))
                {
                    rocsparse::atomic_min(singular_pivot, (row + idx_base));
                }

                if((csr_val[row_diag] = rocsparse::sqrt(rocsparse::abs(diag_val)))
                   == static_cast<T>(0))
                {
                    rocsparse::atomic_min(zero_pivot, (row + idx_base));
                }
            }
        }

        if(lid == WFSIZE - 1)
        {
            // Last lane writes "we are done" flag
            __hip_atomic_store(&done[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void csric0_binsearch_device(J m,
                                                      const I* __restrict__ csr_row_ptr,
                                                      const J* __restrict__ csr_col_ind,
                                                      T* __restrict__ csr_val,
                                                      const I* __restrict__ csr_diag_ind,
                                                      int* __restrict__ done,
                                                      const J* __restrict__ map,
                                                      J* __restrict__ zero_pivot,
                                                      J* __restrict__ singular_pivot,
                                                      double               tol,
                                                      rocsparse_index_base idx_base)
    {
        const auto lid = hipThreadIdx_x & (WFSIZE - 1);
        const auto wid = hipThreadIdx_x / WFSIZE;
        const auto idx = hipBlockIdx_x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(idx >= m)
        {
            return;
        }
        const auto tol_sq = tol * tol;

        // Current row this wavefront is working on
        const J row = map[idx];

        // Diagonal entry point of the current row
        const I row_diag = csr_diag_ind[row];

        // Row entry point
        const I row_begin = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        // Row sum accumulator
        T sum = static_cast<T>(0);

        // Loop over column of current row
        for(I j = row_begin; j < row_diag; ++j)
        {
            // Column index currently being processes
            const J local_col = csr_col_ind[j] - idx_base;

            // Corresponding value
            T local_val = csr_val[j];

            // Beginning of the row that corresponds to local_col
            const I local_begin = csr_row_ptr[local_col] - idx_base;

            // Diagonal entry point of row local_col
            I local_diag = csr_diag_ind[local_col];

            // Local row sum
            T local_sum = static_cast<T>(0);

            // Structural zero pivot, do not process this row
            if(local_diag == -1)
            {
                local_diag = row_diag - 1;
            }

            // Spin loop until dependency has been resolved
            (void)rocsparse::spin_loop<SLEEP>(&done[local_col], __HIP_MEMORY_SCOPE_AGENT);

            // Make sure updated csr_val is visible globally
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

            // Load diagonal entry
            T diag_val = csr_val[local_diag];

            // Row has numerical zero diagonal
            if(diag_val == static_cast<T>(0))
            {
                if(lid == 0)
                {
                    // We are looking for the first zero pivot
                    rocsparse::atomic_min(zero_pivot, local_col + idx_base);
                }

                // Skip this row if it has a zero pivot
                break;
            }

            // Row has numerical singular diagonal
            if((rocsparse::imag(diag_val) == 0) && (rocsparse::real(diag_val) <= tol))
            {
                if(lid == 0)
                {
                    // We are looking for the first singular pivot
                    rocsparse::atomic_min(singular_pivot, local_col + idx_base);
                }

                // Don't skip this row if it has a singular pivot
            }

            // Compute reciprocal
            diag_val = static_cast<T>(1) / diag_val;

            // Loop over the row the current column index depends on
            // Each lane processes one entry
            I l = row_begin;
            for(I k = local_begin + lid; k < local_diag; k += WFSIZE)
            {
                // Perform a binary search to find matching columns
                I       r     = row_end - 1;
                I       m     = (r + l) >> 1;
                J       col_j = csr_col_ind[m];
                const J col_k = csr_col_ind[k];

                // Binary search
                while(l < r)
                {
                    if(col_j < col_k)
                    {
                        l = m + 1;
                    }
                    else
                    {
                        r = m;
                    }

                    m     = (r + l) >> 1;
                    col_j = csr_col_ind[m];
                }

                // Check if a match has been found
                if(col_j == col_k)
                {
                    // If a match has been found, do linear combination
                    local_sum = rocsparse::fma(csr_val[k], rocsparse::conj(csr_val[m]), local_sum);
                }
            }

            // Accumulate row sum
            local_sum = rocsparse::wfreduce_sum<WFSIZE>(local_sum);

            // Last lane id computes the Cholesky factor and writes it to global memory
            if(lid == WFSIZE - 1)
            {
                local_val = (local_val - local_sum) * diag_val;
                sum       = rocsparse::fma(local_val, rocsparse::conj(local_val), sum);

                csr_val[j] = local_val;
            }
        }

        // Last lane processes the diagonal entry
        if(lid == WFSIZE - 1)
        {
            if((row_diag >= 0))
            {
                const T diag_val = csr_val[row_diag] - sum;

                // check for negative value and numerical small value
                if((rocsparse::imag(diag_val) == 0) && (rocsparse::real(diag_val) <= (tol_sq)))
                {
                    rocsparse::atomic_min(singular_pivot, (row + idx_base));
                }

                csr_val[row_diag] = rocsparse::sqrt(rocsparse::abs(csr_val[row_diag] - sum));
                if(csr_val[row_diag] == static_cast<T>(0))
                {
                    rocsparse::atomic_min(zero_pivot, (row + idx_base));
                }
            }
        }

        if(lid == WFSIZE - 1)
        {
            // Last lane writes "we are done" flag
            __hip_atomic_store(&done[row], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csric0_hash_kernel(J m,
                            const I* __restrict__ csr_row_ptr,
                            const J* __restrict__ csr_col_ind,
                            T* __restrict__ csr_val,
                            int64_t csr_val_stride,
                            const I* __restrict__ csr_diag_ind,
                            int32_t* __restrict__ done,
                            int64_t done_stride,
                            const J* __restrict__ map,
                            J* __restrict__ zero_pivot,
                            int64_t zero_pivot_stride,
                            J* __restrict__ singular_pivot,
                            int64_t              singular_pivot_stride,
                            double               tol,
                            rocsparse_index_base idx_base)
    {
        const auto batch_index = hipBlockIdx_y;
        rocsparse::csric0_hash_device<BLOCKSIZE, WFSIZE, HASH>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + batch_index * csr_val_stride,
            csr_diag_ind,
            done + batch_index * done_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            singular_pivot + batch_index * singular_pivot_stride,
            tol,
            idx_base);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csric0_binsearch_kernel(J m,
                                 const I* __restrict__ csr_row_ptr,
                                 const J* __restrict__ csr_col_ind,
                                 T* __restrict__ csr_val,
                                 int64_t csr_val_stride,
                                 const I* __restrict__ csr_diag_ind,
                                 int32_t* __restrict__ done,
                                 int64_t done_stride,
                                 const J* __restrict__ map,
                                 J* __restrict__ zero_pivot,
                                 int64_t zero_pivot_stride,
                                 J* __restrict__ singular_pivot,
                                 int64_t              singular_pivot_stride,
                                 double               tol,
                                 rocsparse_index_base idx_base)
    {
        const auto batch_index = hipBlockIdx_y;
        rocsparse::csric0_binsearch_device<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>(
            m,
            csr_row_ptr,
            csr_col_ind,
            csr_val + batch_index * csr_val_stride,
            csr_diag_ind,
            done + batch_index * done_stride,
            map,
            zero_pivot + batch_index * zero_pivot_stride,
            singular_pivot + batch_index * singular_pivot_stride,
            tol,
            idx_base);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename J>
    static rocsparse_status csric0_launch_hash_kernel(rocsparse_handle handle,
                                                      int64_t          batch_count,
                                                      int64_t          m,
                                                      const void* __restrict__ csr_row_ptr,
                                                      const void* __restrict__ csr_col_ind,
                                                      void* __restrict__ csr_val,
                                                      int64_t csr_val_stride,
                                                      const void* __restrict__ csr_diag_ind,
                                                      int32_t* __restrict__ done,
                                                      int64_t done_stride,
                                                      const void* __restrict__ map,
                                                      void* __restrict__ zero_pivot,
                                                      int64_t zero_pivot_stride,
                                                      void* __restrict__ singular_pivot,
                                                      int64_t              singular_pivot_stride,
                                                      double               tol,
                                                      rocsparse_index_base idx_base)
    {
        const dim3 csric0_blocks((m * handle->wavefront_size - 1) / BLOCKSIZE + 1, batch_count);
        const dim3 csric0_threads(BLOCKSIZE);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csric0_hash_kernel<BLOCKSIZE, WFSIZE, HASH>),
                                           csric0_blocks,
                                           csric0_threads,
                                           0,
                                           handle->stream,
                                           static_cast<J>(m),
                                           reinterpret_cast<const I*>(csr_row_ptr),
                                           reinterpret_cast<const J*>(csr_col_ind),
                                           reinterpret_cast<T*>(csr_val),
                                           csr_val_stride,
                                           reinterpret_cast<const I*>(csr_diag_ind),
                                           done,
                                           done_stride,
                                           reinterpret_cast<const J*>(map),
                                           reinterpret_cast<J*>(zero_pivot),
                                           zero_pivot_stride,
                                           reinterpret_cast<J*>(singular_pivot),
                                           singular_pivot_stride,
                                           tol,
                                           idx_base);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename I, typename J>
    static rocsparse_status csric0_launch_binsearch_kernel(rocsparse_handle handle,
                                                           int64_t          batch_count,
                                                           int64_t          m,
                                                           const void* __restrict__ csr_row_ptr,
                                                           const void* __restrict__ csr_col_ind,
                                                           void* __restrict__ csr_val,
                                                           int64_t csr_val_stride,
                                                           const void* __restrict__ csr_diag_ind,
                                                           int32_t* __restrict__ done,
                                                           int64_t done_stride,
                                                           const void* __restrict__ map,
                                                           void* __restrict__ zero_pivot,
                                                           int64_t zero_pivot_stride,
                                                           void* __restrict__ singular_pivot,
                                                           int64_t singular_pivot_stride,
                                                           double  tol,
                                                           rocsparse_index_base idx_base)
    {
        const dim3 csric0_blocks((m * handle->wavefront_size - 1) / BLOCKSIZE + 1, batch_count);
        const dim3 csric0_threads(BLOCKSIZE);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csric0_binsearch_kernel<BLOCKSIZE, WFSIZE, SLEEP, T, I, J>),
            csric0_blocks,
            csric0_threads,
            0,
            handle->stream,
            m,
            reinterpret_cast<const I*>(csr_row_ptr),
            reinterpret_cast<const J*>(csr_col_ind),
            reinterpret_cast<T*>(csr_val),
            csr_val_stride,
            reinterpret_cast<const I*>(csr_diag_ind),
            done,
            done_stride,
            reinterpret_cast<const J*>(map),
            reinterpret_cast<J*>(zero_pivot),
            zero_pivot_stride,
            reinterpret_cast<J*>(singular_pivot),
            singular_pivot_stride,
            tol,
            idx_base);
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*csric0_launch_kernel_t)(rocsparse_handle handle,
                                                       int64_t          batch_count,
                                                       int64_t          m,
                                                       const void* __restrict__ csr_row_ptr,
                                                       const void* __restrict__ csr_col_ind,
                                                       void* __restrict__ csr_val,
                                                       int64_t csr_val_stride,
                                                       const void* __restrict__ csr_diag_ind,
                                                       int32_t* __restrict__ done,
                                                       int64_t done_stride,
                                                       const void* __restrict__ map,
                                                       void* __restrict__ zero_pivot,
                                                       int64_t zero_pivot_stride,
                                                       void* __restrict__ singular_pivot,
                                                       int64_t              singular_pivot_stride,
                                                       double               tol,
                                                       rocsparse_index_base idx_base);

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     SLEEP,
              typename T,
              typename I,
              typename... P>
    static csric0_launch_kernel_t find_csric0_binsearch_j(rocsparse_indextype j, P... p)
    {
        return (j == rocsparse_indextype_i32) ? rocsparse::
                       csric0_launch_binsearch_kernel<BLOCKSIZE, WF_SIZE, SLEEP, T, I, int32_t>
               : (j == rocsparse_indextype_i64) ? rocsparse::
                       csric0_launch_binsearch_kernel<BLOCKSIZE, WF_SIZE, SLEEP, T, I, int64_t>
                                                : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename T, typename... P>
    static csric0_launch_kernel_t find_csric0_binsearch_i(rocsparse_indextype i, P... p)
    {
        return (i == rocsparse_indextype_i32)
                   ? rocsparse::find_csric0_binsearch_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int32_t>(p...)
               : (i == rocsparse_indextype_i64)
                   ? rocsparse::find_csric0_binsearch_j<BLOCKSIZE, WF_SIZE, SLEEP, T, int64_t>(p...)
                   : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename... P>
    static csric0_launch_kernel_t find_csric0_binsearch_t(rocsparse_datatype i, P... p)
    {
        return (i == rocsparse_datatype_f32_r) //
                   ? rocsparse::find_csric0_binsearch_i<BLOCKSIZE,
                                                        WF_SIZE,
                                                        SLEEP,
                                                        float>(p...) //
                   : (i == rocsparse_datatype_f32_c) //
                         ? rocsparse::find_csric0_binsearch_i<BLOCKSIZE,
                                                              WF_SIZE,
                                                              SLEEP,
                                                              rocsparse_float_complex>(p...) //
                         : (i == rocsparse_datatype_f64_c) //
                               ? rocsparse::find_csric0_binsearch_i<BLOCKSIZE,
                                                                    WF_SIZE,
                                                                    SLEEP,
                                                                    rocsparse_double_complex>(
                                   p...) //
                               : (i == rocsparse_datatype_f64_r) //
                                     ? rocsparse::find_csric0_binsearch_i<BLOCKSIZE,
                                                                          WF_SIZE,
                                                                          SLEEP,
                                                                          double>(p...) //
                                     : nullptr;
    }

    static csric0_launch_kernel_t find_csric0_binsearch_kernel(uint32_t            blocksize_,
                                                               uint32_t            wfsize_,
                                                               bool                sleep_,
                                                               rocsparse_datatype  t_type,
                                                               rocsparse_indextype i_type,
                                                               rocsparse_indextype j_type)
    {
        // Determine gcnArch and ASIC revision
        if(blocksize_ == 256 && ((wfsize_ == 32) && (sleep_ == false)))
        {
            return rocsparse::find_csric0_binsearch_t<256, 32, false>(t_type, i_type, j_type);
        }
        else if(blocksize_ == 256 && ((wfsize_ == 64) && (sleep_ == false)))
        {
            return rocsparse::find_csric0_binsearch_t<256, 64, false>(t_type, i_type, j_type);
        }
        else if(blocksize_ == 256 && ((wfsize_ == 64) && (sleep_ == true)))
        {
            return rocsparse::find_csric0_binsearch_t<256, 64, true>(t_type, i_type, j_type);
        }
        else
        {
            return nullptr;
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t HASH,
              typename T,
              typename I,
              typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_j(const rocsparse_indextype j, P... p)
    {
        return //
            (j == rocsparse_indextype_i32) //
                ? rocsparse::csric0_launch_hash_kernel<BLOCKSIZE,
                                                       WF_SIZE,
                                                       HASH,
                                                       T,
                                                       I,
                                                       int32_t> //
                : (j == rocsparse_indextype_i64) //
                      ? rocsparse::csric0_launch_hash_kernel<BLOCKSIZE,
                                                             WF_SIZE,
                                                             HASH,
                                                             T,
                                                             I,
                                                             int64_t> //
                      : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename T, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_i(const rocsparse_indextype i, P... p)
    {
        return //
            (i == rocsparse_indextype_i32) //
                ? rocsparse::find_csric0_hash_kernel_j<BLOCKSIZE,
                                                       WF_SIZE,
                                                       HASH,
                                                       T,
                                                       int32_t>(p...) //
                : (i == rocsparse_indextype_i64) //
                      ? rocsparse::find_csric0_hash_kernel_j<BLOCKSIZE,
                                                             WF_SIZE,
                                                             HASH,
                                                             T,
                                                             int64_t>(p...) //
                      : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, uint32_t HASH, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_t(const rocsparse_datatype i, P... p)
    {
        return //
            (i == rocsparse_datatype_f32_r) //
                ? rocsparse::find_csric0_hash_kernel_i<BLOCKSIZE,
                                                       WF_SIZE,
                                                       HASH,
                                                       float>(p...) //
                : (i == rocsparse_datatype_f32_c) //
                      ? rocsparse::find_csric0_hash_kernel_i<BLOCKSIZE,
                                                             WF_SIZE,
                                                             HASH,
                                                             rocsparse_float_complex>(p...) //
                      : (i == rocsparse_datatype_f64_c) //
                            ? rocsparse::find_csric0_hash_kernel_i<BLOCKSIZE,
                                                                   WF_SIZE,
                                                                   HASH,
                                                                   rocsparse_double_complex>(
                                p...) //
                            : (i == rocsparse_datatype_f64_r) //
                                  ? rocsparse::
                                      find_csric0_hash_kernel_i<BLOCKSIZE, WF_SIZE, HASH, double>(
                                          p...) //
                                  : nullptr;
    }

    template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_mxnnz(const int32_t max_nnz, P... p)
    {
        return //
            (max_nnz <= 32) //
                ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 1>(p...) //
                : (max_nnz <= 64) //
                      ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 2>(p...) //
                      : (max_nnz <= 128) //
                            ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 4>(p...) //
                            : (max_nnz <= 256) //
                                  ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE, WF_SIZE, 8>(
                                      p...) //
                                  : (max_nnz <= 512) //
                                        ? rocsparse::find_csric0_hash_kernel_t<BLOCKSIZE,
                                                                               WF_SIZE,
                                                                               16>(p...) //
                                        : nullptr;
    }

    template <uint32_t BLOCKSIZE, typename... P>
    static csric0_launch_kernel_t find_csric0_hash_kernel_wf(const int32_t i, P... p)
    {
        return //
            (i == 32) //
                ? rocsparse::find_csric0_hash_kernel_mxnnz<BLOCKSIZE, 32>(p...) //
                : (i == 64) //
                      ? rocsparse::find_csric0_hash_kernel_mxnnz<BLOCKSIZE, 64>(p...) //
                      : nullptr; //
    }

    static csric0_launch_kernel_t find_csric0_hash_kernel(const uint32_t            blocksize_,
                                                          const uint32_t            wfsize,
                                                          const uint32_t            max_nnz,
                                                          const rocsparse_datatype  t_type,
                                                          const rocsparse_indextype i_type,
                                                          const rocsparse_indextype j_type)
    {
        if(blocksize_ == 256)
        {
            return rocsparse::find_csric0_hash_kernel_wf<256>(
                wfsize, max_nnz, t_type, i_type, j_type);
        }
        else
        {
            return nullptr;
        }
    }

}

rocsparse_status rocsparse::csric0_launch_kernel(rocsparse_handle    handle,
                                                 int64_t             batch_count,
                                                 int64_t             m,
                                                 rocsparse_indextype csr_ptr_row_indextype,
                                                 const void* __restrict__ csr_row_ptr,
                                                 rocsparse_indextype csr_col_ind_indextype,
                                                 const void* __restrict__ csr_col_ind,
                                                 rocsparse_datatype csr_val_datatype,
                                                 void* __restrict__ csr_val,
                                                 int64_t csr_val_stride,
                                                 const void* __restrict__ csr_diag_ind,
                                                 int32_t* __restrict__ done,
                                                 int64_t done_stride,
                                                 const void* __restrict__ map,
                                                 void* __restrict__ zero_pivot,
                                                 int64_t zero_pivot_stride,
                                                 void* __restrict__ singular_pivot,
                                                 int64_t              singular_pivot_stride,
                                                 double               tol,
                                                 rocsparse_index_base idx_base,
                                                 int64_t              max_nnz)
{
    const bool sleep = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908
                        && handle->asic_rev < 2);

    auto launch_kernel
        = (sleep || max_nnz > 1024)
              ? rocsparse::find_csric0_binsearch_kernel(256,
                                                        (sleep) ? 64 : handle->wavefront_size,
                                                        sleep,
                                                        csr_val_datatype,
                                                        csr_ptr_row_indextype,
                                                        csr_col_ind_indextype)
              : rocsparse::find_csric0_hash_kernel(256,
                                                   handle->wavefront_size,
                                                   max_nnz,
                                                   csr_val_datatype,
                                                   csr_ptr_row_indextype,
                                                   csr_col_ind_indextype);

    RETURN_IF_ROCSPARSE_ERROR(launch_kernel(handle,
                                            batch_count,
                                            m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            csr_val_stride,
                                            csr_diag_ind,
                                            done,
                                            done_stride,
                                            map,
                                            zero_pivot,
                                            zero_pivot_stride,
                                            singular_pivot,
                                            singular_pivot_stride,
                                            tol,
                                            idx_base));

    return rocsparse_status_success;
}
