/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvn_general_device(bool                 conj,
                                                    J                    m,
                                                    T                    alpha,
                                                    const I*             row_offset_begin,
                                                    const I*             row_offset_end,
                                                    const J*             csr_col_ind,
                                                    const A*             csr_val,
                                                    const X*             x,
                                                    T                    beta,
                                                    Y*                   y,
                                                    rocsparse_index_base idx_base)
    {
        const int lid = hipThreadIdx_x & (WF_SIZE - 1);

        const J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        const J nwf = hipGridDim_x * (BLOCKSIZE / WF_SIZE);

        // Loop over rows
        for(J row = gid / WF_SIZE; row < m; row += nwf)
        {
            // Each wavefront processes one row
            const I row_start = row_offset_begin[row] - idx_base;
            const I row_end   = row_offset_end[row] - idx_base;

            T sum = static_cast<T>(0);

            // Loop over non-zero elements
            for(I j = row_start + lid; j < row_end; j += WF_SIZE)
            {
                sum = rocsparse::fma<T>(alpha * rocsparse::conj_val(csr_val[j], conj),
                                        rocsparse::ldg(x + csr_col_ind[j] - idx_base),
                                        sum);
            }

            // Obtain row sum using parallel reduction
            sum = rocsparse::wfreduce_sum<WF_SIZE>(sum);

            // First thread of each wavefront writes result into global memory
            if(lid == WF_SIZE - 1)
            {
                if(beta == static_cast<T>(0))
                {
                    y[row] = sum;
                }
                else
                {
                    y[row] = rocsparse::fma<T>(beta, y[row], sum);
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvt_general_device(bool                 skip_diag,
                                                    bool                 conj,
                                                    J                    m,
                                                    T                    alpha,
                                                    const I*             csr_row_ptr_begin,
                                                    const I*             csr_row_ptr_end,
                                                    const J*             csr_col_ind,
                                                    const A*             csr_val,
                                                    const X*             x,
                                                    Y*                   y,
                                                    rocsparse_index_base idx_base)
    {
        const int lid = hipThreadIdx_x & (WF_SIZE - 1);

        const J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        const J inc = hipGridDim_x * (BLOCKSIZE / WF_SIZE);

        if(skip_diag)
        {
            for(J row = gid / WF_SIZE; row < m; row += inc)
            {
                const I row_begin = csr_row_ptr_begin[row] - idx_base;
                const I row_end   = csr_row_ptr_end[row] - idx_base;
                const T row_val   = alpha * x[row];

                for(I j = row_begin + lid; j < row_end; j += WF_SIZE)
                {
                    const J col = csr_col_ind[j] - idx_base;

                    if(col != row)
                    {
                        const A val = rocsparse::conj_val(csr_val[j], conj);
                        rocsparse::atomic_add(&y[col], row_val * val);
                    }
                }
            }
        }
        else
        {
            for(J row = gid / WF_SIZE; row < m; row += inc)
            {
                const I row_begin = csr_row_ptr_begin[row] - idx_base;
                const I row_end   = csr_row_ptr_end[row] - idx_base;
                const T row_val   = alpha * x[row];

                for(I j = row_begin + lid; j < row_end; j += WF_SIZE)
                {
                    const J col = csr_col_ind[j] - idx_base;

                    const A val = rocsparse::conj_val(csr_val[j], conj);
                    rocsparse::atomic_add(&y[col], row_val * val);
                }
            }
        }
    }

    // Scales y by beta in the range [0, first_row) and [last_row, m)
    template <uint32_t WG_SIZE, typename J, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void partial_scale_y_device(J m, J first_row, J last_row, T beta, Y* y)
    {
        const J gid              = hipBlockIdx_x * WG_SIZE + hipThreadIdx_x;
        const J required_threads = (first_row - 0) + (m - last_row);

        if(gid >= required_threads)
        {
            return;
        }

        if(gid < first_row)
        {
            if(beta == static_cast<T>(0))
            {
                y[gid] = static_cast<Y>(0);
            }
            else
            {
                y[gid] *= beta;
            }
        }
        else
        {
            if(beta == static_cast<T>(0))
            {
                y[last_row + (gid - first_row)] = static_cast<Y>(0);
            }
            else
            {
                y[last_row + (gid - first_row)] *= beta;
            }
        }
    }

    template <typename I, typename T>
    ROCSPARSE_DEVICE_ILF T sum2_reduce(T cur_sum, T* partial, int lid, I max_size, int reduc_size)
    {
        if(max_size > reduc_size)
        {
            cur_sum = cur_sum + partial[lid + reduc_size];
            __syncthreads();
            partial[lid] = cur_sum;
        }
        return cur_sum;
    }

    template <rocsparse_int BLOCKSIZE,
              rocsparse_int BLOCK_MULTIPLIER,
              rocsparse_int ROWS_FOR_VECTOR,
              rocsparse_int WG_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvn_adaptive_device(bool                 conj,
                                                     I                    nnz,
                                                     const I*             row_blocks,
                                                     uint32_t*            wg_flags,
                                                     const J*             wg_ids,
                                                     T                    alpha,
                                                     const I*             csr_row_ptr,
                                                     const J*             csr_col_ind,
                                                     const A*             csr_val,
                                                     const X*             x,
                                                     T                    beta,
                                                     Y*                   y,
                                                     rocsparse_index_base idx_base)
    {
        __shared__ T partialSums[BLOCKSIZE];

        const int lid = hipThreadIdx_x;
        const int gid = hipBlockIdx_x;

        // The row blocks buffer holds information used to inform each
        // workgroup about how to do its work:
        //
        // The rowBlock entry tell the workgroup the ID of the first
        // row it will be working on. When one workgroup calculates multiple rows, this
        // rowBlock entry and the next one tell it the range of rows to work on.
        // The wg_ids are used whenever multiple workgroups calculate a single long
        // row. This tells each workgroup its ID within that row, so it knows which
        // part of the row to operate on.
        // Alternately, on short row blocks, wg_ids are used to communicate
        // the number of threads that should be used for the reduction. Pre-calculating
        // this on the CPU-side results in a noticeable performance uplift on many matrices.
        // wg_flags contains the flag bit used so that the multiple WGs calculating a long row can
        // know when the first workgroup for that row has finished initializing the output
        // value. While this bit is the same as the first workgroup's flag bit, this
        // workgroup will spin-loop.
        const I row      = row_blocks[gid];
        const I stop_row = row_blocks[gid + 1];
        const J num_rows = stop_row - row;

        // Get the workgroup within this long row ID
        const J wg = wg_ids[gid];

        // Any workgroup only calculates, at most, BLOCK_MULTIPLIER*BLOCKSIZE items in a row.
        // If there are more items in this row, we assign more workgroups.
        const I row_offset = rocsparse::nontemporal_load(csr_row_ptr + row);

        T temp_sum = static_cast<T>(0);

        // If the next row block starts more than 2 rows away, then we choose CSR-Stream.
        // If this is zero (long rows) or one (final workgroup in a long row, or a single
        // row in a row block), we want to use the CSR-Vector algorithm(s).
        // We have found, through experimentation, that CSR-Vector is generally faster
        // when working on 2 rows, due to its simplicity and better reduction method.
        if(num_rows > ROWS_FOR_VECTOR)
        {
            // CSR-Stream case. See Sections III.A and III.B in the SC'14 paper:
            // Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format
            // for a detailed description of CSR-Stream.
            // In a nutshell, the idea is to use all of the threads to stream the matrix
            // values into the local memory in a fast, coalesced manner. After that, the
            // per-row reductions are done out of the local memory, which is designed
            // to handle non-coalsced accesses.

            // The best method for reducing the local memory values depends on the number
            // of rows. The SC'14 paper discusses a CSR-Scalar style reduction where
            // each thread reduces its own row. This yields good performance if there
            // are many (relatively short) rows. However, if they are few (relatively
            // long) rows, it's actually better to perform a tree-style reduction where
            // multiple threads team up to reduce the same row.

            // The calculation below tells you how many threads this workgroup can allocate
            // to each row, assuming that every row gets the same number of threads.
            // We want the closest lower (or equal) power-of-2 to this number --
            // that is how many threads can work in each row's reduction using our algorithm.
            // For instance, with workgroup size 256, 2 rows = 128 threads, 3 rows = 64
            // threads, 4 rows = 64 threads, 5 rows = 32 threads, etc.
            // int numThreadsForRed = get_local_size(0) >> ((CHAR_BIT*sizeof(unsigned
            // int))-clz(num_rows-1));
            const J numThreadsForRed = wg; // Same calculation as above, done on host.

            // Stream all of this row block's matrix values into local memory.
            // Perform the matvec in parallel with this work.
            const I col = row_offset + lid - idx_base;
            if(col + BLOCKSIZE - WG_SIZE < nnz)
            {
                for(J i = 0; i < BLOCKSIZE; i += WG_SIZE)
                {
                    partialSums[lid + i] = alpha * rocsparse::conj_val(csr_val[col + i], conj)
                                           * x[csr_col_ind[col + i] - idx_base];
                }
            }
            else
            {
                // This is required so that we stay in bounds for csr_val[] and csr_col_ind[].
                // Otherwise, if the matrix's endpoints don't line up with BLOCKSIZE,
                // we will buffer overflow. On today's dGPUs, this doesn't cause problems.
                // The values are within a dGPU's page, which is zeroed out on allocation.
                // However, this may change in the future (e.g. with shared virtual memory.)
                // This causes a minor performance loss because this is the last workgroup
                // to be launched, and this loop can't be unrolled.
                for(I i = 0; col + i < csr_row_ptr[stop_row] - idx_base; i += WG_SIZE)
                {
                    partialSums[lid + i] = alpha * rocsparse::conj_val(csr_val[col + i], conj)
                                           * x[csr_col_ind[col + i] - idx_base];
                }
            }
            __syncthreads();

            if(numThreadsForRed > 1)
            {
                // In this case, we want to have the workgroup perform a tree-style reduction
                // of each row. {numThreadsForRed} adjacent threads team up to linearly reduce
                // a row into {numThreadsForRed} locations in local memory.
                // After that, the entire workgroup does a parallel reduction, and each
                // row ends up with an individual answer.

                // {numThreadsForRed} adjacent threads all work on the same row, so their
                // start and end values are the same.
                // numThreadsForRed guaranteed to be a power of two, so the clz code below
                // avoids an integer divide.
                // size_t st = lid/numThreadsForRed;
                const I local_row       = row + (lid >> (31 - __clz(numThreadsForRed)));
                const J local_first_val = csr_row_ptr[local_row] - row_offset;
                const J local_last_val  = csr_row_ptr[local_row + 1] - row_offset;
                const J threadInBlock   = lid & (numThreadsForRed - 1);

                // Not all row blocks are full -- they may have an odd number of rows. As such,
                // we need to ensure that adjacent-groups only work on real data for this rowBlock.
                if(local_row < stop_row)
                {
                    // This is dangerous -- will infinite loop if your last value is within
                    // numThreadsForRed of MAX_UINT. Noticable performance gain to avoid a
                    // long induction variable here, though.
                    for(J local_cur_val = local_first_val + threadInBlock;
                        local_cur_val < local_last_val;
                        local_cur_val += numThreadsForRed)
                    {
                        temp_sum = temp_sum + partialSums[local_cur_val];
                    }
                }
                __syncthreads();

                partialSums[lid] = temp_sum;

                // Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
                // values sitting in the local memory. This means that, roughly, the beginning of
                // LDS is full up to {workgroup size} entries.
                // Now we perform a parallel reduction that sums together the answers for each
                // row in parallel, leaving us an answer in 'temp_sum' for each row.
                for(int i = (WG_SIZE >> 1); i > 0; i >>= 1)
                {
                    __syncthreads();
                    temp_sum = sum2_reduce(temp_sum, partialSums, lid, numThreadsForRed, i);
                }

                if(threadInBlock == 0 && local_row < stop_row)
                {
                    // All of our write-outs check to see if the output vector should first be zeroed.
                    // If so, just do a write rather than a read-write. Measured to be a slight (~5%)
                    // performance improvement.
                    if(beta != static_cast<T>(0))
                    {
                        temp_sum = rocsparse::fma<T>(beta, y[local_row], temp_sum);
                    }
                    y[local_row] = temp_sum;
                }
            }
            else
            {
                // In this case, we want to have each thread perform the reduction for a single row.
                // Essentially, this looks like performing CSR-Scalar, except it is computed out of
                // local memory.
                // However, this reduction is also much faster than CSR-Scalar, because local memory
                // is designed for scatter-gather operations.
                // We need a while loop because there may be more rows than threads in the WG.
                I local_row = row + lid;
                while(local_row < stop_row)
                {
                    const J local_first_val = (csr_row_ptr[local_row] - row_offset);
                    const J local_last_val  = csr_row_ptr[local_row + 1] - row_offset;
                    temp_sum                = static_cast<T>(0);
                    for(J local_cur_val = local_first_val; local_cur_val < local_last_val;
                        ++local_cur_val)
                    {
                        temp_sum = temp_sum + partialSums[local_cur_val];
                    }

                    // After you've done the reduction into the temp_sum register,
                    // put that into the output for each row.
                    if(beta != static_cast<T>(0))
                    {
                        temp_sum = rocsparse::fma<T>(beta, y[local_row], temp_sum);
                    }

                    y[local_row] = temp_sum;
                    local_row += WG_SIZE;
                }
            }
        }
        else if(num_rows >= 1 && !wg) // CSR-Vector case.
        {
            // ^^ The above check says that if this workgroup is supposed to work on <= ROWS_VECTOR
            // number of rows then we should do the CSR-Vector algorithm. If we want this row to be
            // done with CSR-LongRows, then all of its workgroups (except the last one) will have the
            // same stop_row and row. The final workgroup in a LongRow will have stop_row and row
            // different, but the internal wg number will be non-zero.

            // If this workgroup is operating on multiple rows (because CSR-Stream is poor for small
            // numbers of rows), then it needs to iterate until it reaches the stop_row.
            // We don't check <= stop_row because of the potential for unsigned overflow.
            I r = row;
            while(r < stop_row)
            {
                // Any workgroup only calculates, at most, BLOCKSIZE items in this row.
                // If there are more items in this row, we use CSR-LongRows.
                temp_sum         = static_cast<T>(0);
                const I vecStart = csr_row_ptr[r] - idx_base;
                const I vecEnd   = csr_row_ptr[r + 1] - idx_base;

                // Load in a bunch of partial results into your register space, rather than LDS (no
                // contention)
                // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
                // Using a long induction variable to make sure uint32_t overflow doesn't break
                // things.
                for(I j = vecStart + lid; j < vecEnd; j += WG_SIZE)
                {
                    temp_sum = rocsparse::fma<T>(alpha * rocsparse::conj_val(csr_val[j], conj),
                                                 x[csr_col_ind[j] - idx_base],
                                                 temp_sum);
                }

                partialSums[lid] = temp_sum;

                __syncthreads();

                // Reduce partial sums
                rocsparse::blockreduce_sum<WG_SIZE>(lid, partialSums);

                if(lid == 0)
                {
                    temp_sum = partialSums[0];

                    if(beta != static_cast<T>(0))
                    {
                        temp_sum = rocsparse::fma<T>(beta, y[r], temp_sum);
                    }

                    y[r] = temp_sum;
                }
                ++r;
            }
        }
        else
        {
            // In CSR-LongRows, we have more than one workgroup calculating this row.
            // The output values for those types of rows are stored using atomic_add, because
            // more than one parallel workgroup's value makes up the final answer.
            // Unfortunately, this makes it difficult to do y=Ax, rather than y=Ax+y, because
            // the values still left in y will be added in using the atomic_add.
            //
            // Our solution is to have the first workgroup in one of these long-rows cases
            // properly initaizlie the output vector. All the other workgroups working on this
            // row will spin-loop until that workgroup finishes its work.

            // First, figure out which workgroup you are in the row.
            // You can use that to find the global ID for the first workgroup calculating
            // this long row.
            J        first_wg_in_row = gid - wg_ids[gid];
            uint32_t compare_value   = wg_flags[gid];

            // wg_flags[first_wg_in_row] in the first workgroup is the flag that everyone waits on.
            if(gid == first_wg_in_row && lid == 0)
            {
                // The first workgroup handles the output initialization.
                const Y out_val = y[row];
                temp_sum        = (beta - static_cast<T>(1)) * out_val;

                // All inter thread communication is done using atomics, therefore cache flushes or
                // invalidates should not be needed (thus __threadfence() has been removed to regain
                // performance).
                // Because of atomics being relaxed, however, the compiler is allowed to reorder them
                // with respect to ordinary memory accesses (and other relaxed atomic operations).
                // In this case, out_val seem to be reordered with the xor and subsequently, accumulation
                // ends up being wrong.
                // To force the compiler to stick to the order of operations, we need acquire/release fences.
                // Workgroup scope is sufficient for this purpose, to only invalidate L1 and avoid L2
                // invalidations.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define __gfx12__
#endif
#if defined(__gfx12__)
                __threadfence();
#else
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
                __builtin_amdgcn_s_waitcnt(0);
#endif

                // Release other workgroups
                atomicXor(&wg_flags[first_wg_in_row], 1U);
            }

            const I vecStart = (I)wg * (I)BLOCK_MULTIPLIER * BLOCKSIZE + row_offset - idx_base;
            const I vecEnd   = rocsparse::min(csr_row_ptr[row + 1] - idx_base,
                                            vecStart + I(BLOCK_MULTIPLIER * BLOCKSIZE));

            // Load in a bunch of partial results into your register space, rather than LDS (no
            // contention)
            // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
            for(I j = vecStart + lid; j < vecEnd; j += WG_SIZE)
            {
                temp_sum = rocsparse::fma<T>(alpha * rocsparse::conj_val(csr_val[j], conj),
                                             x[csr_col_ind[j] - idx_base],
                                             temp_sum);
            }

            partialSums[lid] = temp_sum;

            __syncthreads();

            // Reduce partial sums
            rocsparse::blockreduce_sum<WG_SIZE>(lid, partialSums);

            // For every other workgroup, wg_flags[first_wg_in_row] holds the value they wait on.
            // If your flag == first_wg's flag, you spin loop.
            // The first workgroup will eventually flip this flag, and you can move forward.
            if(lid == 0)
            {
                if(gid != first_wg_in_row)
                {
                    while(rocsparse::atomic_max(&wg_flags[first_wg_in_row], 0U) == compare_value)
                        ;

                    // __builtin_amdgcn_s_waitcnt(0);
                    // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");

                    // After you've passed the barrier, update your local flag to make sure that
                    // the next time through, you know what to wait on.
                    wg_flags[gid] ^= 1U;
                }

                rocsparse::atomic_add(y + row, partialSums[0]);
            }
        }
    }

    // Compute row lengths, and atomically increment each bin based on length.
    // For each row, use that same atomic op to store the row's intended index within its bin.
    // This permits us to use a single array of length n_rows to store all the bins,
    // given an auxiliary structure storing the start index of each bin in that array.
    // Output:  rows_binoffsets_scratch = <uint32_t* of length csr.rows>. Temp storage for each row's
    //          atomically-calculated offset into its respective row-bin, EXCLUDING the row-bin start idx.
    // Output:  n_rows_bins = <array of 32 uint32_t's>, where the value at index i is the number of rows
    //          in bin i. May want to be particular about spreading across cache banks (w/ padding),
    //          for atomics-load-balancing reasons.
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvn_preprocess_device_32_bins_3phase_phase1(J        m,
                                                        const I* csr_row_ptr,
                                                        J*       rows_binoffsets_scratch,
                                                        J*       n_rows_bins)
    {
        const J gid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;

        for(J i = gid; i < m; i += BLOCKSIZE * hipGridDim_x)
        {
            const I row_len = csr_row_ptr[i + 1] - csr_row_ptr[i];

            const uint32_t target_bin
                = (row_len != 0) ? (uint32_t)rocsparse::ceil(rocsparse::log2(row_len)) : 0;

            rows_binoffsets_scratch[i] = rocsparse::atomic_add(&n_rows_bins[target_bin], (J)1);
        }
    }

    // Compute the start index of each bin, based on the bin lengths we computed in phase 1.
    // At present, for simplicity, we do this in a single GPU thread from a single workgroup,
    // since we only have 32 bins and the computation itself is pretty trivial.
    // This is in a separate kernel right now for synchronization reasons, since we need a barrier
    // across all workgroups. A different, intra-kernel, synchronization method could be used instead
    // (e.g. cooperative groups or global flags with spin-waits) to make the preprocessing a single
    // kernel launch instead of three.
    // Input/output:    n_rows_bins = <array of 32 J's>. Input value at index i is the number of rows
    //                  in bin i; output value is the starting index of each bin in the phase-3 rows_bins
    //                  output array.
    template <typename J>
    ROCSPARSE_KERNEL(1)
    void csrmvn_preprocess_device_32_bins_3phase_phase2(J* n_rows_bins)
    {
        J acc = 0;
        for(int i = 0; i < 32; i++)
        {
            const J tmp    = n_rows_bins[i];
            n_rows_bins[i] = acc;
            acc += tmp;
        }
    }

    // Append the rows to the appropriate bins.
    // Since we use three separate kernel launches, we also recompute the row lengths rather than storing them;
    // this could be removed if using a different synchronization method.
    // Input:   rows_binoffsets_scratch = <J* of length csr.rows>. Temp storage for each row's
    //          atomically-calculated offset into its respective row-bin, EXCLUDING the row-bin start idx.
    // Input:   n_rows_bins = <array of 32 J's>, where the value at index i is the starting index
    //          of each bin in the final output array.
    // Output:  rows_bins = <J* of length csr.rows>. Single array with all bins:
    //          [(bin0 first row #), ... (bin0 last row #), (bin1 first row #), ... (bin1 last row #), ...]
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvn_preprocess_device_32_bins_3phase_phase3(J        m,
                                                        const I* csr_row_ptr,
                                                        const J* rows_binoffsets_scratch,
                                                        const J* n_rows_bins,
                                                        J*       rows_bins)
    {
        const J gid = (BLOCKSIZE * hipBlockIdx_x) + hipThreadIdx_x;

        for(J i = gid; i < m; i += BLOCKSIZE * hipGridDim_x)
        {
            const I row_len = csr_row_ptr[i + 1] - csr_row_ptr[i];

            const uint32_t target_bin
                = (row_len != 0) ? (uint32_t)rocsparse::ceil(rocsparse::log2(row_len)) : 0;

            rows_bins[n_rows_bins[target_bin] + rows_binoffsets_scratch[i]] = i;
        }
    }

    // "Stream" case a la CSR-Adaptive
    template <uint32_t BLOCKSIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvn_lrb_short_rows_device(bool                 conj,
                                                           I                    nnz,
                                                           const J*             rows_bins,
                                                           const J*             n_rows_bins,
                                                           uint32_t             bin_id,
                                                           T                    alpha,
                                                           const I*             csr_row_ptr,
                                                           const J*             csr_col_ind,
                                                           const A*             csr_val,
                                                           const X*             x,
                                                           T                    beta,
                                                           Y*                   y,
                                                           rocsparse_index_base idx_base)
    {
        const uint32_t lid = hipThreadIdx_x;
        const uint32_t gid = hipBlockIdx_x;

        // Allocation from the caller is of size [WG_SIZE << bin_id] elements
        extern __shared__ char shared_memory[];
        T*                     partialSums = (T*)shared_memory;

        const J bin_start    = n_rows_bins[bin_id];
        const J bin_num_rows = n_rows_bins[bin_id + 1] - bin_start;

        const uint32_t wg_row_start = gid * BLOCKSIZE;
        const uint32_t wg_row_end
            = rocsparse::min(wg_row_start + BLOCKSIZE, uint32_t(bin_num_rows));
        const uint32_t wg_num_rows = wg_row_end - wg_row_start;

        // Load a block of row data using all threads in the WG.
        for(uint32_t base_idx = 0; base_idx < (BLOCKSIZE << bin_id); base_idx += BLOCKSIZE)
        {
            uint32_t row_idx = wg_row_start + ((base_idx + lid) >> bin_id);
            if(row_idx < wg_row_start + wg_num_rows)
            {
                const J row_id = rows_bins[row_idx + bin_start];

                const I row_start = csr_row_ptr[row_id] - idx_base;
                const I row_end   = csr_row_ptr[row_id + 1] - idx_base;
                const I row_len   = row_end - row_start;

                const uint32_t col_idx_in_row = lid & ((1 << bin_id) - 1);
                const uint32_t lds_idx        = base_idx + lid;

                if(col_idx_in_row < row_len)
                {
                    const A val = rocsparse::conj_val(csr_val[row_start + col_idx_in_row], conj);
                    partialSums[lds_idx]
                        = alpha * val * x[csr_col_ind[row_start + col_idx_in_row] - idx_base];
                }
                else
                {
                    // lds <- 0
                    partialSums[lds_idx] = 0;
                }
            }
        }
        __syncthreads();

        // For the moment: just have each thread reduce a given row
        // TODO: adaptation as per CSR-Adaptive-Stream
        if(lid < wg_num_rows)
        {
            const uint32_t lds_start_idx = (lid << bin_id);
            const J        row_id        = rows_bins[bin_start + wg_row_start + lid];
            T              acc           = 0;

            for(uint32_t idx = 0; idx < (1 << bin_id); idx++)
            {
                acc += partialSums[lds_start_idx + idx];
            }

            if(beta != 0)
            {
                acc = rocsparse::fma<T>(beta, y[row_id], acc);
            }
            y[row_id] = acc;
        }
    }

    // csrmv_lrb_short_rows_2: Same basic structure as csrmv_lrb_short_rows,
    // but with a fixed-size LDS allocation. Intended for cases where the former's
    // dynamic LDS allocation approach would blow up size requirements beyond reasonable bounds.
    template <uint32_t BLOCKSIZE,
              uint32_t CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvn_lrb_short_rows_2_device(bool                 conj,
                                                             I                    nnz,
                                                             const J*             rows_bins,
                                                             const J*             n_rows_bins,
                                                             const uint32_t       bin_id,
                                                             T                    alpha,
                                                             const I*             csr_row_ptr,
                                                             const J*             csr_col_ind,
                                                             const A*             csr_val,
                                                             const X*             x,
                                                             T                    beta,
                                                             Y*                   y,
                                                             rocsparse_index_base idx_base)
    {
        const uint32_t lid = hipThreadIdx_x;
        const uint32_t gid = hipBlockIdx_x;

        // In LDS-Stream-V2, we have a fixed LDS allocation of CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS elements (example: 1024).
        // So, each thread will load 1024 / BLOCKSIZE elements, which, depending on the bin, corresponds to a different # rows.
        // Bin 0 (1 element) = 1024 rows, bin 1 = 512 rows, bin 2 = 256 rows, bin 3 = 128 rows, etc.
        __shared__ T partialSums[CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS];

        const J        bin_start    = n_rows_bins[bin_id];
        const J        bin_num_rows = n_rows_bins[bin_id + 1] - bin_start;
        const uint32_t rows_per_wg  = CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS >> bin_id;
        const uint32_t wg_row_start = gid * rows_per_wg;
        const uint32_t wg_row_end
            = rocsparse::min(wg_row_start + rows_per_wg, uint32_t(bin_num_rows));
        const uint32_t wg_num_rows = wg_row_end - wg_row_start;

        // Load a block of row data using all threads in the WG.
        for(uint32_t base_idx = 0; base_idx < CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS;
            base_idx += BLOCKSIZE)
        {
            uint32_t row_idx = wg_row_start + ((base_idx + lid) >> bin_id);
            if(row_idx < wg_row_end)
            {
                const J row_id = rows_bins[row_idx + bin_start];

                const I row_start = csr_row_ptr[row_id] - idx_base;
                const I row_end   = csr_row_ptr[row_id + 1] - idx_base;
                const I row_len   = row_end - row_start;

                const uint32_t col_idx_in_row = lid & ((1 << bin_id) - 1);
                const uint32_t lds_idx        = base_idx + lid;
                if(col_idx_in_row < row_len)
                {
                    const A val = rocsparse::conj_val(csr_val[row_start + col_idx_in_row], conj);
                    partialSums[lds_idx]
                        = alpha * val * x[csr_col_ind[row_start + col_idx_in_row] - idx_base];
                }
                else
                {
                    // lds <- 0
                    partialSums[lds_idx] = 0;
                }
            }
        }
        __syncthreads();

        // For the moment: just have each thread reduce a given row
        // TODO: adaptation as per CSR-Adaptive-Stream
        for(uint32_t row_base = 0; row_base < rows_per_wg; row_base += BLOCKSIZE)
        {
            const uint32_t this_row_offset_in_wg = row_base + lid;
            if(this_row_offset_in_wg < wg_num_rows)
            {
                const uint32_t lds_start_idx = (this_row_offset_in_wg << bin_id);
                const J        row_id = rows_bins[bin_start + wg_row_start + this_row_offset_in_wg];

                T acc = 0;
                for(uint32_t idx = 0; idx < (1 << bin_id); idx++)
                {
                    acc += partialSums[lds_start_idx + idx];
                }

                if(beta != 0)
                    acc = rocsparse::fma<T>(beta, y[row_id], acc);
                y[row_id] = acc;
            }
        }
    }

    // "Vector" case a la CSR-Adaptive using one warp per row
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        csrmvn_lrb_medium_rows_warp_reduce_device(bool                 conj,
                                                  I                    nnz,
                                                  int64_t              count,
                                                  const J*             rows_bins,
                                                  const J*             n_rows_bins,
                                                  uint32_t             bin_id,
                                                  T                    alpha,
                                                  const I*             csr_row_ptr,
                                                  const J*             csr_col_ind,
                                                  const A*             csr_val,
                                                  const X*             x,
                                                  T                    beta,
                                                  Y*                   y,
                                                  rocsparse_index_base idx_base)
    {
        const int tid = hipThreadIdx_x;
        const int bid = hipBlockIdx_x;

        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / WF_SIZE;

        const int gid = (BLOCKSIZE / WF_SIZE) * bid + wid;

        if(gid >= count)
        {
            return;
        }

        const J bin_start = n_rows_bins[bin_id];
        const J row       = rows_bins[bin_start + gid];

        T       temp_sum = static_cast<T>(0);
        const I vecStart = csr_row_ptr[row] - idx_base;
        const I vecEnd   = csr_row_ptr[row + 1] - idx_base;

        for(I j = vecStart + lid; j < vecEnd; j += WF_SIZE)
        {
            temp_sum = rocsparse::fma<T>(alpha * rocsparse::conj_val(csr_val[j], conj),
                                         x[csr_col_ind[j] - idx_base],
                                         temp_sum);
        }

        // Obtain row sum using parallel warp reduction
        temp_sum = rocsparse::wfreduce_sum<WF_SIZE>(temp_sum);

        if(lid == WF_SIZE - 1)
        {
            if(beta != static_cast<T>(0))
            {
                temp_sum = rocsparse::fma<T>(beta, y[row], temp_sum);
            }

            y[row] = temp_sum;
        }
    }

    // "Vector" case a la CSR-Adaptive using one block per row
    template <uint32_t BLOCKSIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvn_lrb_medium_rows_device(bool                 conj,
                                                            I                    nnz,
                                                            const J*             rows_bins,
                                                            const J*             n_rows_bins,
                                                            uint32_t             bin_id,
                                                            T                    alpha,
                                                            const I*             csr_row_ptr,
                                                            const J*             csr_col_ind,
                                                            const A*             csr_val,
                                                            const X*             x,
                                                            T                    beta,
                                                            Y*                   y,
                                                            rocsparse_index_base idx_base)
    {
        const int lid = hipThreadIdx_x;
        const int gid = hipBlockIdx_x;

        // LRB-Vector case currently does exact 1:1 WG:row mapping - not doing multiple rows per WG here;
        // we'll work on that later if desired. But, that may be neither needed nor wanted,
        // since we're now launching with LDS and grid size parameters tuned according to row length.

        const J bin_start = n_rows_bins[bin_id];
        const J row       = rows_bins[bin_start + gid];

        // In CSR-Adaptive-Vector, each WG will allocate BLOCKSIZE LDS, but for rows shorter than BLOCKSIZE,
        // this wastes space. So, for LRB-Vector, we allocate from the caller according to bin_id (one LDS
        // element per WG thread).
        // To generalize LRB-Vector for higher #s of rows (since max WG size is finite), we simply limit WG
        // size from the caller; the only requirement is that we have LDS #elems == WG size (max=1024 on gfx90a).
        __shared__ T partialSums[BLOCKSIZE];

        // In its original form, any CSR-Vector workgroup only calculates, at most, BLOCKSIZE items in its row;
        // if there are more items in this row, CSR-LongRows is used instead.
        // In its LRB-ized form, we'll calculate as much as we need, with parallelism bounded only by grid size
        // (iterating over the row as needed for rows with length > WG size).
        // This means that we can more easily process everything with Vector that we would otherwise have done
        // with Longrows - which, in turn, means we can guarantee result reproducibility simply by avoiding Longrows
        // use (-> no non-determinstic atomics), just set the bin threshold for Longrows to "infinity" (or "32").
        T       temp_sum = static_cast<T>(0);
        const I vecStart = csr_row_ptr[row] - idx_base;
        const I vecEnd   = csr_row_ptr[row + 1] - idx_base;

        // Load in a bunch of partial results into your register space, rather than LDS (no contention)
        // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
        for(I j = vecStart + lid; j < vecEnd; j += BLOCKSIZE)
        {
            temp_sum = rocsparse::fma<T>(alpha * rocsparse::conj_val(csr_val[j], conj),
                                         x[csr_col_ind[j] - idx_base],
                                         temp_sum);
        }

        partialSums[lid] = temp_sum;

        __syncthreads();

        // Reduce partial sums
        rocsparse::blockreduce_sum<BLOCKSIZE>(lid, partialSums);

        if(lid == 0)
        {
            temp_sum = partialSums[0];

            if(beta != static_cast<T>(0))
            {
                temp_sum = rocsparse::fma<T>(beta, y[row], temp_sum);
            }

            y[row] = temp_sum;
        }
    }

    // "LongRows" aka "VectorL" case a la CSR-Adaptive
    template <uint32_t BLOCKSIZE,
              uint32_t BLOCK_MULTIPLIER,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmvn_lrb_long_rows_device(bool                 conj,
                                                          I                    nnz,
                                                          uint32_t*            wg_flags,
                                                          const J*             rows_bins,
                                                          const J*             n_rows_bins,
                                                          uint32_t             bin_id,
                                                          T                    alpha,
                                                          const I*             csr_row_ptr,
                                                          const J*             csr_col_ind,
                                                          const A*             csr_val,
                                                          const X*             x,
                                                          T                    beta,
                                                          Y*                   y,
                                                          rocsparse_index_base idx_base)
    {
        __shared__ T partialSums[BLOCKSIZE];

        const int lid = hipThreadIdx_x;
        const int gid = hipBlockIdx_x;

        const J        bin_start       = n_rows_bins[bin_id];
        const uint32_t bin_max_row_len = (1 << bin_id);
        const uint32_t num_wgs_per_row = (bin_max_row_len - 1) / (BLOCK_MULTIPLIER * BLOCKSIZE) + 1;
        const J        row_idx         = gid / num_wgs_per_row;
        const J        wg              = gid % num_wgs_per_row;
        const J        row             = rows_bins[bin_start + row_idx];

        // wg_flags contains the flag bits used so that the multiple WGs calculating a long row can
        // know when the first workgroup for that row has finished initializing the output
        // value. While this bit is the same as the first workgroup's flag bit, this
        // workgroup will spin-loop.

        // TODO: Can the wg_flags-based coordination be done instead with cooperative-groups?

        // Each workgroup computes exactly one row.
        const I vecStart = (I)wg * (I)BLOCK_MULTIPLIER * BLOCKSIZE + csr_row_ptr[row] - idx_base;
        const I vecEnd   = rocsparse::min(csr_row_ptr[row + 1] - idx_base,
                                        vecStart + I(BLOCK_MULTIPLIER * BLOCKSIZE));

        T temp_sum = static_cast<T>(0);

        // In CSR-LongRows, we have more than one workgroup calculating this row.
        // The output values for those types of rows are stored using atomic_add, because
        // more than one parallel workgroup's value makes up the final answer.
        // Unfortunately, this makes it difficult to do y=Ax, rather than y=Ax+y, because
        // the values still left in y will be added in using the atomic_add.
        //
        // Our solution is to have the first workgroup in one of these long-rows cases
        // properly initialize the output vector. All the other workgroups working on this
        // row will spin-loop until that workgroup finishes its work.

        // First, figure out which workgroup you are in the row.
        // You can use that to find the global ID for the first workgroup calculating
        // this long row.
        const J        first_wg_in_row = gid - wg;
        const uint32_t compare_value   = wg_flags[gid];

        // wg_flags[first_wg_in_row] in the first workgroup is the flag that everyone waits on.
        if(gid == first_wg_in_row && lid == 0)
        {
            // The first workgroup handles the output initialization.
            const Y out_val = y[row];
            temp_sum        = (beta - static_cast<T>(1)) * out_val;

            // All inter thread communication is done using atomics, therefore cache flushes or
            // invalidates should not be needed (thus __threadfence() has been removed to regain
            // performance).
            // Because of atomics being relaxed, however, the compiler is allowed to reorder them
            // with respect to ordinary memory accesses (and other relaxed atomic operations).
            // In this case, out_val seem to be reordered with the xor and subsequently, accumulation
            // ends up being wrong.
            // To force the compiler to stick to the order of operations, we need acquire/release fences.
            // Workgroup scope is sufficient for this purpose, to only invalidate L1 and avoid L2
            // invalidations.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define __gfx12__
#endif
#if defined(__gfx12__)
            __threadfence();
#else
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
            __builtin_amdgcn_s_waitcnt(0);
#endif

            // Release other workgroups
            atomicXor(&wg_flags[first_wg_in_row], 1U);
        }

        // Load in a bunch of partial results into your register space, rather than LDS (no
        // contention)
        // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
        for(I j = vecStart + lid; j < vecEnd; j += BLOCKSIZE)
        {
            temp_sum = rocsparse::fma<T>(alpha * rocsparse::conj_val(csr_val[j], conj),
                                         x[csr_col_ind[j] - idx_base],
                                         temp_sum);
        }

        partialSums[lid] = temp_sum;

        __syncthreads();

        // Reduce partial sums
        rocsparse::blockreduce_sum<BLOCKSIZE>(lid, partialSums);

        // For every other workgroup, wg_flags[first_wg_in_row] holds the value they wait on.
        // If your flag == first_wg's flag, you spin loop.
        // The first workgroup will eventually flip this flag, and you can move forward.
        if(lid == 0)
        {
            if(gid != first_wg_in_row)
            {
                while(rocsparse::atomic_max(&wg_flags[first_wg_in_row], 0U) == compare_value)
                    ;

                // __builtin_amdgcn_s_waitcnt(0);
                // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");

                // After you've passed the barrier, update your local flag to make sure that
                // the next time through, you know what to wait on.
                wg_flags[gid] ^= 1U;
            }

            rocsparse::atomic_add((y + row), partialSums[0]);
        }
    }
}
