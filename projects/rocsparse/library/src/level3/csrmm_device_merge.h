/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/thread/thread_search.hpp>

namespace rocsparse
{
    template <typename T>
    struct coordinate_t
    {
        T x;
        T y;
    };

    template <uint32_t BLOCKSIZE, uint32_t ITEMS_PER_THREAD, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmmnn_merge_compute_coords(J M,
                                      I nnz,
                                      const I* __restrict__ csr_row_ptr,
                                      coordinate_t<uint32_t>* __restrict__ coord0,
                                      coordinate_t<uint32_t>* __restrict__ coord1,
                                      rocsparse_index_base idx_base)
    {
        const int bid = blockIdx.x;

        // Search starting/ending coordinates of the range for this block.
        const I diagonal0 = (bid + 0) * ITEMS_PER_THREAD;
        const I diagonal1 = rocsparse::min((I)(M + nnz), (I)((bid + 1) * ITEMS_PER_THREAD));

        rocprim::counting_iterator<I> nnz_indices0(idx_base);
        rocprim::counting_iterator<I> nnz_indices1(idx_base);

        // Search across the diagonals to find coordinates to process.
        merge_path_search(diagonal0, csr_row_ptr + 1, nnz_indices0, I(M), nnz, coord0[bid]);
        merge_path_search(diagonal1, csr_row_ptr + 1, nnz_indices1, I(M), nnz, coord1[bid]);
    }

    // Given the sparse A matrix:
    // 1 2 3 4 0 5 6 7
    // 0 0 2 3 1 0 0 4
    // 0 0 0 0 0 0 0 0
    // 1 0 0 0 0 0 0 7
    // 0 0 3 4 5 6 7 0
    // 0 1 0 0 3 0 0 4
    // 1 2 3 4 5 6 0 0
    // 1 2 3 4 0 0 0 0
    //
    // total work = m + nnz = 8 + 31 = 39
    // block_size = 8
    // block_count = 5
    //
    // This results in the following merge-path
    //   0  1  2  3  4  5  6  7  8
    // 0 |........................  --- (0,0)
    //   |  :  :  :  :  :  :  :  :   |
    // 1 |........................   |
    //   |  :  :  :  :  :  :  :  :   |
    // 2 |........................   |
    //   |  :  :  :  :  :  :  :  :   |
    // 3 |........................   block 0
    //   |  :  :  :  :  :  :  :  :   |
    // 4 |........................   |
    //   |  :  :  :  :  :  :  :  :   |
    // 5 |........................   |
    //   |  :  :  :  :  :  :  :  :   |
    // 6 |___.....................  --- (1,7)
    //   :  |  :  :  :  :  :  :  :   |
    // 7 ...|.....................   |
    //   :  |  :  :  :  :  :  :  :   |
    // 8 ...|.....................   |
    //   :  |  :  :  :  :  :  :  :   |
    // 9 ...|.....................   block 1
    //   :  |  :  :  :  :  :  :  :   |
    //10 ...|______...............   |
    //   :  :  :  |  :  :  :  :  :   |
    //11 .........|...............   |
    //   :  :  :  |  :  :  :  :  :   |
    //12 .........|___............  --- (3,13)
    //   :  :  :  :  |  :  :  :  :   |
    //13 ............|............   |
    //   :  :  :  :  |  :  :  :  :   |
    //14 ............|............   |
    //   :  :  :  :  |  :  :  :  :   |
    //15 ............|............   block 2
    //   :  :  :  :  |  :  :  :  :   |
    //16 ............|............   |
    //   :  :  :  :  |  :  :  :  :   |
    //17 ............|___.........   |
    //   :  :  :  :  :  |  :  :  :   |
    //18 ...............|.........  --- (5,19)
    //   :  :  :  :  :  |  :  :  :   |
    //19 ...............|.........   |
    //   :  :  :  :  :  |  :  :  :   |
    //20 ...............|___......   |
    //   :  :  :  :  :  :  |  :  :   |
    //21 ..................|......   |
    //   :  :  :  :  :  :  |  :  :   block 3
    //22 ..................|......   |
    //   :  :  :  :  :  :  |  :  :   |
    //23 ..................|......   |
    //   :  :  :  :  :  :  |  :  :   |
    //24 ..................|......   |
    //   :  :  :  :  :  :  |  :  :   |
    //25 ..................|......  --- (6,26)
    //   :  :  :  :  :  :  |  :  :   |
    //26 ..................|___...   |
    //   :  :  :  :  :  :  :  |  :   |
    //27 .....................|...   |
    //   :  :  :  :  :  :  :  |  :   block 4
    //28 .....................|...   |
    //   :  :  :  :  :  :  :  |  :   |
    //29 .....................|...   |
    //   :  :  :  :  :  :  :  |  :   |
    //30 .....................|___  --- (8, 30)
    template <uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              uint32_t LOOPS,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmmnt_merge_path_main_device(bool     conj_A,
                                                             bool     conj_B,
                                                             J        ncol_offset,
                                                             J        ncol,
                                                             J        M,
                                                             J        N,
                                                             J        K,
                                                             I        nnz,
                                                             T        alpha,
                                                             const I* csr_row_ptr,
                                                             const J* csr_col_ind,
                                                             const A* csr_val,
                                                             const coordinate_t<uint32_t>* coord0,
                                                             const coordinate_t<uint32_t>* coord1,
                                                             const B*                      dense_B,
                                                             int64_t                       ldb,
                                                             T                             beta,
                                                             C*                            dense_C,
                                                             int64_t                       ldc,
                                                             rocsparse_order               order_C,
                                                             rocsparse_index_base          idx_base)
    {
        const int lid = threadIdx.x & (WF_SIZE - 1);
        const int bid = blockIdx.x;

        const coordinate_t<uint32_t> start_coord = coord0[bid];
        const coordinate_t<uint32_t> end_coord   = coord1[bid];

        J       start_row = start_coord.x;
        const J end_row   = end_coord.x;

        const I ptr_start_row = csr_row_ptr[start_row] - idx_base;
        const I ptr_end_row   = csr_row_ptr[end_row] - idx_base;

        const I start_nz = (start_coord.y == (ptr_start_row)) ? 0 : start_coord.y;
        const I end_nz   = (end_coord.y == (ptr_end_row)) ? 0 : end_coord.y;

        if(start_nz != 0)
        {
            // Entire block is used in the middle part of a long row
            if(start_row == end_row && end_nz != 0)
            {
                for(J l = ncol_offset; l < ncol; l += WF_SIZE * LOOPS)
                {
                    const J colB = l + lid;

                    T sum[LOOPS]{};
                    for(I j = start_nz; j < end_nz; j++)
                    {
                        const T val = conj_val(csr_val[j], conj_A);
                        const J col = (csr_col_ind[j] - idx_base);

                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            sum[p] = rocsparse::fma<T>(
                                val,
                                conj_val(dense_B[ldb * col + (colB + p * WF_SIZE)], conj_B),
                                sum[p]);
                        }
                    }

                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(&dense_C[start_row + ldc * (colB + p * WF_SIZE)],
                                                  static_cast<C>(alpha * sum[p]));
                        }
                    }
                    else
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(&dense_C[ldc * start_row + (colB + p * WF_SIZE)],
                                                  static_cast<C>(alpha * sum[p]));
                        }
                    }
                }

                return;
            }
            else // last part of long row where the block ends on a subsequent row and therefore more work to be done.
            {
                for(J l = ncol_offset; l < ncol; l += WF_SIZE * LOOPS)
                {
                    const J colB = l + lid;

                    T       sum[LOOPS]{};
                    const I end = csr_row_ptr[start_row + 1] - idx_base;
                    for(I j = start_nz; j < end; j++)
                    {
                        const T val = conj_val(csr_val[j], conj_A);
                        const J col = (csr_col_ind[j] - idx_base);

                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            sum[p] = rocsparse::fma<T>(
                                val,
                                conj_val(dense_B[ldb * col + (colB + p * WF_SIZE)], conj_B),
                                sum[p]);
                        }
                    }

                    if(order_C == rocsparse_order_column)
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(&dense_C[start_row + ldc * (colB + p * WF_SIZE)],
                                                  static_cast<C>(alpha * sum[p]));
                        }
                    }
                    else
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            rocsparse::atomic_add(&dense_C[ldc * start_row + (colB + p * WF_SIZE)],
                                                  static_cast<C>(alpha * sum[p]));
                        }
                    }
                }

                start_row += 1;
            }
        }

        // Complete rows therefore no atomics required
        for(J i = start_row; i < end_row; i++)
        {
            const I row_begin = csr_row_ptr[i] - idx_base;
            const I row_end   = csr_row_ptr[i + 1] - idx_base;
            const I count     = row_end - row_begin;

            for(J l = ncol_offset; l < ncol; l += WF_SIZE * LOOPS)
            {
                const J colB = l + lid;

                T sum[LOOPS]{};

                for(I j = 0; j < count; j++)
                {
                    const T val = conj_val(csr_val[row_begin + j], conj_A);
                    const J col = (csr_col_ind[row_begin + j] - idx_base);

                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse::fma<T>(
                            val,
                            conj_val(dense_B[ldb * col + (colB + p * WF_SIZE)], conj_B),
                            sum[p]);
                    }
                }

                if(order_C == rocsparse_order_column)
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[i + ldc * (colB + p * WF_SIZE)] = rocsparse::fma<T>(
                            alpha, sum[p], dense_C[i + ldc * (colB + p * WF_SIZE)]);
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[ldc * i + (colB + p * WF_SIZE)] = rocsparse::fma<T>(
                            alpha, sum[p], dense_C[ldc * i + (colB + p * WF_SIZE)]);
                    }
                }
            }
        }

        // first part of row
        if(end_nz != 0)
        {
            for(J l = ncol_offset; l < ncol; l += WF_SIZE * LOOPS)
            {
                const J colB = l + lid;

                T sum[LOOPS]{};
                for(I j = ptr_end_row; j < end_nz; j++)
                {
                    const T val = conj_val(csr_val[j], conj_A);
                    const J col = (csr_col_ind[j] - idx_base);

                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse::fma<T>(
                            val,
                            conj_val(dense_B[ldb * col + (colB + p * WF_SIZE)], conj_B),
                            sum[p]);
                    }
                }

                if(order_C == rocsparse_order_column)
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        rocsparse::atomic_add(&dense_C[end_row + ldc * (colB + p * WF_SIZE)],
                                              static_cast<C>(alpha * sum[p]));
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        rocsparse::atomic_add(&dense_C[ldc * end_row + (colB + p * WF_SIZE)],
                                              static_cast<C>(alpha * sum[p]));
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        csrmmnt_merge_path_main_multi_rows_device(bool                          conj_A,
                                                  bool                          conj_B,
                                                  J                             ncol_offset,
                                                  J                             ncol,
                                                  J                             M,
                                                  J                             N,
                                                  J                             K,
                                                  I                             nnz,
                                                  T                             alpha,
                                                  const I*                      csr_row_ptr,
                                                  const J*                      csr_col_ind,
                                                  const A*                      csr_val,
                                                  const coordinate_t<uint32_t>* coord0,
                                                  const coordinate_t<uint32_t>* coord1,
                                                  const B*                      dense_B,
                                                  int64_t                       ldb,
                                                  T                             beta,
                                                  C*                            dense_C,
                                                  int64_t                       ldc,
                                                  rocsparse_order               order_C,
                                                  rocsparse_index_base          idx_base)
    {
        const int tid = threadIdx.x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / (WF_SIZE);

        const int bid = (BLOCKSIZE / WF_SIZE) * blockIdx.x + wid;

        const uint64_t total_work  = static_cast<uint64_t>(M + nnz);
        const uint64_t block_count = (total_work - 1) / ITEMS_PER_THREAD + 1;

        if(bid < block_count)
        {
            const coordinate_t<uint32_t> start_coord = coord0[bid];
            const coordinate_t<uint32_t> end_coord   = coord1[bid];

            J       start_row = start_coord.x;
            const J end_row   = end_coord.x;

            const I ptr_start_row = csr_row_ptr[start_row] - idx_base;
            const I ptr_end_row   = csr_row_ptr[end_row] - idx_base;

            const I start_nz = (start_coord.y == ptr_start_row) ? 0 : start_coord.y;
            const I end_nz   = (end_coord.y == ptr_end_row) ? 0 : end_coord.y;

            if(start_nz != 0)
            {
                // Entire block is used in the middle part of a long row
                if(start_row == end_row && end_nz != 0)
                {
                    for(J l = ncol_offset; l < ncol; l += WF_SIZE)
                    {
                        const J colB = l + lid;

                        T sum = static_cast<T>(0);
                        for(I j = start_nz; j < end_nz; j++)
                        {
                            const T val = conj_val(csr_val[j], conj_A);
                            const J col = (csr_col_ind[j] - idx_base);

                            sum = rocsparse::fma<T>(
                                val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                        }

                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[start_row + ldc * colB],
                                                  static_cast<C>(alpha * sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[ldc * start_row + colB],
                                                  static_cast<C>(alpha * sum));
                        }
                    }

                    return;
                }
                else // last part of long row where the block ends on a subsequent row and therefore more work to be done.
                {
                    for(J l = ncol_offset; l < ncol; l += WF_SIZE)
                    {
                        const J colB = l + lid;

                        T       sum = static_cast<T>(0);
                        const I end = csr_row_ptr[start_row + 1] - idx_base;
                        for(I j = start_nz; j < end; j++)
                        {
                            const T val = conj_val(csr_val[j], conj_A);
                            const J col = (csr_col_ind[j] - idx_base);

                            sum = rocsparse::fma<T>(
                                val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                        }

                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[start_row + ldc * colB],
                                                  static_cast<C>(alpha * sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[ldc * start_row + colB],
                                                  static_cast<C>(alpha * sum));
                        }
                    }

                    start_row += 1;
                }
            }

            // Complete rows therefore no atomics required
            for(J i = start_row; i < end_row; i++)
            {
                const I row_begin = csr_row_ptr[i] - idx_base;
                const I row_end   = csr_row_ptr[i + 1] - idx_base;
                const I count     = row_end - row_begin;

                for(J l = ncol_offset; l < ncol; l += WF_SIZE)
                {
                    const J colB = l + lid;

                    T sum = static_cast<T>(0);
                    for(I j = 0; j < count; j++)
                    {
                        const T val = conj_val(csr_val[row_begin + j], conj_A);
                        const J col = (csr_col_ind[row_begin + j] - idx_base);

                        sum = rocsparse::fma<T>(
                            val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                    }

                    if(order_C == rocsparse_order_column)
                    {
                        dense_C[i + ldc * colB]
                            = rocsparse::fma<T>(alpha, sum, dense_C[i + ldc * colB]);
                    }
                    else
                    {
                        dense_C[ldc * i + colB]
                            = rocsparse::fma<T>(alpha, sum, dense_C[ldc * i + colB]);
                    }
                }
            }

            // first part of row
            if(end_nz != 0)
            {
                for(J l = ncol_offset; l < ncol; l += WF_SIZE)
                {
                    const J colB = l + lid;

                    T sum = static_cast<T>(0);
                    for(I j = ptr_end_row; j < end_nz; j++)
                    {
                        const T val = conj_val(csr_val[j], conj_A);
                        const J col = (csr_col_ind[j] - idx_base);

                        sum = rocsparse::fma<T>(
                            val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                    }

                    if(order_C == rocsparse_order_column)
                    {
                        rocsparse::atomic_add(&dense_C[end_row + ldc * colB],
                                              static_cast<C>(alpha * sum));
                    }
                    else
                    {
                        rocsparse::atomic_add(&dense_C[ldc * end_row + colB],
                                              static_cast<C>(alpha * sum));
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_DEVICE_ILF void
        csrmmnt_merge_path_remainder_device(bool                          conj_A,
                                            bool                          conj_B,
                                            J                             ncol_offset,
                                            J                             M,
                                            J                             N,
                                            J                             K,
                                            I                             nnz,
                                            T                             alpha,
                                            const I*                      csr_row_ptr,
                                            const J*                      csr_col_ind,
                                            const A*                      csr_val,
                                            const coordinate_t<uint32_t>* coord0,
                                            const coordinate_t<uint32_t>* coord1,
                                            const B*                      dense_B,
                                            int64_t                       ldb,
                                            T                             beta,
                                            C*                            dense_C,
                                            int64_t                       ldc,
                                            rocsparse_order               order_C,
                                            rocsparse_index_base          idx_base)
    {
        const int tid = threadIdx.x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / (WF_SIZE);

        const int bid = (BLOCKSIZE / WF_SIZE) * blockIdx.x + wid;

        const uint64_t total_work  = static_cast<uint64_t>(M + nnz);
        const uint64_t block_count = (total_work - 1) / ITEMS_PER_THREAD + 1;

        if(bid < block_count)
        {
            const coordinate_t<uint32_t> start_coord = coord0[bid];
            const coordinate_t<uint32_t> end_coord   = coord1[bid];

            J       start_row = start_coord.x;
            const J end_row   = end_coord.x;

            const I ptr_start_row = csr_row_ptr[start_row] - idx_base;
            const I ptr_end_row   = csr_row_ptr[end_row] - idx_base;

            const I start_nz = (start_coord.y == ptr_start_row) ? 0 : start_coord.y;
            const I end_nz   = (end_coord.y == ptr_end_row) ? 0 : end_coord.y;

            if(start_nz != 0)
            {
                // Entire block is used in the middle part of a long row
                if(start_row == end_row && end_nz != 0)
                {
                    for(J l = ncol_offset; l < N; l += WF_SIZE)
                    {
                        const J colB = l + lid;

                        if(colB < N)
                        {
                            T sum = static_cast<T>(0);
                            for(I j = start_nz; j < end_nz; j++)
                            {
                                const T val = conj_val(csr_val[j], conj_A);
                                const J col = (csr_col_ind[j] - idx_base);

                                sum = rocsparse::fma<T>(
                                    val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                            }

                            if(order_C == rocsparse_order_column)
                            {
                                rocsparse::atomic_add(&dense_C[start_row + ldc * colB],
                                                      static_cast<C>(alpha * sum));
                            }
                            else
                            {
                                rocsparse::atomic_add(&dense_C[ldc * start_row + colB],
                                                      static_cast<C>(alpha * sum));
                            }
                        }
                    }

                    return;
                }
                else // last part of long row where the block ends on a subsequent row and therefore more work to be done.
                {
                    for(J l = ncol_offset; l < N; l += WF_SIZE)
                    {
                        const J colB = l + lid;

                        if(colB < N)
                        {
                            T       sum = static_cast<T>(0);
                            const I end = csr_row_ptr[start_row + 1] - idx_base;
                            for(I j = start_nz; j < end; j++)
                            {
                                const T val = conj_val(csr_val[j], conj_A);
                                const J col = (csr_col_ind[j] - idx_base);

                                sum = rocsparse::fma<T>(
                                    val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                            }

                            if(order_C == rocsparse_order_column)
                            {
                                rocsparse::atomic_add(&dense_C[start_row + ldc * colB],
                                                      static_cast<C>(alpha * sum));
                            }
                            else
                            {
                                rocsparse::atomic_add(&dense_C[ldc * start_row + colB],
                                                      static_cast<C>(alpha * sum));
                            }
                        }
                    }

                    start_row += 1;
                }
            }

            // Complete rows therefore no atomics required
            for(J i = start_row; i < end_row; i++)
            {
                const I row_begin = csr_row_ptr[i] - idx_base;
                const I row_end   = csr_row_ptr[i + 1] - idx_base;
                const I count     = row_end - row_begin;

                for(J l = ncol_offset; l < N; l += WF_SIZE)
                {
                    const J colB = l + lid;

                    if(colB < N)
                    {
                        T sum = static_cast<T>(0);
                        for(I j = 0; j < count; j++)
                        {
                            const T val = conj_val(csr_val[row_begin + j], conj_A);
                            const J col = (csr_col_ind[row_begin + j] - idx_base);

                            sum = rocsparse::fma<T>(
                                val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                        }

                        if(order_C == rocsparse_order_column)
                        {
                            dense_C[i + ldc * colB]
                                = rocsparse::fma<T>(alpha, sum, dense_C[i + ldc * colB]);
                        }
                        else
                        {
                            dense_C[ldc * i + colB]
                                = rocsparse::fma<T>(alpha, sum, dense_C[ldc * i + colB]);
                        }
                    }
                }
            }

            // first part of row
            if(end_nz != 0)
            {
                for(J l = ncol_offset; l < N; l += WF_SIZE)
                {
                    const J colB = l + lid;

                    if(colB < N)
                    {
                        T sum = static_cast<T>(0);
                        for(I j = ptr_end_row; j < end_nz; j++)
                        {
                            const T val = conj_val(csr_val[j], conj_A);
                            const J col = (csr_col_ind[j] - idx_base);

                            sum = rocsparse::fma<T>(
                                val, conj_val(dense_B[ldb * col + colB], conj_B), sum);
                        }

                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[end_row + ldc * colB],
                                                  static_cast<C>(alpha * sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[ldc * end_row + colB],
                                                  static_cast<C>(alpha * sum));
                        }
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_DEVICE_ILF void csrmmnn_merge_path_device(bool                          conj_A,
                                                        bool                          conj_B,
                                                        J                             M,
                                                        J                             N,
                                                        J                             K,
                                                        I                             nnz,
                                                        T                             alpha,
                                                        const I*                      csr_row_ptr,
                                                        const J*                      csr_col_ind,
                                                        const A*                      csr_val,
                                                        const coordinate_t<uint32_t>* coord0,
                                                        const coordinate_t<uint32_t>* coord1,
                                                        const B*                      dense_B,
                                                        int64_t                       ldb,
                                                        T                             beta,
                                                        C*                            dense_C,
                                                        int64_t                       ldc,
                                                        rocsparse_order               order_C,
                                                        rocsparse_index_base          idx_base)
    {
        const int tid = threadIdx.x;
        const int lid = tid & (WF_SIZE - 1);
        const int wid = tid / (WF_SIZE);

        const int bid = (BLOCKSIZE / WF_SIZE) * blockIdx.x + wid;

        const uint64_t total_work  = static_cast<uint64_t>(M + nnz);
        const uint64_t block_count = (total_work - 1) / ITEMS_PER_THREAD + 1;

        if(bid < block_count)
        {
            const coordinate_t<uint32_t> start_coord = coord0[bid];
            const coordinate_t<uint32_t> end_coord   = coord1[bid];

            J       start_row = start_coord.x;
            const J end_row   = end_coord.x;

            const I ptr_start_row = csr_row_ptr[start_row] - idx_base;
            const I ptr_end_row   = csr_row_ptr[end_row] - idx_base;

            const I start_nz = (start_coord.y == ptr_start_row) ? 0 : start_coord.y;
            const I end_nz   = (end_coord.y == ptr_end_row) ? 0 : end_coord.y;

            if(start_nz != 0)
            {
                // Entire block is used in the middle part of a long row
                if(start_row == end_row && end_nz != 0)
                {
                    for(J l = 0; l < N; l += WF_SIZE)
                    {
                        const J colB = l + lid;

                        if(colB < N)
                        {
                            T sum = static_cast<T>(0);
                            for(I j = start_nz; j < end_nz; j++)
                            {
                                const T val = conj_val(csr_val[j], conj_A);
                                const J col = (csr_col_ind[j] - idx_base);

                                sum = rocsparse::fma<T>(
                                    val, conj_val(dense_B[col + ldb * colB], conj_B), sum);
                            }

                            if(order_C == rocsparse_order_column)
                            {
                                rocsparse::atomic_add(&dense_C[start_row + ldc * colB],
                                                      static_cast<C>(alpha * sum));
                            }
                            else
                            {
                                rocsparse::atomic_add(&dense_C[ldc * start_row + colB],
                                                      static_cast<C>(alpha * sum));
                            }
                        }
                    }

                    return;
                }
                else // last part of long row where the block ends on a subsequent row and therefore more work to be done.
                {
                    for(J l = 0; l < N; l += WF_SIZE)
                    {
                        const J colB = l + lid;

                        if(colB < N)
                        {
                            T       sum = static_cast<T>(0);
                            const I end = csr_row_ptr[start_row + 1] - idx_base;
                            for(I j = start_nz; j < end; j++)
                            {
                                const T val = conj_val(csr_val[j], conj_A);
                                const J col = (csr_col_ind[j] - idx_base);

                                sum = rocsparse::fma<T>(
                                    val, conj_val(dense_B[col + ldb * colB], conj_B), sum);
                            }

                            if(order_C == rocsparse_order_column)
                            {
                                rocsparse::atomic_add(&dense_C[start_row + ldc * colB],
                                                      static_cast<C>(alpha * sum));
                            }
                            else
                            {
                                rocsparse::atomic_add(&dense_C[ldc * start_row + colB],
                                                      static_cast<C>(alpha * sum));
                            }
                        }
                    }

                    start_row += 1;
                }
            }

            // Complete rows therefore no atomics required
            for(J i = start_row; i < end_row; i++)
            {
                const I row_begin = csr_row_ptr[i] - idx_base;
                const I row_end   = csr_row_ptr[i + 1] - idx_base;
                const I count     = row_end - row_begin;

                for(J l = 0; l < N; l += WF_SIZE)
                {
                    const J colB = l + lid;

                    if(colB < N)
                    {
                        T sum = static_cast<T>(0);
                        for(I j = 0; j < count; j++)
                        {
                            const T val = conj_val(csr_val[row_begin + j], conj_A);
                            const J col = (csr_col_ind[row_begin + j] - idx_base);

                            sum = rocsparse::fma<T>(
                                val, conj_val(dense_B[col + ldb * colB], conj_B), sum);
                        }

                        if(order_C == rocsparse_order_column)
                        {
                            dense_C[i + ldc * colB]
                                = rocsparse::fma<T>(alpha, sum, dense_C[i + ldc * colB]);
                        }
                        else
                        {
                            dense_C[ldc * i + colB]
                                = rocsparse::fma<T>(alpha, sum, dense_C[ldc * i + colB]);
                        }
                    }
                }
            }

            // first part of row
            if(end_nz != 0)
            {
                for(J l = 0; l < N; l += WF_SIZE)
                {
                    const J colB = l + lid;

                    if(colB < N)
                    {
                        T sum = static_cast<T>(0);
                        for(I j = ptr_end_row; j < end_nz; j++)
                        {
                            const T val = conj_val(csr_val[j], conj_A);
                            const J col = (csr_col_ind[j] - idx_base);

                            sum = rocsparse::fma<T>(
                                val, conj_val(dense_B[col + ldb * colB], conj_B), sum);
                        }

                        if(order_C == rocsparse_order_column)
                        {
                            rocsparse::atomic_add(&dense_C[end_row + ldc * colB],
                                                  static_cast<C>(alpha * sum));
                        }
                        else
                        {
                            rocsparse::atomic_add(&dense_C[ldc * end_row + colB],
                                                  static_cast<C>(alpha * sum));
                        }
                    }
                }
            }
        }
    }
}
