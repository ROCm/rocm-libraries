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

#include <type_traits>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnn_row_split_shared_device(T    alpha,
                                                              T    beta,
                                                              bool conj_A,
                                                              bool conj_B,
                                                              J    M,
                                                              J    N,
                                                              const I* __restrict__ csr_row_ptr,
                                                              const J* __restrict__ csr_col_ind,
                                                              const A* __restrict__ csr_val,
                                                              const B* __restrict__ dense_B,
                                                              int64_t ldb,
                                                              C* __restrict__ dense_C,
                                                              int64_t              ldc,
                                                              rocsparse_order      order_C,
                                                              rocsparse_index_base idx_base,
                                                              J                    col_panel)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid = gid & (WF_SIZE - 1);
        const J        row = gid / WF_SIZE;
        const J        col = lid + col_panel;

        if(row >= M)
        {
            return;
        }

        const int64_t colB = col * ldb;

        const I row_start = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        T sum = static_cast<T>(0);

        for(I j = row_start; j < row_end; j += WF_SIZE)
        {
            const I k = j + lid;

            const J my_col = (k < row_end) ? csr_col_ind[k] - idx_base : 0;
            const T my_val = (k < row_end) ? static_cast<T>(rocsparse::conj_val(csr_val[k], conj_A))
                                           : static_cast<T>(0);

            for(uint32_t i = 0; i < WF_SIZE; ++i)
            {
                const J sc = rocsparse::shfl(my_col, i, WF_SIZE);
                const T sv = rocsparse::shfl(my_val, i, WF_SIZE);
                if(col < N)
                {
                    sum = rocsparse::fma<T>(
                        sv, rocsparse::conj_val(dense_B[sc + colB], conj_B), sum);
                }
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc] = alpha * sum;
                }
                else
                {
                    dense_C[row * ldc + col] = alpha * sum;
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    dense_C[row + col * ldc]
                        = rocsparse::fma<T>(beta, dense_C[row + col * ldc], alpha * sum);
                }
                else
                {
                    dense_C[row * ldc + col]
                        = rocsparse::fma<T>(beta, dense_C[row * ldc + col], alpha * sum);
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnn_row_split_device(T    alpha,
                                                       T    beta,
                                                       bool conj_A,
                                                       bool conj_B,
                                                       J    col_panel,
                                                       J    M,
                                                       J    N,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       const A* __restrict__ csr_val,
                                                       const B* __restrict__ dense_B,
                                                       int64_t ldb,
                                                       C* __restrict__ dense_C,
                                                       int64_t              ldc,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t tid  = hipThreadIdx_x;
        const J        gid  = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid  = gid & (WF_SIZE - 1);
        const J        row  = gid / WF_SIZE;
        const J        colB = col_panel;

        if(row >= M)
        {
            return;
        }

        const I row_start = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        T sum[LOOPS]{};

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j] - idx_base;
            const T val = rocsparse::conj_val(csr_val[j], conj_A);

            for(uint32_t p = 0; p < LOOPS; p++)
            {
                sum[p] = rocsparse::fma<T>(
                    val,
                    rocsparse::conj_val(rocsparse::ldg(dense_B + col + (colB + p) * ldb), conj_B),
                    sum[p]);
            }
        }
        for(uint32_t p = 0; p < LOOPS; p++)
        {
            sum[p] = rocsparse::wfreduce_sum<WF_SIZE>(sum[p]);
        }

        if(lid == WF_SIZE - 1)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p) * ldc] = alpha * sum[p];
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + (colB + p)] = alpha * sum[p];
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row + (colB + p) * ldc] = rocsparse::fma<T>(
                            beta, dense_C[row + (colB + p) * ldc], alpha * sum[p]);
                    }
                }
                else
                {
                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        dense_C[row * ldc + (colB + p)] = rocsparse::fma<T>(
                            beta, dense_C[row * ldc + (colB + p)], alpha * sum[p]);
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t SUBWFSIZE,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void
        csrmmnt_row_split_subwfsize_x_loop_columns_device(T       alpha,
                                                          T       beta,
                                                          J       col_start,
                                                          J       col_end,
                                                          J       M,
                                                          J       N,
                                                          int64_t ldb,
                                                          int64_t ldc,
                                                          const I* __restrict__ csr_row_ptr,
                                                          const J* __restrict__ csr_col_ind,
                                                          const A* __restrict__ csr_val,
                                                          const B* __restrict__ dense_B,
                                                          C* __restrict__ dense_C,
                                                          rocsparse_order      order_C,
                                                          rocsparse_index_base idx_base,
                                                          bool                 conj_A,
                                                          bool                 conj_B)
    {
        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const J        row = gid / WFSIZE;
        const uint32_t lid = tid & (WFSIZE - 1);

        const uint32_t slid = lid & (SUBWFSIZE - 1);
        const uint32_t swid = lid / SUBWFSIZE;

        if(row >= M)
        {
            return;
        }

        const I row_start = rocsparse::nontemporal_load(csr_row_ptr + row) - idx_base;
        const I row_end   = rocsparse::nontemporal_load(csr_row_ptr + row + 1) - idx_base;

        for(J l = col_start; l < col_end; l += SUBWFSIZE * LOOPS)
        {
            const J col = l + slid;

            T sum[LOOPS]{};

            for(I j = row_start; j < row_end; j += WFSIZE)
            {
                const I k = j + lid;

                const J c
                    = (k < row_end) ? (rocsparse::nontemporal_load(csr_col_ind + k) - idx_base) : 0;

                const T v = (k < row_end) ? static_cast<T>(rocsparse::conj_val(
                                                rocsparse::nontemporal_load(csr_val + k), conj_A))
                                          : static_cast<T>(0);

                for(uint32_t i = 0; i < SUBWFSIZE; ++i)
                {
                    const int64_t sc = ldb * rocsparse::shfl(c, SUBWFSIZE * swid + i, SUBWFSIZE);
                    const T       sv = rocsparse::shfl(v, SUBWFSIZE * swid + i, SUBWFSIZE);

                    for(uint32_t p = 0; p < LOOPS; p++)
                    {
                        sum[p] = rocsparse::fma<T>(
                            sv,
                            rocsparse::conj_val(rocsparse::ldg(dense_B + col + p * SUBWFSIZE + sc),
                                                conj_B),
                            sum[p]);
                    }
                }
            }

            if(SUBWFSIZE < WFSIZE)
            {
                for(uint32_t p = 0; p < LOOPS; p++)
                {
                    sum[p] = rocsparse::wfreduce_partial_sum<WFSIZE, SUBWFSIZE>(sum[p]);
                }
            }

            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row + (col + p * SUBWFSIZE) * ldc] = alpha * sum[p];
                        }
                    }
                }
                else
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row * ldc + col + p * SUBWFSIZE] = alpha * sum[p];
                        }
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row + (col + p * SUBWFSIZE) * ldc] = rocsparse::fma<T>(
                                beta, dense_C[row + (col + p * SUBWFSIZE) * ldc], alpha * sum[p]);
                        }
                    }
                }
                else
                {
                    if(lid >= (WFSIZE - SUBWFSIZE))
                    {
                        for(uint32_t p = 0; p < LOOPS; p++)
                        {
                            dense_C[row * ldc + col + p * SUBWFSIZE] = rocsparse::fma<T>(
                                beta, dense_C[row * ldc + col + p * SUBWFSIZE], alpha * sum[p]);
                        }
                    }
                }
            }
        }
    }

    // csrmvn_general_device extended to a fixed number of B columns in NT layout.
    // Each lane keeps one accumulator per column (T sum[COL]) and walks the row once.
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t COL,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_row_split_general_device(T    alpha,
                                                               T    beta,
                                                               bool conj_A,
                                                               bool conj_B,
                                                               J    col_start,
                                                               J    M,
                                                               const I* __restrict__ csr_row_ptr,
                                                               const J* __restrict__ csr_col_ind,
                                                               const A* __restrict__ csr_val,
                                                               const B* __restrict__ dense_B,
                                                               int64_t ldb,
                                                               C* __restrict__ dense_C,
                                                               int64_t              ldc,
                                                               rocsparse_order      order_C,
                                                               rocsparse_index_base idx_base)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");
        static_assert(COL > 0, "COL must be positive.");

        const int lid = hipThreadIdx_x & (WF_SIZE - 1);

        const J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        const J row = gid / WF_SIZE;

        if(row >= M)
        {
            return;
        }

        const I row_start = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        T sum[COL]{};

        // Loop over non-zero elements. Each lane accumulates every B column.
        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J  col_k = csr_col_ind[j] - idx_base;
            const T  val   = static_cast<T>(rocsparse::conj_val(csr_val[j], conj_A));
            const B* b_k   = dense_B + (static_cast<int64_t>(col_k) * ldb + col_start);

#pragma unroll
            for(uint32_t p = 0; p < COL; p++)
            {
                sum[p] = rocsparse::fma<T>(
                    val,
                    static_cast<T>(rocsparse::conj_val(rocsparse::ldg(b_k + p), conj_B)),
                    sum[p]);
            }
        }

#pragma unroll
        for(uint32_t p = 0; p < COL; p++)
        {
            sum[p] = rocsparse::wfreduce_sum<WF_SIZE>(sum[p]);
        }

        // Last lane of the wavefront writes the row result.
        if(lid == WF_SIZE - 1)
        {
            if(beta == static_cast<T>(0))
            {
                if(order_C == rocsparse_order_column)
                {
#pragma unroll
                    for(uint32_t p = 0; p < COL; p++)
                    {
                        dense_C[row + (col_start + p) * ldc] = alpha * sum[p];
                    }
                }
                else
                {
#pragma unroll
                    for(uint32_t p = 0; p < COL; p++)
                    {
                        dense_C[row * ldc + (col_start + p)] = alpha * sum[p];
                    }
                }
            }
            else
            {
                if(order_C == rocsparse_order_column)
                {
#pragma unroll
                    for(uint32_t p = 0; p < COL; p++)
                    {
                        dense_C[row + (col_start + p) * ldc] = rocsparse::fma<T>(
                            beta, dense_C[row + (col_start + p) * ldc], alpha * sum[p]);
                    }
                }
                else
                {
#pragma unroll
                    for(uint32_t p = 0; p < COL; p++)
                    {
                        dense_C[row * ldc + (col_start + p)] = rocsparse::fma<T>(
                            beta, dense_C[row * ldc + (col_start + p)], alpha * sum[p]);
                    }
                }
            }
        }
    }

    // First column owned by `lane` after a reduce-scatter across WF_SIZE lanes.
    // The first (COL % WF_SIZE) lanes own one extra column. Lane WF_SIZE owns nothing
    // and returns COL, so a lane's range is [begin(lane), begin(lane + 1)).
    template <uint32_t WF_SIZE, uint32_t COL>
    __host__ __device__ constexpr uint32_t csrmmnn_owned_begin(uint32_t lane)
    {
        constexpr uint32_t chunk = COL / WF_SIZE;
        constexpr uint32_t rem   = COL % WF_SIZE;
        if(lane >= WF_SIZE)
        {
            return COL;
        }
        if(lane < rem)
        {
            return lane * (chunk + 1);
        }
        return rem * (chunk + 1) + (lane - rem) * chunk;
    }

    template <uint32_t WF_SIZE, uint32_t COL, uint32_t DIST>
    __host__ __device__ constexpr uint32_t csrmmnn_max_span()
    {
        uint32_t m = 0;
        for(uint32_t b = 0; b < WF_SIZE; b += DIST)
        {
            const uint32_t span = csrmmnn_owned_begin<WF_SIZE, COL>(b + DIST)
                                  - csrmmnn_owned_begin<WF_SIZE, COL>(b);
            if(span > m)
            {
                m = span;
            }
        }
        return m;
    }

    ROCSPARSE_DEVICE_ILF double csrmmnn_shfl_xor(double value, int lane)
    {
        return __shfl_xor(value, lane);
    }

    ROCSPARSE_DEVICE_ILF rocsparse_double_complex csrmmnn_shfl_xor(rocsparse_double_complex value,
                                                                   int                      lane)
    {
        return rocsparse_double_complex(__shfl_xor(std::real(value), lane),
                                        __shfl_xor(std::imag(value), lane));
    }

    // One step of a butterfly reduce-scatter. Lane groups of 2*DIST split in half.
    // Each lane sends the column its partner is accumulating and adds the column it keeps,
    // so one shuffle moves a different column in each direction and every lane's FP64 add
    // is live. Distances 1, 2 and 4 stay inside a software wavefront of 8 lanes.
    template <uint32_t WF_SIZE, uint32_t COL, uint32_t DIST, typename T>
    ROCSPARSE_DEVICE_ILF void csrmmnn_scatter_step(T (&sum)[COL], int lid)
    {
        if constexpr(DIST > 0)
        {
            constexpr uint32_t max_span  = csrmmnn_max_span<WF_SIZE, COL, DIST>();
            const uint32_t     lane_base = static_cast<uint32_t>(lid) & ~(DIST - 1u);
            const uint32_t     my_begin  = csrmmnn_owned_begin<WF_SIZE, COL>(lane_base);
            const uint32_t     my_count
                = csrmmnn_owned_begin<WF_SIZE, COL>(lane_base + DIST) - my_begin;

            const int      partner       = lid ^ static_cast<int>(DIST);
            const uint32_t partner_base  = static_cast<uint32_t>(partner) & ~(DIST - 1u);
            const uint32_t partner_begin = csrmmnn_owned_begin<WF_SIZE, COL>(partner_base);
            const uint32_t partner_count
                = csrmmnn_owned_begin<WF_SIZE, COL>(partner_base + DIST) - partner_begin;

#pragma unroll
            for(uint32_t i = 0; i < max_span; ++i)
            {
                const uint32_t send_p = (i < partner_count) ? (partner_begin + i) : 0u;
                const T        recv   = csrmmnn_shfl_xor(sum[send_p], static_cast<int>(DIST));
                if(i < my_count)
                {
                    sum[my_begin + i] += recv;
                }
            }

            csrmmnn_scatter_step<WF_SIZE, COL, DIST / 2, T>(sum, lid);
        }
    }

    // Reduce-scatter COL columns as tiles of WF_SIZE. A tile fits in a few registers, so the
    // indexed shuffle does not move the per-column accumulators out to scratch. Columns in
    // different tiles never mix, so the tiles are independent.
    template <uint32_t WF_SIZE, uint32_t COL, uint32_t BASE, typename T, typename Store>
    ROCSPARSE_DEVICE_ILF void csrmmnn_scatter_tiles(T (&sum)[COL], int lid, Store store)
    {
        if constexpr(BASE < COL)
        {
            constexpr uint32_t remain = COL - BASE;
            constexpr uint32_t TILE   = (remain >= WF_SIZE) ? WF_SIZE : remain;

            T tmp[TILE];
#pragma unroll
            for(uint32_t i = 0; i < TILE; ++i)
            {
                tmp[i] = sum[BASE + i];
            }

            if constexpr(WF_SIZE > 1)
            {
                csrmmnn_scatter_step<WF_SIZE, TILE, WF_SIZE / 2, T>(tmp, lid);
            }

            constexpr uint32_t max_owned = (TILE + WF_SIZE - 1) / WF_SIZE;
            const uint32_t begin = csrmmnn_owned_begin<WF_SIZE, TILE>(static_cast<uint32_t>(lid));
            const uint32_t count
                = csrmmnn_owned_begin<WF_SIZE, TILE>(static_cast<uint32_t>(lid) + 1) - begin;

#pragma unroll
            for(uint32_t i = 0; i < max_owned; ++i)
            {
                if(i < count)
                {
                    store(BASE + begin + i, tmp[begin + i]);
                }
            }

            csrmmnn_scatter_tiles<WF_SIZE, COL, BASE + TILE, T, Store>(sum, lid, store);
        }
    }

    // csrmvn_general_device extended to a fixed number of B columns in NN layout.
    // Each lane keeps one accumulator per column (T sum[COL]) and walks the row once.
    // B is indexed as dense_B[col_k + (col_start + p) * ldb]: the K index (col_k) is
    // stride-1 and the N index (col_start + p) carries stride ldb. This corresponds to
    // (order_B == rocsparse_order_column && trans_B == rocsparse_operation_none) or
    // (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose /
    //    conjugate_transpose).
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t COL,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnn_row_split_general_device(T    alpha,
                                                               T    beta,
                                                               bool conj_A,
                                                               bool conj_B,
                                                               J    col_start,
                                                               J    M,
                                                               const I* __restrict__ csr_row_ptr,
                                                               const J* __restrict__ csr_col_ind,
                                                               const A* __restrict__ csr_val,
                                                               const B* __restrict__ dense_B,
                                                               int64_t ldb,
                                                               C* __restrict__ dense_C,
                                                               int64_t              ldc,
                                                               rocsparse_order      order_C,
                                                               rocsparse_index_base idx_base)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");
        static_assert(COL > 0, "COL must be positive.");

        const int lid = hipThreadIdx_x & (WF_SIZE - 1);

        const J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        const J row = gid / WF_SIZE;

        if(row >= M)
        {
            return;
        }

        const I row_start = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        T sum[COL]{};

        // Loop over non-zero elements. Each lane accumulates every B column.
        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J  col_k = csr_col_ind[j] - idx_base;
            const T  val   = static_cast<T>(rocsparse::conj_val(csr_val[j], conj_A));
            const B* b_k
                = dense_B + (static_cast<int64_t>(col_k) + static_cast<int64_t>(col_start) * ldb);

#pragma unroll
            for(uint32_t p = 0; p < COL; p++)
            {
                sum[p] = rocsparse::fma<T>(
                    val,
                    static_cast<T>(rocsparse::conj_val(
                        rocsparse::ldg(b_k + static_cast<int64_t>(p) * ldb), conj_B)),
                    sum[p]);
            }
        }

        // FP64 adds are the expensive part of the all-reduce: every lane adds every column and
        // only one lane keeps the result. Double and double complex reduce-scatter instead, so
        // each add stays live and each lane writes the columns it owns. Float keeps the
        // all-reduce; those adds are hidden by the FP32 rate.
        if constexpr(std::is_same<T, double>::value
                     || std::is_same<T, rocsparse_double_complex>::value)
        {
            if constexpr(WF_SIZE > 1)
            {
                csrmmnn_scatter_tiles<WF_SIZE, COL, 0, T>(sum, lid, [&](uint32_t p, T acc) {
                    if(beta == static_cast<T>(0))
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            dense_C[row + (col_start + p) * ldc] = alpha * acc;
                        }
                        else
                        {
                            dense_C[row * ldc + (col_start + p)] = alpha * acc;
                        }
                    }
                    else if(order_C == rocsparse_order_column)
                    {
                        dense_C[row + (col_start + p) * ldc] = rocsparse::fma<T>(
                            beta, dense_C[row + (col_start + p) * ldc], alpha * acc);
                    }
                    else
                    {
                        dense_C[row * ldc + (col_start + p)] = rocsparse::fma<T>(
                            beta, dense_C[row * ldc + (col_start + p)], alpha * acc);
                    }
                });
            }
            else
            {
#pragma unroll
                for(uint32_t p = 0; p < COL; p++)
                {
                    if(beta == static_cast<T>(0))
                    {
                        if(order_C == rocsparse_order_column)
                        {
                            dense_C[row + (col_start + p) * ldc] = alpha * sum[p];
                        }
                        else
                        {
                            dense_C[row * ldc + (col_start + p)] = alpha * sum[p];
                        }
                    }
                    else if(order_C == rocsparse_order_column)
                    {
                        dense_C[row + (col_start + p) * ldc] = rocsparse::fma<T>(
                            beta, dense_C[row + (col_start + p) * ldc], alpha * sum[p]);
                    }
                    else
                    {
                        dense_C[row * ldc + (col_start + p)] = rocsparse::fma<T>(
                            beta, dense_C[row * ldc + (col_start + p)], alpha * sum[p]);
                    }
                }
            }
        }
        else
        {
#pragma unroll
            for(uint32_t p = 0; p < COL; p++)
            {
                sum[p] = rocsparse::wfreduce_sum<WF_SIZE>(sum[p]);
            }

            // Last lane of the wavefront writes the row result.
            if(lid == WF_SIZE - 1)
            {
                if(beta == static_cast<T>(0))
                {
                    if(order_C == rocsparse_order_column)
                    {
#pragma unroll
                        for(uint32_t p = 0; p < COL; p++)
                        {
                            dense_C[row + (col_start + p) * ldc] = alpha * sum[p];
                        }
                    }
                    else
                    {
#pragma unroll
                        for(uint32_t p = 0; p < COL; p++)
                        {
                            dense_C[row * ldc + (col_start + p)] = alpha * sum[p];
                        }
                    }
                }
                else
                {
                    if(order_C == rocsparse_order_column)
                    {
#pragma unroll
                        for(uint32_t p = 0; p < COL; p++)
                        {
                            dense_C[row + (col_start + p) * ldc] = rocsparse::fma<T>(
                                beta, dense_C[row + (col_start + p) * ldc], alpha * sum[p]);
                        }
                    }
                    else
                    {
#pragma unroll
                        for(uint32_t p = 0; p < COL; p++)
                        {
                            dense_C[row * ldc + (col_start + p)] = rocsparse::fma<T>(
                                beta, dense_C[row * ldc + (col_start + p)], alpha * sum[p]);
                        }
                    }
                }
            }
        }
    }

    template <typename T, uint32_t WFSIZE>
    ROCSPARSE_DEVICE_ILF void device_partial_sum(T* sum) {};

    template <typename T, uint32_t WFSIZE, uint32_t N, uint32_t... R>
    ROCSPARSE_DEVICE_ILF void device_partial_sum(T* sum)
    {
        if(N < WFSIZE)
        {
            sum[0] = rocsparse::wfreduce_partial_sum<WFSIZE, N>(sum[0]);
        }
        device_partial_sum<T, WFSIZE, R...>(++sum);
    };

    template <std::size_t N, typename T>
    ROCSPARSE_DEVICE_ILF constexpr std::size_t sizeofarray(T (&)[N])
    {
        return N;
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t SUBWFSIZE,
              uint32_t LOOP,
              uint32_t... SUBWFSIZES_LIST,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmnt_row_split_subwfsize_x_loop_plus_swfs_columns_device(
        T alpha,
        T beta,
        J col_start,
        J col_end,
        J M,
        J N,
        const I* __restrict__ csr_row_ptr,
        const J* __restrict__ csr_col_ind,
        const A* __restrict__ csr_val,
        const B* __restrict__ dense_B,
        int64_t ldb,
        C* __restrict__ dense_C,
        int64_t              ldc,
        rocsparse_order      order_C,
        rocsparse_index_base idx_base,
        bool                 conj_A,
        bool                 conj_B)
    {
        static_assert(WFSIZE > 0 && (WFSIZE & (WFSIZE - 1)) == 0, "WFSIZE must be a power of two.");
        static_assert(SUBWFSIZE > 0 && (SUBWFSIZE & (SUBWFSIZE - 1)) == 0,
                      "SUBWFSIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WFSIZE == 0, "BLOCKSIZE must be a multiple of WFSIZE.");

        static constexpr uint32_t SUBWFSIZES[]   = {SUBWFSIZES_LIST...};
        static constexpr uint32_t NUM_SUBWFSIZES = sizeofarray(SUBWFSIZES);

        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const J        row = gid / WFSIZE;
        const uint32_t lid = tid & (WFSIZE - 1);

        const uint32_t slid = lid & (SUBWFSIZE - 1);
        const uint32_t swid = lid / SUBWFSIZE;

        uint32_t slid_swf[NUM_SUBWFSIZES];
        uint32_t swid_swf[NUM_SUBWFSIZES];

        for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
        {
            slid_swf[iswf] = lid & (SUBWFSIZES[iswf] - 1);
            swid_swf[iswf] = lid / SUBWFSIZES[iswf];
        }

        if(row >= M)
        {
            return;
        }

        const I row_start = rocsparse::nontemporal_load(csr_row_ptr + row) - idx_base;
        const I row_end   = rocsparse::nontemporal_load(csr_row_ptr + row + 1) - idx_base;

        const J col_loop = col_start + slid;

        J col_swf[NUM_SUBWFSIZES];

        if(NUM_SUBWFSIZES > 0)
        {
            col_swf[0] = col_start + SUBWFSIZE * LOOP;
            for(uint32_t iswf = 1; iswf < NUM_SUBWFSIZES; ++iswf)
            {
                col_swf[iswf] = col_swf[iswf - 1] + SUBWFSIZES[iswf - 1];
            }
            for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
            {
                col_swf[iswf] += slid_swf[iswf];
            }
        }

        T sum_loop[LOOP]{};
        T sum_swf[NUM_SUBWFSIZES]{};

        for(I j = row_start; j < row_end; j += WFSIZE)
        {
            const I k = j + lid;

            const J c
                = (k < row_end) ? (rocsparse::nontemporal_load(csr_col_ind + k) - idx_base) : 0;

            const T v = (k < row_end) ? static_cast<T>(rocsparse::conj_val(
                                            rocsparse::nontemporal_load(csr_val + k), conj_A))
                                      : static_cast<T>(0);

            for(uint32_t i = 0; i < SUBWFSIZE; ++i)
            {
                const int64_t sc = ldb * rocsparse::shfl(c, SUBWFSIZE * swid + i, SUBWFSIZE);
                const T       sv = rocsparse::shfl(v, SUBWFSIZE * swid + i, SUBWFSIZE);

                for(uint32_t p = 0; p < LOOP; p++)
                {
                    sum_loop[p] = rocsparse::fma<T>(
                        sv,
                        rocsparse::conj_val(rocsparse::ldg(dense_B + col_loop + p * SUBWFSIZE + sc),
                                            conj_B),
                        sum_loop[p]);
                }
            }

            for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
            {
                for(uint32_t i = 0; i < SUBWFSIZES[iswf]; ++i)
                {
                    const J sc = rocsparse::shfl(
                        c, SUBWFSIZES[iswf] * swid_swf[iswf] + i, SUBWFSIZES[iswf]);
                    const T sv = rocsparse::shfl(
                        v, SUBWFSIZES[iswf] * swid_swf[iswf] + i, SUBWFSIZES[iswf]);

                    sum_swf[iswf] = rocsparse::fma<T>(
                        sv,
                        rocsparse::conj_val(rocsparse::ldg(dense_B + col_swf[iswf] + ldb * sc),
                                            conj_B),
                        sum_swf[iswf]);
                }
            }
        }

        if(SUBWFSIZE < WFSIZE)
        {
            for(uint32_t p = 0; p < LOOP; p++)
            {
                sum_loop[p] = rocsparse::wfreduce_partial_sum<WFSIZE, SUBWFSIZE>(sum_loop[p]);
            }
        }

        device_partial_sum<T, WFSIZE, SUBWFSIZES_LIST...>(sum_swf);

        if(beta == static_cast<T>(0))
        {
            if(order_C == rocsparse_order_column)
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row + (col_loop + p * SUBWFSIZE) * ldc] = alpha * sum_loop[p];
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row + col_swf[iswf] * ldc] = alpha * sum_swf[iswf];
                    }
                }
            }
            else
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row * ldc + (col_loop + p * SUBWFSIZE)] = alpha * sum_loop[p];
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row * ldc + col_swf[iswf]] = alpha * sum_swf[iswf];
                    }
                }
            }
        }
        else
        {
            if(order_C == rocsparse_order_column)
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row + (col_loop + p * SUBWFSIZE) * ldc]
                            = rocsparse::fma<T>(beta,
                                                dense_C[row + (col_loop + p * SUBWFSIZE) * ldc],
                                                alpha * sum_loop[p]);
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row + col_swf[iswf] * ldc] = rocsparse::fma<T>(
                            beta, dense_C[row + col_swf[iswf] * ldc], alpha * sum_swf[iswf]);
                    }
                }
            }
            else
            {
                if(lid >= (WFSIZE - SUBWFSIZE))
                {
                    for(uint32_t p = 0; p < LOOP; p++)
                    {
                        dense_C[row * ldc + (col_loop + p * SUBWFSIZE)]
                            = rocsparse::fma<T>(beta,
                                                dense_C[row * ldc + (col_loop + p * SUBWFSIZE)],
                                                alpha * sum_loop[p]);
                    }
                }

                for(uint32_t iswf = 0; iswf < NUM_SUBWFSIZES; ++iswf)
                {
                    if(lid >= (WFSIZE - SUBWFSIZES[iswf]))
                    {
                        dense_C[row * ldc + col_swf[iswf]] = rocsparse::fma<T>(
                            beta, dense_C[row * ldc + col_swf[iswf]], alpha * sum_swf[iswf]);
                    }
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmtn_row_split_device(T    alpha,
                                                       bool conj_A,
                                                       bool conj_B,
                                                       J    M,
                                                       J    N,
                                                       J    K,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       const A* __restrict__ csr_val,
                                                       const B* __restrict__ dense_B,
                                                       int64_t ldb,
                                                       C* __restrict__ dense_C,
                                                       int64_t              ldc,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base,
                                                       J                    col_panel)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid = gid & (WF_SIZE - 1);
        const uint32_t wid = tid / WF_SIZE;

        const J row = gid / WF_SIZE;

        if(row >= M)
        {
            return;
        }

        // Compute size of dense_C for 4-argument atomic_add. C is K x N because op(A) = A^T.
        const int64_t dense_C_size = (order_C == rocsparse_order_column) ? (ldc * N) : (K * ldc);

        const J       cid  = lid + col_panel;
        const int64_t colB = cid * ldb;

        __shared__ T shared_B[BLOCKSIZE / WF_SIZE][WF_SIZE];
        shared_B[wid][lid]
            = (cid < N) ? rocsparse::conj_val(dense_B[row + colB], conj_B) : static_cast<B>(0);

        __threadfence_block();
        const I row_start = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j] - idx_base;
            const T val = alpha * rocsparse::conj_val(csr_val[j], conj_A);

            if(order_C == rocsparse_order_column)
            {
                for(J i = 0; i < WF_SIZE && (i + col_panel) < N; ++i)
                {
                    rocsparse::atomic_add(dense_C,
                                          col + (i + col_panel) * ldc,
                                          dense_C_size,
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
            else
            {
                for(J i = 0; i < WF_SIZE && (i + col_panel) < N; ++i)
                {
                    rocsparse::atomic_add(dense_C,
                                          col * ldc + i + col_panel,
                                          dense_C_size,
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_DEVICE_ILF void csrmmtt_row_split_device(T    alpha,
                                                       bool conj_A,
                                                       bool conj_B,
                                                       J    M,
                                                       J    N,
                                                       J    K,
                                                       const I* __restrict__ csr_row_ptr,
                                                       const J* __restrict__ csr_col_ind,
                                                       const A* __restrict__ csr_val,
                                                       const B* __restrict__ dense_B,
                                                       int64_t ldb,
                                                       C* __restrict__ dense_C,
                                                       int64_t              ldc,
                                                       rocsparse_order      order_C,
                                                       rocsparse_index_base idx_base,
                                                       J                    col_panel)
    {
        static_assert(WF_SIZE > 0 && (WF_SIZE & (WF_SIZE - 1)) == 0,
                      "WF_SIZE must be a power of two.");
        static_assert(BLOCKSIZE > 0, "BLOCKSIZE must be positive.");
        static_assert(BLOCKSIZE % WF_SIZE == 0, "BLOCKSIZE must be a multiple of WF_SIZE.");

        const uint32_t tid = hipThreadIdx_x;
        const J        gid = hipBlockIdx_x * BLOCKSIZE + tid;
        const uint32_t lid = gid & (WF_SIZE - 1);
        const uint32_t wid = tid / WF_SIZE;

        const J row = gid / WF_SIZE;
        const J cid = lid + col_panel;

        if(row >= M)
        {
            return;
        }

        // Compute size of dense_C for 4-argument atomic_add. C is K x N because op(A) = A^T.
        const int64_t dense_C_size = (order_C == rocsparse_order_column) ? (ldc * N) : (K * ldc);

        __shared__ T shared_B[BLOCKSIZE / WF_SIZE][WF_SIZE];

        shared_B[wid][lid]
            = (cid < N) ? rocsparse::conj_val(dense_B[ldb * row + cid], conj_B) : static_cast<B>(0);

        __threadfence_block();
        const I row_start = csr_row_ptr[row] - idx_base;
        const I row_end   = csr_row_ptr[row + 1] - idx_base;

        for(I j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j] - idx_base;
            const T val = alpha * rocsparse::conj_val(csr_val[j], conj_A);

            if(order_C == rocsparse_order_column)
            {
                for(J i = 0; i < WF_SIZE && (i + col_panel) < N; ++i)
                {
                    rocsparse::atomic_add(dense_C,
                                          col + (i + col_panel) * ldc,
                                          dense_C_size,
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
            else
            {
                for(J i = 0; i < WF_SIZE && (i + col_panel) < N; ++i)
                {
                    rocsparse::atomic_add(dense_C,
                                          col * ldc + i + col_panel,
                                          dense_C_size,
                                          static_cast<C>(val * shared_B[wid][i]));
                }
            }
        }
    }
}
