// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * scale_kernels.hpp
 *
 * GPU kernels — multi-modulus scaling (separate per path).
 *
 * These kernels extract the per-modulus INT8 residue slices for the batched
 * INT8 GEMM passes.
 *
 * Four kernels named by the transpose value they handle:
 *   scale_A_T_kernel  TRANS_A=true  (k-fast coalesced, blockDim=512, TILE_M=8)
 *   scale_A_N_kernel  TRANS_A=false (SHMEM transposition, blockDim=256, TILE_M=16)
 *   scale_B_N_kernel  TRANS_B=false (j-fast coalesced, blockDim=512, TILE_M=8)
 *   scale_B_T_kernel  TRANS_B=true  (SHMEM transposition, blockDim=256, TILE_M=16)
 *
 * Coalesced kernels: no SHMEM → low LDS → good latency hiding.
 * SHMEM kernels: TILE_M=16, K_UNROLL=4 → blockDim=256; 8.7 KB LDS per block.
 *
 * This header must be compiled as HIP.
 */

#include "kernel_common.hpp"
#include "tables.hpp" /* neg_mod, inv_mod, inv_mod_f */

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace FixedPointEmulation
{
    /* ── A_T: TRANS_A=true, k-fast coalesced, blockDim=512, TILE_M=8, K_UNROLL=4 ──
     * Each thread processes FOUR adjacent k-positions: j0..j0+3.
     * j0 = blockIdx.x * 4*TILE_K + (t%TILE_K)*4  →  always 4-aligned.
     * Loads:  two double2 reads (j0..j0+1 and j0+2..j0+3), each 128-bit, each aligned.
     * Stores: one uint32_t NT store per modulus (packs INT8[j0..j0+3]) = 128 bytes/warp.
     * Moduli loop is OUTER so nm/im/imf are loaded once per modulus, and only
     * TILE_M=8 NT write-combine buffers are needed simultaneously.               */
    template <unsigned T_COUNT, typename T>
    __global__ static void scale_A_T_kernel(const T* __restrict__ A,
                                            int64_t m,
                                            int64_t lda,
                                            int8_t* __restrict__ A8i,
                                            size_t lda8i,
                                            size_t cola8i,
                                            const int16_t* __restrict__ sftA,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M = OZ2_SCALE_COALESC_TILE_M; /* 8 */
        const int            t      = static_cast<int>(threadIdx.x);
        const int64_t        m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        /* j0 is always 4-aligned → both vector loads and the uint32_t store are aligned. */
        const int64_t j0 = static_cast<int64_t>(blockIdx.x) * (TILE_K * 4)
                           + static_cast<int64_t>(t % TILE_K) * 4;
        const int64_t i = m_base + (t / TILE_K);
        if(i >= m || j0 >= k)
            return;
        const int sft = static_cast<int>(sftA[i]);
        /* Load and widen to double immediately — all residue arithmetic stays in double.
         * FP64: double2 (2 × 8 bytes = 128-bit).  FP32: float2 (2 × 4 bytes = 64-bit). */
        double ival[4];
        if constexpr(std::is_same_v<T, double>)
        {
            /* Load j0/j0+1 as double2 (j0 is 4-aligned → even → 16-byte aligned). */
            {
                const double2 vv = *reinterpret_cast<const double2*>(A + i * lda + j0);
                ival[0]          = trunc(ldexp(vv.x, sft));
                ival[1]          = (j0 + 1 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
            }
            /* Load j0+2/j0+3 as double2 (j0+2 is even → aligned) only when valid. */
            if(j0 + 2 < k)
            {
                const double2 vv = *reinterpret_cast<const double2*>(A + i * lda + j0 + 2);
                ival[2]          = trunc(ldexp(vv.x, sft));
                ival[3]          = (j0 + 3 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
            }
            else
            {
                ival[2] = 0.0;
                ival[3] = 0.0;
            }
        }
        else /* float: float2 (2 × 4 bytes = 64-bit), still coalesced */
        {
            /* Load j0/j0+1 as float2 (j0 is 4-aligned → 8-byte aligned). */
            {
                const float2 vv = *reinterpret_cast<const float2*>(A + i * lda + j0);
                ival[0]         = trunc(ldexp(static_cast<double>(vv.x), sft));
                ival[1]         = (j0 + 1 < k) ? trunc(ldexp(static_cast<double>(vv.y), sft)) : 0.0;
            }
            /* Load j0+2/j0+3 as float2 (j0+2 is even → 8-byte aligned) only when valid. */
            if(j0 + 2 < k)
            {
                const float2 vv = *reinterpret_cast<const float2*>(A + i * lda + j0 + 2);
                ival[2]         = trunc(ldexp(static_cast<double>(vv.x), sft));
                ival[3]         = (j0 + 3 < k) ? trunc(ldexp(static_cast<double>(vv.y), sft)) : 0.0;
            }
            else
            {
                ival[2] = 0.0;
                ival[3] = 0.0;
            }
        }
        const size_t stride   = lda8i * cola8i;
        const size_t off_base = static_cast<size_t>(i) * lda8i;
        const size_t off0 = static_cast<size_t>(j0) + off_base; /* 4-aligned → uint32_t aligned */
        /* Outer loop over moduli — sequential (no unroll) so nm/im/imf loaded once per modulus,
         * keeping register pressure low and WCB usage at TILE_M=8 entries simultaneously.   */
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
        {
            const unsigned tidx = t_start + t_local;
            const double   nm   = neg_mod(tidx);
            const double   im   = inv_mod(tidx);
            const float    imf  = inv_mod_f(tidx);
            /* Compute INT8 residues for all four k-positions. */
            int8_t b[4];
#pragma unroll
            for(int p = 0; p < 4; ++p)
            {
                const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                const float  rfp = static_cast<float>(rp);
                b[p]             = static_cast<int8_t>(
                    static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
            }
            /* Pack and store: always write 4 bytes as uint32_t.
             * Out-of-bounds b[p] are zero (ival[p] was pre-zeroed for p >= valid count),
             * so the extra bytes land harmlessly in the [k, lda8i) padding region.
             * j0 is always 4-aligned, so dst is always uint32_t-aligned.
             * This eliminates branch divergence in tail k-tiles.              */
            int8_t* const  dst    = A8i + t_local * stride + off0;
            const uint32_t packed = static_cast<uint8_t>(b[0])
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
            __builtin_nontemporal_store(packed, reinterpret_cast<uint32_t*>(dst));
        }
    }

    /* ── A_N: TRANS_A=false, SHMEM transposition, blockDim=256, TILE_M=16, K_UNROLL=4 ──
     * Each thread loads K_UNROLL=4 consecutive k-positions into SHMEM, then in the
     * write phase reads 4 SHMEM entries, computes 4 int8 residues, and stores them as a
     * single uint32_t NT write (4 bytes per store instead of 1).
     * blockDim = (TILE_K / K_UNROLL) * TILE_M = 16 * 16 = 256.
     *
     * Load phase: k_grp = t/TILE_M (0..15), m_loc = t%TILE_M (0..15).
     *   Each thread fills shmem[k_grp*4+p][m_loc] for p=0..3 (4 coalesced column loads).
     * Write phase: k_wb = (t%16)*4 (always 4-aligned), m_wr = t/16 (0..15).
     *   Thread reads 4 shmem entries for the same row, packs 4 int8 → uint32_t.
     *   j_out = blockIdx.x*TILE_K + k_wb is always 4-aligned → uint32_t aligned.
     *   Out-of-bounds shmem entries are pre-zeroed in the load phase (stored 0.0 for j≥k),
     *   so the extra bytes in the padding region [k, lda8i) are harmlessly zero. */
    template <unsigned T_COUNT, typename T>
    __global__ static void scale_A_N_kernel(const T* __restrict__ A,
                                            int64_t m,
                                            int64_t lda,
                                            int8_t* __restrict__ A8i,
                                            size_t lda8i,
                                            size_t cola8i,
                                            const int16_t* __restrict__ sftA,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K   = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M   = OZ2_SCALE_SHMEM_TILE_M; /* 16 */
        static constexpr int K_UNROLL = 4;
        /* blockDim = OZ2_SCALE_SHMEM_BLOCK_DIM = (TILE_K/K_UNROLL)*TILE_M = 256 */
        __shared__ double  shmem[TILE_K][TILE_M + 1]; /* always double — values widened at load */
        __shared__ int16_t s_sft[TILE_M];

        const int     t      = static_cast<int>(threadIdx.x);
        const int64_t m_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        const int     k_grp  = t / TILE_M; /* 0..TILE_K/K_UNROLL-1 = 0..15 */
        const int     m_loc  = t % TILE_M; /* 0..TILE_M-1 = 0..15           */
        const int64_t i      = m_base + m_loc;

        /* Load phase: widen T→double immediately at load; SHMEM stays double.
         * Access A[i + j*lda] (m-fast across m_loc) is coalesced within each
         * group of TILE_M threads sharing the same k_grp.                   */
#pragma unroll
        for(int p = 0; p < K_UNROLL; p++)
        {
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + k_grp * K_UNROLL + p;
            shmem[k_grp * K_UNROLL + p][m_loc]
                = (i < m && j < k) ? static_cast<double>(A[i + j * lda]) : 0.0;
        }
        if(k_grp == 0 && i < m)
            s_sft[m_loc] = sftA[i];
        __syncthreads();

        /* Write phase: k_wb is always K_UNROLL-aligned so j_out is always 4-aligned.
         * lda8i = pad(k) is 128-aligned → offset is 4-aligned → uint32_t aligned.
         * Out-of-bounds positions have shmem=0.0 → ival=0.0 → b[p]=0.            */
        const int     k_wb  = (t % (TILE_K / K_UNROLL)) * K_UNROLL; /* 0,4,8,...,60 */
        const int     m_wr  = t / (TILE_K / K_UNROLL); /* 0..15         */
        const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_wb;
        const int64_t i_out = m_base + m_wr;
        if(i_out < m && j_out < k)
        {
            const int sft = static_cast<int>(s_sft[m_wr]);
            /* Read K_UNROLL scaled values from SHMEM; shmem already contains 0.0 for
             * out-of-bounds j≥k positions set during the load phase above.         */
            double ival[K_UNROLL];
#pragma unroll
            for(int p = 0; p < K_UNROLL; p++)
                ival[p] = trunc(ldexp(shmem[k_wb + p][m_wr], sft));

            const size_t stride = lda8i * cola8i;
            const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(i_out) * lda8i;
            /* Outer loop over moduli: nm/im/imf loaded once per modulus, reused for
             * all K_UNROLL=4 k-positions.  Pack 4 int8 bytes as one uint32_t NT store. */
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
            {
                const unsigned tidx = t_start + t_local;
                const double   nm   = neg_mod(tidx);
                const double   im   = inv_mod(tidx);
                const float    imf  = inv_mod_f(tidx);
                int8_t         b[K_UNROLL];
#pragma unroll
                for(int p = 0; p < K_UNROLL; p++)
                {
                    const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                    const float  rfp = static_cast<float>(rp);
                    b[p]             = static_cast<int8_t>(
                        static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
                }
                /* Pack and store: always write 4 bytes as uint32_t.
                 * Out-of-bounds b[p] are zero (ival[p]=0.0 from pre-zeroed shmem),
                 * so extra bytes land harmlessly in the [k, lda8i) padding region. */
                const uint32_t packed = static_cast<uint8_t>(b[0])
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
                __builtin_nontemporal_store(
                    packed, reinterpret_cast<uint32_t*>(A8i + t_local * stride + offset));
            }
        }
    }

    /* ── B_N: TRANS_B=false, j-fast coalesced, blockDim=512, TILE_M=8, K_UNROLL=4 ──
     * Mirrors scale_A_T_kernel: two vector loads (j0..j0+1 and j0+2..j0+3) + uint32_t packed store.
     * FP64: double2 (128-bit).  FP32: float2 (64-bit), widened to double before arithmetic. */
    template <unsigned T_COUNT, typename T>
    __global__ static void scale_B_N_kernel(const T* __restrict__ B,
                                            int64_t n,
                                            int64_t ldb,
                                            int8_t* __restrict__ B8i,
                                            size_t ldb8i,
                                            const int16_t* __restrict__ sftB,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M = OZ2_SCALE_COALESC_TILE_M; /* 8 */
        const int            t      = static_cast<int>(threadIdx.x);
        const int64_t        n_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        const int64_t        j0     = static_cast<int64_t>(blockIdx.x) * (TILE_K * 4)
                           + static_cast<int64_t>(t % TILE_K) * 4;
        const int64_t col = n_base + (t / TILE_K);
        if(col >= n || j0 >= k)
            return;
        const int sft = static_cast<int>(sftB[col]);
        /* Load and widen to double immediately — all residue arithmetic stays in double. */
        double ival[4];
        if constexpr(std::is_same_v<T, double>)
        {
            {
                const double2 vv = *reinterpret_cast<const double2*>(B + col * ldb + j0);
                ival[0]          = trunc(ldexp(vv.x, sft));
                ival[1]          = (j0 + 1 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
            }
            if(j0 + 2 < k)
            {
                const double2 vv = *reinterpret_cast<const double2*>(B + col * ldb + j0 + 2);
                ival[2]          = trunc(ldexp(vv.x, sft));
                ival[3]          = (j0 + 3 < k) ? trunc(ldexp(vv.y, sft)) : 0.0;
            }
            else
            {
                ival[2] = 0.0;
                ival[3] = 0.0;
            }
        }
        else /* float: float2 (2 × 4 bytes = 64-bit), still coalesced */
        {
            {
                const float2 vv = *reinterpret_cast<const float2*>(B + col * ldb + j0);
                ival[0]         = trunc(ldexp(static_cast<double>(vv.x), sft));
                ival[1]         = (j0 + 1 < k) ? trunc(ldexp(static_cast<double>(vv.y), sft)) : 0.0;
            }
            if(j0 + 2 < k)
            {
                const float2 vv = *reinterpret_cast<const float2*>(B + col * ldb + j0 + 2);
                ival[2]         = trunc(ldexp(static_cast<double>(vv.x), sft));
                ival[3]         = (j0 + 3 < k) ? trunc(ldexp(static_cast<double>(vv.y), sft)) : 0.0;
            }
            else
            {
                ival[2] = 0.0;
                ival[3] = 0.0;
            }
        }
        const size_t stride   = ldb8i * static_cast<size_t>(n);
        const size_t off_base = static_cast<size_t>(col) * ldb8i;
        const size_t off0 = static_cast<size_t>(j0) + off_base; /* 4-aligned → uint32_t aligned */
        for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
        {
            const unsigned tidx = t_start + t_local;
            const double   nm   = neg_mod(tidx);
            const double   im   = inv_mod(tidx);
            const float    imf  = inv_mod_f(tidx);
            int8_t         b[4];
#pragma unroll
            for(int p = 0; p < 4; ++p)
            {
                const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                const float  rfp = static_cast<float>(rp);
                b[p]             = static_cast<int8_t>(
                    static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
            }
            /* Pack and store: always write 4 bytes as uint32_t.
             * Out-of-bounds b[p] are zero (ival[p] was pre-zeroed for p >= valid count),
             * so the extra bytes land harmlessly in the [k, ldb8i) padding region.
             * j0 is always 4-aligned, so dst is always uint32_t-aligned.
             * This eliminates branch divergence in tail k-tiles.              */
            int8_t* const  dst    = B8i + t_local * stride + off0;
            const uint32_t packed = static_cast<uint8_t>(b[0])
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                    | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
            __builtin_nontemporal_store(packed, reinterpret_cast<uint32_t*>(dst));
        }
    }

    /* ── B_T: TRANS_B=true, SHMEM transposition, blockDim=256, TILE_M=16, K_UNROLL=4 ──
     * Mirrors scale_A_N_kernel exactly, but operates on B (col-fast layout).
     * Each thread loads K_UNROLL=4 consecutive k-positions for one column into SHMEM,
     * then in the write phase reads 4 SHMEM entries and packs them as a uint32_t NT store.
     * blockDim = OZ2_SCALE_SHMEM_BLOCK_DIM = (TILE_K/K_UNROLL)*TILE_M = 256.
     *
     * Load phase: k_grp = t/TILE_M (0..15), l_loc = t%TILE_M (0..15).
     *   Each thread fills shmem[k_grp*4+p][l_loc] for p=0..3 (coalesced col-fast loads).
     * Write phase: k_wb = (t%16)*4 (always 4-aligned), l_wr = t/16 (0..15).
     *   Thread reads 4 shmem entries for the same column, packs 4 int8 → uint32_t.
     *   j_out = blockIdx.x*TILE_K + k_wb is always 4-aligned → uint32_t aligned.
     *   Out-of-bounds shmem entries are pre-zeroed in the load phase (stored 0.0 for j≥k),
     *   so extra bytes in the padding region [k, ldb8i) are harmlessly zero.           */
    template <unsigned T_COUNT, typename T>
    __global__ static void scale_B_T_kernel(const T* __restrict__ B,
                                            int64_t n,
                                            int64_t ldb,
                                            int8_t* __restrict__ B8i,
                                            size_t ldb8i,
                                            const int16_t* __restrict__ sftB,
                                            int64_t  k,
                                            unsigned t_start)
    {
        static constexpr int TILE_K   = OZ2_SCALE_TILE_K; /* 64 */
        static constexpr int TILE_M   = OZ2_SCALE_SHMEM_TILE_M; /* 16 */
        static constexpr int K_UNROLL = 4;
        /* blockDim = OZ2_SCALE_SHMEM_BLOCK_DIM = (TILE_K/K_UNROLL)*TILE_M = 256 */
        __shared__ double  shmem[TILE_K][TILE_M + 1]; /* always double — values widened at load */
        __shared__ int16_t s_sft[TILE_M];

        const int     t      = static_cast<int>(threadIdx.x);
        const int64_t n_base = static_cast<int64_t>(blockIdx.y) * TILE_M;
        const int     k_grp  = t / TILE_M; /* 0..TILE_K/K_UNROLL-1 = 0..15 */
        const int     l_loc  = t % TILE_M; /* 0..TILE_M-1 = 0..15           */
        const int64_t col    = n_base + l_loc;

        /* Load phase: widen T→double immediately at load; SHMEM stays double.
         * Access B[col + j*ldb] (col-fast across l_loc) is coalesced within each
         * group of TILE_M threads sharing the same k_grp.                   */
#pragma unroll
        for(int p = 0; p < K_UNROLL; p++)
        {
            const int64_t j = static_cast<int64_t>(blockIdx.x) * TILE_K + k_grp * K_UNROLL + p;
            shmem[k_grp * K_UNROLL + p][l_loc]
                = (col < n && j < k) ? static_cast<double>(B[col + j * ldb]) : 0.0;
        }
        if(k_grp == 0 && col < n)
            s_sft[l_loc] = sftB[col];
        __syncthreads();

        /* Write phase: k_wb is always K_UNROLL-aligned so j_out is always 4-aligned.
         * ldb8i = pad(k) is 128-aligned → offset is 4-aligned → uint32_t aligned.
         * Out-of-bounds positions have shmem=0.0 → ival=0.0 → b[p]=0.            */
        const int     k_wb  = (t % (TILE_K / K_UNROLL)) * K_UNROLL; /* 0,4,8,...,60 */
        const int     l_wr  = t / (TILE_K / K_UNROLL); /* 0..15         */
        const int64_t j_out = static_cast<int64_t>(blockIdx.x) * TILE_K + k_wb;
        const int64_t c_out = n_base + l_wr;
        if(c_out < n && j_out < k)
        {
            const int sft = static_cast<int>(s_sft[l_wr]);
            /* Read K_UNROLL scaled values from SHMEM; shmem already contains 0.0 for
             * out-of-bounds j≥k positions set during the load phase above.         */
            double ival[K_UNROLL];
#pragma unroll
            for(int p = 0; p < K_UNROLL; p++)
                ival[p] = trunc(ldexp(shmem[k_wb + p][l_wr], sft));

            const size_t stride = ldb8i * static_cast<size_t>(n);
            const size_t offset = static_cast<size_t>(j_out) + static_cast<size_t>(c_out) * ldb8i;
            /* Outer loop over moduli: nm/im/imf loaded once per modulus, reused for
             * all K_UNROLL=4 k-positions.  Pack 4 int8 bytes as one uint32_t NT store. */
            for(unsigned t_local = 0; t_local < T_COUNT; ++t_local)
            {
                const unsigned tidx = t_start + t_local;
                const double   nm   = neg_mod(tidx);
                const double   im   = inv_mod(tidx);
                const float    imf  = inv_mod_f(tidx);
                int8_t         b[K_UNROLL];
#pragma unroll
                for(int p = 0; p < K_UNROLL; p++)
                {
                    const double rp  = fma(nm, rint(ival[p] * im), ival[p]);
                    const float  rfp = static_cast<float>(rp);
                    b[p]             = static_cast<int8_t>(
                        static_cast<int32_t>(fmaf(rintf(rfp * imf), static_cast<float>(nm), rfp)));
                }
                /* Pack and store: always write 4 bytes as uint32_t.
                 * Out-of-bounds b[p] are zero (ival[p]=0.0 from pre-zeroed shmem),
                 * so extra bytes land harmlessly in the [k, ldb8i) padding region. */
                const uint32_t packed = static_cast<uint8_t>(b[0])
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
                                        | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
                __builtin_nontemporal_store(
                    packed, reinterpret_cast<uint32_t*>(B8i + t_local * stride + offset));
            }
        }
    }

} // namespace FixedPointEmulation
