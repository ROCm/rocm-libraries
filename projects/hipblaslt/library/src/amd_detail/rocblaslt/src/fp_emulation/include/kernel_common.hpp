// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * kernel_common.hpp
 *
 * Shared device-side helpers and tuning constants for the fixed-point
 * emulation GPU kernels (prelim / shift-refine / scale / reconstruct).
 *
 * This header contains only __device__ helper functions and compile-time
 * constants; it is included by the individual kernel headers and by
 * emulation.cpp.  It must be compiled as HIP.
 */

#include "tables.hpp" /* moduli, inv_mod_f, limb_weight, neg_mod, inv_mod */

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace FixedPointEmulation
{
    /* =========================================================================
     * Tuning constants — preliminary shift + INT8 extraction kernels
     * ========================================================================= */
    static constexpr int OZ2_PRELIM_TILE_K = 64; /* k-tile size for SHMEM paths */
    static constexpr int OZ2_PRELIM_SHMEM_TILE_M
        = 16; /* rows/cols per SHMEM block (blockDim=1024) */
    static constexpr int OZ2_PRELIM_COALESC_THRS = 256; /* threads for coalesced paths */
    static constexpr int OZ2_MIN_WARP_SIZE = 32; /* minimum warpSize across supported devices */

    /* =========================================================================
     * Tuning constants — multi-modulus scaling kernels
     * ========================================================================= */
    static constexpr int OZ2_SCALE_TILE_K = 64;
    /* Coalesced kernels (A_T, B_N) process 4 k-positions per thread (K_UNROLL=4):
     * effective k-tile per block = OZ2_SCALE_TILE_K * 4 = 256.
     * 32 threads × 4 bytes = 128 bytes = one full HBM cache line per warp.      */
    static constexpr int OZ2_SCALE_COALESC_TILE_M = 8; /* blockDim=512, no SHMEM */
    static constexpr int OZ2_SCALE_SHMEM_TILE_M   = 16; /* TILE_M for SHMEM kernels */
    /* SHMEM kernels (A_N, B_T) also use K_UNROLL=4: each thread loads/stores 4
     * consecutive k-positions, packs 4 int8 bytes as uint32_t per store.
     * TILE_K = 128: one row of the tile writes 32 threads × 4 bytes = 128 bytes
     * (one full cache line) per modulus.
     * blockDim = (TILE_K / 4) * TILE_M = 32 * 16 = 512 threads.               */
    static constexpr int OZ2_SCALE_SHMEM_TILE_K = 128;
    static constexpr int OZ2_SCALE_SHMEM_BLOCK_DIM
        = (OZ2_SCALE_SHMEM_TILE_K / 4) * OZ2_SCALE_SHMEM_TILE_M; /* 512 */

    /* =========================================================================
     * Device-side reduction helpers
     * ========================================================================= */
    static __device__ __forceinline__ int floor_log2(double x)
    {
        unsigned long long bits;
        __builtin_memcpy(&bits, &x, 8);
        return static_cast<int>((bits >> 52) & 0x7FFull) - 1023;
    }

    static __device__ __forceinline__ int floor_log2(float x)
    {
        unsigned bits;
        __builtin_memcpy(&bits, &x, 4);
        return static_cast<int>((bits >> 23) & 0xFFu) - 127;
    }

    static __device__ __forceinline__ double warp_reduce_max_abs(double val)
    {
        val = fabs(val);
        unsigned long long bits;
        __builtin_memcpy(&bits, &val, 8);
        for(int off = warpSize >> 1; off > 0; off >>= 1)
        {
            unsigned long long other = __shfl_down(bits, off);
            if(other > bits)
                bits = other;
        }
        double res;
        __builtin_memcpy(&res, &bits, 8);
        return res;
    }

    static __device__ __forceinline__ float warp_reduce_max_abs(float val)
    {
        val      = fabsf(val);
        int bits = __float_as_int(val);
        for(int off = warpSize >> 1; off > 0; off >>= 1)
        {
            int other = __shfl_down(bits, off);
            if(other > bits)
                bits = other;
        }
        return __int_as_float(bits);
    }

    static __device__ __forceinline__ int32_t warp_reduce_max_abs(int32_t val)
    {
        if(val < 0)
            val = -val;
        for(int off = warpSize >> 1; off > 0; off >>= 1)
        {
            int32_t other = __shfl_down(val, off);
            if(other > val)
                val = other;
        }
        return val;
    }

    template <typename T>
    static __device__ __forceinline__ T block_reduce_max(T warp_max, T* __restrict__ s_wmax)
    {
        if(threadIdx.x % warpSize == 0)
            s_wmax[threadIdx.x / warpSize] = warp_max;
        __syncthreads();
        T result = T{0};
        if(threadIdx.x == 0)
        {
            const int nw = (blockDim.x + warpSize - 1) / warpSize;
            result       = s_wmax[0];
            for(int w = 1; w < nw; ++w)
                if(s_wmax[w] > result)
                    result = s_wmax[w];
        }
        return result;
    }

    /* atomicMax for non-negative floats: IEEE 754 positive floats are totally ordered
     * by their integer bit representation, so int-based atomicMax is correct.
     * The caller must ensure val ≥ 0 (we bias crt_capacity_bits_req values by +200 to guarantee this). */
    static __device__ __forceinline__ void adp_atomicMaxF(float* addr, float val)
    {
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
    }

    /* =========================================================================
     * Integer residue helpers
     *
     * The residues are computed with integer and FP32 arithmetic instead of
     * FP64.  On GPUs with a low FP64 rate (Radeon) the FP64 versions made the
     * scale and accumulation kernels FP64-bound.  All steps are exact:
     * balanced_residue_i32 matches the FP64 formula it replaces for every
     * int32 input, and int8_residue matches it for scaled values below 2^75.
     * Above 2^75 the FP64 formula can return an unbalanced or wrong residue;
     * int8_residue stays exact up to 2^88.
     * ========================================================================= */

    /* |y| split into four LIMB_BITS-bit limbs, for an integer-valued double y.
     * fits is false when |y| >= 2^88 (or y is Inf/NaN); callers then use the
     * FP64 formula.                                                             */
    struct Int88Limbs
    {
        uint32_t x[4];
        bool     neg;
        bool     fits;
    };

    static __device__ __forceinline__ Int88Limbs split_int88(double y)
    {
        Int88Limbs     r{{0u, 0u, 0u, 0u}, false, true};
        const uint64_t bits = static_cast<uint64_t>(__double_as_longlong(y));
        r.neg               = (bits >> 63) != 0u;
        const int ef        = static_cast<int>((bits >> 52) & 0x7FFu);
        if(ef == 0)
            return r; /* +-0: y is integer-valued, so it is never subnormal */
        /* |y| = M * 2^s with a 53-bit integer M. */
        const uint64_t M = (bits & ((uint64_t(1) << 52) - 1u)) | (uint64_t(1) << 52);
        const int      s = ef - 1075;
        if(s > 88 - 53)
        {
            r.fits = false;
            return r;
        }
        /* 128-bit value hi:lo = |y|.  For s < 0 the low -s bits of M are zero
         * (y is integer-valued), so the right shift is exact.                  */
        uint64_t lo = 0u, hi = 0u;
        if(s <= 0)
            lo = (s <= -53) ? 0u : (M >> -s);
        else
        {
            lo = M << s;
            hi = M >> (64 - s);
        }
        constexpr uint64_t mask = (uint64_t(1) << LIMB_BITS) - 1u;
        const uint64_t     mid  = (lo >> (2 * LIMB_BITS)) | (hi << (64 - 2 * LIMB_BITS));
        r.x[0]                  = static_cast<uint32_t>(lo & mask);
        r.x[1]                  = static_cast<uint32_t>((lo >> LIMB_BITS) & mask);
        r.x[2]                  = static_cast<uint32_t>(mid & mask);
        r.x[3]                  = static_cast<uint32_t>((hi >> (3 * LIMB_BITS - 64)) & mask);
        return r;
    }

    /* INT8 balanced residue of the integer-valued double y modulo m_t, given
     * v = split_int88(y).  Out of range (v.fits false), it falls back to the
     * original FP64 formula.                                                    */
    static __device__ __forceinline__ int8_t int8_residue(const Int88Limbs& v, double y, unsigned t)
    {
        const float imf = inv_mod_f(t);
        if(!v.fits)
        {
            const double rp  = fma(neg_mod(t), rint(y * inv_mod(t)), y);
            const float  rfp = static_cast<float>(rp);
            return static_cast<int8_t>(static_cast<int32_t>(
                fmaf(rintf(rfp * imf), static_cast<float>(neg_mod(t)), rfp)));
        }
        const uint32_t m = static_cast<uint32_t>(moduli[t]);
        /* S < 2^22 + 3 * 255 * 2^22 < 2^32.  __umul24 needs operands < 2^24.  */
        const uint32_t S = v.x[0] + __umul24(v.x[1], limb_weight(t, 1))
                           + __umul24(v.x[2], limb_weight(t, 2))
                           + __umul24(v.x[3], limb_weight(t, 3));
        /* Approximate quotient, off by at most 5 (float rounding of S and 1/m,
         * plus truncation), so |S - q*m| < 6m.  The unsigned difference wraps,
         * and the int32 cast recovers it exactly.                              */
        const uint32_t q = static_cast<uint32_t>(static_cast<float>(S) * imf);
        int32_t        r = static_cast<int32_t>(S - q * m);
        if(v.neg)
            r = -r;
        /* Balance into [-m/2, m/2]; exact because |r| < 2^24.                  */
        const float rf = static_cast<float>(r);
        return static_cast<int8_t>(
            static_cast<int32_t>(fmaf(rintf(rf * imf), -static_cast<float>(m), rf)));
    }

    /* Balanced residue of the int32 c modulo m_t, identical to
     * fma(neg_mod(t), rint(double(c) * inv_mod(t)), double(c)), including the
     * round-half-to-even choice of +-m/2 for the even modulus 256.             */
    static __device__ __forceinline__ int32_t balanced_residue_i32(int32_t c, unsigned t)
    {
        const int32_t m   = moduli[t];
        const float   imf = inv_mod_f(t);
        /* Coarse quotient, off by at most 3 (float rounding of c and 1/m);
         * unsigned arithmetic avoids overflow and the int32 cast recovers the
         * small remainder exactly.                                             */
        const int32_t  q1  = static_cast<int32_t>(rintf(static_cast<float>(c) * imf));
        const uint32_t qm1 = static_cast<uint32_t>(q1) * static_cast<uint32_t>(m);
        const int32_t  r1  = static_cast<int32_t>(static_cast<uint32_t>(c) - qm1);
        /* Floor quotient of the small remainder, corrected by at most one step:
         * c = (q1 + q2) * m + r0 with r0 in [0, m).                             */
        int32_t q2 = static_cast<int32_t>(floorf(static_cast<float>(r1) * imf));
        int32_t r0 = r1 - q2 * m;
        if(r0 < 0)
        {
            r0 += m;
            --q2;
        }
        if(r0 >= m)
        {
            r0 -= m;
            ++q2;
        }
        /* rint(c / m): round up above m/2; on a tie, round to the even quotient. */
        const int32_t twice = 2 * r0;
        if(twice > m || (twice == m && ((q1 + q2) & 1)))
            return r0 - m;
        return r0;
    }

} // namespace FixedPointEmulation
