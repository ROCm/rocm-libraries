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
     * blockDim = (TILE_K / 4) * TILE_M = 16 * 16 = 256 threads.
     * LDS bank conflicts drop from 4-way to 2-way.    */
    static constexpr int OZ2_SCALE_SHMEM_BLOCK_DIM
        = (OZ2_SCALE_TILE_K / 4) * OZ2_SCALE_SHMEM_TILE_M; /* 256 */

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
     * The caller must ensure val ≥ 0 (we bias log2P_req values by +200 to guarantee this). */
    static __device__ __forceinline__ void adp_atomicMaxF(float* addr, float val)
    {
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
    }

} // namespace FixedPointEmulation
