// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * reconstruct_kernels.hpp
 *
 * GPU kernels — chunked CRT accumulation and finalization (reconstruct).
 *
 *   chunk_accum_kernel     accumulate a chunk of moduli into Zhi/Zlo
 *   accum_finalize_kernel  accumulate the final chunk and write D
 *
 * These kernels reconstruct the exact GEMM result from its CRT residues, then
 * range-reduce and scale it back to the output type.  When all moduli are
 * processed in one chunk, accum_finalize_kernel sums exactly in integer limbs;
 * with several chunks the partial sums are double-double (TwoSum) values.
 *
 * This header must be compiled as HIP.
 */

#include "kernel_common.hpp" /* balanced_residue_i32 */
#include "tables.hpp" /* qpi, inv_P, P_dd */

#include <hip/hip_runtime.h>

#include <cstdint>

namespace FixedPointEmulation
{
    template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK>
    __global__ static void chunk_accum_kernel(const int32_t* __restrict__ C32i_batch,
                                              double* __restrict__ Zhi_out,
                                              double* __restrict__ Zlo_out,
                                              int64_t  m,
                                              int64_t  n,
                                              size_t   ldc32i,
                                              unsigned chunk_start,
                                              unsigned effective_s)
    {
        const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        if(i >= m || l >= n)
            return;
        const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
        const size_t slice_stride = ldc32i * static_cast<size_t>(n);
        /* Two-pass: dc_s[CS] caches balanced residues from independent HBM loads.
         * Pass 1: all CS loads + residue reductions issued simultaneously (MLP, ~200cy
         *   latency); the integer/FP32 reduction (balanced_residue_i32) overlaps HBM
         *   latency. Uses CS×2 VGPRs (~52 for CS=16).
         * Pass 2: pure register TwoSum, loop-carried dep through Zhi only.
         * Note: storing dc as int32 (dc_int_s) was tested — same performance at all
         * sizes on gfx950 (kernel is HBM-bandwidth-limited there; higher occupancy does
         * not help). */
        double dc_s[CHUNK_SIZE];
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t = chunk_start + t_local;
            dc_s[t_local]    = static_cast<double>(
                balanced_residue_i32(C32i_batch[t_local * slice_stride + idx], t));
        }
        double Zhi = 0.0, Zlo = 0.0;
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t      = chunk_start + t_local;
            const double   dc     = dc_s[t_local];
            const auto [qhi, qlo] = qpi(effective_s - 2, t);
            const double hi       = dc * qhi;
            const double new_hi   = Zhi + hi;
            // TwoSum (Knuth) — exact for ANY operand ordering.  FastTwoSum
            // (err = hi - (new_hi - Zhi)) is only exact when |Zhi| >= |hi|; the
            // running |Zhi| and per-term |hi| are not monotonic (the qpi_hi
            // magnitudes vary non-monotonically across the moduli), so a plain
            // FastTwoSum silently drops ~1 ULP whenever |hi| > |Zhi| at a step.
            // TwoSum removes that precondition and makes the accumulation
            // order-independent, which keeps the result robust for any qpi split
            // regardless of its trailing-zero slack.  Measured overhead vs
            // FastTwoSum is ~3% of the (bandwidth-bound) accum kernel and
            // <0.5% of total emulation time — negligible for the robustness gained.
            const double bb  = new_hi - Zhi;
            const double err = (Zhi - (new_hi - bb)) + (hi - bb);
            Zhi              = new_hi;
            if constexpr(HAS_LO)
                Zlo = fma(dc, qlo, Zlo + err);
            else
                Zlo += err;
        }
        if constexpr(IS_FIRST_CHUNK)
        {
            __builtin_nontemporal_store(Zhi, Zhi_out + idx);
            __builtin_nontemporal_store(Zlo, Zlo_out + idx);
        }
        else
        {
            const double old_hi = Zhi_out[idx];
            const double s_hi   = old_hi + Zhi;
            const double err    = Zhi - (s_hi - old_hi);
            Zhi_out[idx]        = s_hi;
            Zlo_out[idx] += err + Zlo;
        }
    }

    /* ── Exact integer CRT sum (single-chunk finalize) ─────────────────────────
     * Z = sum_t dc_t * qpi_t is held in 16-bit limbs (int32 per limb), folded
     * into X = Z - rint(Z / P) * P, and converted to double with one rounding,
     * so X is exact before the final rounding.  All operations are integer
     * except a few FP64 ones per element, instead of about ten per modulus.
     *
     * Limb count for CS moduli: |dc_t| <= 128 and qpi_t < P < 2^(8 CS), so
     * |Z| < CS * 128 * 2^(8 CS) needs at most 8 CS + 12 bits plus a sign bit.
     * Per limb, |sum_t dc_t * qpi_limb| < 20 * 2^7 * 2^16 < 2^31.             */
    template <unsigned CS>
    __device__ constexpr unsigned crt_sum_limbs() noexcept
    {
        return (8u * CS + 28u) / 16u;
    }

    /* Carry-propagates signed limbs: on return limbs [0, NL-2] are in
     * [0, 2^16) and the top limb carries the sign of the represented value.   */
    template <unsigned NL>
    __device__ __forceinline__ void crt_normalize(int32_t (&z)[NL])
    {
        int32_t carry = 0;
#pragma unroll
        for(unsigned j = 0; j + 1 < NL; ++j)
        {
            const int32_t x = z[j] + carry;
            z[j]            = x & 0xFFFF;
            carry           = x >> 16; /* arithmetic shift: floor division */
        }
        z[NL - 1] += carry;
    }

    /* Correctly rounded ldexp(X, e) for the integer X held in normalized limbs
     * with |X| < 2^(16 (NL - 1)): the highest nonzero limb and the four below
     * it form a 64-bit mantissa, all lower bits fold into a sticky bit, and the
     * uint64 -> double conversion rounds once.  Indices stay compile-time
     * constants so the limbs remain in registers.                             */
    template <unsigned NL>
    __device__ __forceinline__ double crt_to_double(int32_t (&z)[NL], int e)
    {
        const bool neg = z[NL - 1] < 0;
        if(neg)
        {
#pragma unroll
            for(unsigned j = 0; j < NL; ++j)
                z[j] = -z[j];
            crt_normalize(z);
        }
        /* below: limbs j-1..j-4 (j-1 in the top 16 bits) while scanning upward. */
        uint64_t below = 0u, low = 0u;
        uint32_t top          = 0u;
        bool     sticky_below = false, sticky = false;
        int      h            = -1;
#pragma unroll
        for(unsigned j = 0; j < NL; ++j)
        {
            const uint32_t lj = static_cast<uint32_t>(z[j]);
            if(lj != 0u)
            {
                top    = lj;
                low    = below;
                sticky = sticky_below;
                h      = static_cast<int>(j);
            }
            sticky_below = sticky_below || (below & 0xFFFFu) != 0u;
            below        = (below >> 16) | (static_cast<uint64_t>(lj) << 48);
        }
        if(h < 0)
            return 0.0;
        const int lz   = __clz(top) - 16; /* top in [1, 2^16) */
        uint64_t  mant = (static_cast<uint64_t>(top) << (48 + lz)) | (low >> (16 - lz));
        if(sticky || (low & ((uint64_t(1) << (16 - lz)) - 1u)) != 0u)
            mant |= 1u;
        const double d = ldexp(static_cast<double>(mant), 16 * (h - 4) + 16 - lz + e);
        return neg ? -d : d;
    }

    template <bool HAS_LO, unsigned CHUNK_SIZE, bool IS_FIRST_CHUNK, typename T_out = double>
    __global__ static void accum_finalize_kernel(const int32_t* __restrict__ C32i_batch,
                                                 const double* __restrict__ Zhi_in,
                                                 const double* __restrict__ Zlo_in,
                                                 const T_out* __restrict__ C,
                                                 T_out* __restrict__ D,
                                                 int64_t m,
                                                 int64_t n,
                                                 size_t  ldc32i,
                                                 int64_t ldc,
                                                 int64_t ldd,
                                                 double  alpha,
                                                 double  beta,
                                                 const int16_t* __restrict__ sftA,
                                                 const int16_t* __restrict__ sftB,
                                                 unsigned chunk_start,
                                                 unsigned effective_s)
    {
        const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t l = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        if(i >= m || l >= n)
            return;
        const size_t idx          = static_cast<size_t>(i) + static_cast<size_t>(l) * ldc32i;
        const size_t slice_stride = ldc32i * static_cast<size_t>(n);
        const int    inv_sft = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
        double       x_scaled; /* ldexp(X, inv_sft), X = the reconstructed integer */
        if constexpr(IS_FIRST_CHUNK)
        {
            /* Single chunk: chunk_start == 0 and CHUNK_SIZE == effective_s.  Exact
             * integer CRT sum (see crt_sum_limbs); all CS loads are issued first
             * so they overlap.                                                    */
            constexpr unsigned NL    = crt_sum_limbs<CHUNK_SIZE>();
            const unsigned     s_idx = effective_s - 2;
            int32_t            dc_s[CHUNK_SIZE];
#pragma unroll
            for(unsigned t = 0; t < CHUNK_SIZE; ++t)
                dc_s[t] = balanced_residue_i32(C32i_batch[t * slice_stride + idx], t);
            int32_t z[NL];
#pragma unroll
            for(unsigned j = 0; j < NL; ++j)
                z[j] = 0;
#pragma unroll
            for(unsigned t = 0; t < CHUNK_SIZE; ++t)
#pragma unroll
                for(unsigned j = 0; j < NL && j < CRT_LIMBS; ++j)
                    z[j] += __mul24(dc_s[t], static_cast<int32_t>(qpi_limb(s_idx, t, j)));
            crt_normalize(z);
            /* q = rint(Z / P) from the top five limbs.  |X| <= P/4 (the one-bit
             * margin in crt_capacity_bits), so this approximation of Z / P is
             * far from a rounding boundary.  |q| <= CS * 128, so q * P_limb fits
             * __mul24 and int32.                                                   */
            constexpr unsigned lo_limb = (NL > 5u) ? NL - 5u : 0u;
            double             zd      = 0.0;
#pragma unroll
            for(unsigned j = NL; j-- > lo_limb;)
                zd = fma(zd, 65536.0, static_cast<double>(z[j]));
            zd              = ldexp(zd, 16 * static_cast<int>(lo_limb));
            const int32_t q = static_cast<int32_t>(rint(zd * inv_P(s_idx)));
#pragma unroll
            for(unsigned j = 0; j < NL && j < CRT_LIMBS; ++j)
                z[j] -= __mul24(q, static_cast<int32_t>(P_limb(s_idx, j)));
            crt_normalize(z);
            x_scaled = crt_to_double(z, inv_sft);
        }
        else
        {
            double dc_s[CHUNK_SIZE];
#pragma unroll
            for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
            {
                const unsigned t = chunk_start + t_local;
                dc_s[t_local]    = static_cast<double>(
                    balanced_residue_i32(C32i_batch[t_local * slice_stride + idx], t));
            }
            double Zhi = 0.0, Zlo = 0.0;
#pragma unroll
            for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
            {
                const unsigned t      = chunk_start + t_local;
                const double   dc     = dc_s[t_local];
                const auto [qhi, qlo] = qpi(effective_s - 2, t);
                const double hi       = dc * qhi;
                const double new_hi   = Zhi + hi;
                // TwoSum (Knuth) — exact for ANY operand ordering (see chunk_accum_kernel).
                const double bb  = new_hi - Zhi;
                const double err = (Zhi - (new_hi - bb)) + (hi - bb);
                Zhi              = new_hi;
                if constexpr(HAS_LO)
                    Zlo = fma(dc, qlo, Zlo + err);
                else
                    Zlo += err;
            }
            /* Fold the prior chunks' Zhi_in/Zlo_in into Zhi/Zlo in-place. */
            {
                const double old_hi = Zhi_in[idx];
                const double s_hi   = old_hi + Zhi;
                const double err    = Zhi - (s_hi - old_hi);
                Zhi                 = s_hi;
                Zlo                 = Zlo_in[idx] + err + Zlo;
            }
            const double q        = rint((Zhi + Zlo) * inv_P(effective_s - 2));
            const auto [phi, plo] = P_dd(effective_s - 2);
            const double X        = fma(plo, q, fma(phi, q, Zhi) + Zlo);
            x_scaled              = ldexp(X, inv_sft);
        }
        const size_t d_idx
            = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
        double d_val = alpha * x_scaled;
        if(beta != 0.0)
        {
            const size_t c_idx
                = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
            d_val += beta * static_cast<double>(C[c_idx]);
        }
        __builtin_nontemporal_store(static_cast<T_out>(d_val), D + d_idx);
    }

} // namespace FixedPointEmulation
