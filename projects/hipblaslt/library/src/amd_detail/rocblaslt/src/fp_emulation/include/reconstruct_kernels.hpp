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
 * These kernels reconstruct the exact GEMM result from its CRT residues
 * (accumulated via TwoSum), then range-reduce and scale it back to the
 * output type.
 *
 * This header must be compiled as HIP.
 */

#include "tables.hpp" /* neg_mod, inv_mod, qpi, inv_P, P_dd */

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
         * Pass 1: all CS loads + rint+fma issued simultaneously (MLP, ~200cy latency);
         *   FP64 compute overlaps HBM latency. Uses CS×2 VGPRs (~52 for CS=16).
         * Pass 2: pure register TwoSum, loop-carried dep through Zhi only.
         * Note: storing dc as int32 (dc_int_s) was tested — same performance at all
         * sizes (kernel is HBM-bandwidth-limited; higher occupancy does not help). */
        double dc_s[CHUNK_SIZE];
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t      = chunk_start + t_local;
            const double   dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
            dc_s[t_local]         = fma(neg_mod(t), rint(dc_raw * inv_mod(t)), dc_raw);
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
        double       dc_s[CHUNK_SIZE];
#pragma unroll
        for(unsigned t_local = 0; t_local < CHUNK_SIZE; ++t_local)
        {
            const unsigned t      = chunk_start + t_local;
            const double   dc_raw = static_cast<double>(C32i_batch[t_local * slice_stride + idx]);
            dc_s[t_local]         = fma(neg_mod(t), rint(dc_raw * inv_mod(t)), dc_raw);
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
        /* IS_FIRST_CHUNK=true: Zhi/Zlo already hold the final value (no prior chunk).
         * IS_FIRST_CHUNK=false: fold the prior Zhi_in/Zlo_in into Zhi/Zlo in-place. */
        if constexpr(!IS_FIRST_CHUNK)
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
        const int    inv_sft  = -(static_cast<int>(sftA[i]) + static_cast<int>(sftB[l]));
        const size_t d_idx
            = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldd);
        double d_val = alpha * ldexp(X, inv_sft);
        if(beta != 0.0)
        {
            const size_t c_idx
                = static_cast<size_t>(i) + static_cast<size_t>(l) * static_cast<size_t>(ldc);
            d_val += beta * static_cast<double>(C[c_idx]);
        }
        __builtin_nontemporal_store(static_cast<T_out>(d_val), D + d_idx);
    }

} // namespace FixedPointEmulation
