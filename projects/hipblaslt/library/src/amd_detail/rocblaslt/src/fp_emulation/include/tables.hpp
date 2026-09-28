// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * tables.hpp
 *
 * Constexpr device functions returning the CRT constant data used by the
 * fixed-point emulation kernels (FP32 and FP64).  Embedding the data here
 * (rather than in __constant__ globals) avoids all cross-translation-unit
 * device-symbol issues and requires no initialization step.
 *
 * Naming convention for the s-dependent functions:
 *   s_idx = num_moduli - 2  (0-based index into the per-s tables)
 *
 * This file may be included by any HIP translation unit; it is intentionally
 * separate from emulation.hpp (which must stay free of __device__
 * code so it can be included from plain C++ units).
 *
 * Constants (tables) are based on the open-source GEMMul8 implementation
 * (Y. Uchino, RIKEN R-CCS, https://github.com/RIKEN-RCCS/GEMMul8).
 */

#include <array>
#include <cmath>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace FixedPointEmulation
{
    /* Maximum number of moduli for FP64 (full precision range). */
    static constexpr unsigned S_MAX_FP64 = 20u;
    /* Maximum number of moduli for FP32 (reduced range; 12 moduli give
     * ~46 bits of CRT capacity, far exceeding FP32's 23-bit precision). */
    static constexpr unsigned S_MAX_FP32 = 12u;
    /* S_MAX is an alias for S_MAX_FP64 and is used for table sizing only. */
    static constexpr unsigned S_MAX = S_MAX_FP64;

    /* Returns the maximum number of moduli for the given floating-point type T.
     * Use these functions in kernels and workspace-size calculations instead of
     * the raw constants, so that FP32 paths use the smaller S_MAX_FP32. */
    template <typename T>
    __host__ __device__ constexpr unsigned max_num_moduli() noexcept
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "max_num_moduli<T>: T must be float or double");
        return S_MAX_FP64; // specializations below override for float and double
    }
    template <>
    __host__ __device__ constexpr unsigned max_num_moduli<float>() noexcept
    {
        return S_MAX_FP32;
    }
    template <>
    __host__ __device__ constexpr unsigned max_num_moduli<double>() noexcept
    {
        return S_MAX_FP64;
    }

    /* hipDataType overload for non-templated callers. */
    inline __host__ __device__ constexpr unsigned max_num_moduli(hipDataType t) noexcept
    {
        return (t == HIP_R_32F) ? S_MAX_FP32 : (t == HIP_R_64F) ? S_MAX_FP64 : 0u;
    }

    /* Alignment for INT8 arrays (128 bytes = 128 INT8 elements). */
    static constexpr size_t ALIGN = 128u;

    /* Round n up to the nearest multiple of ALIGN. */
    inline __host__ __device__ size_t pad(size_t n) noexcept
    {
        return (n + ALIGN - 1u) / ALIGN * ALIGN;
    }

    /* The S_MAX pairwise-coprime CRT moduli m_i, as positive integers.
     * All derived tables (neg_mod, inv_mod, inv_mod_f) are computed from
     * these values. */
    static constexpr int moduli[S_MAX] = {
        256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
        223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
    };

    /* Negative of the i-th CRT modulus m_i, i.e. -m_i, as a double. */
    __device__ __forceinline__ constexpr double neg_mod(unsigned i) noexcept
    {
        return -(double)moduli[i];
    }

    /* Nearest double to 1/m_i, derived at compile time from moduli[].
     * Used in the fast reduction r_i = X - m_i * round(X * inv_mod[i]). */
    __device__ __forceinline__ constexpr double inv_mod(unsigned i) noexcept
    {
        constexpr auto v = []() constexpr {
            std::array<double, S_MAX> a{};
            for(unsigned k = 0; k < S_MAX; ++k)
                a[k] = 1.0 / (double)moduli[k];
            return a;
        }();
        return v[i];
    }

    /* Nearest float to 1/m_i, derived at compile time from inv_mod[].
     * Used in the fast reduction r_i = X - m_i * round(X * inv_mod_f[i]). */
    __device__ __forceinline__ constexpr float inv_mod_f(unsigned i) noexcept
    {
        constexpr auto v = []() constexpr {
            std::array<float, S_MAX> a{};
            for(unsigned k = 0; k < S_MAX; ++k)
                a[k] = static_cast<float>(inv_mod(k));
            return a;
        }();
        return v[i];
    }

    // UInt256: compile-time 256-bit unsigned integer (4 x uint64_t, little-endian).
    struct UInt256
    {
        uint64_t d[4]{};
        constexpr UInt256() noexcept = default;
        constexpr explicit UInt256(uint64_t v) noexcept
            : d{v, 0, 0, 0}
        {
        }

        constexpr bool bit(int p) const noexcept
        {
            if(p < 0 || p >= 256)
                return false;
            return (d[p >> 6] >> (p & 63)) & 1u;
        }

        constexpr int msb_pos() const noexcept
        {
            for(int i = 3; i >= 0; --i)
            {
                if(d[i] != 0)
                {
                    uint64_t v = d[i];
                    int      b = 0;
                    if(v >> 32)
                    {
                        b += 32;
                        v >>= 32;
                    }
                    if(v >> 16)
                    {
                        b += 16;
                        v >>= 16;
                    }
                    if(v >> 8)
                    {
                        b += 8;
                        v >>= 8;
                    }
                    if(v >> 4)
                    {
                        b += 4;
                        v >>= 4;
                    }
                    if(v >> 2)
                    {
                        b += 2;
                        v >>= 2;
                    }
                    b += (int)(v >> 1);
                    return i * 64 + b;
                }
            }
            return -1;
        }

        constexpr uint64_t extract_bits(int high, int low) const noexcept
        {
            int      ll = low >> 6, lo = low & 63, hl = high >> 6;
            uint64_t r = d[ll] >> lo;
            if(lo != 0 && hl > ll && hl < 4)
                r |= d[hl] << (64 - lo);
            int len = high - low + 1;
            if(len < 64)
                r &= (1ULL << len) - 1u;
            return r;
        }

        constexpr bool any_bit_set_below(int p) const noexcept
        {
            if(p < 0)
                return false;
            int limb = p >> 6, bo = p & 63;
            for(int i = 0; i < limb && i < 4; ++i)
                if(d[i])
                    return true;
            if(limb < 4)
            {
                uint64_t mask = (bo < 63) ? ((1ULL << (bo + 1)) - 1u) : ~uint64_t(0);
                if(d[limb] & mask)
                    return true;
            }
            return false;
        }

        constexpr UInt256 mul(uint32_t rhs) const noexcept
        {
            UInt256  res;
            uint64_t carry = 0;
            for(int i = 0; i < 4; ++i)
            {
                uint64_t lo = d[i] & 0xFFFFFFFFu, hi = d[i] >> 32;
                uint64_t p0 = lo * rhs + (carry & 0xFFFFFFFFu),
                         p1 = hi * rhs + (carry >> 32) + (p0 >> 32);
                res.d[i]    = (p0 & 0xFFFFFFFFu) | ((p1 & 0xFFFFFFFFu) << 32);
                carry       = p1 >> 32;
            }
            return res;
        }

        constexpr UInt256 shl(int n) const noexcept
        {
            if(n == 0)
                return *this;
            UInt256 res;
            int     ls = n >> 6, bs = n & 63;
            for(int i = 3; i >= 0; --i)
            {
                int src = i - ls;
                if(src < 0)
                    res.d[i] = 0;
                else if(bs == 0)
                    res.d[i] = d[src];
                else
                {
                    res.d[i] = d[src] << bs;
                    if(src > 0)
                        res.d[i] |= d[src - 1] >> (64 - bs);
                }
            }
            return res;
        }

        constexpr UInt256 shr(int n) const noexcept
        {
            if(n == 0)
                return *this;
            UInt256 res;
            int     ls = n >> 6, bs = n & 63;
            for(int i = 0; i < 4; ++i)
            {
                int src = i + ls;
                if(src >= 4)
                    res.d[i] = 0;
                else if(bs == 0)
                    res.d[i] = d[src];
                else
                {
                    res.d[i] = d[src] >> bs;
                    if(src + 1 < 4)
                        res.d[i] |= d[src + 1] << (64 - bs);
                }
            }
            return res;
        }

        constexpr bool ge(const UInt256& r) const noexcept
        {
            for(int i = 3; i >= 0; --i)
            {
                if(d[i] > r.d[i])
                    return true;
                if(d[i] < r.d[i])
                    return false;
            }
            return true;
        }

        constexpr UInt256 sub(const UInt256& r) const noexcept
        {
            UInt256  res;
            uint64_t borrow = 0;
            for(int i = 0; i < 4; ++i)
            {
                uint64_t a = d[i], b = r.d[i], d1 = a - borrow, b1 = (a < borrow) ? 1u : 0u,
                         d2 = d1 - b, b2 = (d1 < b) ? 1u : 0u;
                res.d[i] = d2;
                borrow   = b1 + b2;
            }
            return res;
        }

        // Exact division by a 32-bit divisor (caller must ensure zero remainder).
        constexpr UInt256 div(uint32_t rhs) const noexcept
        {
            UInt256  q;
            uint64_t rem = 0;
            for(int i = 3; i >= 0; --i)
            {
                uint64_t hi = (rem << 32) | (d[i] >> 32);
                uint64_t qh = hi / rhs, rh = hi % rhs;
                uint64_t lo = (rh << 32) | (d[i] & 0xFFFFFFFFu);
                uint64_t ql = lo / rhs;
                q.d[i]      = (qh << 32) | ql;
                rem         = lo % rhs;
            }
            return q;
        }

        // Compute this mod rhs (32-bit divisor), returning remainder.
        constexpr uint32_t mod(uint32_t rhs) const noexcept
        {
            uint64_t rem = 0;
            for(int i = 3; i >= 0; --i)
            {
                uint64_t hi = (rem << 32) | (d[i] >> 32);
                uint64_t rh = hi % rhs;
                uint64_t lo = (rh << 32) | (d[i] & 0xFFFFFFFFu);
                rem         = lo % rhs;
            }
            return static_cast<uint32_t>(rem);
        }

        constexpr double to_double() const noexcept
        {
            int top = msb_pos();
            if(top < 0)
                return 0.0;
            if(top <= 52)
                return static_cast<double>(d[0]);
            int      shift = top - 52;
            uint64_t m     = extract_bits(top, top - 52);
            bool     rb = bit(top - 53), st = any_bit_set_below(top - 54);
            if(rb && (st || (m & 1u)))
                ++m;
            double r = static_cast<double>(m);
            int    s = shift;
            while(s >= 32)
            {
                r *= 4294967296.0;
                s -= 32;
            }
            if(s > 0)
                r *= static_cast<double>(1u << s);
            return r;
        }

        // Returns qpi_hi = the value with its 53-bit mantissa truncated toward
        // zero to (53 - QPI_N_TZ) significant bits (its low QPI_N_TZ bits cleared).
        //
        // Algorithm:
        //   1. Round Q to the nearest 53-bit double (round-to-nearest-even).
        //   2. Truncate ("floor") the stored 53-bit mantissa by QPI_N_TZ bits.
        //   3. If the result exceeds Q (can happen when step 1 rounded UP and the
        //      increment crossed an alignment boundary), subtract one alignment
        //      unit to restore result <= Q.
        // "floor" (truncation toward zero) guarantees qpi_hi <= qpi_exact, hence
        // qpi_lo = qpi_exact - qpi_hi >= 0 — a non-negative lo avoids cancellation
        // in the FMA accumulation of chunk_accum_kernel.
        //
        // For small values (top <= 52) the CRT coefficient is an exact integer
        // < 2^53, so it is returned exactly (qpi_lo = 0).  This is required
        // because the kernel uses HAS_LO=false (ignoring qpi_lo) when effective_s
        // <= 7.
        //
        // ── Choice of QPI_N_TZ ────────────────────────────────────────────────
        // QPI_N_TZ = number of trailing zero bits forced into the 53-bit mantissa
        // of qpi_hi, so qpi_hi carries 53-QPI_N_TZ significant bits.  Two lower
        // bounds constrain it:
        //   * Per-term exactness of dc*qpi_hi: a balanced residue has
        //     |dc| <= max(m)/2 = 128 = 2^7, so sig(dc) <= 7 and we need
        //     sig(qpi_hi) <= 53 - 7 = 46, i.e. QPI_N_TZ >= 7.
        //   * Whole-sum headroom: the accumulated high word sum_t(dc_t*qpi_hi_t)
        //     can carry up to ceil(log2(sum_t floor(m_t/2))) = ceil(log2(2188))
        //     = 12 extra bits over a single term, so QPI_N_TZ >= 12 (41
        //     significant bits) keeps it representable in one double.
        //
        // We use QPI_N_TZ = 13 (40 significant bits) — one bit above the minimum.
        // With a TwoSum accumulator 12 (41 bits) would already be correct; the
        // extra trailing zero is a cheap safety/accuracy margin:
        //   * Every product hi = dc*qpi_hi then carries >= 1 trailing zero, so the
        //     running-sum error term stays exact even under a plain FastTwoSum
        //     (whose err = hi - (new_hi - Zhi) is otherwise exact only when
        //     |Zhi| >= |hi|) — i.e. the accumulation becomes order-independent.
        //   * Empirically it tightens the final reconstruction error
        //     |Zhi+Zlo - Z_exact|/P from ~1e-14 (NTZ=12) to ~1e-25 (NTZ=13); this
        //     summed CRT remainder is the only quantity that matters (per-
        //     coefficient accuracy is irrelevant).
        //   * It costs nothing: dc*qpi_hi is still FP64-exact (40 + 7 = 47 <= 53).
        // The kernel additionally uses TwoSum for full order-independence.
        static constexpr int QPI_N_TZ = 13; // 40 significant bits (see note above)
        constexpr double     to_double_aligned() const noexcept
        {
            int top = msb_pos();
            if(top < 0)
                return 0.0;
            // For top <= 52 the value is an exact integer < 2^53, perfectly
            // representable as a double.  Return it exactly so qpi_lo = 0.
            if(top <= 52)
                return static_cast<double>(d[0]);
            int      shift = top - 52;
            uint64_t m     = extract_bits(top, top - 52);
            bool     rb = bit(top - 53), st = any_bit_set_below(top - 54);
            if(rb && (st || (m & 1u)))
                ++m;
            if(m >= (1ULL << 53))
            {
                m >>= 1;
                ++shift;
            }
            // Truncate the 53-bit mantissa by QPI_N_TZ bits (floor).
            uint64_t ma = (m >> QPI_N_TZ) << QPI_N_TZ;
            // Step 3: overshoot correction.
            // If ma * 2^shift > Q (can happen when step 1 rounded UP and the carry
            // crossed an alignment boundary), subtract one alignment unit to ensure
            // qpi_hi <= qpi_exact, guaranteeing qpi_lo = qpi_exact - qpi_hi >= 0.
            {
                UInt256 ms = UInt256(ma).shl(shift);
                if(!ge(ms))
                    ma -= (1u << QPI_N_TZ);
            }
            double r = static_cast<double>(ma);
            int    s = shift;
            while(s >= 32)
            {
                r *= 4294967296.0;
                s -= 32;
            }
            if(s > 0)
                r *= static_cast<double>(1u << s);
            return r;
        }
    };

    // Compute qpi_lo = Q - to_double_aligned(Q) as a double.
    // Uses the same floor split as to_double_aligned(), so qpi_lo >= 0 always.
    constexpr double qpi_lo_val(const UInt256& Q) noexcept
    {
        int top = Q.msb_pos();
        if(top < 0)
            return 0.0;
        // to_double_aligned() returns d[0] exactly for top <= 52,
        // so qpi_lo = Q - qpi_hi = 0 in that case.
        if(top <= 52)
            return 0.0;
        int      shift = top - 52;
        uint64_t m     = Q.extract_bits(top, top - 52);
        bool     rb = Q.bit(top - 53), st = Q.any_bit_set_below(top - 54);
        if(rb && (st || (m & 1u)))
            ++m;
        if(m >= (1ULL << 53))
        {
            m >>= 1;
            ++shift;
        }
        uint64_t ma = (m >> UInt256::QPI_N_TZ) << UInt256::QPI_N_TZ;
        UInt256  ms = UInt256(ma).shl(shift);
        // Ensure ms <= Q (the overshoot case: when nearest-double rounded UP
        // and the increment crossed an alignment boundary).
        if(!Q.ge(ms))
        {
            ma -= (1u << UInt256::QPI_N_TZ);
            ms = UInt256(ma).shl(shift);
        }
        UInt256 diff = Q.sub(ms); // Q >= ms guaranteed (qpi_hi <= qpi_exact)
        // Correctly-rounded nearest double of the exact residual (more accurate
        // than the old (double)d0 + (double)d1*2^64 two-step, which lost bits for
        // residuals >= 2^64).  qpi_lo >= 0 is preserved by the floor-based hi.
        return diff.to_double();
    }

    constexpr uint32_t mod_inverse(uint32_t a, uint32_t m) noexcept
    {
        int32_t t = 0, newt = 1, r = (int32_t)m, newr = (int32_t)a;
        while(newr != 0)
        {
            int32_t q = r / newr, tmp = t - q * newt;
            t    = newt;
            newt = tmp;
            tmp  = r - q * newr;
            r    = newr;
            newr = tmp;
        }
        if(t < 0)
            t += (int32_t)m;
        return (uint32_t)t;
    }

    constexpr std::array<UInt256, S_MAX - 1> make_P_table() noexcept
    {
        std::array<UInt256, S_MAX - 1> t{};
        UInt256                        prod(static_cast<uint64_t>(moduli[0]));
        for(unsigned i = 0; i < S_MAX - 1; ++i)
        {
            prod = prod.mul(static_cast<uint32_t>(moduli[i + 1]));
            t[i] = prod;
        }
        return t;
    }

    constexpr double p_lo_double(const UInt256& P) noexcept
    {
        int top = P.msb_pos();
        if(top < 0 || top <= 52)
            return 0.0;
        int      shift = top - 52;
        uint64_t m     = P.extract_bits(top, top - 52);
        bool     rb = P.bit(top - 53), st = P.any_bit_set_below(top - 54);
        if(rb && (st || (m & 1u)))
            ++m;
        UInt256 ms  = UInt256(m).shl(shift);
        double  res = 0.0;
        if(P.ge(ms))
        {
            UInt256 diff = P.sub(ms);
            res          = static_cast<double>(diff.d[0])
                  + static_cast<double>(diff.d[1]) * 18446744073709551616.0;
        }
        else
        {
            UInt256 diff = ms.sub(P);
            res          = -(static_cast<double>(diff.d[0])
                    + static_cast<double>(diff.d[1]) * 18446744073709551616.0);
        }
        return -res;
    }

    // Double-double representation of -P_s = {hi, lo} where hi = nearest_double(-P_s)
    // and lo = -P_s - hi (the rounding residual).  Returns hi and lo as a pair so
    // callers always get both components from a single table lookup, preventing
    // mismatches between independently loaded hi and lo values.
    __device__ __forceinline__ std::pair<double, double> P_dd(unsigned s_idx) noexcept
    {
        static constexpr auto v = []() constexpr {
            auto                                             P = make_P_table();
            std::array<std::pair<double, double>, S_MAX - 1> a{};
            for(unsigned i = 0; i < S_MAX - 1; ++i)
            {
                a[i].first  = -P[i].to_double();
                a[i].second = p_lo_double(P[i]);
            }
            return a;
        }();
        return v[s_idx];
    }

    // Nearest double to 1/P_s.
    // Used in the range-reduction step: q = rint((Zhi + Zlo) * inv_P(s))
    // folds the accumulated CRT sum Z into the symmetric range (-P_s/2, P_s/2].
    __device__ __forceinline__ double inv_P(unsigned s_idx) noexcept
    {
        static constexpr auto v = []() constexpr {
            auto                          P = make_P_table();
            std::array<double, S_MAX - 1> a{};
            for(unsigned i = 0; i < S_MAX - 1; ++i)
                a[i] = 1.0 / P[i].to_double();
            return a;
        }();
        return v[s_idx];
    }

    // qpi_k = (P(s)/m_k) * mod_inverse((P(s)/m_k) mod m_k, m_k)
    // This is the CRT coefficient for index k: summing dc[k]*qpi_k over k
    // reconstructs x from its residues dc[k] = x mod m_k.
    //
    // Returns {hi, lo} where:
    //   hi: first (53-QPI_N_TZ) significant bits of qpi_k (floor, always <= qpi_k),
    //       making dc*hi FP64-exact for all balanced residues |dc| <= max(moduli)/2.
    //   lo: double-double residual qpi_k - hi >= 0.
    //
    // Both words are computed entirely at compile time from the CRT moduli via
    // make_P_table() / to_double_aligned() / qpi_lo_val() — see the algorithm
    // note on UInt256::to_double_aligned() (QPI_N_TZ trailing-zero split).  For
    // s <= 6 (s_idx <= 4) the CRT coefficient is an exact integer < 2^53, so
    // lo = 0 (the HAS_LO=false kernel path applies).
    __device__ __forceinline__ std::pair<double, double> qpi(unsigned s_idx,
                                                             unsigned mod_idx) noexcept
    {
        // Computed constexpr table: floor hi (qpi_hi <= qpi, so qpi_lo >= 0)
        // with QPI_N_TZ trailing zeros, plus the correctly-rounded nearest
        // double of the exact residual as lo.  Built once at compile time.
        static constexpr auto vc = []() constexpr {
            std::array<std::array<std::pair<double, double>, S_MAX>, S_MAX - 1> t{};
            auto                                                                Pt = make_P_table();
            for(unsigned s = 1; s < S_MAX; ++s)
                for(unsigned m = 0; m <= s; ++m)
                {
                    uint32_t mk        = static_cast<uint32_t>(moduli[m]);
                    UInt256  Mk        = Pt[s - 1].div(mk);
                    uint32_t rk        = Mk.mod(mk);
                    uint32_t Nk        = mod_inverse(rk, mk);
                    UInt256  qp        = Mk.mul(Nk);
                    t[s - 1][m].first  = qp.to_double_aligned();
                    t[s - 1][m].second = qpi_lo_val(qp);
                }
            return t;
        }();
        return vc[s_idx][mod_idx];
    }

    // log2P(s_idx): usable CRT capacity (bits) for (s_idx+2) moduli.
    //
    //   log2(P_s)/2 - 0.5: per-side CRT capacity.  The 1/2 splits the single
    //     CRT budget log2(M_s/2) equally between the A and B shifts; the -0.5
    //     folds in the M_s/2 safety divisor.
    //   - 1.0: REQUIRED one-bit CRT safety margin (do NOT remove).
    //
    // Used both to pick effective_s (min s with log2P(s-2) >= log2P_needed)
    // and as refine_log2P in refine_sftA/B_kernel, which folds it into the
    // per-row/col shifts (sft += floor(-0.5*log2(max_val) + log2P)).
    // Reconstruction folds Z into (-M/2, M/2] via q=rint(Z/M), X=Z-q*M, so
    // correctness needs |X_true| <= M/2.  But sftA/sftB are set BEFORE X_true
    // is known, from preliminary GEMM maxima + a float __log2f that can
    // under-predict by ~1 bit; the -1.0 targets |X_true| <= M/4 so worst-case
    // error still lands inside M/2.
    //
    // Computed lazily on first host call; host-only.
    inline double log2P(unsigned s_idx) noexcept
    {
        static const auto v = []() {
            auto                          P = make_P_table();
            std::array<double, S_MAX - 1> a{};
            for(unsigned i = 0; i < S_MAX - 1; ++i)
                a[i] = std::log2(P[i].to_double()) / 2.0 - 0.5 - 1.0;
            return a;
        }();
        return v[s_idx];
    }

} // namespace FixedPointEmulation
