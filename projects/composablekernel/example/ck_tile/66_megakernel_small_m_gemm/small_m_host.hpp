// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of example 66: bf16 helpers, the double reference, and an emulation of
// SmallMGemmPhase's data path (lane pieces, WMMA operand assembly, C layout, wave and split-K
// reduction order) against ck_tile's gfx125 bf16 WMMA layout model. Uses no HIP API, so the
// emulation builds and runs on a host without a GPU.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/gemm/small_m_gemm_phase.hpp"

namespace small_m_host {

using index_t = ck_tile::index_t;
namespace sm  = ck_tile::small_m;

inline float bf(uint16_t b) { return sm::bf16_bits_to_float(b); }

inline std::vector<uint16_t> random_bf16(size_t n, float sigma, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, sigma);
    std::vector<uint16_t> v(n);
    for(auto& x : v)
        x = sm::bf16_rne_bits(dist(rng));
    return v;
}

/// Split count for @p grid workgroups. Cost model, in window steps of one wave:
///   rounds(S) * (windows / S / waves + stages)
/// where rounds = ceil(n_blocks S / grid) and `stages` stands for the fixed cost of an item (ring
/// fill latency, reduction, hand-off). Keeps at least one window per wave; ties go to fewer
/// splits. A first guess: sweep -splitk on the GPU.
inline index_t
auto_splits(index_t n_blocks, index_t windows, index_t waves, index_t stages, index_t grid)
{
    index_t best       = 1;
    double best_c      = 1e30;
    const index_t smax = std::max<index_t>(1, std::min<index_t>(64, windows / waves));
    for(index_t s = 1; s <= smax; ++s)
    {
        const double rounds = std::ceil(double(n_blocks) * s / grid);
        const double c      = rounds * (double(windows) / s / waves + stages);
        if(c < best_c * (1 - 1e-9))
        {
            best_c = c;
            best   = s;
        }
    }
    return best;
}

/// Epilogue arguments as the example uses them: none, or residual add with alpha = 1.
struct HostEpi
{
    const uint16_t* residual = nullptr; // [M, ld]
    index_t ld               = 0;
};

/// sum_k x[r, k] w[n, k] in double, plus sum |x w| for the tolerance.
inline void ref_dot(const std::vector<uint16_t>& x,
                    const std::vector<uint16_t>& w,
                    index_t K,
                    index_t r,
                    index_t n,
                    double& ref,
                    double& mag)
{
    ref = 0;
    mag = 0;
    for(index_t k = 0; k < K; ++k)
    {
        const double p = double(bf(x[size_t(r) * K + k])) * bf(w[size_t(n) * K + k]);
        ref += p;
        mag += std::fabs(p);
    }
}

/// Checks rows @p rows of got [M, N] (bf16 bits) against the double reference of x * w^T
/// (+ residual). Tolerance: |got - ref| <= |ref| 2^-7 + sum|x w| 2^-14.
inline int check_rows(const char* name,
                      const std::vector<uint16_t>& got,
                      const std::vector<uint16_t>& x,
                      const std::vector<uint16_t>& w,
                      index_t N,
                      index_t K,
                      const std::vector<index_t>& rows,
                      const HostEpi& epi = {})
{
    int bad = 0;
    for(index_t r : rows)
    {
        for(index_t n = 0; n < N; ++n)
        {
            double ref, mag;
            ref_dot(x, w, K, r, n, ref, mag);
            if(epi.residual)
            {
                const double res = bf(epi.residual[size_t(r) * epi.ld + n]);
                ref += res;
                mag += std::fabs(res);
            }
            const double g   = bf(got[size_t(r) * N + n]);
            const double tol = std::fabs(ref) * std::ldexp(1.0, -7) + mag * std::ldexp(1.0, -14);
            if(!(std::fabs(g - ref) <= tol))
            {
                if(bad < 5)
                    std::printf("  %s[%d, %d] = %g, ref %g, tol %g\n", name, r, n, g, ref, tol);
                ++bad;
            }
        }
    }
    return bad;
}

/// Emulates Phase on x [M, K], w [N, K] with k_splits splits, following the device code's index
/// math (small_m::piece_k, wmma_piece, split_begin, wave_windows, the C layout and the reduction
/// orders), with the WMMA itself modelled after ck_tile's WmmaTraits for gfx125 bf16 16x16x32:
/// element e of lane half h is logical K (e / K1) * (2 K1) + h K1 + e % K1 on A and B; C element
/// (row 8 h + v, column c) in lane c + 16 h, element v. Returns Y as fp32 before rounding.
template <typename Phase>
std::vector<float> emulate(const std::vector<uint16_t>& x,
                           const std::vector<uint16_t>& w,
                           index_t M,
                           index_t N,
                           index_t K,
                           index_t k_splits,
                           const HostEpi& epi)
{
    using Wmma            = typename Phase::Wmma;
    constexpr index_t K1  = Wmma::kAK1PerLane;
    constexpr index_t KL  = Wmma::kABKLane;
    constexpr index_t kK  = Wmma::kK;
    static_assert(K1 == Wmma::kBK1PerLane && KL == 2 && kK == 32, "bf16 16x16x32 model");
    constexpr index_t kNT = Phase::kNTiles, kMT = Phase::kMTiles, kW = Phase::kWaves;
    constexpr index_t kNR = Phase::kNRows, kMaxM = Phase::kMaxM;

    const index_t windows  = K / sm::kWindowK;
    const index_t n_blocks = N / kNR;
    std::vector<float> y(size_t(M) * N, 0.f);
    std::vector<float> partial(size_t(k_splits) * n_blocks * M * kNR, 0.f);

    for(index_t nb = 0; nb < n_blocks; ++nb)
    {
        for(index_t s = 0; s < k_splits; ++s)
        {
            const index_t w0 = sm::split_begin(windows, k_splits, s);
            const index_t w1 = sm::split_begin(windows, k_splits, s + 1);
            // red[wave][m][n] as in LDS (rows of all kMaxM tokens).
            std::vector<float> red(size_t(kW) * kMaxM * kNR, 0.f);
            for(index_t wave = 0; wave < kW; ++wave)
            {
                const index_t nw = sm::wave_windows(w0, w1, wave, kW);
                // acc[t][mt][lane][v]
                std::vector<float> acc(size_t(kNT) * kMT * 32 * 8, 0.f);
                for(index_t i = 0; i < nw; ++i)
                {
                    const index_t win = w0 + wave + i * kW;
                    // Lane pieces: piece[q][lane][8 bf16], q < kPiecesW weights, then tokens.
                    std::vector<uint16_t> piece(size_t(Phase::kPieces) * 32 * 8);
                    for(index_t lane = 0; lane < 32; ++lane)
                    {
                        const index_t lr = lane % 16, half = lane / 16;
                        for(index_t c = 0; c < sm::kChunks; ++c)
                        {
                            const index_t k = win * sm::kWindowK + sm::piece_k(half, c);
                            for(index_t t = 0; t < kNT; ++t)
                                for(index_t e = 0; e < 8; ++e)
                                    piece[((t * sm::kChunks + c) * 32 + lane) * 8 + e] =
                                        w[size_t(nb * kNR + t * 16 + lr) * K + k + e];
                            for(index_t mt = 0; mt < kMT; ++mt)
                            {
                                const index_t row = std::min(mt * 16 + lr, M - 1);
                                for(index_t e = 0; e < 8; ++e)
                                    piece[((Phase::kPiecesW + mt * sm::kChunks + c) * 32 + lane) *
                                              8 +
                                          e] = x[size_t(row) * K + k + e];
                            }
                        }
                    }
                    // Two WMMAs a window: operand element e of lane = piece wmma_piece(j, e),
                    // element e % 8.
                    for(index_t j = 0; j < 2; ++j)
                    {
                        auto elem = [&](index_t q0, index_t lane, index_t e) {
                            const index_t q = q0 + sm::wmma_piece(j, e) - sm::wmma_piece(j, 0);
                            return bf(piece[(size_t(q) * 32 + lane) * 8 + e % 8]);
                        };
                        for(index_t t = 0; t < kNT; ++t)
                        {
                            for(index_t mt = 0; mt < kMT; ++mt)
                            {
                                const index_t qa = t * sm::kChunks + sm::wmma_piece(j, 0);
                                const index_t qb =
                                    Phase::kPiecesW + mt * sm::kChunks + sm::wmma_piece(j, 0);
                                // Logical A [16][32] and B [32][16] from the lane registers.
                                float A[16][32], B[32][16];
                                for(index_t lane = 0; lane < 32; ++lane)
                                {
                                    const index_t r = lane % 16, h = lane / 16;
                                    for(index_t e = 0; e < 16; ++e)
                                    {
                                        const index_t kl = (e / K1) * (KL * K1) + h * K1 + e % K1;
                                        A[r][kl]         = elem(qa, lane, e);
                                        B[kl][r]         = elem(qb, lane, e);
                                    }
                                }
                                for(index_t lane = 0; lane < 32; ++lane)
                                {
                                    const index_t col = lane % 16, h = lane / 16;
                                    for(index_t v = 0; v < 8; ++v)
                                    {
                                        float d = 0.f;
                                        for(index_t kl = 0; kl < kK; ++kl)
                                            d += A[8 * h + v][kl] * B[kl][col];
                                        acc[((size_t(t) * kMT + mt) * 32 + lane) * 8 + v] += d;
                                    }
                                }
                            }
                        }
                    }
                }
                // C -> red: lane (tok, half) of (t, mt) holds features 16 t + 8 half + v.
                for(index_t t = 0; t < kNT; ++t)
                    for(index_t mt = 0; mt < kMT; ++mt)
                        for(index_t lane = 0; lane < 32; ++lane)
                            for(index_t v = 0; v < 8; ++v)
                            {
                                const index_t m = mt * 16 + lane % 16;
                                const index_t n = t * 16 + 8 * (lane / 16) + v;
                                red[(size_t(wave) * kMaxM + m) * kNR + n] =
                                    acc[((size_t(t) * kMT + mt) * 32 + lane) * 8 + v];
                            }
            }
            // Waves in order, into the split's partial.
            for(index_t m = 0; m < M; ++m)
                for(index_t n = 0; n < kNR; ++n)
                {
                    float v = 0.f;
                    for(index_t wave = 0; wave < kW; ++wave)
                        v += red[(size_t(wave) * kMaxM + m) * kNR + n];
                    partial[((size_t(s) * n_blocks + nb) * M + m) * kNR + n] = v;
                }
        }
        // Splits in order, then the epilogue.
        for(index_t m = 0; m < M; ++m)
            for(index_t n = 0; n < kNR; ++n)
            {
                float v = 0.f;
                for(index_t s = 0; s < k_splits; ++s)
                    v += partial[((size_t(s) * n_blocks + nb) * M + m) * kNR + n];
                if(epi.residual)
                    v += bf(epi.residual[size_t(m) * epi.ld + nb * kNR + n]);
                y[size_t(m) * N + nb * kNR + n] = v;
            }
    }
    return y;
}

/// Runs emulate() on small random problems (several M, split counts) and compares with the
/// double reference. Also cross-checks small_m::wave_windows against a direct enumeration.
template <typename Phase>
bool selftest(const char* name, uint64_t seed)
{
    bool ok = true;
    // Window distribution: every window of every split exactly once.
    for(index_t windows : {8, 17, 64, 130})
        for(index_t S : {1, 2, 3, 7})
        {
            if(S > windows)
                continue;
            std::vector<int> seen(windows, 0);
            for(index_t s = 0; s < S; ++s)
            {
                const index_t w0 = sm::split_begin(windows, S, s);
                const index_t w1 = sm::split_begin(windows, S, s + 1);
                for(index_t wave = 0; wave < Phase::kWaves; ++wave)
                    for(index_t i = 0; i < sm::wave_windows(w0, w1, wave, Phase::kWaves); ++i)
                    {
                        const index_t win = w0 + wave + i * Phase::kWaves;
                        if(win >= w1)
                            ok = false;
                        else
                            ++seen[win];
                    }
            }
            for(int c : seen)
                ok = ok && c == 1;
        }
    if(!ok)
        std::printf("%s: window distribution FAILED\n", name);

    const index_t N = 2 * Phase::kNRows;
    const index_t K = 17 * sm::kWindowK; // windows not a multiple of the wave count
    int bad_total   = 0;
    for(index_t M : {1, 8, Phase::kMaxM})
    {
        for(index_t S : {1, 3})
        {
            for(int use_res = 0; use_res < 2; ++use_res)
            {
                const auto x   = random_bf16(size_t(M) * K, 1.f, seed + 1);
                const auto w   = random_bf16(size_t(N) * K, 1.f / std::sqrt(float(K)), seed + 2);
                const auto res = random_bf16(size_t(M) * N, 1.f, seed + 3);
                HostEpi epi;
                if(use_res)
                {
                    epi.residual = res.data();
                    epi.ld       = N;
                }
                const auto yf = emulate<Phase>(x, w, M, N, K, S, epi);
                std::vector<uint16_t> y(yf.size());
                for(size_t i = 0; i < y.size(); ++i)
                    y[i] = sm::bf16_rne_bits(yf[i]);
                std::vector<index_t> rows(M);
                for(index_t r = 0; r < M; ++r)
                    rows[r] = r;
                const int bad = check_rows("Yemu", y, x, w, N, K, rows, epi);
                if(bad)
                    std::printf(
                        "%s: M %d splits %d residual %d: %d bad\n", name, M, S, use_res, bad);
                bad_total += bad;
            }
        }
    }
    ok = ok && bad_total == 0;
    std::printf("%s selftest (kMaxM %d, kNTiles %d, %d waves): %s\n",
                name,
                Phase::kMaxM,
                Phase::kNTiles,
                Phase::kWaves,
                ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace small_m_host
