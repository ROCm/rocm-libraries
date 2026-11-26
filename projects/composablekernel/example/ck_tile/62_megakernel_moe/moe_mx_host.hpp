// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of example 62: MX numerics (e2m1, e4m3, e8m0) and an emulation of the MoE GEMV's
// lane-level fragment assembly against ck_tile's gfx125 WMMA layout model. Uses no HIP API, so
// the emulation can be built and run on a host without a GPU.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <cstring>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/moe/mx_expert_gemv.hpp"

namespace moe_host {

using index_t = ck_tile::index_t;

inline double e8m0_to_double(uint8_t s) { return std::ldexp(1.0, int(s) - 127); }

inline double e2m1_to_double(uint8_t nib)
{
    static constexpr double mag[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return (nib & 8 ? -1 : 1) * mag[nib & 7];
}

inline double e4m3_to_double(uint8_t b)
{
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if(e == 15 && m == 7)
        return NAN;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1 + m / 8.0, e - 7);
    return s ? -v : v;
}

/// OCP e4m3, round to nearest even, saturating to 448.
inline uint8_t double_to_e4m3(double x)
{
    const uint8_t sign = std::signbit(x) ? 0x80 : 0;
    const double a     = std::fabs(x);
    if(!(a == a))
        return 0x7F;
    int e;
    std::frexp(a, &e);                            // a = f * 2^e, f in [0.5, 1)
    int exp = std::max(e - 1, -6);                // unbiased exponent; -6 also covers subnormals
    long q  = std::lrint(std::ldexp(a, 3 - exp)); // mantissa units, ties to even
    if(q == 16)
    {
        q = 8;
        ++exp;
    }
    if(q < 8) // subnormal (exp == -6) or zero
        return sign | static_cast<uint8_t>(q);
    if(exp > 8 || (exp == 8 && q > 14))
        return sign | 0x7E;
    return sign | static_cast<uint8_t>(((exp + 7) << 3) | (q - 8));
}

/// MX fp8 quantisation of one row of K values, as the device requant does (float scale math).
inline void quantize_row_mxfp8(const float* x, index_t K, uint8_t* q, uint8_t* scale)
{
    for(index_t b = 0; b < K / 32; ++b)
    {
        float amax = 0;
        for(index_t i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(x[32 * b + i]));
        const uint32_t sbits = ck_tile::moe_mx::mx_scale_bits<ck_tile::fp8_t>(amax);
        const float inv      = ck_tile::bit_cast<float>((254u << 23) - sbits);
        scale[b]             = static_cast<uint8_t>(sbits >> 23);
        for(index_t i = 0; i < 32; ++i)
            q[32 * b + i] = double_to_e4m3(x[32 * b + i] * inv);
    }
}

/// A [rows, K] MXFP4 matrix: e2m1 row-major [rows, K/2] (element 2i in the low nibble) and e8m0
/// scales [rows, K/32].
struct Mxfp4Matrix
{
    std::vector<uint8_t> q, s;
    double at(index_t K, index_t r, index_t k) const
    {
        const uint8_t byte = q[size_t(r) * (K / 2) + k / 2];
        return e2m1_to_double(k % 2 ? byte >> 4 : byte & 15) *
               e8m0_to_double(s[size_t(r) * (K / 32) + k / 32]);
    }
};

/// Random weights with rms about 2^-7.5 * 2.9, i.e. about 1/sqrt(4096) for K = 4096.
inline Mxfp4Matrix random_mxfp4(size_t rows, index_t K, std::mt19937_64& rng)
{
    Mxfp4Matrix m;
    m.q.resize(rows * (K / 2));
    m.s.resize(rows * (K / 32));
    std::uniform_int_distribution<int> byte(0, 255), sc(119, 120);
    for(auto& b : m.q)
        b = static_cast<uint8_t>(byte(rng));
    for(auto& b : m.s)
        b = static_cast<uint8_t>(sc(rng));
    return m;
}

/// Emulates one wave of mx_gemv_wave<fp8_t, 1>: weight rows [0, 16) x token rows [0, 16) over K,
/// assembling every lane's operands with the device helpers and executing the WMMA by ck_tile's
/// gfx125 layout model. Returns the largest |emulated - direct| / (sum |w x| + tiny).
inline double
emulate_wave(const uint8_t* w, const uint8_t* ws, const uint8_t* x, const uint8_t* xs, index_t K)
{
    namespace mx     = ck_tile::moe_mx;
    double c[16][16] = {};
    for(index_t k = 0; k < K; k += mx::kWindowK)
    {
        for(index_t j = 0; j < 2; ++j)
        {
            // Each lane's operands, gathered with the device's offsets, decoded by the layout
            // model.
            double A[16][128] = {}, B[128][16] = {};
            for(index_t lane = 0; lane < 32; ++lane)
            {
                const index_t m = lane % 16, h = lane / 16;
                uint8_t a[32], b[64];
                for(index_t ch = 0; ch < 2; ++ch)
                    std::memcpy(a + 16 * ch,
                                w + size_t(m) * (K / 2) + k / 2 + mx::a_chunk_byte(j, ch, h),
                                16);
                for(index_t sl = 0; sl < 4; ++sl)
                    std::memcpy(
                        b + 16 * sl, x + size_t(m) * K + k + mx::b_chunk_byte(j, sl, h), 16);
                // Scales come from lanes 0-15 (lane select 0), which load the same bytes.
                uint32_t wlo, whi, xlo, xhi;
                std::memcpy(&wlo, ws + size_t(m) * (K / 32) + k / 32, 4);
                std::memcpy(&whi, ws + size_t(m) * (K / 32) + k / 32 + 4, 4);
                std::memcpy(&xlo, xs + size_t(m) * (K / 32) + k / 32, 4);
                std::memcpy(&xhi, xs + size_t(m) * (K / 32) + k / 32 + 4, 4);
                const uint32_t sa = mx::select_scales(wlo, whi, j);
                const uint32_t sb = mx::select_scales(xlo, xhi, j);
                for(index_t i = 0; i < 64; ++i) // e2m1 nibble i of the 8 dwords
                {
                    const index_t d = i / 8, kk = 64 * (d / 4) + 32 * h + 8 * (d % 4) + i % 8;
                    const uint8_t nib = (a[i / 2] >> (4 * (i % 2))) & 15;
                    A[m][kk] = e2m1_to_double(nib) * e8m0_to_double((sa >> (8 * (kk / 32))) & 255);
                }
                for(index_t i = 0; i < 64; ++i)
                {
                    const index_t kk = 32 * (i / 16) + 16 * h + i % 16;
                    B[kk][m] = e4m3_to_double(b[i]) * e8m0_to_double((sb >> (8 * (kk / 32))) & 255);
                }
            }
            for(index_t m = 0; m < 16; ++m)
                for(index_t n = 0; n < 16; ++n)
                    for(index_t kk = 0; kk < 128; ++kk)
                        c[m][n] += A[m][kk] * B[kk][n];
        }
    }
    double worst = 0;
    for(index_t m = 0; m < 16; ++m)
        for(index_t n = 0; n < 16; ++n)
        {
            double ref = 0, mag = 0;
            for(index_t k = 0; k < K; ++k)
            {
                const uint8_t byte = w[size_t(m) * (K / 2) + k / 2];
                const double wv    = e2m1_to_double(k % 2 ? byte >> 4 : byte & 15) *
                                  e8m0_to_double(ws[size_t(m) * (K / 32) + k / 32]);
                const double xv = e4m3_to_double(x[size_t(n) * K + k]) *
                                  e8m0_to_double(xs[size_t(n) * (K / 32) + k / 32]);
                ref += wv * xv;
                mag += std::fabs(wv * xv);
            }
            worst = std::max(worst, std::fabs(c[m][n] - ref) / (mag + 1e-30));
        }
    return worst;
}

/// Random 16 x K weights and tokens through emulate_wave; true when the fragment assembly matches.
inline bool selftest_fragments(index_t K, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    const Mxfp4Matrix wm = random_mxfp4(16, K, rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf(size_t(16) * K);
    for(auto& v : xf)
        v = nd(rng);
    std::vector<uint8_t> x(xf.size()), xs(size_t(16) * (K / 32));
    for(index_t r = 0; r < 16; ++r)
        quantize_row_mxfp8(&xf[size_t(r) * K], K, &x[size_t(r) * K], &xs[size_t(r) * (K / 32)]);
    const double err = emulate_wave(wm.q.data(), wm.s.data(), x.data(), xs.data(), K);
    std::printf(
        "fragment self-test (host emulation of the WMMA layout model), K %d: rel err %.3g\n",
        K,
        err);
    return err < 1e-12;
}

} // namespace moe_host
