// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host reference of the target model's fused epilogues, written from the HF module code
// (the model's modeling file) op by op: fp32 math, bf16 rounding wherever HF produces a bf16
// tensor. Self-contained (no ck_tile / HIP) so the host-only self-check builds with any C++17
// compiler. Encoders are written independently of the device ones.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mk_ref {

constexpr int kHeadDim          = 64;
constexpr float kRmsEps         = 1e-5f;
constexpr float kPostNormEps    = 1e-8f;
constexpr float kQkScale        = 5.4730064863875f;
constexpr float kRopeTheta      = 5e5f;
constexpr float kGateTemp       = 0.3f;
constexpr float kGateAlphaFloor = 1e-3f;

inline uint32_t f2u(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float u2f(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t bf16_bits(float f)
{
    const uint32_t u = f2u(f);
    if((u & 0x7FFFFFFFu) > 0x7F800000u)
        return static_cast<uint16_t>((u >> 16) | 0x40u); // quiet NaN
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_float(uint16_t b) { return u2f(static_cast<uint32_t>(b) << 16); }
inline float rb(float f) { return bf16_float(bf16_bits(f)); } // round to bf16

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// ---- encoders ------------------------------------------------------------------------------

/// e4m3fn value of a code (no inf; 0x7F / 0xFF are NaN).
inline float fp8_decode(uint8_t c)
{
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    if(e == 15 && m == 7)
        return NAN;
    const float v =
        e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}

/// fp32 -> e4m3fn, round to nearest even, saturate to +-448 (NaN stays NaN).
inline uint8_t fp8_encode(float x)
{
    if(std::isnan(x))
        return 0x7F;
    const uint8_t s = std::signbit(x) ? 0x80 : 0;
    const double a  = std::min(std::fabs(static_cast<double>(x)), 448.0);
    // Codes 0..126 are monotonic in value: pick the nearest, ties to the even code.
    int lo = 0, hi = 126;
    while(hi - lo > 1)
    {
        const int mid                                          = (lo + hi) / 2;
        (fp8_decode(static_cast<uint8_t>(mid)) <= a ? lo : hi) = mid;
    }
    const double dl = a - fp8_decode(static_cast<uint8_t>(lo));
    const double dh = fp8_decode(static_cast<uint8_t>(hi)) - a;
    const int c     = dl < dh ? lo : (dh < dl ? hi : (lo % 2 == 0 ? lo : hi));
    return static_cast<uint8_t>(s | c);
}

inline float e2m1_decode(uint8_t c)
{
    static const float kGrid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (c & 8) ? -kGrid[c & 7] : kGrid[c & 7];
}

/// fp32 -> e2m1 nibble, nearest on the grid, ties to the even code, saturate to +-6.
inline uint8_t e2m1_encode(float x)
{
    const float a = std::min(std::fabs(x), 6.f);
    int best      = 0;
    for(int c = 1; c < 8; ++c)
    {
        const float d = std::fabs(e2m1_decode(static_cast<uint8_t>(c)) - a);
        const float b = std::fabs(e2m1_decode(static_cast<uint8_t>(best)) - a);
        if(d < b || (d <= b && c % 2 == 0)) // tie: even code
            best = c;
    }
    return static_cast<uint8_t>(best | (std::signbit(x) ? 8 : 0));
}

/// e8m0 exponent of an MX block: floor(log2(amax rounded up at mantissa 1.75)) - emax_elem.
inline uint8_t mx_scale(float amax, int emax_elem)
{
    int e = 0;
    if(amax > 0.f)
    {
        const float fr = std::frexp(amax, &e); // amax = fr * 2^e, fr in [0.5, 1)
        e              = (fr >= 0.875f ? e : e - 1) + 127 - emax_elem;
    }
    return static_cast<uint8_t>(std::clamp(e, 0, 254));
}
inline float mx_value(uint8_t e8m0) { return std::ldexp(1.f, static_cast<int>(e8m0) - 127); }

// ---- ops -----------------------------------------------------------------------------------

/// HF rotary table row: cos / sin of pos * inv_freq[i], fp32 as torch computes them.
inline void rope_row(int pos, float (&c)[32], float (&s)[32])
{
    for(int i = 0; i < 32; ++i)
    {
        const float inv_freq = 1.f / std::pow(kRopeTheta, static_cast<float>(2 * i) / 64.f);
        const float f        = static_cast<float>(pos) * inv_freq;
        c[i]                 = std::cos(f);
        s[i]                 = std::sin(f);
    }
}

/// q / k head: qk_norm (bf16), [* qk_scale (bf16)], [RoPE with bf16 cos / sin, bf16 per op].
inline void qk_head(float* x, bool is_q, bool rope, int pos)
{
    float ss = 0.f;
    for(int d = 0; d < kHeadDim; ++d)
        ss += x[d] * x[d];
    const float r = std::pow(ss / kHeadDim + kRmsEps, -0.5f);
    for(int d = 0; d < kHeadDim; ++d)
    {
        x[d] = rb(x[d] * r);
        if(is_q)
            x[d] = rb(x[d] * kQkScale);
    }
    if(!rope)
        return;
    float c[32], s[32], y[kHeadDim];
    rope_row(pos, c, s);
    for(int d = 0; d < kHeadDim; ++d)
    {
        const float rot = d < 32 ? -x[d + 32] : x[d - 32];
        y[d]            = rb(rb(x[d] * rb(c[d % 32])) + rb(rot * rb(s[d % 32])));
    }
    std::copy(y, y + kHeadDim, x);
}

/// o_norm then * sigmoid(gate); o, gate are bf16 values.
inline void attn_gate_head(float* o, const float* gate)
{
    float ss = 0.f;
    for(int d = 0; d < kHeadDim; ++d)
        ss += o[d] * o[d];
    const float r = std::pow(ss / kHeadDim + kRmsEps, -0.5f);
    for(int d = 0; d < kHeadDim; ++d)
        o[d] = rb(rb(o[d] * r) * rb(sigmoid(gate[d])));
}

inline void residual_gate_coeff(float g, float& alpha, float& beta)
{
    const float w = g / kGateTemp;
    beta          = sigmoid(w);
    alpha         = std::sqrt(std::max(sigmoid(-w) * (1.f + beta), kGateAlphaFloor));
}

/// CenteredRMSNorm: bf16((x * rsqrt(mean(x^2) + eps)) * bf16(1 + w)).
inline void centered_rmsnorm(const float* x, const float* w, int n, float eps, float* out)
{
    float ss = 0.f;
    for(int i = 0; i < n; ++i)
        ss += x[i] * x[i];
    const float r = 1.f / std::sqrt(ss / n + eps);
    for(int i = 0; i < n; ++i)
        out[i] = rb((x[i] * r) * rb(1.f + w[i]));
}

/// Post norm + residual gate + next CenteredRMSNorm of one row (see row_norm_phase.hpp).
/// in_sum: the fp32 sum of the partials. input_scaled: post_feedforward (with first_w).
///
/// x_lo / x_hi (optional): the smallest / largest x over r in [r (1 - r_slack), r (1 + r_slack)],
/// r the post-norm rsqrt. The normed row is rounded to bf16 *before* the residual gate, so an
/// fp32-level difference in r (summation order over the row, rsqrt vs pow) moves an element that
/// sits on a bf16 rounding boundary by one bf16 ulp; x = alpha * residual + beta * an can cancel,
/// which turns that ulp of an into several ulps of x. A checker accepts any x in the range. x is
/// monotone in r (beta > 0), so the endpoints bound it.
inline void residual_norm_row(const float* in_sum,
                              bool input_scaled,
                              const float* first_w,
                              const float* gate,
                              const float* residual,
                              const float* next_w,
                              int n,
                              float* x_out,
                              float* h_out,
                              float r_slack = 0.f,
                              float* x_lo   = nullptr,
                              float* x_hi   = nullptr)
{
    std::vector<float> a(n);
    // The sum of squares in double and r rounded once: the reference for r is the exact value.
    // A serial fp32 sum over 8192 terms is off by ~1e-6 relative, enough to move elements on a
    // bf16 boundary of a (see x_lo / x_hi); HF's fp32 tree reduction and the GPU's are closer.
    double ss = 0.;
    for(int i = 0; i < n; ++i)
    {
        a[i] = rb(in_sum[i]);
        if(input_scaled)
            a[i] = rb(a[i] * rb(1.f + first_w[i]));
        ss += static_cast<double>(a[i]) * a[i];
    }
    const float r = static_cast<float>(1.0 / std::sqrt(ss / n + static_cast<double>(kPostNormEps)));
    for(int i = 0; i < n; ++i)
    {
        float alpha, beta;
        residual_gate_coeff(gate[i], alpha, beta);
        x_out[i] = rb(alpha * residual[i] + beta * rb(a[i] * r));
        if(x_lo != nullptr && x_hi != nullptr)
        {
            const float xa = rb(alpha * residual[i] + beta * rb(a[i] * (r * (1.f - r_slack))));
            const float xb = rb(alpha * residual[i] + beta * rb(a[i] * (r * (1.f + r_slack))));
            x_lo[i]        = std::min({xa, xb, x_out[i]});
            x_hi[i]        = std::max({xa, xb, x_out[i]});
        }
    }
    if(h_out)
        centered_rmsnorm(x_out, next_w, n, kRmsEps, h_out);
}

} // namespace mk_ref
