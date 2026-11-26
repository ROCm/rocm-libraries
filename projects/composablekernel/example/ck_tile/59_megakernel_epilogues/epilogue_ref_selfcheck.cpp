// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only self-check of epilogue_reference.hpp (no GPU, no HIP): the encoders against their
// definitions, and the op-by-op bf16 reference against the spec formulas evaluated in double
// without intermediate rounding. Exit code 0 pass, 1 fail.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "epilogue_reference.hpp"

namespace {

int g_fail = 0;

void expect(bool ok, const char* what, double got = 0, double bound = 0)
{
    std::printf("  %-58s %s (%.3g, bound %.3g)\n", what, ok ? "ok" : "FAIL", got, bound);
    g_fail += !ok;
}

void check_encoders()
{
    using namespace mk_ref;
    std::printf("encoders\n");
    int bad = 0;
    for(int c = 0; c < 256; ++c)
    {
        const float v = fp8_decode(static_cast<uint8_t>(c));
        if(!std::isnan(v) && !(v == 0.f && c == 0x80) && fp8_encode(v) != c)
            ++bad;
    }
    expect(bad == 0, "fp8 e4m3fn: decode -> encode is the identity", bad, 0);
    // Midpoints between neighbours go to the even code; beyond 448 saturates.
    bad = 0;
    for(int c = 0; c < 126; ++c)
    {
        const double mid = 0.5 * (fp8_decode(c) + fp8_decode(c + 1));
        const int want   = (c % 2 == 0) ? c : c + 1;
        if(fp8_encode(static_cast<float>(mid)) != want)
            ++bad;
    }
    expect(bad == 0 && fp8_encode(1e6f) == 0x7E && fp8_encode(-500.f) == 0xFE,
           "fp8: ties to even, saturation to 448",
           bad,
           0);
    bad = 0;
    for(int c = 0; c < 16; ++c)
        if(c != 8 && e2m1_encode(e2m1_decode(static_cast<uint8_t>(c))) != c)
            ++bad;
    const float ties[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
    const int tie_ok[7] = {0, 2, 2, 4, 4, 6, 6};
    for(int i = 0; i < 7; ++i)
        bad += e2m1_encode(ties[i]) != tie_ok[i];
    bad += e2m1_encode(100.f) != 7 || e2m1_encode(-6.5f) != 15;
    expect(bad == 0, "e2m1: identity, ties to even, saturation to 6", bad, 0);

    // MX scale: the scaled block max lands in [elem_max / 2, elem_max] (fp8) or [., 7) (fp4).
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ex(-30.f, 30.f);
    double worst8 = 0, worst4 = 0, low8 = 1e9, low4 = 1e9;
    for(int i = 0; i < 100000; ++i)
    {
        const float amax = std::exp2(ex(rng));
        const double s8  = amax / mx_value(mx_scale(amax, 8));
        const double s4  = amax / mx_value(mx_scale(amax, 2));
        worst8           = std::max(worst8, s8);
        low8             = std::min(low8, s8);
        worst4           = std::max(worst4, s4);
        low4             = std::min(low4, s4);
    }
    expect(worst8 <= 448 && low8 >= 224, "MXFP8 scale: amax / scale in [224, 448]", worst8, 448);
    expect(worst4 < 7 && low4 >= 3.5, "MXFP4 scale: amax / scale in [3.5, 7)", worst4, 7);
}

// Spec formulas in double, no intermediate rounding.
void spec_qk_head(const float* x, bool is_q, bool rope, int pos, double* out)
{
    double ss = 0;
    for(int d = 0; d < 64; ++d)
        ss += double(x[d]) * x[d];
    const double r = 1 / std::sqrt(ss / 64 + mk_ref::kRmsEps);
    double y[64];
    for(int d = 0; d < 64; ++d)
        y[d] = x[d] * r * (is_q ? mk_ref::kQkScale : 1.0);
    for(int d = 0; d < 64; ++d)
    {
        if(!rope)
        {
            out[d] = y[d];
            continue;
        }
        const double f   = pos * std::pow(5e5, -(2.0 * (d % 32)) / 64);
        const double rot = d < 32 ? -y[d + 32] : y[d - 32];
        out[d]           = y[d] * std::cos(f) + rot * std::sin(f);
    }
}

void check_heads()
{
    std::printf("q/k heads, attention gate\n");
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    double err_q = 0, err_k = 0, err_g = 0, max_score = 0;
    for(int t = 0; t < 2000; ++t)
    {
        float q[64], k[64], o[64], g[64];
        for(int d = 0; d < 64; ++d)
        {
            q[d] = mk_ref::rb(nd(rng) * 3.f);
            k[d] = mk_ref::rb(nd(rng) * 0.5f);
            o[d] = mk_ref::rb(nd(rng) * 0.1f);
            g[d] = mk_ref::rb(nd(rng) * 2.f);
        }
        const int pos = static_cast<int>(rng() % 200000);
        double qs[64], ks[64];
        spec_qk_head(q, true, true, pos, qs);
        spec_qk_head(k, false, true, pos, ks);
        float qr[64], kr[64];
        std::copy(q, q + 64, qr);
        std::copy(k, k + 64, kr);
        mk_ref::qk_head(qr, true, true, pos);
        mk_ref::qk_head(kr, false, true, pos);
        double score = 0;
        for(int d = 0; d < 64; ++d)
        {
            // Errors in units of the head's RMS (qk_scale for q, 1 for k).
            err_q = std::max(err_q, std::fabs(qr[d] - qs[d]) / mk_ref::kQkScale);
            err_k = std::max(err_k, std::fabs(kr[d] - ks[d]));
            score += double(qr[d]) * kr[d];
        }
        max_score = std::max(max_score, std::fabs(score) / 8);

        double ss = 0;
        for(int d = 0; d < 64; ++d)
            ss += double(o[d]) * o[d];
        const double r = 1 / std::sqrt(ss / 64 + mk_ref::kRmsEps);
        float og[64];
        std::copy(o, o + 64, og);
        mk_ref::attn_gate_head(og, g);
        for(int d = 0; d < 64; ++d)
        {
            const double want = o[d] * r / (1 + std::exp(-double(g[d])));
            err_g             = std::max(err_g, std::fabs(og[d] - want));
        }
    }
    // HF rounds to bf16 after every op (norm, scale, cos/sin, products, sum): a few bf16 ulps
    // of the head RMS (2^-8 relative per rounding).
    expect(err_q < 0.05, "q head (norm, scale, RoPE) vs spec, |err| / qk_scale", err_q, 0.05);
    expect(err_k < 0.05, "k head (norm, RoPE) vs spec", err_k, 0.05);
    expect(err_g < 0.03, "o_norm * sigmoid(gate) vs spec", err_g, 0.03);
    expect(max_score <= 5.4730064863875 * 64 / 8 * 1.01,
           "|q.k| / 8 within the spec bound 43.8",
           max_score,
           43.8);
}

void check_rows()
{
    std::printf("row norms, residual gate\n");
    float alpha, beta;
    mk_ref::residual_gate_coeff(0.f, alpha, beta);
    expect(std::fabs(beta - 0.5f) < 1e-7f && std::fabs(alpha - std::sqrt(0.75f)) < 1e-7f,
           "residual gate at g = 0: beta 0.5, alpha sqrt(0.75)",
           alpha,
           std::sqrt(0.75));
    mk_ref::residual_gate_coeff(-10.f, alpha, beta); // sigmoid(-w) (1 + beta) -> 1
    expect(std::fabs(alpha - 1.f) < 1e-4f, "residual gate at g = -10: alpha -> 1", alpha, 1);
    mk_ref::residual_gate_coeff(10.f, alpha, beta); // floor at 1e-3
    expect(std::fabs(alpha - std::sqrt(1e-3f)) < 1e-6f,
           "residual gate at g = 10: alpha = sqrt(1e-3) (eps floor)",
           alpha,
           std::sqrt(1e-3));

    const int n = 8192;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> in(n), w1(n), g(n), res(n), w2(n), x(n), h(n);
    double err_x = 0, err_h = 0;
    for(int t = 0; t < 8; ++t)
    {
        const bool scaled = t % 2;
        for(int i = 0; i < n; ++i)
        {
            in[i]  = nd(rng) * 7.f;
            w1[i]  = mk_ref::rb(nd(rng) * 0.1f);
            g[i]   = nd(rng) * 0.5f;
            res[i] = mk_ref::rb(nd(rng) * 3.f);
            w2[i]  = mk_ref::rb(nd(rng) * 0.1f);
        }
        mk_ref::residual_norm_row(
            in.data(), scaled, w1.data(), g.data(), res.data(), w2.data(), n, x.data(), h.data());
        std::vector<double> xs(n);
        double ss = 0;
        for(int i = 0; i < n; ++i)
        {
            const double y = scaled ? in[i] * (1.0 + w1[i]) : in[i];
            ss += y * y;
        }
        const double r = 1 / std::sqrt(ss / n + 1e-8);
        double ss2     = 0;
        for(int i = 0; i < n; ++i)
        {
            const double y      = scaled ? in[i] * (1.0 + w1[i]) : in[i];
            const double w      = g[i] / 0.3;
            const double beta_d = 1 / (1 + std::exp(-w));
            const double a      = std::sqrt(std::max(1 / (1 + std::exp(w)) * (1 + beta_d), 1e-3));
            xs[i]               = a * res[i] + beta_d * y * r;
            ss2 += xs[i] * xs[i];
            err_x = std::max(err_x, std::fabs(x[i] - xs[i]) / (std::fabs(xs[i]) + 1));
        }
        const double r2 = 1 / std::sqrt(ss2 / n + 1e-5);
        for(int i = 0; i < n; ++i)
            err_h = std::max(err_h, std::fabs(h[i] - xs[i] * r2 * (1 + w2[i])));
    }
    expect(err_x < 0.03, "post norm + residual gate vs spec, |err| / (|x| + 1)", err_x, 0.03);
    expect(err_h < 0.05, "next CenteredRMSNorm vs spec", err_h, 0.05);
}

} // namespace

int main()
{
    check_encoders();
    check_heads();
    check_rows();
    std::printf("%s: %d failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
