// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host reference of the target model's head (no HIP calls, so it can be tested on the CPU alone).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "ck_tile/ops/megakernel/head/head_common.hpp"

namespace head_ref {

inline float bf16_to_float(uint16_t b)
{
    return __builtin_bit_cast(float, static_cast<uint32_t>(b) << 16);
}
inline uint16_t float_to_bf16(float f)
{
    return static_cast<uint16_t>(__builtin_bit_cast(uint32_t, ck_tile::head_round_bf16(f)) >> 16);
}

/// xn = bf16(x * (mean(x^2) + eps)^-0.5 * w), in double.
inline std::vector<uint16_t>
rmsnorm(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, int M, int H, double eps)
{
    std::vector<uint16_t> y(size_t(M) * H);
    for(int r = 0; r < M; ++r)
    {
        double ss = 0;
        for(int k = 0; k < H; ++k)
        {
            const double v = bf16_to_float(x[size_t(r) * H + k]);
            ss += v * v;
        }
        const double rstd = 1.0 / std::sqrt(ss / H + eps);
        for(int k = 0; k < H; ++k)
            y[size_t(r) * H + k] = float_to_bf16(static_cast<float>(
                bf16_to_float(x[size_t(r) * H + k]) * rstd * bf16_to_float(w[k])));
    }
    return y;
}

/// Raw logits of one row in double, with sum |a * b| as the error scale of an fp32 GPU sum.
struct RowLogits
{
    std::vector<double> acc;
    std::vector<double> mag;
};

inline std::vector<RowLogits> logits(const std::vector<uint16_t>& xn,
                                     const std::vector<uint16_t>& wt,
                                     const std::vector<int>& rows,
                                     int N,
                                     int K)
{
    std::vector<RowLogits> out(rows.size());
    for(auto& o : out)
    {
        o.acc.assign(N, 0.0);
        o.mag.assign(N, 0.0);
    }
    std::vector<std::vector<double>> a(rows.size(), std::vector<double>(K));
    for(size_t i = 0; i < rows.size(); ++i)
        for(int k = 0; k < K; ++k)
            a[i][k] = bf16_to_float(xn[size_t(rows[i]) * K + k]);

    const unsigned nt = std::max(1u, std::min(64u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for(unsigned t = 0; t < nt; ++t)
    {
        th.emplace_back([&, t]() {
            std::vector<double> b(K);
            for(int n = static_cast<int>(t); n < N; n += static_cast<int>(nt))
            {
                for(int k = 0; k < K; ++k)
                    b[k] = bf16_to_float(wt[size_t(n) * K + k]);
                for(size_t i = 0; i < rows.size(); ++i)
                {
                    double s = 0, m = 0;
                    for(int k = 0; k < K; ++k)
                    {
                        const double p = a[i][k] * b[k];
                        s += p;
                        m += std::fabs(p);
                    }
                    out[i].acc[n] = s;
                    out[i].mag[n] = m;
                }
            }
        });
    }
    for(auto& t : th)
        t.join();
    return out;
}

/// Softcap of an fp32 sum known to lie in [acc - err, acc + err]: the bounds of the GPU value.
/// The softcap (with or without bf16 rounding) is monotone, so the bounds map to bounds. In fp32
/// the host and device tanhf may differ by an ulp or two, so the bounds widen by 4 ulp of the cap;
/// with bf16 rounding they do not (a tanhf ulp flips a bf16 rounding only at a rounding boundary).
template <ck_tile::head_softcap_numerics Numerics>
struct CappedBounds
{
    double lo, hi;
    CappedBounds(double acc, double err, float mult, float cap)
        : lo(ck_tile::head_softcap<Numerics>(static_cast<float>(acc - err), mult, cap)),
          hi(ck_tile::head_softcap<Numerics>(static_cast<float>(acc + err), mult, cap))
    {
        if constexpr(Numerics == ck_tile::head_softcap_numerics::fp32)
        {
            lo -= std::ldexp(static_cast<double>(cap), -21);
            hi += std::ldexp(static_cast<double>(cap), -21);
        }
    }
};

/// Relative error of the GPU's fp32 dot product, against sum |a * b|.
inline double fp32_dot_err(double mag) { return mag * std::ldexp(1.0, -16) + 1e-30; }

struct ArgmaxCheck
{
    bool ok;
    int exact_ties; // tokens other than the expected one with the same reference sum (duplicates)
    int ref_token;  // argmax of the reference centre values, lowest id on ties
};

/// Checks a GPU argmax token (shard-local index @p t) of one row. The GPU value of token j lies in
/// [lo_j, hi_j]. t is wrong if some j is surely larger (lo_j > hi_t), or surely equal or larger
/// and of lower id (lo_j >= hi_t with j < t), or exactly equal and of lower id (same double sum:
/// duplicated lm_head rows, which the GPU sums identically).
template <ck_tile::head_softcap_numerics Numerics>
ArgmaxCheck check_argmax(const RowLogits& r, int t, float mult, float cap)
{
    const int N = static_cast<int>(r.acc.size());
    ArgmaxCheck c{true, 0, 0};
    if(t < 0 || t >= N)
    {
        c.ok = false;
        return c;
    }
    const CappedBounds<Numerics> bt(r.acc[t], fp32_dot_err(r.mag[t]), mult, cap);
    double best = -1e300;
    for(int j = 0; j < N; ++j)
    {
        const double centre =
            ck_tile::head_softcap<Numerics>(static_cast<float>(r.acc[j]), mult, cap);
        if(centre > best)
        {
            best        = centre;
            c.ref_token = j;
        }
        if(j == t)
            continue;
        if(!(r.acc[j] < r.acc[t]) && !(r.acc[t] < r.acc[j]))
        {
            ++c.exact_ties;
            if(j < t)
                c.ok = false;
            continue;
        }
        const CappedBounds<Numerics> bj(r.acc[j], fp32_dot_err(r.mag[j]), mult, cap);
        if(bj.lo > bt.hi || (j < t && bj.lo >= bt.hi))
            c.ok = false;
    }
    return c;
}

struct Accepted
{
    std::vector<int32_t> tokens; // [seqs, window]
    std::vector<int32_t> count;  // [seqs]
};

/// DFlash2 accept of all windows from the argmax tokens.
inline Accepted
accept(const std::vector<int32_t>& tokens, const std::vector<int32_t>& drafts, int window)
{
    const int seqs = static_cast<int>(tokens.size()) / window;
    Accepted a{std::vector<int32_t>(size_t(seqs) * window, -1), std::vector<int32_t>(seqs, 0)};
    for(int s = 0; s < seqs; ++s)
        a.count[s] = ck_tile::head_dflash_accept(tokens.data() + size_t(s) * window,
                                                 drafts.data() + size_t(s) * (window - 1),
                                                 window,
                                                 a.tokens.data() + size_t(s) * window);
    return a;
}

} // namespace head_ref
