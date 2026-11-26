// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of example 60: inputs in the phase's layouts, a double reference, and a float
// emulation of the phase's split / mask / combine decomposition (it calls the phase's own
// Locate / IsMasked / ChooseNumSplits). No HIP calls, so it also builds host-only.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ck_tile/core.hpp"

namespace attn_ref {

using index_t = ck_tile::index_t;

constexpr index_t kHeadDim = 64;
constexpr index_t kGroup   = 8;
constexpr double kQkScale  = 5.4730064863875; // qk_scale_factor
constexpr double kScale    = 0.125;           // head_dim^-0.5

inline uint16_t bf16_bits(float f)
{
    return ck_tile::bit_cast<uint16_t>(
        ck_tile::float_to_bf16<ck_tile::bf16_rounding_mode::standard>(f));
}
inline float bf16_val(uint16_t b) { return ck_tile::bit_cast<float>(uint32_t(b) << 16); }
inline float round_bf16(float f) { return bf16_val(bf16_bits(f)); }
inline uint8_t fp8_bits(float f)
{
    return ck_tile::bit_cast<uint8_t>(ck_tile::type_convert<ck_tile::fp8_t>(f));
}
inline float fp8_val(uint8_t b)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::fp8_t>(b));
}

struct Config
{
    index_t num_requests = 1;
    index_t q_len        = 16;
    index_t ctx          = 4096; // kv_len of request 0; request b has ctx - 37 * b
    index_t window       = 2048; // 0: global
    index_t q_heads      = 16;
    index_t kv_heads     = 2;
    index_t page_size    = 16;
    bool fp8             = false;
    uint64_t seed        = 7;

    index_t M() const { return num_requests * q_len; }
    index_t kv_len(index_t b) const { return ctx - 37 * b; }
};

/// Inputs in the device layouts plus the dequantised K / V the reference reads.
struct Data
{
    using long_index_t = ck_tile::long_index_t;

    Config cfg;
    index_t stride_q = 0, stride_gate = 0;
    std::vector<uint16_t> q, gate; // [M][q_heads * 64] bf16
    index_t pages_per_req = 0, num_pages = 0;
    std::vector<int32_t> block_table, kv_lens;
    std::vector<uint8_t> k_cache, v_cache; // [page][kv_head][slot][64], 2 or 1 bytes / element
    long_index_t page_stride = 0;          // elements
    index_t head_stride = 0, token_stride = kHeadDim;
    float k_scale = 1.f, v_scale = 1.f;
    std::vector<float> k, v; // logical [b][pos][kv_head][64], dequantised

    index_t elem_bytes() const { return cfg.fp8 ? 1 : 2; }
    size_t kv_index(index_t b, index_t pos, index_t g) const
    {
        return ((size_t(b) * cfg.ctx + pos) * cfg.kv_heads + g) * kHeadDim;
    }
    long_index_t cache_offset(index_t b, index_t pos, index_t g) const
    {
        const index_t page = block_table[size_t(b) * pages_per_req + pos / cfg.page_size];
        return page * page_stride + g * head_stride + (pos % cfg.page_size) * token_stride;
    }
};

// RMSNorm over each 64-wide head (no weight), times scale, bf16.
inline void norm_heads(std::vector<float>& x, double scale)
{
    for(size_t h0 = 0; h0 < x.size(); h0 += kHeadDim)
    {
        double ss = 0;
        for(index_t d = 0; d < kHeadDim; ++d)
            ss += double(x[h0 + d]) * x[h0 + d];
        const double r = 1.0 / std::sqrt(ss / kHeadDim + 1e-5);
        for(index_t d = 0; d < kHeadDim; ++d)
            x[h0 + d] = round_bf16(float(round_bf16(float(x[h0 + d] * r)) * scale));
    }
}

inline Data make_data(const Config& cfg)
{
    Data D;
    D.cfg = cfg;
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<float> nd(0.f, 1.f);

    const index_t M = cfg.M();
    D.stride_q      = cfg.q_heads * kHeadDim;
    D.stride_gate   = cfg.q_heads * kHeadDim;
    std::vector<float> qf(size_t(M) * D.stride_q);
    for(auto& x : qf)
        x = nd(rng);
    norm_heads(qf, kQkScale);
    D.q.resize(qf.size());
    for(size_t i = 0; i < qf.size(); ++i)
        D.q[i] = bf16_bits(qf[i]);
    D.gate.resize(size_t(M) * D.stride_gate);
    for(auto& g : D.gate)
        g = bf16_bits(2.f * nd(rng));

    // Logical K (normed) and V, bf16.
    D.k.resize(size_t(cfg.num_requests) * cfg.ctx * cfg.kv_heads * kHeadDim);
    D.v.resize(D.k.size());
    for(auto& x : D.k)
        x = nd(rng);
    norm_heads(D.k, 1.0);
    for(auto& x : D.v)
        x = round_bf16(nd(rng));

    // Pages: a random permutation over all requests, plus spare pages nobody owns.
    D.pages_per_req = ck_tile::integer_divide_ceil(cfg.ctx, cfg.page_size);
    D.num_pages     = cfg.num_requests * D.pages_per_req + 3;
    std::vector<int32_t> perm(D.num_pages);
    for(index_t i = 0; i < D.num_pages; ++i)
        perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    D.block_table.assign(perm.begin(), perm.begin() + cfg.num_requests * D.pages_per_req);
    D.kv_lens.resize(cfg.num_requests);
    for(index_t b = 0; b < cfg.num_requests; ++b)
        D.kv_lens[b] = cfg.kv_len(b);

    D.head_stride = cfg.page_size * kHeadDim;
    D.page_stride = ck_tile::long_index_t(cfg.kv_heads) * D.head_stride;

    if(cfg.fp8)
    {
        float kmax = 0, vmax = 0;
        for(float x : D.k)
            kmax = std::max(kmax, std::fabs(x));
        for(float x : D.v)
            vmax = std::max(vmax, std::fabs(x));
        D.k_scale = kmax / 448.f;
        D.v_scale = vmax / 448.f;
    }

    // Unwritten slots hold NaN (0xFF bytes: bf16 and e4m3fn NaN) to catch unmasked reads.
    const size_t cache_elems = size_t(D.num_pages) * D.page_stride;
    D.k_cache.assign(cache_elems * D.elem_bytes(), 0xFF);
    D.v_cache.assign(cache_elems * D.elem_bytes(), 0xFF);
    for(index_t b = 0; b < cfg.num_requests; ++b)
        for(index_t pos = 0; pos < cfg.kv_len(b); ++pos)
            for(index_t g = 0; g < cfg.kv_heads; ++g)
            {
                const size_t li = D.kv_index(b, pos, g);
                const auto off  = D.cache_offset(b, pos, g);
                for(index_t d = 0; d < kHeadDim; ++d)
                {
                    float& kk = D.k[li + d];
                    float& vv = D.v[li + d];
                    if(cfg.fp8)
                    {
                        const uint8_t kb   = fp8_bits(kk / D.k_scale);
                        const uint8_t vb   = fp8_bits(vv / D.v_scale);
                        D.k_cache[off + d] = kb;
                        D.v_cache[off + d] = vb;
                        kk                 = fp8_val(kb) * D.k_scale;
                        vv                 = fp8_val(vb) * D.v_scale;
                    }
                    else
                    {
                        const uint16_t kb = bf16_bits(kk), vb = bf16_bits(vv);
                        std::memcpy(&D.k_cache[(off + d) * 2], &kb, 2);
                        std::memcpy(&D.v_cache[(off + d) * 2], &vb, 2);
                    }
                }
            }
    return D;
}

/// Double reference: out[token][q_head][64] = softmax(q k^T / 8, causal [+ window]) v.
inline std::vector<double> reference(const Data& D)
{
    const Config& c = D.cfg;
    std::vector<double> out(size_t(c.M()) * c.q_heads * kHeadDim, 0.0);
    std::vector<double> s;
    for(index_t b = 0; b < c.num_requests; ++b)
        for(index_t i = 0; i < c.q_len; ++i)
        {
            const index_t token = b * c.q_len + i;
            const index_t p     = c.kv_len(b) - c.q_len + i;
            const index_t lo    = c.window > 0 ? std::max<index_t>(0, p - c.window + 1) : 0;
            for(index_t h = 0; h < c.q_heads; ++h)
            {
                const index_t g   = h / kGroup;
                const uint16_t* q = &D.q[size_t(token) * D.stride_q + h * kHeadDim];
                s.assign(p + 1 - lo, 0.0);
                double mx = -1e300;
                for(index_t j = lo; j <= p; ++j)
                {
                    const float* k = &D.k[D.kv_index(b, j, g)];
                    double acc     = 0;
                    for(index_t d = 0; d < kHeadDim; ++d)
                        acc += double(bf16_val(q[d])) * k[d];
                    s[j - lo] = acc * kScale;
                    mx        = std::max(mx, s[j - lo]);
                }
                double sum = 0;
                for(auto& x : s)
                    sum += (x = std::exp(x - mx));
                double* o = &out[(size_t(token) * c.q_heads + h) * kHeadDim];
                for(index_t j = lo; j <= p; ++j)
                {
                    const float* v = &D.v[D.kv_index(b, j, g)];
                    for(index_t d = 0; d < kHeadDim; ++d)
                        o[d] += s[j - lo] / sum * v[d];
                }
            }
        }
    return out;
}

/// The epilogue the NormGate functor applies, on a reference row.
inline void norm_gate(double* o, const uint16_t* gate)
{
    float x[kHeadDim];
    float ss = 0;
    for(index_t d = 0; d < kHeadDim; ++d)
    {
        x[d] = round_bf16(float(o[d]));
        ss += x[d] * x[d];
    }
    const float r = 1.f / std::sqrt(ss / kHeadDim + 1e-5f);
    for(index_t d = 0; d < kHeadDim; ++d)
    {
        const float g  = bf16_val(gate[d]);
        const float sg = round_bf16(1.f / (1.f + std::exp(-g)));
        o[d]           = round_bf16(round_bf16(x[d] * r) * sg);
    }
}

/// Float emulation of Phase's decomposition: every work item's split range and mask from the
/// phase's own Locate / IsMasked, the no-max (or running-max) exp2 softmax, lse partials and
/// the last-split combine. Also checks every visible (row, key) is visited exactly once.
/// Returns out[token][q_head][64] (before the output functor), or empty on a coverage error.
template <typename Phase>
std::vector<double> emulate(const Data& D, const typename Phase::Args& a)
{
    const Config& c       = D.cfg;
    constexpr index_t kM0 = Phase::kM0, kN0 = Phase::kN0, R = Phase::kRowsPerHead;
    const index_t S      = a.num_splits;
    const float scale_l2 = a.scale * float(M_LOG2E);
    std::vector<double> out(size_t(c.M()) * c.q_heads * kHeadDim, 0.0);
    std::vector<float> o_part(size_t(S) * kM0 * kHeadDim), lse_part(size_t(S) * kM0);
    long bad_cover = 0;

    for(index_t tile = 0; tile < Phase::NumTiles(a); ++tile)
    {
        typename Phase::Item it{};
        std::vector<int> visits;
        for(index_t split = 0; split < S; ++split)
        {
            const index_t w = tile * S + split;
            const index_t b = w / S / a.num_kv_heads / Phase::QTiles(a);
            it              = Phase::Locate(a, w, c.kv_len(b));
            if(visits.empty())
                visits.assign(size_t(kM0) * c.kv_len(it.b), 0);
            for(index_t r = 0; r < kM0; ++r)
            {
                const index_t h = r / R, i = r % R;
                const index_t token = it.b * c.q_len + it.row0 + std::min(i, it.rows - 1);
                const uint16_t* q =
                    &D.q[size_t(token) * D.stride_q + (it.g * kGroup + h) * kHeadDim];
                float l = 0, m = -INFINITY;
                std::vector<float> o(kHeadDim, 0.f);
                for(index_t blk = it.kv_block_begin; blk < it.kv_block_end; ++blk)
                    for(index_t jj = 0; jj < kN0; ++jj)
                    {
                        const index_t j = blk * kN0 + jj;
                        if(j >= c.kv_len(it.b) || Phase::IsMasked(a, it, r, j))
                            continue;
                        if(i < it.rows)
                            ++visits[size_t(r) * c.kv_len(it.b) + j];
                        const float* k = &D.k[D.kv_index(it.b, j, it.g)];
                        float sc       = 0;
                        for(index_t d = 0; d < kHeadDim; ++d)
                            sc += bf16_val(q[d]) * k[d];
                        float pr;
                        if constexpr(Phase::kRunningMax)
                        {
                            const float mn   = std::max(m, sc);
                            const float corr = std::exp2(scale_l2 * (m - mn));
                            l *= corr;
                            for(auto& x : o)
                                x *= corr;
                            m  = mn;
                            pr = std::exp2(scale_l2 * (sc - m));
                        }
                        else
                        {
                            pr = std::exp2(scale_l2 * sc);
                        }
                        l += pr;
                        const float* v = &D.v[D.kv_index(it.b, j, it.g)];
                        for(index_t d = 0; d < kHeadDim; ++d)
                            o[d] += pr * v[d];
                    }
                const float m_v = Phase::kRunningMax && l > 0 ? scale_l2 * m : 0.f;
                lse_part[size_t(split) * kM0 + r] = l > 0 ? m_v + std::log2(l) : -INFINITY;
                for(index_t d = 0; d < kHeadDim; ++d)
                    o_part[(size_t(split) * kM0 + r) * kHeadDim + d] = l > 0 ? o[d] / l : 0.f;
            }
        }
        // Coverage: each visible (row, key) exactly once over the splits.
        for(index_t r = 0; r < kM0; ++r)
        {
            if(r % R >= it.rows)
                continue;
            const index_t p  = it.pos0 + r % R;
            const index_t lo = c.window > 0 ? std::max<index_t>(0, p - c.window + 1) : 0;
            for(index_t j = 0; j < c.kv_len(it.b); ++j)
                bad_cover +=
                    visits[size_t(r) * c.kv_len(it.b) + j] != ((j >= lo && j <= p) ? 1 : 0);
        }
        // Combine, as Phase::Combine.
        for(index_t r = 0; r < kM0; ++r)
        {
            if(r % R >= it.rows)
                continue;
            float mx = -INFINITY;
            for(index_t s = 0; s < S; ++s)
                mx = std::max(mx, lse_part[size_t(s) * kM0 + r]);
            double sum = 0;
            std::vector<double> o(kHeadDim, 0.0);
            for(index_t s = 0; s < S; ++s)
            {
                const float ls = lse_part[size_t(s) * kM0 + r];
                if(std::isinf(ls))
                    continue;
                const double w = std::exp2(double(ls) - mx);
                sum += w;
                for(index_t d = 0; d < kHeadDim; ++d)
                    o[d] += w * o_part[(size_t(s) * kM0 + r) * kHeadDim + d];
            }
            const index_t token = it.b * c.q_len + it.row0 + r % R;
            const index_t head  = it.g * kGroup + r / R;
            for(index_t d = 0; d < kHeadDim; ++d)
                out[(size_t(token) * c.q_heads + head) * kHeadDim + d] = o[d] / sum;
        }
    }
    if(bad_cover != 0)
    {
        std::printf("  emulation: %ld (row, key) pairs visited != once\n", bad_cover);
        return {};
    }
    return out;
}

/// Count of |got - ref| > atol + rtol * |ref|; prints the first few.
inline long compare(const char* name,
                    const std::vector<double>& got,
                    const std::vector<double>& ref,
                    double atol,
                    double rtol)
{
    long bad       = 0;
    double max_err = 0;
    for(size_t i = 0; i < ref.size(); ++i)
    {
        const double e = std::fabs(got[i] - ref[i]);
        max_err        = std::max(max_err, std::isnan(e) ? INFINITY : e);
        if(!(e <= atol + rtol * std::fabs(ref[i])))
        {
            if(bad < 5)
                std::printf("  %s[%zu] = %g, ref %g\n", name, i, got[i], ref[i]);
            ++bad;
        }
    }
    std::printf(
        "  %s: max |err| %.3g, %ld / %zu out of tolerance\n", name, max_err, bad, ref.size());
    return bad;
}

} // namespace attn_ref
