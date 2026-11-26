// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host reference and checks of example 67 (one decoder layer of the target model per megakernel
// launch). Two kinds of reference:
//
//  * stage-local (-check_stages=1): every stage against a reference fed with the GPU's own input
//    to that stage, with the tolerance of the example that verified the phase on its own
//    (58-66). A failure points at one stage.
//  * end to end (always): the whole layer on the host from x and the weights, fp32 / double math
//    with the model's bf16 / fp8 / MXFP8 rounding points (the same quantised weights and the
//    activation quantisation the kernel uses). Only the router's top-8 is taken from the GPU,
//    and only on rows where the two sets differ by a near-tie (choice within 1e-5 of the k-th).
//
// The per-op host math is reused from the phase examples: epilogue_reference.hpp (59: norms,
// QK-norm / RoPE, attention gate, residual gate), router_reference.hpp (61: logits, top-k,
// route table), moe_mx_host.hpp (62: MX decode / requant) and 62's WmmaErr bound (copied below).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/epilogue/epilogue_common.hpp"
#include "ck_tile/ops/megakernel/moe/mx_expert_gemv.hpp"

#include "decoder_layer_common.hpp"
#include "../59_megakernel_epilogues/epilogue_reference.hpp"
#include "../61_megakernel_router/router_reference.hpp"
#include "../62_megakernel_moe/moe_mx_host.hpp"

namespace layer67 {

namespace {

using ck_tile::index_t;

constexpr double kOutMul = 0.17677669529663687; // router output_multiplier

/// Element checker: |got - ref| <= tol (per element); tracks the worst error and error / tol.
struct Check
{
    const char* name;
    long n = 0, bad = 0;
    double max_err = 0, max_ratio = 0;
    explicit Check(const char* s) : name(s) {}
    void operator()(double got, double ref, double tol, long where)
    {
        ++n;
        const double e = std::fabs(got - ref);
        const double r = tol > 0 ? e / tol : (e > 0 ? INFINITY : 0);
        max_err        = std::max(max_err, std::isnan(e) ? INFINITY : e);
        max_ratio      = std::max(max_ratio, std::isnan(r) ? INFINITY : r);
        if(!(e <= tol))
        {
            if(bad < 4)
                std::printf("    %s[%ld] = %.6g, ref %.6g, tol %.3g\n", name, where, got, ref, tol);
            ++bad;
        }
    }
    /// got anywhere in [lo, hi] widened by tol.
    void range(double got, double lo, double hi, double ref, double tol, long where)
    {
        const double c = std::min(std::max(got, lo), hi);
        (*this)(got, got >= lo && got <= hi ? got : c, tol, where);
        ignore_ref(ref);
    }
    static void ignore_ref(double) {}
    int report() const
    {
        std::printf("  %-26s %9ld elems  max|err| %-10.3g max err/tol %-8.3g %s\n",
                    name,
                    n,
                    max_err,
                    max_ratio,
                    bad == 0 ? "ok" : "FAIL");
        if(bad)
            std::printf("  %-26s %ld out of tolerance\n", "", bad);
        return bad != 0;
    }
};

// ---- 62's bound on the gfx1250 scaled-WMMA truncation (copied from megakernel_moe.cpp) --------
/// Error bound of the GPU's MX GEMV against the exact sum. gfx1250's
/// v_wmma_scale_f32_16x16x128_f8f6f4 does not add its 128 scaled products t_k exactly: the result
/// comes out low by at most 0.44 * sum_k (t_k - floor(t_k / g) g), g = 2^(e - 13), e the frexp
/// exponent of the WMMA's largest |t_k|. Feed the products in K order.
struct WmmaErr
{
    double bound = 0;
    double t[ck_tile::moe_mx::kWindowK];
    void add(index_t k, double v)
    {
        namespace mx        = ck_tile::moe_mx;
        t[k % mx::kWindowK] = v;
        if((k + 1) % mx::kWindowK != 0)
            return;
        for(index_t j = 0; j < 2; ++j)
        {
            auto at = [&](index_t b, index_t i) {
                return t[mx::window_block(j, b) * mx::kScaleBlock + i];
            };
            double tmax = 0;
            for(index_t b = 0; b < 4; ++b)
                for(index_t i = 0; i < mx::kScaleBlock; ++i)
                    tmax = std::max(tmax, std::fabs(at(b, i)));
            int e = 0;
            std::frexp(tmax, &e);
            const double g = std::ldexp(1.0, e - 13);
            for(index_t b = 0; b < 4; ++b)
                for(index_t i = 0; i < mx::kScaleBlock; ++i)
                    bound += at(b, i) - std::floor(at(b, i) / g) * g;
        }
    }
};

std::vector<float> widen(const uint16_t* p, size_t n)
{
    std::vector<float> v(n);
    for(size_t i = 0; i < n; ++i)
        v[i] = bf16_float(p[i]);
    return v;
}

/// Row-major double GEMM y[m, n] = sum_k x[m, k] w[n, k] (w bf16 bits, row stride ldw, column
/// offset wcol), plus mag = sum |x w|.
struct GemmOut
{
    std::vector<double> y, mag;
};
GemmOut gemm_ref(const std::vector<float>& x,
                 int M,
                 int K,
                 int ldx,
                 int xcol,
                 const std::vector<uint16_t>& w,
                 int N,
                 int ldw,
                 int wrow,
                 int wcol)
{
    GemmOut o;
    o.y.assign(size_t(M) * N, 0.0);
    o.mag.assign(size_t(M) * N, 0.0);
    std::vector<double>& y   = o.y;
    std::vector<double>* mag = &o.mag;
    parallel_for(size_t(N), [&](size_t n) {
        std::vector<float> wr(K);
        for(int k = 0; k < K; ++k)
            wr[k] = bf16_float(w[size_t(wrow + n) * ldw + wcol + k]);
        for(int m = 0; m < M; ++m)
        {
            double acc = 0, a = 0;
            const float* xr = &x[size_t(m) * ldx + xcol];
            for(int k = 0; k < K; ++k)
            {
                const double p = double(xr[k]) * wr[k];
                acc += p;
                a += std::fabs(p);
            }
            y[size_t(m) * N + n] = acc;
            if(mag)
                (*mag)[size_t(m) * N + n] = a;
        }
    });
    return o;
}

/// Attention of one rank's (or all) heads: q [M][nq * 64] (bf16 values), K / V [kv_len][nkv][64]
/// (as the cache holds them, dequantised). Token i sits at position ctx + i. Double softmax.
std::vector<double> attention_ref(const std::vector<float>& q,
                                  int M,
                                  int nq,
                                  int nkv,
                                  const std::vector<float>& K,
                                  const std::vector<float>& V,
                                  int ctx,
                                  int window)
{
    std::vector<double> out(size_t(M) * nq * kHeadDim, 0.0);
    parallel_for(size_t(M) * nq, [&](size_t t) {
        const int i = int(t / nq), h = int(t % nq), g = h / 8;
        const int p     = ctx + i;
        const int lo    = window > 0 ? std::max(0, p - window + 1) : 0;
        const float* qr = &q[(size_t(i) * nq + h) * kHeadDim];
        std::vector<double> s(p + 1 - lo);
        double mx = -1e300;
        for(int j = lo; j <= p; ++j)
        {
            const float* kr = &K[(size_t(j) * nkv + g) * kHeadDim];
            double acc      = 0;
            for(int d = 0; d < kHeadDim; ++d)
                acc += double(qr[d]) * kr[d];
            s[j - lo] = acc * 0.125;
            mx        = std::max(mx, s[j - lo]);
        }
        double sum = 0;
        for(auto& x : s)
            sum += (x = std::exp(x - mx));
        double* o = &out[(size_t(i) * nq + h) * kHeadDim];
        for(int j = lo; j <= p; ++j)
        {
            const float* vr = &V[(size_t(j) * nkv + g) * kHeadDim];
            for(int d = 0; d < kHeadDim; ++d)
                o[d] += s[j - lo] / sum * vr[d];
        }
    });
    return out;
}

/// o_norm + sigmoid gate of every head (59's attn_gate_head) on the bf16-rounded attention.
std::vector<float> gate_heads(const std::vector<double>& att,
                              int M,
                              int nq,
                              const std::vector<float>& gate /* [M][nq * 64] */)
{
    std::vector<float> o(att.size());
    for(size_t t = 0; t < size_t(M) * nq; ++t)
    {
        float oh[kHeadDim], gh[kHeadDim];
        for(int d = 0; d < kHeadDim; ++d)
        {
            oh[d] = round_bf16(float(att[t * kHeadDim + d]));
            gh[d] = gate[t * kHeadDim + d];
        }
        mk_ref::attn_gate_head(oh, gh);
        std::copy(oh, oh + kHeadDim, &o[t * kHeadDim]);
    }
    return o;
}

float kv_decode(const uint8_t* raw, size_t elem, bool fp8, float scale)
{
    if(fp8)
        return mk_ref::fp8_decode(raw[elem]) * scale;
    uint16_t b;
    std::memcpy(&b, raw + 2 * elem, 2);
    return bf16_float(b);
}

/// A KV value as the cache stores it (bf16, or e4m3 of x / scale), dequantised.
float kv_store(float x, bool fp8, float scale)
{
    return fp8 ? mk_ref::fp8_decode(mk_ref::fp8_encode(x * (1.f / scale))) * scale : x;
}

/// Logical [kv_len][nkv][64] K or V of rank r from its paged cache.
std::vector<float>
read_cache(const LayerDims& d, const HostModel& m, const std::vector<uint8_t>& c, float scale)
{
    const int nkv = d.KVHr();
    std::vector<float> v(size_t(d.kv_len()) * nkv * kHeadDim);
    for(int pos = 0; pos < d.kv_len(); ++pos)
        for(int g = 0; g < nkv; ++g)
        {
            const size_t e0 =
                ((size_t(m.block_table[pos / kPage]) * nkv + g) * kPage + pos % kPage) * kHeadDim;
            for(int dd = 0; dd < kHeadDim; ++dd)
                v[(size_t(pos) * nkv + g) * kHeadDim + dd] =
                    kv_decode(c.data(), e0 + dd, d.fp8, scale);
        }
    return v;
}

double rms_of(const float* x, int n)
{
    double s = 0;
    for(int i = 0; i < n; ++i)
        s += double(x[i]) * x[i];
    return std::sqrt(s / n);
}

/// MX GEMV of one expert matrix (MXFP4 rows [row0, row0 + N) of [., K]) against activations
/// x[route][K] (dequantised doubles) for the given routes: y[route][N], optional WmmaErr bound.
void expert_gemv(const std::vector<uint8_t>& wq,
                 const std::vector<uint8_t>& ws,
                 int K,
                 size_t row0,
                 int N,
                 const std::vector<int>& routes,
                 const std::vector<std::vector<double>>& x,
                 std::vector<std::vector<double>>& y,
                 std::vector<std::vector<double>>* bound)
{
    for(int r : routes)
    {
        y[r].assign(N, 0.0);
        if(bound)
            (*bound)[r].assign(N, 0.0);
    }
    parallel_for(size_t(N), [&](size_t n) {
        std::vector<double> w(K);
        const size_t row = row0 + n;
        for(int k = 0; k < K; ++k)
        {
            const uint8_t byte = wq[row * (K / 2) + k / 2];
            w[k]               = moe_host::e2m1_to_double(k % 2 ? byte >> 4 : byte & 15) *
                   moe_host::e8m0_to_double(ws[row * (K / 32) + k / 32]);
        }
        for(int r : routes)
        {
            double acc = 0;
            WmmaErr e;
            for(int k = 0; k < K; ++k)
            {
                const double t = w[k] * x[r][k];
                acc += t;
                if(bound)
                    e.add(k, t);
            }
            y[r][n] = acc;
            if(bound)
                (*bound)[r][n] = e.bound;
        }
    });
}

/// Host emulation of MoeCombinePhase on one row (kernel rounding points: v = bf16(y (w + 1)),
/// n = bf16(v rsqrt(mean v^2 + eps)), acc = sum w n in fp32, out bf16).
void combine_row(const LayerDims& d,
                 const HostModel& m,
                 const float* y /* [8][P] */,
                 const double* wts,
                 float* out,
                 double* mag)
{
    const int P = d.s.P;
    std::vector<double> acc(P, 0.0);
    std::fill(mag, mag + P, 0.0);
    for(int s = 0; s < kTopK; ++s)
    {
        std::vector<float> v(P);
        double ss = 0;
        for(int n = 0; n < P; ++n)
        {
            v[n] = round_bf16(y[size_t(s) * P + n] * (bf16_float(m.w_postexp[n]) + 1.0f));
            ss += double(v[n]) * v[n];
        }
        const float rs = float(1.0 / std::sqrt(ss / P + 1e-8));
        for(int n = 0; n < P; ++n)
        {
            const double t = wts[s] * round_bf16(v[n] * rs);
            acc[n] += t;
            mag[n] += std::fabs(t);
        }
    }
    for(int n = 0; n < P; ++n)
        out[n] = round_bf16(float(acc[n]));
}

/// MXFP8 block adoption for the end-to-end chain. A quantiser is discontinuous: where the chain's
/// value and the GPU's (which saw slightly different inputs) straddle a rounding boundary, the
/// codes differ by one e4m3 step (up to 2^-3 relative), and two such quantisations in the MoE
/// would amplify sub-ulp input differences into percent-level output noise. So, as with the
/// router's near-ties, the chain continues with the GPU's block (scale + 32 codes) when it is an
/// admissible quantisation of the chain's own values: scale exponent within one binade of the
/// chain's (an amax near a power of two) and every code a neighbouring grid point of the value
/// (|code - v / scale| <= one e4m3 spacing). Otherwise the chain keeps its own block and the
/// block counts as inadmissible (a failure). Returns the adopted dequantised values in out.
struct MxAdopt
{
    long blocks = 0, adopted_blocks = 0, flipped_codes = 0, bad_blocks = 0;
    /// slack (optional): extra absolute error per element the producer may have (62's WmmaErr
    /// bound of the gfx1250 scaled WMMA, propagated through SwiGLU).
    void block(const float* v,
               uint32_t e_ref,
               const uint8_t* gq,
               uint8_t ge,
               double* out,
               const double* slack = nullptr)
    {
        ++blocks;
        const double sg = std::ldexp(1.0, int(ge) - 127);
        bool ok         = std::abs(int(ge) - int(e_ref)) <= 1;
        long flips      = 0;
        for(int j = 0; j < 32 && ok; ++j)
        {
            const double t = v[j] / sg, d = moe_host::e4m3_to_double(gq[j]);
            const double a       = std::max(std::fabs(t), std::fabs(d));
            const double spacing = a < std::ldexp(1.0, -6)
                                       ? std::ldexp(1.0, -9)
                                       : std::ldexp(1.0, int(std::floor(std::log2(a))) - 3);
            ok = ok && std::fabs(d - t) <= spacing * (1 + 1e-9) + (slack ? slack[j] / sg : 0.0);
            flips += moe_host::double_to_e4m3(t) != gq[j];
        }
        if(!ok)
        {
            if(bad_blocks < 2 && std::getenv("L67_DEBUG"))
            {
                std::printf("    inadmissible block: e_ref %u e_gpu %u\n", e_ref, unsigned(ge));
                for(int j = 0; j < 32; ++j)
                    std::printf("      v %.6g t %.6g gpu %.6g code %02x\n",
                                v[j],
                                v[j] / sg,
                                moe_host::e4m3_to_double(gq[j]),
                                gq[j]);
            }
            ++bad_blocks;
            return; // keep the chain's own quantisation in out
        }
        for(int j = 0; j < 32; ++j)
            out[j] = moe_host::e4m3_to_double(gq[j]) * sg;
        adopted_blocks += flips > 0 || ge != e_ref;
        flipped_codes += flips;
    }
};

} // namespace

int check_layer(const LayerDims& d, const HostModel& m, const DeviceRun& run, bool check_stages)
{
    const int M = d.M, H = d.s.H, P = d.s.P, I = d.s.I, E = d.s.E, W = d.world;
    const int QH = d.s.QH, KVH = d.s.KVH, QHr = d.QHr(), KVHr = d.KVHr(), NQ = d.NQkvg();
    const int NR = d.NQkvgR(), R = M * kTopK;
    const RankDump& g0 = run.ranks[0];
    int fails          = 0;

    // ---- rank consistency: replicated tensors must be bit-identical on every rank
    if(W > 1)
    {
        int bad = 0;
        for(int r = 1; r < W; ++r)
        {
            const RankDump& g = run.ranks[r];
            auto cmp          = [&](const char* what, bool eq) {
                if(!eq)
                {
                    std::printf("  rank %d %s differs from rank 0\n", r, what);
                    ++bad;
                }
            };
            cmp("h", g.h == g0.h);
            cmp("x1", g.x1 == g0.x1);
            cmp("h2", g.h2 == g0.h2);
            cmp("logits", g.logits == g0.logits);
            cmp("topk", g.topk_ids == g0.topk_ids && g.topk_w == g0.topk_w);
            cmp("p", g.p == g0.p);
            cmp("pq", g.pq == g0.pq && g.ps == g0.ps);
            cmp("z", g.z == g0.z && g.zs == g0.zs);
            cmp("y", g.y == g0.y);
            cmp("acc", g.acc == g0.acc);
            cmp("out", g.out == g0.out);
        }
        std::printf("  %-26s %s\n", "replicated tensors equal", bad == 0 ? "ok" : "FAIL");
        fails += bad != 0;
    }

    // GPU tensors in global head order.
    auto gpu_qkvg = [&]() {
        std::vector<float> v(size_t(M) * NQ);
        const int gk = QH * 64, gv = gk + KVH * 64, gg = gv + KVH * 64;
        for(int r = 0; r < W; ++r)
            for(int i = 0; i < M; ++i)
            {
                const uint16_t* row = &run.ranks[r].qkvg[size_t(i) * NR];
                auto put            = [&](int src, int dst, int n) {
                    for(int c = 0; c < n; ++c)
                        v[size_t(i) * NQ + dst + c] = bf16_float(row[src + c]);
                };
                put(0, r * QHr * 64, QHr * 64);
                put(QHr * 64, gk + r * KVHr * 64, KVHr * 64);
                put((QHr + KVHr) * 64, gv + r * KVHr * 64, KVHr * 64);
                put((QHr + 2 * KVHr) * 64, gg + r * QHr * 64, QHr * 64);
            }
        return v;
    }();
    auto gpu_heads = [&](auto member) {
        std::vector<float> v(size_t(M) * QH * 64);
        for(int r = 0; r < W; ++r)
            for(int i = 0; i < M; ++i)
                for(int c = 0; c < QHr * 64; ++c)
                    v[size_t(i) * QH * 64 + r * QHr * 64 + c] =
                        bf16_float((run.ranks[r].*member)[size_t(i) * QHr * 64 + c]);
        return v;
    };
    const std::vector<float> gpu_q = gpu_heads(&RankDump::q);
    const std::vector<float> gpu_o = gpu_heads(&RankDump::o);
    const std::vector<float> xs    = widen(m.x.data(), m.x.size());
    const auto w_in = widen(m.w_in.data(), H), w_preff = widen(m.w_preff.data(), H),
               w_postff = widen(m.w_postff.data(), H), w_preexp = widen(m.w_preexp.data(), P);

    // =========================================================================================
    // Stage-local checks
    if(check_stages)
    {
        std::printf(" stage-local checks (each stage from the GPU's own input):\n");
        const double u7 = std::ldexp(1.0, -7), u10 = std::ldexp(1.0, -10);
        {
            // 0: input CenteredRMSNorm (59: one bf16 ulp + 2^-10).
            Check c("0 in_norm h");
            std::vector<float> hr(H);
            for(int i = 0; i < M; ++i)
            {
                mk_ref::centered_rmsnorm(
                    &xs[size_t(i) * H], w_in.data(), H, mk_ref::kRmsEps, hr.data());
                for(int n = 0; n < H; ++n)
                    c(bf16_float(g0.h[size_t(i) * H + n]),
                      hr[n],
                      u7 * std::fabs(hr[n]) + u10,
                      long(i) * H + n);
            }
            fails += c.report();
        }
        const auto gh = widen(g0.h.data(), g0.h.size());
        {
            // 1: qkvg GEMM per rank shard (66: |ref| 2^-7 + sum|x w| 2^-14).
            Check c("1 qkvg gemm");
            std::vector<double> y, mag;
            {
                auto go_ = gemm_ref(gh, M, H, H, 0, m.w_qkvg, NQ, H, 0, 0);
                y        = std::move(go_.y);
                mag      = std::move(go_.mag);
            }
            for(size_t e = 0; e < y.size(); ++e)
                c(gpu_qkvg[e], y[e], std::fabs(y[e]) * u7 + mag[e] * std::ldexp(1.0, -14), long(e));
            fails += c.report();
        }
        {
            // 2: QK-norm / RoPE / KV write from the GPU's qkvg (59's tolerances).
            Check cq("2 q (norm, rope)"), ck("2 K cache new rows"), cv("2 V cache new rows"),
                cx("2 cache other slots");
            const int gk = QH * 64, gv = gk + KVH * 64;
            for(int r = 0; r < W; ++r)
            {
                const RankDump& g = run.ranks[r];
                std::vector<char> written(g.kcache.size() / (d.fp8 ? 1 : 2), 0);
                for(int i = 0; i < M; ++i)
                {
                    const int pos = m.positions[i], slot = m.slot_mapping[i];
                    for(int h = 0; h < QHr; ++h)
                    {
                        float x[64];
                        std::copy_n(&gpu_qkvg[size_t(i) * NQ + (r * QHr + h) * 64], 64, x);
                        mk_ref::qk_head(x, true, d.sliding, pos);
                        for(int dd = 0; dd < 64; ++dd)
                            cq(bf16_float(g.q[size_t(i) * QHr * 64 + h * 64 + dd]),
                               x[dd],
                               u7 * std::fabs(x[dd]) + mk_ref::kQkScale * u7,
                               long(i));
                    }
                    for(int kv = 0; kv < KVHr; ++kv)
                    {
                        float k[64], v[64];
                        std::copy_n(&gpu_qkvg[size_t(i) * NQ + gk + (r * KVHr + kv) * 64], 64, k);
                        std::copy_n(&gpu_qkvg[size_t(i) * NQ + gv + (r * KVHr + kv) * 64], 64, v);
                        mk_ref::qk_head(k, false, d.sliding, pos);
                        const size_t e0 =
                            ((size_t(slot / kPage) * KVHr + kv) * kPage + slot % kPage) * 64;
                        for(int dd = 0; dd < 64; ++dd)
                        {
                            written[e0 + dd] = 1;
                            const float kg = kv_decode(g.kcache.data(), e0 + dd, d.fp8, m.k_scale);
                            const float vg = kv_decode(g.vcache.data(), e0 + dd, d.fp8, m.v_scale);
                            if(d.fp8)
                            {
                                ck(kg,
                                   kv_store(k[dd], true, m.k_scale),
                                   std::ldexp(1.0, -3) * std::fabs(k[dd]) + std::ldexp(1.0, -9),
                                   long(e0 + dd));
                                cv(vg,
                                   kv_store(v[dd], true, m.v_scale),
                                   std::ldexp(1.0, -3) * std::fabs(v[dd]) + std::ldexp(1.0, -9),
                                   long(e0 + dd));
                            }
                            else
                            {
                                ck(kg, k[dd], u7 * std::fabs(k[dd]) + u7, long(e0 + dd));
                                cv(vg, v[dd], 0, long(e0 + dd));
                            }
                        }
                    }
                }
                // Context slots and unowned pages must hold what the host put there.
                const int es = d.fp8 ? 1 : 2;
                for(size_t e = 0; e < written.size(); ++e)
                    if(!written[e])
                    {
                        const bool same =
                            std::memcmp(&g.kcache[e * es], &g.kcache_init[e * es], es) == 0 &&
                            std::memcmp(&g.vcache[e * es], &g.vcache_init[e * es], es) == 0;
                        cx(same ? 0 : 1, 0, 0, long(e));
                    }
            }
            fails += cq.report() + ck.report() + cv.report() + cx.report();
        }
        {
            // 3: attention + o_norm + gate from the GPU's q and cache, per rank (60: atol 1/32,
            // rtol 1/128 with the norm-gate epilogue).
            Check c("3 attention+gate o");
            for(int r = 0; r < W; ++r)
            {
                const RankDump& g = run.ranks[r];
                const auto K      = read_cache(d, m, g.kcache, m.k_scale);
                const auto V      = read_cache(d, m, g.vcache, m.v_scale);
                const auto q      = widen(g.q.data(), g.q.size());
                const auto att    = attention_ref(q, M, QHr, KVHr, K, V, d.ctx, d.window());
                std::vector<float> gate(size_t(M) * QHr * 64);
                for(int i = 0; i < M; ++i)
                    for(int cc = 0; cc < QHr * 64; ++cc)
                        gate[size_t(i) * QHr * 64 + cc] =
                            bf16_float(g.qkvg[size_t(i) * NR + (QHr + 2 * KVHr) * 64 + cc]);
                const auto o = gate_heads(att, M, QHr, gate);
                for(size_t e = 0; e < o.size(); ++e)
                    c(bf16_float(g.o[e]), o[e], 1.0 / 32 + std::fabs(o[e]) / 128, long(e));
            }
            fails += c.report();
        }
        {
            // 4: o_proj partial per rank (fp32 out: sum|x w| 2^-14 + |ref| 2^-22).
            Check c("4 o_proj partial (fp32)");
            for(int r = 0; r < W; ++r)
            {
                const auto o = widen(run.ranks[r].o.data(), run.ranks[r].o.size());
                std::vector<double> y, mag;
                {
                    auto go_ =
                        gemm_ref(o, M, QHr * 64, QHr * 64, 0, m.w_o, H, QH * 64, 0, r * QHr * 64);
                    y   = std::move(go_.y);
                    mag = std::move(go_.mag);
                }
                for(size_t e = 0; e < y.size(); ++e)
                    c(run.ranks[r].opart[e],
                      y[e],
                      mag[e] * std::ldexp(1.0, -14) + std::fabs(y[e]) * std::ldexp(1.0, -22),
                      long(e));
            }
            fails += c.report();
        }
        // 5 / 18: all-reduce + post norm + residual gate (+ next norm), 59's range check.
        auto residual_check = [&](const char* nx,
                                  const char* nh,
                                  bool scaled,
                                  const std::vector<float> RankDump::*part,
                                  const std::vector<float>& gate,
                                  const std::vector<float>& first_w,
                                  const std::vector<float>& residual,
                                  const std::vector<uint16_t>& gx,
                                  const std::vector<uint16_t>* gh2) {
            Check cx(nx), ch(nh ? nh : "-");
            constexpr float kRSlack = 1.f / 262144.f;
            std::vector<float> sum(H), xr(H), hr(H), xlo(H), xhi(H), hg(H);
            for(int i = 0; i < M; ++i)
            {
                for(int n = 0; n < H; ++n)
                {
                    float s = 0.f; // fp32, rank order, as the all-reduce
                    for(int r = 0; r < W; ++r)
                        s += (run.ranks[r].*part)[size_t(i) * H + n];
                    sum[n] = s;
                }
                mk_ref::residual_norm_row(sum.data(),
                                            scaled,
                                            scaled ? first_w.data() : nullptr,
                                            gate.data(),
                                            &residual[size_t(i) * H],
                                            w_preff.data(),
                                            H,
                                            xr.data(),
                                            nullptr,
                                            kRSlack,
                                            xlo.data(),
                                            xhi.data());
                const auto xg = widen(&gx[size_t(i) * H], H);
                for(int n = 0; n < H; ++n)
                    cx.range(xg[n],
                             xlo[n],
                             xhi[n],
                             xr[n],
                             u7 * std::max(std::fabs(xlo[n]), std::fabs(xhi[n])) + u10,
                             long(i) * H + n);
                if(gh2)
                {
                    mk_ref::centered_rmsnorm(
                        xg.data(), w_preff.data(), H, mk_ref::kRmsEps, hg.data());
                    for(int n = 0; n < H; ++n)
                        ch(bf16_float((*gh2)[size_t(i) * H + n]),
                           hg[n],
                           u7 * std::fabs(hg[n]) + u10,
                           long(i) * H + n);
                }
            }
            int f = cx.report();
            if(gh2)
                f += ch.report();
            return f;
        };
        fails += residual_check("5 x1 (post_attn+gate)",
                                "5 h2 (pre_ff norm)",
                                false,
                                &RankDump::opart,
                                m.g_attn,
                                w_postff,
                                xs,
                                g0.x1,
                                &g0.h2);
        const auto gh2 = widen(g0.h2.data(), g0.h2.size());
        router_ref::Routing ref_route;
        {
            // 6: router logits (61: sum|h w| 1e-5 + 1e-6). 7: top-8 (61's compare_topk) and the
            // route table rebuilt from the GPU's ids.
            Check c("6 router logits");
            const auto L = router_ref::router_logits(g0.h2, m.w_router, M, H, E, kOutMul);
            for(size_t e = 0; e < L.logits.size(); ++e)
                c(g0.logits[e], L.logits[e], L.mag[e] * 1e-5 + 1e-6, long(e));
            fails += c.report();
            ref_route = router_ref::route_from_logits(L.logits, m.router_bias, M, E, kTopK);
            int ties  = 0;
            const int bad =
                router_ref::compare_topk(ref_route, g0.topk_ids, g0.topk_w, 1e-5, 1e-5, &ties);
            const auto t = router_ref::build_route_table(g0.topk_ids, M, kTopK, E, M);
            bool tab =
                g0.num_active[0] == t.num_active && g0.count == t.count && g0.active == t.active;
            for(int e = 0; e < E && tab; ++e)
                for(int j = 0; j < t.count[e]; ++j)
                    tab = tab && g0.routes[size_t(e) * M + j] == t.routes[size_t(e) * M + j];
            std::printf("  %-26s %d rows bad, %d near-ties accepted; route table %s; %d active "
                        "experts  %s\n",
                        "7 router top-8",
                        bad,
                        ties,
                        tab ? "consistent" : "INCONSISTENT",
                        g0.num_active[0],
                        bad == 0 && tab ? "ok" : "FAIL");
            fails += bad != 0 || !tab;
        }
        {
            // 8/9: pooler (N-sharded) + all-gather: every column of p from the GPU's h2.
            Check c("8+9 pooler p (gathered)");
            std::vector<double> y, mag;
            {
                auto go_ = gemm_ref(gh2, M, H, H, 0, m.w_pool, P, H, 0, 0);
                y        = std::move(go_.y);
                mag      = std::move(go_.mag);
            }
            for(size_t e = 0; e < y.size(); ++e)
                c(bf16_float(g0.p[e]),
                  y[e],
                  std::fabs(y[e]) * u7 + mag[e] * std::ldexp(1.0, -14),
                  long(e));
            fails += c.report();
        }
        {
            // 10: pre_expert_norm (one bf16 ulp) and MXFP8: the e8m0 scale exactly as the
            // epilogue computes it from the GPU's pn, the codes within half an e4m3 ulp (2^-4
            // relative) or the subnormal step (2^-9 of the block scale).
            Check cn("10 pre_expert_norm pn"), cs("10 MXFP8 scales"), cq("10 MXFP8 values");
            std::vector<float> pr(P);
            for(int i = 0; i < M; ++i)
            {
                const auto pg = widen(&g0.p[size_t(i) * P], P);
                mk_ref::centered_rmsnorm(
                    pg.data(), w_preexp.data(), P, mk_ref::kRmsEps, pr.data());
                for(int n = 0; n < P; ++n)
                    cn(bf16_float(g0.pn[size_t(i) * P + n]),
                       pr[n],
                       u7 * std::fabs(pr[n]) + u10,
                       long(i) * P + n);
                for(int b = 0; b < P / 32; ++b)
                {
                    float amax = 0;
                    for(int j = 0; j < 32; ++j)
                        amax = std::max(amax,
                                        std::fabs(bf16_float(g0.pn[size_t(i) * P + 32 * b + j])));
                    const uint32_t e = ck_tile::mk_mx_scale_e8m0<8>(amax);
                    cs(g0.ps[size_t(i) * (P / 32) + b], double(e), 0, long(i) * (P / 32) + b);
                    const double sc = std::ldexp(1.0, int(g0.ps[size_t(i) * (P / 32) + b]) - 127);
                    for(int j = 0; j < 32; ++j)
                    {
                        const size_t idx = size_t(i) * P + 32 * b + j;
                        const double v   = bf16_float(g0.pn[idx]);
                        cq(moe_host::e4m3_to_double(g0.pq[idx]) * sc,
                           v,
                           std::fabs(v) * 0.0626 + sc * std::ldexp(1.0, -9),
                           long(idx));
                    }
                }
            }
            fails += cn.report() + cs.report() + cq.report();
        }
        {
            // 11-16: MoE from the GPU's inputs (62's tolerances, WmmaErr per output):
            //  z (gate/up + SwiGLU + MXFP8 requant, gathered), y (down, gathered), acc (combine).
            Check cz("11+12+13 moe z (gathered)"), cy("14+15 moe y (gathered)"),
                ca("16 moe combine acc");
            std::map<int, std::vector<int>> by_e;
            for(int rt = 0; rt < R; ++rt)
                by_e[g0.topk_ids[rt]].push_back(rt);
            std::vector<std::vector<double>> px(R), zx(R), gy(R), uy(R), dy(R), gb(R), ub(R), db(R);
            for(int rt = 0; rt < R; ++rt)
            {
                const int i = rt / kTopK;
                px[rt].resize(P);
                for(int k = 0; k < P; ++k)
                    px[rt][k] = moe_host::e4m3_to_double(g0.pq[size_t(i) * P + k]) *
                                moe_host::e8m0_to_double(g0.ps[size_t(i) * (P / 32) + k / 32]);
                zx[rt].resize(I);
                for(int k = 0; k < I; ++k)
                    zx[rt][k] = moe_host::e4m3_to_double(g0.z[size_t(rt) * I + k]) *
                                moe_host::e8m0_to_double(g0.zs[size_t(rt) * (I / 32) + k / 32]);
            }
            for(const auto& [e, rts] : by_e)
            {
                if(e < 0 || e >= E)
                    continue;
                expert_gemv(m.gate_q, m.gate_s, P, size_t(e) * I, I, rts, px, gy, &gb);
                expert_gemv(m.up_q, m.up_s, P, size_t(e) * I, I, rts, px, uy, &ub);
                expert_gemv(m.down_q, m.down_s, I, size_t(e) * P, P, rts, zx, dy, &db);
            }
            for(int rt = 0; rt < R; ++rt)
            {
                if(gy[rt].empty())
                    continue;
                std::vector<double> zr(I);
                for(int n = 0; n < I; ++n)
                    zr[n] = gy[rt][n] / (1 + std::exp(-gy[rt][n])) * uy[rt][n];
                for(int n = 0; n < I; ++n)
                {
                    double amax = 0;
                    for(int j = n / 32 * 32; j < n / 32 * 32 + 32; ++j)
                        amax = std::max(amax, std::fabs(zr[j]));
                    const double sg = gy[rt][n] / (1 + std::exp(-gy[rt][n]));
                    const double wb = 1.1 * gb[rt][n] * std::fabs(uy[rt][n]) +
                                      ub[rt][n] * std::fabs(sg) + 1.1 * gb[rt][n] * ub[rt][n];
                    const double tol =
                        std::fabs(zr[n]) * 0.0626 + amax / 448 * std::ldexp(1.0, -8) + wb;
                    cz(zx[rt][n], zr[n], tol, long(rt) * I + n);
                }
                for(int n = 0; n < P; ++n)
                    cy(bf16_float(g0.y[size_t(rt) * P + n]),
                       dy[rt][n],
                       std::fabs(dy[rt][n]) * std::ldexp(1.0, -8) + db[rt][n],
                       long(rt) * P + n);
            }
            std::vector<float> yg(size_t(kTopK) * P), accr(P);
            std::vector<double> mag(P), wts(kTopK);
            for(int i = 0; i < M; ++i)
            {
                for(size_t e = 0; e < yg.size(); ++e)
                    yg[e] = bf16_float(g0.y[size_t(i) * kTopK * P + e]);
                for(int s = 0; s < kTopK; ++s)
                    wts[s] = g0.topk_w[size_t(i) * kTopK + s];
                combine_row(d, m, yg.data(), wts.data(), accr.data(), mag.data());
                for(int n = 0; n < P; ++n)
                    ca(bf16_float(g0.acc[size_t(i) * P + n]),
                       accr[n],
                       mag[n] * std::ldexp(1.0, -7) + std::fabs(accr[n]) * std::ldexp(1.0, -8),
                       long(i) * P + n);
            }
            fails += cz.report() + cy.report() + ca.report();
        }
        {
            // 17: unpooler partial per rank (fp32 out, as stage 4).
            Check c("17 unpooler partial (fp32)");
            const auto acc = widen(g0.acc.data(), g0.acc.size());
            for(int r = 0; r < W; ++r)
            {
                std::vector<double> y, mag;
                {
                    auto go_ =
                        gemm_ref(acc, M, P / W, P, r * (P / W), m.w_unpool, H, P, 0, r * (P / W));
                    y   = std::move(go_.y);
                    mag = std::move(go_.mag);
                }
                for(size_t e = 0; e < y.size(); ++e)
                    c(run.ranks[r].upart[e],
                      y[e],
                      mag[e] * std::ldexp(1.0, -14) + std::fabs(y[e]) * std::ldexp(1.0, -22),
                      long(e));
            }
            fails += c.report();
        }
        const auto gx1 = widen(g0.x1.data(), g0.x1.size());
        fails += residual_check("18 out (post_ff+gate)",
                                nullptr,
                                true,
                                &RankDump::upart,
                                m.g_ffn,
                                w_postff,
                                gx1,
                                g0.out,
                                nullptr);
    }

    // =========================================================================================
    // End-to-end host layer
    std::printf(
        " end-to-end host reference (x and weights only; GPU top-8 adopted on near-ties):\n");
    std::vector<float> h(size_t(M) * H);
    for(int i = 0; i < M; ++i)
        mk_ref::centered_rmsnorm(
            &xs[size_t(i) * H], w_in.data(), H, mk_ref::kRmsEps, &h[size_t(i) * H]);
    const GemmOut qkvg_d = gemm_ref(h, M, H, H, 0, m.w_qkvg, NQ, H, 0, 0);
    std::vector<float> qkvg(qkvg_d.y.size());
    // qkvg feeds the attention-sensitive q / k (see below): continue with the GPU's element
    // where it is within the GEMM's tolerance (66: 2^-7 |v| + sum|x w| 2^-14) of the chain's.
    long qkvg_adopted = 0, qkvg_bad = 0;
    for(size_t e = 0; e < qkvg.size(); ++e)
    {
        qkvg[e] = round_bf16(float(qkvg_d.y[e]));
        const double tol =
            std::ldexp(1.0, -7) * std::fabs(qkvg[e]) + qkvg_d.mag[e] * std::ldexp(1.0, -14);
        if(std::fabs(double(gpu_qkvg[e]) - qkvg[e]) <= tol)
        {
            qkvg_adopted += bf16_bits(gpu_qkvg[e]) != bf16_bits(qkvg[e]);
            qkvg[e] = gpu_qkvg[e];
        }
        else
            ++qkvg_bad;
    }
    // q, new K / V, as stored. The attention is very sensitive to q and k: QK-norm with the 5.47
    // q scale makes |q.k| / 8 reach 43.8, so a one-ulp bf16 difference of an element moves a
    // score by ~0.1 and the softmax weights by ~10 %. Two correct implementations whose qkvg
    // differ by an ulp (fp32 vs double GEMM sums) would thus differ by ~0.1 in the attention
    // output. So, like the top-8 and the MXFP8 blocks, the chain continues with the GPU's q / K /
    // V element where it is within the stage's rounding tolerance of the chain's own value (59's
    // q tolerance: 2^-7 |v| + 5.47 2^-7; K bf16 2^-7 |v| + 2^-7, V bf16 exact; fp8 one e4m3 step,
    // 2^-3 |v| + 2^-9 scale); anything else is inadmissible (a failure).
    std::vector<float> q(size_t(M) * QH * 64), gate(size_t(M) * QH * 64);
    std::vector<float> K(size_t(d.kv_len()) * KVH * 64), V(K.size());
    std::copy(m.k_ctx.begin(), m.k_ctx.end(), K.begin());
    std::copy(m.v_ctx.begin(), m.v_ctx.end(), V.begin());
    long qkv_adopted = 0, qkv_bad = 0;
    auto adopt = [&](float chain, float gpu, double tol) {
        if(std::fabs(double(gpu) - chain) <= tol)
        {
            qkv_adopted += bf16_bits(gpu) != bf16_bits(chain);
            return gpu;
        }
        ++qkv_bad;
        return chain;
    };
    {
        const int gk = QH * 64, gv = gk + KVH * 64, gg = gv + KVH * 64;
        const double u7 = std::ldexp(1.0, -7);
        for(int i = 0; i < M; ++i)
        {
            const float* row = &qkvg[size_t(i) * NQ];
            for(int hh = 0; hh < QH; ++hh)
            {
                float x[64];
                std::copy_n(row + hh * 64, 64, x);
                mk_ref::qk_head(x, true, d.sliding, m.positions[i]);
                for(int dd = 0; dd < 64; ++dd)
                {
                    const size_t e = size_t(i) * QH * 64 + hh * 64 + dd;
                    q[e] = adopt(x[dd], gpu_q[e], u7 * std::fabs(x[dd]) + mk_ref::kQkScale * u7);
                }
                std::copy_n(row + gg + hh * 64, 64, &gate[size_t(i) * QH * 64 + hh * 64]);
            }
            const int slot = m.slot_mapping[i];
            for(int kv = 0; kv < KVH; ++kv)
            {
                float x[64];
                std::copy_n(row + gk + kv * 64, 64, x);
                mk_ref::qk_head(x, false, d.sliding, m.positions[i]);
                const RankDump& g = run.ranks[kv / KVHr];
                const size_t e0 =
                    ((size_t(slot / kPage) * KVHr + kv % KVHr) * kPage + slot % kPage) * 64;
                for(int dd = 0; dd < 64; ++dd)
                {
                    const size_t e = (size_t(d.ctx + i) * KVH + kv) * 64 + dd;
                    const float kc = kv_store(x[dd], d.fp8, m.k_scale);
                    const float vc = kv_store(row[gv + kv * 64 + dd], d.fp8, m.v_scale);
                    const float kg = kv_decode(g.kcache.data(), e0 + dd, d.fp8, m.k_scale);
                    const float vg = kv_decode(g.vcache.data(), e0 + dd, d.fp8, m.v_scale);
                    if(d.fp8)
                    {
                        K[e] = adopt(kc,
                                     kg,
                                     std::ldexp(1.0, -3) * std::fabs(kc) +
                                         std::ldexp(1.0, -9) * m.k_scale);
                        V[e] = adopt(vc,
                                     vg,
                                     std::ldexp(1.0, -3) * std::fabs(vc) +
                                         std::ldexp(1.0, -9) * m.v_scale);
                    }
                    else
                    {
                        K[e] = adopt(kc, kg, u7 * std::fabs(kc) + u7);
                        V[e] = adopt(vc, vg, u7 * std::fabs(vc));
                    }
                }
            }
        }
    }
    const auto att             = attention_ref(q, M, QH, KVH, K, V, d.ctx, d.window());
    const auto o               = gate_heads(att, M, QH, gate);
    std::vector<double> osum_d = gemm_ref(o, M, QH * 64, QH * 64, 0, m.w_o, H, QH * 64, 0, 0).y;
    std::vector<float> osum(osum_d.begin(), osum_d.end()), x1(size_t(M) * H), h2(size_t(M) * H);
    for(int i = 0; i < M; ++i)
        mk_ref::residual_norm_row(&osum[size_t(i) * H],
                                    false,
                                    nullptr,
                                    m.g_attn.data(),
                                    &xs[size_t(i) * H],
                                    w_preff.data(),
                                    H,
                                    &x1[size_t(i) * H],
                                    &h2[size_t(i) * H]);
    // ---- MoE half (router .. layer output) from a given h2 and residual x1
    struct MoeHalf
    {
        std::vector<float> p, acc, out;
        int adopted = 0, route_mismatch = 0;
        MxAdopt adopt_p, adopt_z;
    };
    auto moe_half = [&](const std::vector<float>& h2_in, const std::vector<float>& x1_in) {
        MoeHalf res;
        int& adopted        = res.adopted;
        int& route_mismatch = res.route_mismatch;
        MxAdopt& adopt_p    = res.adopt_p;
        MxAdopt& adopt_z    = res.adopt_z;
        // router
        std::vector<uint16_t> h2b(h2_in.size());
        for(size_t e = 0; e < h2_in.size(); ++e)
            h2b[e] = bf16_bits(h2_in[e]);
        const auto L  = router_ref::router_logits(h2b, m.w_router, M, H, E, kOutMul);
        const auto rt = router_ref::route_from_logits(L.logits, m.router_bias, M, E, kTopK);
        std::vector<int> ids(R);
        std::vector<double> wts(R);
        for(int i = 0; i < M; ++i)
        {
            std::vector<int> want(&rt.ids[size_t(i) * kTopK], &rt.ids[size_t(i) * kTopK] + kTopK);
            std::vector<int> got(&g0.topk_ids[size_t(i) * kTopK],
                                 &g0.topk_ids[size_t(i) * kTopK] + kTopK);
            std::vector<int> ws = want, gs = got;
            std::sort(ws.begin(), ws.end());
            std::sort(gs.begin(), gs.end());
            bool use_gpu = ws == gs;
            if(!use_gpu)
            {
                // Near-tie given the propagated error: the GPU's router saw its own h2, which
                // differs from the chain's by bf16 ulps (and by the attention phase's bf16-P error
                // upstream). dlogit[e] = scale |sum_k (h2_gpu - h2_ref)[k] w[e, k]| is that
                // difference's exact effect on logit e; sigmoid' <= 1/4. An expert in the set
                // difference is accepted when its choice is within that slack (twice the largest
                // such shift in the row, plus 61's 1e-5 for the fp32 GEMM) of the k-th choice.
                const double* c  = &rt.choice[size_t(i) * E];
                const double kth = c[want.back()];
                double slack     = 0;
                for(int e = 0; e < E; ++e)
                {
                    double dl = 0;
                    for(int k = 0; k < H; ++k)
                        dl += (double(bf16_float(g0.h2[size_t(i) * H + k])) -
                               h2_in[size_t(i) * H + k]) *
                              m.w_router[size_t(e) * H + k];
                    slack = std::max(slack, 0.25 * kOutMul * std::fabs(dl));
                }
                const double tol = 2 * slack + 1e-5;
                bool tie =
                    std::all_of(gs.begin(), gs.end(), [&](int e) { return e >= 0 && e < E; });
                for(int e : gs)
                    if(tie && !std::binary_search(ws.begin(), ws.end(), e))
                        tie = std::fabs(c[e] - kth) <= tol;
                for(int e : ws)
                    if(tie && !std::binary_search(gs.begin(), gs.end(), e))
                        tie = std::fabs(c[e] - kth) <= tol;
                use_gpu = tie;
                adopted += tie;
                route_mismatch += !tie;
            }
            const std::vector<int>& pick = use_gpu ? got : want;
            double sum                   = 0;
            for(int e : pick)
                sum += rt.s[size_t(i) * E + e];
            for(int s = 0; s < kTopK; ++s)
            {
                ids[size_t(i) * kTopK + s] = pick[s];
                wts[size_t(i) * kTopK + s] = rt.s[size_t(i) * E + pick[s]] / (sum + 1e-15);
            }
        }
        // pooler, pre_expert_norm, MXFP8 (the epilogue's scale rule, RNE e4m3)
        std::vector<double> p_d = gemm_ref(h2_in, M, H, H, 0, m.w_pool, P, H, 0, 0).y;
        std::vector<float> p(p_d.size()), pn(p_d.size());
        for(size_t e = 0; e < p.size(); ++e)
            p[e] = round_bf16(float(p_d[e]));
        std::vector<std::vector<double>> px(R), zx(R), gy(R), uy(R), dy(R), gbd(R), ubd(R);
        std::vector<std::vector<double>> pq_rows(M, std::vector<double>(P));
        for(int i = 0; i < M; ++i)
        {
            mk_ref::centered_rmsnorm(
                &p[size_t(i) * P], w_preexp.data(), P, mk_ref::kRmsEps, &pn[size_t(i) * P]);
            for(int b = 0; b < P / 32; ++b)
            {
                const float* v = &pn[size_t(i) * P + 32 * b];
                float amax     = 0;
                for(int j = 0; j < 32; ++j)
                    amax = std::max(amax, std::fabs(v[j]));
                const uint32_t e = ck_tile::mk_mx_scale_e8m0<8>(amax);
                const float inv  = ck_tile::mk_mx_inv_scale(e);
                const double sc  = std::ldexp(1.0, int(e) - 127);
                for(int j = 0; j < 32; ++j)
                    pq_rows[i][32 * b + j] =
                        mk_ref::fp8_decode(mk_ref::fp8_encode(v[j] * inv)) * sc;
                adopt_p.block(v,
                              e,
                              &g0.pq[size_t(i) * P + 32 * b],
                              g0.ps[size_t(i) * (P / 32) + b],
                              &pq_rows[i][32 * b]);
            }
        }
        std::map<int, std::vector<int>> by_e;
        for(int r_ = 0; r_ < R; ++r_)
        {
            by_e[ids[r_]].push_back(r_);
            px[r_] = pq_rows[r_ / kTopK];
        }
        for(const auto& [e, rts] : by_e)
        {
            expert_gemv(m.gate_q, m.gate_s, P, size_t(e) * I, I, rts, px, gy, &gbd);
            expert_gemv(m.up_q, m.up_s, P, size_t(e) * I, I, rts, px, uy, &ubd);
        }
        for(int r_ = 0; r_ < R; ++r_)
        {
            std::vector<float> z(I);
            for(int n = 0; n < I; ++n)
            {
                const float g = float(gy[r_][n]), u = float(uy[r_][n]);
                z[n] = g / (1.0f + std::exp(-g)) * u;
            }
            std::vector<uint8_t> zq(I), zs(I / 32);
            moe_host::quantize_row_mxfp8(z.data(), I, zq.data(), zs.data());
            zx[r_].resize(I);
            for(int n = 0; n < I; ++n)
                zx[r_][n] = moe_host::e4m3_to_double(zq[n]) * moe_host::e8m0_to_double(zs[n / 32]);
            // 62's z slack: silu' <= 1.1, z = silu(g) u.
            std::vector<double> zslack(I);
            for(int n = 0; n < I; ++n)
            {
                const double g = gy[r_][n], sg = g / (1 + std::exp(-g));
                zslack[n] = 1.1 * gbd[r_][n] * std::fabs(uy[r_][n]) + ubd[r_][n] * std::fabs(sg) +
                            1.1 * gbd[r_][n] * ubd[r_][n];
            }
            if(ids[r_] == g0.topk_ids[r_]) // same expert on this route
                for(int b = 0; b < I / 32; ++b)
                    adopt_z.block(&z[32 * b],
                                  zs[b],
                                  &g0.z[size_t(r_) * I + 32 * b],
                                  g0.zs[size_t(r_) * (I / 32) + b],
                                  &zx[r_][32 * b],
                                  &zslack[32 * b]);
        }
        for(const auto& [e, rts] : by_e)
            expert_gemv(m.down_q, m.down_s, I, size_t(e) * P, P, rts, zx, dy, nullptr);
        std::vector<float> y(size_t(R) * P), acc(size_t(M) * P);
        for(int r_ = 0; r_ < R; ++r_)
            for(int n = 0; n < P; ++n)
                y[size_t(r_) * P + n] = round_bf16(float(dy[r_][n]));
        {
            std::vector<double> mag(P);
            for(int i = 0; i < M; ++i)
                combine_row(d,
                            m,
                            &y[size_t(i) * kTopK * P],
                            &wts[size_t(i) * kTopK],
                            &acc[size_t(i) * P],
                            mag.data());
        }
        std::vector<double> u_d = gemm_ref(acc, M, P, P, 0, m.w_unpool, H, P, 0, 0).y;
        std::vector<float> usum(u_d.begin(), u_d.end()), out(size_t(M) * H);
        for(int i = 0; i < M; ++i)
            mk_ref::residual_norm_row(&usum[size_t(i) * H],
                                        true,
                                        w_postff.data(),
                                        m.g_ffn.data(),
                                        &x1_in[size_t(i) * H],
                                        nullptr,
                                        H,
                                        &out[size_t(i) * H],
                                        nullptr);

        res.p   = std::move(p);
        res.acc = std::move(acc);
        res.out = std::move(out);
        return res;
    };
    // Independent: the whole layer from x (information; see below).
    const MoeHalf ind = moe_half(h2, x1);
    // Re-synchronised at the halves' boundary: the MoE half from the GPU's own h2 and x1.
    const auto gh2v = widen(g0.h2.data(), g0.h2.size()), gx1v = widen(g0.x1.data(), g0.x1.size());
    const MoeHalf syn = moe_half(gh2v, gx1v);

    // Per-stage distance of the GPU from the end-to-end chain, in units of the tensor's rms
    // (informational: shows where differences enter and how they propagate).
    auto dist = [&](const char* name, const std::vector<float>& ref, auto gpu_at, size_t n) {
        double ss = 0, worst = 0;
        for(size_t e = 0; e < n; ++e)
            ss += double(ref[e]) * ref[e];
        const double rms = std::sqrt(ss / n);
        for(size_t e = 0; e < n; ++e)
            worst = std::max(worst, std::fabs(gpu_at(e) - ref[e]));
        std::printf("  %-26s max|gpu - ref| %-10.3g = %.3g x rms (rms %.3g)\n",
                    name,
                    worst,
                    worst / rms,
                    rms);
    };
    auto bf = [](const std::vector<uint16_t>& v) {
        return [&v](size_t e) { return double(bf16_float(v[e])); };
    };
    dist("h (in_norm)", h, bf(g0.h), h.size());
    dist("qkvg", qkvg, [&](size_t e) { return double(gpu_qkvg[e]); }, qkvg.size());
    dist("o (attention+gate)", o, [&](size_t e) { return double(gpu_o[e]); }, o.size());
    dist("x1 (residual 1)", x1, bf(g0.x1), x1.size());
    dist("h2 (pre_ff norm)", h2, bf(g0.h2), h2.size());
    dist("out (independent chain)", ind.out, bf(g0.out), ind.out.size());
    std::printf(
        "  %-26s qkvg: %ld GPU elements adopted, %ld inadmissible; q / new K / V: %ld adopted, "
        "%ld inadmissible  %s\n",
        "attention half",
        qkvg_adopted,
        qkvg_bad,
        qkv_adopted,
        qkv_bad,
        qkv_bad == 0 && qkvg_bad == 0 ? "ok" : "FAIL");
    fails += qkv_bad != 0 || qkvg_bad != 0;
    auto adopt_report = [&](const char* tag, const MoeHalf& r) {
        std::printf(
            "  %-26s router: %d rows took the GPU's near-tie top-8, %d differ beyond a tie\n",
            tag,
            r.adopted,
            r.route_mismatch);
        for(auto [name, a] :
            {std::make_pair("MXFP8 p", &r.adopt_p), std::make_pair("MXFP8 z", &r.adopt_z)})
            std::printf("  %-26s %s: %ld blocks, %ld GPU blocks adopted (%ld boundary flips), %ld "
                        "inadmissible\n",
                        tag,
                        name,
                        a->blocks,
                        a->adopted_blocks,
                        a->flipped_codes,
                        a->bad_blocks);
    };
    adopt_report("MoE half (independent)", ind);
    adopt_report("MoE half (from GPU h2)", syn);
    dist("p (from GPU h2)", syn.p, bf(g0.p), syn.p.size());
    dist("acc (from GPU h2)", syn.acc, bf(g0.acc), syn.acc.size());
    dist("out (from GPU h2, x1)", syn.out, bf(g0.out), syn.out.size());
    const bool syn_ok =
        syn.route_mismatch == 0 && syn.adopt_p.bad_blocks == 0 && syn.adopt_z.bad_blocks == 0;
    std::printf("  %-26s %s\n", "MoE half decisions", syn_ok ? "all admissible  ok" : "FAIL");
    fails += !syn_ok;

    // Final checks, per row (rms over the row's elements):
    //  * attention half, x from the inputs: x1. The chain and the GPU round at the same points;
    //    what separates them is the attention phase's own precision (bf16 K / V and bf16 softmax
    //    probabilities in the PV WMMA: example 60 accepts 1/32 absolute on unit-scale heads),
    //    rounding-boundary flips and fp32 accumulation. o_proj sums thousands of these with
    //    random-sign weights of rms 1/sqrt(K), so after the post norm they are noise of about the
    //    o difference's relative size:   rms(err) <= 2^-6 rms(ref),
    //    |err| <= 2^-7 |ref| + 2^-4 rms(ref)  (2 bf16 ulps + ~4 sigma of that noise).
    //  * MoE half from the GPU's h2 / x1: out. Same rounding points, all discrete decisions
    //    admissible, so only fp32 / truncating-WMMA accumulation and isolated boundary flips
    //    remain: rms(err) <= 2^-8 rms(ref), |err| <= 2^-7 (|ref| + |beta an|) + 2^-6 rms(ref),
    //    beta an = ref - alpha x1 the bf16-rounded normed MoE term inside the residual gate
    //    (one ulp flip of each of the two rounded terms, as 59's r_slack range).
    //  * the independent chain's out is printed above for information only: with x1 already
    //    ~0.3 % off (attention precision), the two MXFP8 quantisations in the MoE turn that into
    //    percent-level differences (flip probability ~ |dx| / e4m3 step, step up to 2^-3).
    //  A wrong buffer, shard, stage order or missing partial gives O(1) relative errors.
    Check cx1("e2e x1 (from x)"), cout("e2e out (from GPU h2, x1)");
    Check cx1r("e2e x1 row rms err/rms"), coutr("e2e out row rms err/rms");
    for(int i = 0; i < M; ++i)
    {
        const double r1 = rms_of(&x1[size_t(i) * H], H), r2 = rms_of(&syn.out[size_t(i) * H], H);
        double e1 = 0, e2 = 0;
        for(int n = 0; n < H; ++n)
        {
            const size_t e  = size_t(i) * H + n;
            const double g1 = bf16_float(g0.x1[e]), g2 = bf16_float(g0.out[e]);
            cx1(g1, x1[e], std::ldexp(1.0, -7) * std::fabs(x1[e]) + std::ldexp(r1, -4), long(e));
            // beta * bf16(an) = out - alpha * x1: one ulp of it and one of out (2^-8 each,
            // doubled at a binade edge) + the accumulation noise.
            const double an = std::fabs(syn.out[e] - m.a_ffn[n] * gx1v[e]);
            cout(g2,
                 syn.out[e],
                 std::ldexp(1.0, -7) * (std::fabs(syn.out[e]) + an) + std::ldexp(r2, -6),
                 long(e));
            e1 += (g1 - x1[e]) * (g1 - x1[e]);
            e2 += (g2 - syn.out[e]) * (g2 - syn.out[e]);
        }
        cx1r(std::sqrt(e1 / H) / r1, 0, std::ldexp(1.0, -6), i);
        coutr(std::sqrt(e2 / H) / r2, 0, std::ldexp(1.0, -8), i);
    }
    fails += cx1r.report() + coutr.report();
    fails += cx1.report() + cout.report();
    return fails;
}

} // namespace layer67
