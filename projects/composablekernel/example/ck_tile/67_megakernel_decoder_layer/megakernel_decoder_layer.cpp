// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// One decoder layer of the target model as ONE persistent ck_tile::Megakernel launch per TP rank
// (MI450 / gfx1250), chaining the phases of examples 58-66, checked against a host reference.
//
// Stages (dependency; sharding over the TP world W = 1 or 2, one process driving W devices with
// peer access, as example 65). All stages run 4 waves (128 threads): the attention phase is fixed
// at 4 waves, so every other phase is instantiated with the same wave count.
//   0  in_norm        CenteredRMSNorm(x, input_layernorm)               none        replicated
//   1  qkvg           h @ W_qkvg^T, bf16                                 after_all 0 N: rank's
//   heads
//                     (rank's columns [q QH/W | k KVH/W | v KVH/W | gate QH/W] x 64)
//   2  qkv_rowops     QK-norm, x qk_scale, RoPE (sliding), paged KV write after_all 1 rank's heads
//   3  attention      split-KV decode attention + combine + o_norm +     after_all 2 rank's heads
//                     sigmoid gate (AttentionOutputNormGate)
//   4  o_proj         o @ W_o[:, rank's heads]^T -> fp32 partial         after_all 3 K-sharded
//   5  ar_attn        all-reduce (Lamport, fp32) fused with               after_all 4 all-reduce
//                     post_attention RMSNorm + residual gate -> x1 and pre_feedforward
//                     CenteredRMSNorm -> h2 (MkResidualNormPhase as the row epilogue)
//   6  router_gemm    fp32 logits h2 @ W_r^T * 0.1768 (all experts)      after_all 5 replicated
//   7  router_topk    sigmoid + bias top-8 -> moe_route_table            after_all 6 replicated
//   8  pooler         h2 @ W_pool^T -> p[:, rank's P/W columns]          after_all 7 N-sharded
//   9  ag_p           all-gather p (bf16 sentinel codec, in place)       after_all 8 all-gather
//   10 pre_exp_quant  pre_expert CenteredRMSNorm + MXFP8 quant           after_rows 9 replicated
//   11 moe_gate_up    gate/up + SwiGLU + MXFP8 requant, z[:, rank's I/W] after_all 10 N-sharded
//   12 ag_z           all-gather z (fp8 bytes, LL codec, in place)       after_all 11 all-gather
//   13 ag_zs          all-gather z scales (e8m0 bytes, LL codec)         after_rows 12 all-gather
//   14 moe_down       y[:, rank's P/W] = z @ down^T (full K)             after_all 13 N-sharded
//   15 ag_y           all-gather y (bf16 sentinel, in place)             after_all 14 all-gather
//   16 moe_combine    post_expert InputScaledRMSNorm + weighted sum      after_all 15 replicated
//   17 unpooler       acc[:, rank's P/W] @ W_unpool[:, rank's P/W]^T     after_all 16 K-sharded
//                     -> fp32 partial
//   18 ar_ffn         all-reduce fused with post_feedforward              after_all 17 all-reduce
//                     InputScaledRMSNorm + residual gate -> layer output
// A stage waits on one earlier stage; inputs from further back are covered transitively (e.g. the
// pooler waits on the router top-8, which waited on the router GEMM, which waited on ar_attn, so
// both h2 and the route table are visible to the MoE stages).
//
// Checks: every launch replays without a reset (counters, exchanges, tickets, KV cache) and must
// reproduce launch 0 bit for bit; end-to-end host reference of the whole layer; with
// -check_stages=1 every stage against a reference fed with the GPU's own input to that stage.
// Exit code: 0 pass, 1 wrong result, 2 unsupported, 3 error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/host/arg_parser.hpp"

#include "decoder_layer_common.hpp"
#include "../59_megakernel_epilogues/epilogue_reference.hpp"

namespace layer67 {

int check_layer(const LayerDims& d, const HostModel& m, const DeviceRun& run, bool check_stages);

#define LAYER67_DECLARE(s, w, f, mm)          \
    int run_layer_s##s##_w##w##_f##f##_m##mm( \
        const LayerDims&, const HostModel&, const RunOptions&, DeviceRun&);
#define LAYER67_FOR_ALL(X)                                                                        \
    X(0, 1, 0, 16)                                                                                \
    X(0, 1, 1, 16) X(0, 2, 0, 16) X(0, 2, 1, 16) X(1, 1, 0, 16) X(1, 1, 1, 16) X(1, 2, 0, 16)     \
        X(1, 2, 1, 16) X(0, 1, 0, 64) X(0, 1, 1, 64) X(0, 2, 0, 64) X(0, 2, 1, 64) X(1, 1, 0, 64) \
            X(1, 1, 1, 64) X(1, 2, 0, 64) X(1, 2, 1, 64)
LAYER67_FOR_ALL(LAYER67_DECLARE)

RunFn find_instance(bool small, int world, bool fp8, int max_m)
{
#define LAYER67_MATCH(s, w, f, mm)                                      \
    if(small == bool(s) && world == w && fp8 == bool(f) && max_m == mm) \
        return &run_layer_s##s##_w##w##_f##f##_m##mm;
    LAYER67_FOR_ALL(LAYER67_MATCH)
#undef LAYER67_MATCH
    return nullptr;
}

namespace {

// ---- fast parallel random fill (splitmix64 per 64 Ki chunk, Box-Muller normals)
uint64_t splitmix(uint64_t& s)
{
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
double unif(uint64_t& s) { return (double(splitmix(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

template <typename F>
void fill_chunks(size_t n, uint64_t seed, F&& f) // f(i, state)
{
    constexpr size_t kChunk = 1 << 16;
    parallel_for((n + kChunk - 1) / kChunk, [&](size_t c) {
        uint64_t s = seed * 0x632BE59BD9B4E019ull + c * 0x9E3779B97F4A7C15ull + 1;
        for(size_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i)
            f(i, s);
    });
}
double normal(uint64_t& s)
{
    return std::sqrt(-2.0 * std::log(unif(s))) * std::cos(6.283185307179586 * unif(s));
}
std::vector<uint16_t> rand_bf16(size_t n, double sigma, uint64_t seed)
{
    std::vector<uint16_t> v(n);
    fill_chunks(
        n, seed, [&](size_t i, uint64_t& s) { v[i] = bf16_bits(float(sigma * normal(s))); });
    return v;
}
std::vector<float> rand_f32(size_t n, double sigma, uint64_t seed)
{
    std::vector<float> v(n);
    fill_chunks(n, seed, [&](size_t i, uint64_t& s) { v[i] = float(sigma * normal(s)); });
    return v;
}
/// MXFP4 matrix of `rows` x K as example 62 (random e2m1 bytes, e8m0 scales 119 / 120: rms about
/// 1 / 64, i.e. 1 / sqrt(4096) for K = 4096).
struct Mxfp4
{
    std::vector<uint8_t> q, s;
};
Mxfp4 rand_mxfp4(size_t rows, int K, uint64_t seed)
{
    Mxfp4 r;
    r.q.resize(rows * (K / 2));
    r.s.resize(rows * (K / 32));
    std::vector<uint8_t>& q  = r.q;
    std::vector<uint8_t>& sc = r.s;
    fill_chunks(q.size(), seed, [&](size_t i, uint64_t& s) { q[i] = uint8_t(splitmix(s)); });
    fill_chunks(sc.size(), seed + 1, [&](size_t i, uint64_t& s) {
        sc[i] = uint8_t(119 + (splitmix(s) & 1));
    });
    return r;
}

/// a and b already sized like g.
void gate_coeffs(const std::vector<float>& g, std::vector<float>& a, std::vector<float>& b)
{
    for(size_t i = 0; i < g.size(); ++i)
        mk_ref::residual_gate_coeff(g[i], a[i], b[i]);
}

HostModel make_model(const LayerDims& d, uint64_t seed)
{
    const int H = d.s.H, P = d.s.P, I = d.s.I, E = d.s.E, QH = d.s.QH, KVH = d.s.KVH;
    HostModel m;
    uint64_t k = seed * 1000;
    m.w_in     = rand_bf16(H, 0.1, ++k);
    m.w_qkvg   = rand_bf16(size_t(d.NQkvg()) * H, 1.0 / std::sqrt(double(H)), ++k);
    m.w_o      = rand_bf16(size_t(H) * QH * 64, 1.0 / std::sqrt(double(QH * 64)), ++k);
    m.g_attn   = rand_f32(H, 1.0, ++k);
    m.w_preff  = rand_bf16(H, 0.1, ++k);
    // Router: scaled logits with a std of about 2 (full) so the top-8 is well separated.
    m.w_router    = rand_f32(size_t(E) * H, 0.125, ++k);
    m.router_bias = rand_f32(E, 0.05, ++k);
    m.w_pool      = rand_bf16(size_t(P) * H, 1.0 / std::sqrt(double(H)), ++k);
    m.w_preexp    = rand_bf16(P, 0.1, ++k);
    {
        Mxfp4 g = rand_mxfp4(size_t(E) * I, P, ++k), u = rand_mxfp4(size_t(E) * I, P, k += 2),
              dn = rand_mxfp4(size_t(E) * P, I, k += 2);
        m.gate_q = std::move(g.q);
        m.gate_s = std::move(g.s);
        m.up_q   = std::move(u.q);
        m.up_s   = std::move(u.s);
        m.down_q = std::move(dn.q);
        m.down_s = std::move(dn.s);
    }
    m.w_postexp = rand_bf16(P, 0.1, ++k);
    m.w_unpool  = rand_bf16(size_t(H) * P, 1.0 / std::sqrt(double(P)), ++k);
    m.w_postff  = rand_bf16(H, 0.1, ++k);
    m.g_ffn     = rand_f32(H, 1.0, ++k);
    m.a_attn.resize(H);
    m.b_attn.resize(H);
    m.a_ffn.resize(H);
    m.b_ffn.resize(H);
    gate_coeffs(m.g_attn, m.a_attn, m.b_attn);
    gate_coeffs(m.g_ffn, m.a_ffn, m.b_ffn);

    m.x = rand_bf16(size_t(d.M) * H, 1.0, ++k);

    // Context K (RMS-normed per head, as QK-norm leaves it; |k| <= 8) and V (unit normal).
    // fp8 cache: per-tensor scales 8 / 448, so neither saturates.
    m.k_scale      = d.fp8 ? 8.f / 448.f : 1.f;
    m.v_scale      = d.fp8 ? 8.f / 448.f : 1.f;
    const size_t n = size_t(d.ctx) * KVH * 64;
    m.k_ctx        = rand_f32(n, 1.0, ++k);
    m.v_ctx        = rand_f32(n, 1.0, ++k);
    const int es   = d.fp8 ? 1 : 2;
    m.k_ctx_raw.resize(n * es);
    m.v_ctx_raw.resize(n * es);
    parallel_for(n / 64, [&](size_t hrow) {
        float* kk = &m.k_ctx[hrow * 64];
        double ss = 0;
        for(int i = 0; i < 64; ++i)
            ss += double(kk[i]) * kk[i];
        const double r = 1.0 / std::sqrt(ss / 64 + 1e-5);
        for(int i = 0; i < 64; ++i)
        {
            float& kv      = kk[i];
            float& vv      = m.v_ctx[hrow * 64 + i];
            kv             = round_bf16(float(kv * r));
            vv             = round_bf16(vv);
            const size_t e = hrow * 64 + i;
            if(d.fp8)
            {
                m.k_ctx_raw[e] = mk_ref::fp8_encode(kv / m.k_scale);
                m.v_ctx_raw[e] = mk_ref::fp8_encode(vv / m.v_scale);
                kv             = mk_ref::fp8_decode(m.k_ctx_raw[e]) * m.k_scale;
                vv             = mk_ref::fp8_decode(m.v_ctx_raw[e]) * m.v_scale;
            }
            else
            {
                const uint16_t kb = bf16_bits(kv), vb = bf16_bits(vv);
                std::memcpy(&m.k_ctx_raw[2 * e], &kb, 2);
                std::memcpy(&m.v_ctx_raw[2 * e], &vb, 2);
            }
        }
    });

    // Pages: a random permutation; the request owns pages_needed of them.
    std::vector<int32_t> perm(d.num_pages());
    std::iota(perm.begin(), perm.end(), 0);
    uint64_t s = seed + 77;
    for(size_t i = perm.size() - 1; i > 0; --i)
        std::swap(perm[i], perm[splitmix(s) % (i + 1)]);
    m.block_table.assign(perm.begin(), perm.begin() + d.pages_needed());
    for(int i = 0; i < d.M; ++i)
    {
        const int pos = d.ctx + i;
        m.positions.push_back(pos);
        m.slot_mapping.push_back(m.block_table[pos / kPage] * kPage + pos % kPage);
    }
    return m;
}

} // namespace
} // namespace layer67

int main(int argc, char* argv[])
{
    using namespace layer67;
    ck_tile::ArgParser arg;
    arg.insert("world", "1", "TP ranks (1 or 2), one device each, peer access (example 65)")
        .insert("m", "16", "rows (verify tokens): 8, 16 or 64")
        .insert("layer_type", "sliding", "sliding (window 2048 + RoPE) | global (NoPE)")
        .insert("ctx", "4096", "KV positions already in the cache (random)")
        .insert("kv", "bf16", "KV cache type: bf16 | fp8")
        .insert("small", "0", "1: reduced model (H 2048, P 2048, I 1024, 128 experts, 32 q heads)")
        .insert("launches", "3", "validated launches, replayed with no reset in between")
        .insert("repeat", "20", "timed back-to-back launches (0: no timing)")
        .insert("grid", "0", "workgroups per device; 0 = resident maximum")
        .insert("seed", "1", "random seed")
        .insert("check_stages", "0", "1: also check every stage from the GPU's own inputs")
        .insert("poison", "1", "NaN-fill activations before each validated launch")
        .insert("spin_limit", "16777216", "polls before an exchange wait traps (0 = unbounded)");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        LayerDims d;
        d.small   = arg.get_int("small") != 0;
        d.s       = model_shape(d.small);
        d.world   = arg.get_int("world");
        d.M       = arg.get_int("m");
        d.ctx     = arg.get_int("ctx");
        d.sliding = arg.get_str("layer_type") != "global";
        d.fp8     = arg.get_str("kv") == "fp8";
        d.max_m   = d.M <= 16 ? 16 : 64;
        if(d.world < 1 || d.world > 2 || d.M < 1 || d.M > 64 || d.ctx < 0)
        {
            std::printf("unsupported: -world 1|2, -m 1..64, -ctx >= 0\n");
            return 2;
        }
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess || count < d.world)
        {
            std::printf("need %d visible GPU(s), have %d\n", d.world, count);
            return 2;
        }
        for(int i = 0; i < d.world; ++i)
            for(int j = 0; j < d.world; ++j)
                if(i != j)
                {
                    int can = 0;
                    (void)hipDeviceCanAccessPeer(&can, i, j);
                    if(!can)
                    {
                        std::printf("GPU %d cannot access GPU %d\n", i, j);
                        return 2;
                    }
                    (void)hipSetDevice(i);
                    const hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                    if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                        throw std::runtime_error("hipDeviceEnablePeerAccess failed");
                }
        const RunFn fn = find_instance(d.small, d.world, d.fp8, d.max_m);
        if(fn == nullptr)
        {
            std::printf("no instance for this configuration\n");
            return 2;
        }
        std::printf(
            "Decoder layer (%s), %s, world %d, M %d, ctx %d, KV %s: H %d P %d "
            "I %d E %d q/kv heads %d/%d (%d/%d per rank)\n",
            d.small ? "small" : "full",
            d.sliding ? "sliding window 2048 + RoPE" : "global (NoPE)",
            d.world,
            d.M,
            d.ctx,
            d.fp8 ? "fp8" : "bf16",
            d.s.H,
            d.s.P,
            d.s.I,
            d.s.E,
            d.s.QH,
            d.s.KVH,
            d.QHr(),
            d.KVHr());
        auto t0           = std::chrono::steady_clock::now();
        const HostModel m = make_model(d, arg.get_int("seed"));
        std::printf("host model generated in %.1f s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

        RunOptions o;
        o.launches = std::max(1, arg.get_int("launches"));
        o.repeat   = arg.get_int("repeat");
        o.grid     = arg.get_int("grid");
        o.spin     = arg.get_uint32("spin_limit");
        o.poison   = arg.get_int("poison") != 0;
        DeviceRun run;
        const int rc = fn(d, m, o, run);
        if(rc != 0)
            return rc;
        const RunInfo& in = run.info;
        std::printf(
            "megakernel: %d stages, grid %d (resident %d), 128 threads, LDS %d B, max work %d\n",
            in.stages,
            in.grid,
            in.resident,
            in.lds,
            in.max_work);
        std::printf("splits: qkvg %d, o_proj %d, pooler %d, unpooler %d, router %d, attention %d\n",
                    in.qkvg_splits,
                    in.o_splits,
                    in.pool_splits,
                    in.unpool_splits,
                    in.router_splits,
                    in.attn_splits);
        if(o.repeat > 0)
            std::printf(
                "time: %.1f us per launch (one layer, %d back-to-back launches, max over ranks)\n",
                in.us_per_launch,
                o.repeat);
        std::printf(
            "replay: %d mismatches over %d validated + %d timed launches; epoch errors %d\n",
            in.replay_mismatches,
            o.launches,
            std::max(0, o.repeat),
            in.epoch_errors);

        t0        = std::chrono::steady_clock::now();
        int fails = check_layer(d, m, run, arg.get_int("check_stages") != 0);
        std::printf("host checks took %.1f s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        fails += in.replay_mismatches + in.epoch_errors;
        std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
        return fails == 0 ? 0 : 1;
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
