// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-side types shared by the device runner (decoder_layer_device.hpp, one instantiation per
// translation unit) and the host reference (decoder_layer_ref.cpp): model shape, the full
// (unsharded) weights and inputs of one decoder layer, and what a run brings back per rank.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace layer67 {

/// Shape of the layer. Full: the target model's shapes. Small (-small=1): the same structure with fewer
/// heads, a narrower hidden size and fewer, narrower experts, for fast debugging.
struct ModelShape
{
    int H;   // hidden
    int P;   // moe_hidden (pooler output)
    int I;   // expert intermediate
    int E;   // experts
    int QH;  // q heads (= gate heads)
    int KVH; // kv heads (GQA 8)
};

constexpr ModelShape model_shape(bool small)
{
    return small ? ModelShape{2048, 2048, 1024, 128, 32, 4}
                 : ModelShape{8192, 4096, 4096, 256, 128, 16};
}

constexpr int kTopK     = 8;
constexpr int kHeadDim  = 64;
constexpr int kPage     = 16;
constexpr int kWindow   = 2048;
constexpr int kMaxWorld = 2;

struct LayerDims
{
    ModelShape s{};
    bool small   = false;
    int world    = 1;
    int M        = 16;
    int ctx      = 4096; // KV positions already in the cache
    bool sliding = true; // sliding window 2048 + RoPE; false: global (NoPE)
    bool fp8     = false;
    int max_m    = 16; // kMaxM of the instantiation

    int QHr() const { return s.QH / world; }
    int KVHr() const { return s.KVH / world; }
    int NQkvg() const { return (2 * s.QH + 2 * s.KVH) * kHeadDim; } // full fused N
    int NQkvgR() const { return NQkvg() / world; }                  // per rank
    int kv_len() const { return ctx + M; }
    int pages_needed() const { return (kv_len() + kPage - 1) / kPage; }
    int num_pages() const { return pages_needed() + 3; } // + spare pages nobody owns
    int window() const { return sliding ? kWindow : 0; }
};

/// Full weights (global head order) and inputs. bf16 tensors are raw bits.
struct HostModel
{
    // weights
    std::vector<uint16_t> w_in; // input_layernorm [H]
    std::vector<uint16_t>
        w_qkvg;                // [NQkvg, H]: rows [q heads | k heads | v heads | gate heads] x 64
    std::vector<uint16_t> w_o; // o_proj [H, QH * 64]
    std::vector<float> g_attn; // residual gate (attention) [H], fp32
    std::vector<float> a_attn, b_attn;               // its alpha / beta (mk_residual_gate_coeffs)
    std::vector<uint16_t> w_preff;                   // pre_feedforward_layernorm [H]
    std::vector<float> w_router;                     // [E, H] fp32
    std::vector<float> router_bias;                  // e_score_correction_bias [E]
    std::vector<uint16_t> w_pool;                    // pre_expert_proj [P, H]
    std::vector<uint16_t> w_preexp;                  // pre_expert_norm [P]
    std::vector<uint8_t> gate_q, gate_s, up_q, up_s; // [E * I, P / 2], [E * I, P / 32]
    std::vector<uint8_t> down_q, down_s;             // [E * P, I / 2], [E * P, I / 32]
    std::vector<uint16_t> w_postexp;                 // post_expert_norm [P]
    std::vector<uint16_t> w_unpool;                  // post_expert_proj [H, P]
    std::vector<uint16_t> w_postff;                  // post_feedforward_layernorm [H]
    std::vector<float> g_ffn, a_ffn, b_ffn;
    // inputs
    std::vector<uint16_t> x; // [M, H]
    std::vector<float> k_ctx,
        v_ctx; // [ctx][KVH][64] logical context K / V (as stored: dequantised)
    std::vector<uint8_t> k_ctx_raw, v_ctx_raw; // [ctx][KVH][64] stored codes (bf16: 2 bytes each)
    float k_scale = 1.f, v_scale = 1.f;
    std::vector<int32_t> block_table;             // [pages_needed]
    std::vector<int32_t> slot_mapping, positions; // [M]
};

/// Everything a run brings back from one rank (after the last validated launch).
struct RankDump
{
    std::vector<uint16_t> h, qkvg, q, o, x1, h2, p, pn, y, acc, out;
    std::vector<uint8_t> kcache, vcache, pq, ps, z, zs;
    std::vector<uint8_t> kcache_init, vcache_init; // what the host uploaded
    std::vector<float> opart, upart, logits, topk_w;
    std::vector<int32_t> topk_ids, count, routes, active, num_active;
};

struct RunOptions
{
    int launches  = 3;
    int repeat    = 20;
    int grid      = 0;
    uint32_t spin = 1u << 24;
    bool poison   = true; // NaN-fill the activations before each validated launch
};

struct RunInfo
{
    int grid = 0, resident = 0, lds = 0, stages = 0, max_work = 0;
    int qkvg_splits = 0, o_splits = 0, pool_splits = 0, unpool_splits = 0, router_splits = 0,
        attn_splits       = 0;
    double us_per_launch  = 0;
    int replay_mismatches = 0;
    int epoch_errors      = 0;
    std::string kernel_name;
};

struct DeviceRun
{
    std::vector<RankDump> ranks;
    RunInfo info;
};

using RunFn = int (*)(const LayerDims&, const HostModel&, const RunOptions&, DeviceRun&);

/// Instances (one translation unit each, generated by CMake); null when not built.
RunFn find_instance(bool small, int world, bool fp8, int max_m);

// ---------------------------------------------------------------------------------------------
// Small host helpers

inline uint16_t bf16_bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if((u & 0x7FFFFFFFu) > 0x7F800000u)
        return static_cast<uint16_t>((u >> 16) | 0x40u);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_float(uint16_t b)
{
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline float round_bf16(float f) { return bf16_float(bf16_bits(f)); }

/// Runs f(i) for i in [0, n) on all host threads (dynamic chunks of one index).
template <typename F>
void parallel_for(size_t n, F&& f)
{
    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t nt = std::min<size_t>(std::min<size_t>(hw, 256), n);
    if(nt <= 1)
    {
        for(size_t i = 0; i < n; ++i)
            f(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> ts;
    for(size_t t = 0; t < nt; ++t)
        ts.emplace_back([&]() {
            for(size_t i; (i = next.fetch_add(1)) < n;)
                f(i);
        });
    for(auto& t : ts)
        t.join();
}

} // namespace layer67
