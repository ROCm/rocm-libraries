// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// One decoder layer of the target model as ONE persistent ck_tile::Megakernel launch per rank, built
// from the phases of examples 58-66. See the stage table in megakernel_decoder_layer.cpp.
// Included by the generated instance translation units with LAYER67_SMALL, LAYER67_WORLD,
// LAYER67_FP8, LAYER67_MAXM defined.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/megakernel.hpp"

#include "decoder_layer_common.hpp"

namespace layer67 {

using index_t      = ck_tile::index_t;
using long_index_t = ck_tile::long_index_t;
using bf16_t       = ck_tile::bf16_t;
using fp8_t        = ck_tile::fp8_t;

#define L67_HIP_CHECK(cmd)                                                                     \
    do                                                                                         \
    {                                                                                          \
        const hipError_t e_ = (cmd);                                                           \
        if(e_ != hipSuccess)                                                                   \
            throw std::runtime_error(std::string(#cmd) + ": " + hipGetErrorString(e_) + " (" + \
                                     __FILE__ + ":" + std::to_string(__LINE__) + ")");         \
    } while(0)

struct DevBuf
{
    int dev      = 0;
    void* p      = nullptr;
    size_t bytes = 0;
    DevBuf()     = default;
    DevBuf(int d, size_t n, bool uncached = false) { alloc(d, n, uncached); }
    void alloc(int d, size_t n, bool uncached = false)
    {
        dev   = d;
        bytes = std::max<size_t>(n, 16);
        L67_HIP_CHECK(hipSetDevice(dev));
        if(uncached)
            L67_HIP_CHECK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
        else
            L67_HIP_CHECK(hipMalloc(&p, bytes));
    }
    ~DevBuf()
    {
        if(p != nullptr)
        {
            (void)hipSetDevice(dev);
            (void)hipFree(p);
        }
    }
    DevBuf(const DevBuf&)            = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : dev(o.dev), p(o.p), bytes(o.bytes) { o.p = nullptr; }
    DevBuf& operator=(DevBuf&& o) noexcept
    {
        std::swap(dev, o.dev);
        std::swap(p, o.p);
        std::swap(bytes, o.bytes);
        return *this;
    }

    template <typename T>
    T* as() const
    {
        return static_cast<T*>(p);
    }
    void upload(const void* src, size_t n) const
    {
        L67_HIP_CHECK(hipSetDevice(dev));
        L67_HIP_CHECK(hipMemcpy(p, src, n, hipMemcpyHostToDevice));
    }
    template <typename T>
    void upload(const std::vector<T>& v) const
    {
        upload(v.data(), v.size() * sizeof(T));
    }
    template <typename T>
    std::vector<T> get(size_t n) const
    {
        std::vector<T> v(n);
        L67_HIP_CHECK(hipSetDevice(dev));
        L67_HIP_CHECK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
        return v;
    }
    void fill(uint8_t b, size_t n = 0) const
    {
        L67_HIP_CHECK(hipSetDevice(dev));
        L67_HIP_CHECK(hipMemset(p, b, n ? n : bytes));
    }
};

template <typename T>
DevBuf make_buf(int dev, const std::vector<T>& v)
{
    DevBuf b(dev, v.size() * sizeof(T));
    b.upload(v);
    return b;
}

/// One exchange of the xgpu phases: a buffer per rank (uncached, filled once with the empty
/// pattern), and per rank a device table of the peers' buffers (peer access is enabled up front).
struct Exchange
{
    std::vector<DevBuf> bufs, tables;
    uint32_t slot_bytes = 0, buffer_bytes = 0;

    Exchange(int world, uint32_t slot, uint32_t total, uint32_t fill_word)
        : slot_bytes(slot), buffer_bytes(total)
    {
        std::vector<void*> ptrs;
        for(int d = 0; d < world; ++d)
        {
            bufs.emplace_back(d, total, true);
            L67_HIP_CHECK(hipMemsetD32(
                static_cast<hipDeviceptr_t>(bufs.back().p), fill_word, total / sizeof(uint32_t)));
            ptrs.push_back(bufs.back().p);
        }
        ptrs.resize(kMaxWorld, nullptr);
        for(int d = 0; d < world; ++d)
        {
            tables.emplace_back(d, ptrs.size() * sizeof(void*));
            tables.back().upload(ptrs.data(), ptrs.size() * sizeof(void*));
        }
    }

    ck_tile::xgpu_exchange<kMaxWorld> view(int rank, const uint32_t* state, uint32_t spin) const
    {
        ck_tile::xgpu_exchange<kMaxWorld> x{};
        x.peers           = tables[static_cast<size_t>(rank)].as<void* const>();
        x.rank            = rank;
        x.world           = static_cast<index_t>(bufs.size());
        x.slot_bytes      = slot_bytes;
        x.buffer_bytes    = buffer_bytes;
        x.launch_state    = state;
        x.uses_per_launch = 1;
        x.use_index       = 0;
        x.spin_limit      = spin;
        return x;
    }
};

/// Split count of a SmallMGemmPhase (cost model of example 66's auto_splits).
inline index_t
gemm_splits(index_t n_blocks, index_t windows, index_t waves, index_t stages, index_t grid)
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

/// Device buffers of one rank.
struct Rank
{
    // weights
    DevBuf w_in, w_qkvg, w_o, a_attn, b_attn, w_preff, w_router, router_bias, w_pool, w_preexp,
        gate_q, gate_s, up_q, up_s, down_q, down_s, w_postexp, w_unpool, w_postff, a_ffn, b_ffn;
    // inputs
    DevBuf x, kcache, vcache, block_table, kv_lens, slot_mapping, positions, rope;
    // activations
    DevBuf h, qkvg, q, o, opart, x1, h2, logits, topk_ids, topk_w, count, routes, active,
        num_active, p, pn, pq, ps, z, zs, y, acc, upart, out;
    // scratch
    DevBuf gp[4], gt[4], attn_o, attn_lse, attn_cnt, rpart, rtick, counters;
    std::vector<uint8_t> kcache0, vcache0;
};

template <bool kSmall, int kWorld, bool kFp8, int kMaxM>
struct DecoderLayer
{
    static constexpr ModelShape S = model_shape(kSmall);
    static constexpr index_t H = S.H, P = S.P, I = S.I, E = S.E;
    static constexpr index_t QHr = S.QH / kWorld, KVHr = S.KVH / kWorld;
    static constexpr index_t NQkvgR = (2 * QHr + 2 * KVHr) * kHeadDim;

    // One workgroup size for every stage: the attention phase is fixed at 4 waves (128 Q rows,
    // one combine thread each), so all phases are instantiated with 4 waves = 128 threads on
    // gfx1250. kWaves is a wave count (both compile passes agree); every phase derives its
    // kBlockSize = kWaves * get_warp_size() itself, and the launch uses MK::BlockSize().
    static constexpr index_t kWaves   = 4;
    static constexpr index_t kThreads = kWaves * 32; // device threads (wave32); phase template arg
    static constexpr index_t kStagesG = kMaxM <= 16 ? 4 : 3;

    using KV  = std::conditional_t<kFp8, fp8_t, bf16_t>;
    using Epi = ck_tile::small_m::EpiPassThrough;

    using InNorm   = ck_tile::MkNormQuantPhase<H, kWaves, ck_tile::MkActQuant::kNone>;
    using GemmBf16 = ck_tile::SmallMGemmPhase<kMaxM, 2, kStagesG, bf16_t, Epi, true, kWaves>;
    using GemmF32  = ck_tile::SmallMGemmPhase<kMaxM, 2, kStagesG, float, Epi, true, kWaves>;
    using QkvOps   = ck_tile::MkQkvRowOpsPhase<kWaves, KV, kPage, QHr, KVHr>;
    using Attn = ck_tile::DecodeAttentionPhase<KV, ck_tile::AttentionOutputNormGate, kPage, false>;
    using PostAttn =
        ck_tile::MkResidualNormPhase<H, kWaves, ck_tile::MkFirstNorm::kRms, float, kMaxWorld>;
    using ArAttn = ck_tile::
        XgpuAllReducePhase<float, kWaves, ck_tile::xgpu_row_phase_epilogue<PostAttn>, kMaxWorld>;
    using RouterG = ck_tile::RouterGemmPhase<kThreads, kMaxM, 2, 6>;
    using RouterF = ck_tile::RouterFinalizePhase<kThreads, kMaxM, E, kTopK>;
    using AgBf16 =
        ck_tile::XgpuAllGatherPhase<ck_tile::xgpu_sentinel_codec<bf16_t>, kWaves, kMaxWorld>;
    using AgBytes = ck_tile::XgpuAllGatherPhase<ck_tile::xgpu_ll_codec, kWaves, kMaxWorld>;
    using PreExp  = ck_tile::MkNormQuantPhase<P, kWaves, ck_tile::MkActQuant::kMxFp8>;
    using GateUp  = ck_tile::MoeGateUpPhase<fp8_t, P, 32, 2, kThreads>;
    using Down    = ck_tile::MoeDownPhase<fp8_t, I, 32, 2, kThreads>;
    using Combine = ck_tile::MoeCombinePhase<bf16_t, P, kTopK, kThreads>;
    using PostFf  = ck_tile::
        MkResidualNormPhase<H, kWaves, ck_tile::MkFirstNorm::kInputScaled, float, kMaxWorld>;
    using ArFfn = ck_tile::
        XgpuAllReducePhase<float, kWaves, ck_tile::xgpu_row_phase_epilogue<PostFf>, kMaxWorld>;

    template <typename Ph, typename Dep = ck_tile::no_dependency>
    using St = ck_tile::megakernel_stage<Ph, Dep>;
    template <index_t I_>
    using All = ck_tile::after_all<I_>;
    template <index_t I_>
    using Rows = ck_tile::after_rows<I_>;

    // Stage list (index: phase, dependency). A stage has one dependency; inputs produced further
    // back are covered transitively (each producer item itself waited on its own producer
    // before it signalled; release / acquire at agent scope is transitive).
    using MK = ck_tile::Megakernel<St<InNorm>,           //  0 input CenteredRMSNorm
                                   St<GemmBf16, All<0>>, //  1 qkvg GEMM (rank's heads)
                                   St<QkvOps, All<1>>,   //  2 QK-norm, RoPE, KV write
                                   St<Attn, All<2>>,     //  3 split-KV attention + o_norm + gate
                                   St<GemmF32, All<3>>,  //  4 o_proj, K-sharded, fp32 partial
                                   St<ArAttn, All<4>>,   //  5 all-reduce + post_attn norm + gate +
                                                         //  pre_ff norm
                                   St<RouterG, All<5>>,  //  6 router logits (replicated)
                                   St<RouterF, All<6>>,  //  7 sigmoid + bias top-8 -> route table
                                   St<GemmBf16, All<7>>, //  8 pooler, N-sharded
                                   St<AgBf16, All<8>>,   //  9 all-gather p
                                   St<PreExp, Rows<9>>,  // 10 pre_expert_norm + MXFP8 quant
                                   St<GateUp, All<10>>,  // 11 gate/up + SwiGLU + requant, N-sharded
                                   St<AgBytes, All<11>>, // 12 all-gather z (fp8)
                                   St<AgBytes, Rows<12>>, // 13 all-gather z scales (e8m0)
                                   St<Down, All<13>>,     // 14 down, N-sharded, full K
                                   St<AgBf16, All<14>>,   // 15 all-gather y
                                   St<Combine, All<15>>,  // 16 post_expert_norm + weighted sum
                                   St<GemmF32, All<16>>,  // 17 unpooler, K-sharded, fp32 partial
                                   St<ArFfn, All<17>>>;   // 18 all-reduce + post_ff norm + gate

    static constexpr const char* kStageNames[] = {"in_norm",
                                                  "qkvg",
                                                  "qkv_rowops",
                                                  "attention",
                                                  "o_proj",
                                                  "ar_attn+norms",
                                                  "router_gemm",
                                                  "router_topk",
                                                  "pooler",
                                                  "ag_p",
                                                  "pre_exp_quant",
                                                  "moe_gate_up",
                                                  "ag_z",
                                                  "ag_zs",
                                                  "moe_down",
                                                  "ag_y",
                                                  "moe_combine",
                                                  "unpooler",
                                                  "ar_ffn+norms"};

    static int run(const LayerDims& d, const HostModel& m, const RunOptions& opt, DeviceRun& out)
    {
        static_assert(model_shape(kSmall).QH % kWorld == 0, "heads per rank");
        if(d.world != kWorld || d.small != kSmall || d.fp8 != kFp8 || d.M > kMaxM)
            throw std::runtime_error("instance does not match the options");
        const index_t M       = d.M;
        const index_t R       = M * kTopK; // routes
        const int W           = kWorld;
        const size_t kv_elems = size_t(d.num_pages()) * KVHr * kPage * kHeadDim;
        const size_t kv_bytes = kv_elems * sizeof(KV);

        std::vector<Rank> rk(W);
        const auto rope = ck_tile::mk_make_rope_table(d.kv_len());
        for(int r = 0; r < W; ++r)
        {
            Rank& b = rk[r];
            // qkvg shard: this rank's q / k / v / gate heads, in that order.
            std::vector<uint16_t> qkvg_w(size_t(NQkvgR) * H);
            {
                const index_t gk = S.QH * kHeadDim, gv = gk + S.KVH * kHeadDim,
                              gg = gv + S.KVH * kHeadDim;
                index_t dst      = 0;
                auto take        = [&](index_t row0, index_t rows) {
                    std::copy(m.w_qkvg.begin() + size_t(row0) * H,
                              m.w_qkvg.begin() + size_t(row0 + rows) * H,
                              qkvg_w.begin() + size_t(dst) * H);
                    dst += rows;
                };
                take(r * QHr * kHeadDim, QHr * kHeadDim);
                take(gk + r * KVHr * kHeadDim, KVHr * kHeadDim);
                take(gv + r * KVHr * kHeadDim, KVHr * kHeadDim);
                take(gg + r * QHr * kHeadDim, QHr * kHeadDim);
            }
            b.w_in        = make_buf(r, m.w_in);
            b.w_qkvg      = make_buf(r, qkvg_w);
            b.w_o         = make_buf(r, m.w_o);
            b.a_attn      = make_buf(r, m.a_attn);
            b.b_attn      = make_buf(r, m.b_attn);
            b.w_preff     = make_buf(r, m.w_preff);
            b.w_router    = make_buf(r, m.w_router);
            b.router_bias = make_buf(r, m.router_bias);
            b.w_pool      = make_buf(r, m.w_pool);
            b.w_preexp    = make_buf(r, m.w_preexp);
            b.gate_q      = make_buf(r, m.gate_q);
            b.gate_s      = make_buf(r, m.gate_s);
            b.up_q        = make_buf(r, m.up_q);
            b.up_s        = make_buf(r, m.up_s);
            b.down_q      = make_buf(r, m.down_q);
            b.down_s      = make_buf(r, m.down_s);
            b.w_postexp   = make_buf(r, m.w_postexp);
            b.w_unpool    = make_buf(r, m.w_unpool);
            b.w_postff    = make_buf(r, m.w_postff);
            b.a_ffn       = make_buf(r, m.a_ffn);
            b.b_ffn       = make_buf(r, m.b_ffn);

            b.x = make_buf(r, m.x);
            // Paged cache of this rank's kv heads: context positions from the host, the rest NaN
            // (0xFF bytes: bf16 and e4m3fn NaN) so an unmasked or stray read shows.
            b.kcache0.assign(kv_bytes, 0xFF);
            b.vcache0.assign(kv_bytes, 0xFF);
            for(index_t pos = 0; pos < d.ctx; ++pos)
                for(index_t g = 0; g < KVHr; ++g)
                {
                    const index_t page = m.block_table[pos / kPage];
                    const size_t dst =
                        ((size_t(page) * KVHr + g) * kPage + pos % kPage) * kHeadDim * sizeof(KV);
                    const size_t src = (size_t(pos) * S.KVH + r * KVHr + g) * kHeadDim * sizeof(KV);
                    std::memcpy(&b.kcache0[dst], &m.k_ctx_raw[src], kHeadDim * sizeof(KV));
                    std::memcpy(&b.vcache0[dst], &m.v_ctx_raw[src], kHeadDim * sizeof(KV));
                }
            b.kcache       = make_buf(r, b.kcache0);
            b.vcache       = make_buf(r, b.vcache0);
            b.block_table  = make_buf(r, m.block_table);
            b.kv_lens      = make_buf(r, std::vector<int32_t>{d.kv_len()});
            b.slot_mapping = make_buf(r, m.slot_mapping);
            b.positions    = make_buf(r, m.positions);
            b.rope         = make_buf(r, rope);

            b.h.alloc(r, size_t(M) * H * 2);
            b.qkvg.alloc(r, size_t(M) * NQkvgR * 2);
            b.q.alloc(r, size_t(M) * QHr * kHeadDim * 2);
            b.o.alloc(r, size_t(M) * QHr * kHeadDim * 2);
            b.opart.alloc(r, size_t(M) * H * 4);
            b.x1.alloc(r, size_t(M) * H * 2);
            b.h2.alloc(r, size_t(M) * H * 2);
            b.logits.alloc(r, size_t(M) * E * 4);
            b.topk_ids.alloc(r, size_t(R) * 4);
            b.topk_w.alloc(r, size_t(R) * 4);
            b.count.alloc(r, size_t(E) * 4);
            b.routes.alloc(r, size_t(E) * M * 4);
            b.active.alloc(r, size_t(E) * 4);
            b.num_active.alloc(r, 4);
            b.p.alloc(r, size_t(M) * P * 2);
            b.pn.alloc(r, size_t(M) * P * 2);
            b.pq.alloc(r, size_t(M) * P);
            b.ps.alloc(r, size_t(M) * (P / 32));
            b.z.alloc(r, size_t(R) * I);
            b.zs.alloc(r, size_t(R) * (I / 32));
            b.y.alloc(r, size_t(R) * P * 2);
            b.acc.alloc(r, size_t(M) * P * 2);
            b.upart.alloc(r, size_t(M) * H * 4);
            b.out.alloc(r, size_t(M) * H * 2);
        }

        // Exchanges: one per collective (each used once per launch).
        const uint32_t slot_ar = ArAttn::SlotBytes(M, H);
        Exchange ex_ar1(W, slot_ar, ArAttn::ExchangeBytes(W, slot_ar), ArAttn::kFillWord);
        Exchange ex_ar2(W, slot_ar, ArFfn::ExchangeBytes(W, slot_ar), ArFfn::kFillWord);
        const uint32_t slot_p = AgBf16::SlotBytes(M, P / W * 2);
        Exchange ex_p(W, slot_p, AgBf16::ExchangeBytes(W, slot_p), AgBf16::kFillWord);
        const uint32_t slot_z = AgBytes::SlotBytes(R, I / W);
        Exchange ex_z(W, slot_z, AgBytes::ExchangeBytes(W, slot_z), AgBytes::kFillWord);
        const uint32_t slot_zs = AgBytes::SlotBytes(R, I / W / 32);
        Exchange ex_zs(W, slot_zs, AgBytes::ExchangeBytes(W, slot_zs), AgBytes::kFillWord);
        const uint32_t slot_y = AgBf16::SlotBytes(R, P / W * 2);
        Exchange ex_y(W, slot_y, AgBf16::ExchangeBytes(W, slot_y), AgBf16::kFillWord);

        L67_HIP_CHECK(hipSetDevice(0));
        const index_t resident = MK::MaxResidentGridSize(ck_tile::stream_config{nullptr});
        const index_t grid     = opt.grid > 0 ? std::min<index_t>(opt.grid, resident) : resident;

        // Split counts (functions of the grid, same on every rank).
        const index_t qkvg_splits =
            gemm_splits(NQkvgR / GemmBf16::kNRows, H / 64, kWaves, kStagesG, grid);
        const index_t o_splits =
            gemm_splits(H / GemmF32::kNRows, QHr * kHeadDim / 64, kWaves, kStagesG, grid);
        const index_t pool_splits =
            gemm_splits(P / W / GemmBf16::kNRows, H / 64, kWaves, kStagesG, grid);
        const index_t unp_splits =
            gemm_splits(H / GemmF32::kNRows, P / W / 64, kWaves, kStagesG, grid);
        const index_t r_tiles = E / RouterG::kExpertsPerItem;
        const index_t r_splits =
            std::min<index_t>(std::max<index_t>(1, grid / r_tiles), H / RouterG::kKUnit);
        const index_t attn_tiles = ck_tile::integer_divide_ceil(M, Attn::kRowsPerHead) * KVHr;
        const index_t attn_splits =
            Attn::ChooseNumSplits(attn_tiles, Attn::MaxSpanKeys(d.kv_len(), M, d.window()), grid);

        auto gemm_args = [&](auto tag,
                             const void* x,
                             index_t sx,
                             const void* w,
                             index_t sw,
                             void* y,
                             index_t ldy,
                             index_t N,
                             index_t K,
                             index_t splits,
                             Rank& b,
                             int gi) {
            using G = typename decltype(tag)::type;
            typename G::Args a{};
            a.x                   = static_cast<const bf16_t*>(x);
            a.w                   = static_cast<const bf16_t*>(w);
            a.y                   = static_cast<typename G::OutType*>(y);
            a.M                   = M;
            a.N                   = N;
            a.K                   = K;
            a.stride_x            = sx;
            a.stride_w            = sw;
            a.ld_y                = ldy;
            a.k_splits            = splits;
            a.l2_prefetch_windows = 0;
            if(splits > 1)
            {
                if(b.gp[gi].p == nullptr)
                {
                    b.gp[gi].alloc(b.x.dev, size_t(G::PartialSize(a)) * 4);
                    b.gt[gi].alloc(b.x.dev, size_t(G::NumTickets(a)) * 4);
                    b.gt[gi].fill(0);
                }
                a.partial = b.gp[gi].template as<float>();
                a.tickets = b.gt[gi].template as<uint32_t>();
            }
            return a;
        };
        struct tag_bf16
        {
            using type = GemmBf16;
        };
        struct tag_f32
        {
            using type = GemmF32;
        };

        auto make_args = [&](int r, const uint32_t* state) {
            Rank& b = rk[r];
            // 0 input norm
            typename InNorm::Args a0{};
            a0.in        = b.x.as<bf16_t>();
            a0.in_stride = H;
            a0.w         = b.w_in.as<bf16_t>();
            a0.eps       = 1e-5f;
            a0.p_out     = b.h.as<bf16_t>();
            a0.p_stride  = H;
            a0.M         = M;
            // 1 qkvg
            auto a1 = gemm_args(tag_bf16{},
                                b.h.p,
                                H,
                                b.w_qkvg.p,
                                H,
                                b.qkvg.p,
                                NQkvgR,
                                NQkvgR,
                                H,
                                qkvg_splits,
                                b,
                                0);
            // 2 qkv row ops
            typename QkvOps::Args a2{};
            a2.qkvg         = b.qkvg.as<bf16_t>();
            a2.qkvg_stride  = NQkvgR;
            a2.q_out        = b.q.as<bf16_t>();
            a2.q_stride     = QHr * kHeadDim;
            a2.k_cache      = b.kcache.as<KV>();
            a2.v_cache      = b.vcache.as<KV>();
            a2.slot_mapping = b.slot_mapping.as<int32_t>();
            a2.positions    = b.positions.as<int32_t>();
            a2.rope_table   = b.rope.as<float>();
            a2.k_scale      = m.k_scale;
            a2.v_scale      = m.v_scale;
            a2.M            = M;
            a2.use_rope     = d.sliding ? 1 : 0;
            // 3 attention (+ o_norm + sigmoid gate)
            typename Attn::Args a3{};
            a3.q                  = b.q.as<bf16_t>();
            a3.stride_q           = QHr * kHeadDim;
            a3.k_cache            = b.kcache.as<KV>();
            a3.v_cache            = b.vcache.as<KV>();
            a3.head_stride        = kPage * kHeadDim;
            a3.page_stride        = long_index_t(KVHr) * kPage * kHeadDim;
            a3.token_stride       = kHeadDim;
            a3.block_table        = b.block_table.as<int32_t>();
            a3.block_table_stride = d.pages_needed();
            a3.kv_lens            = b.kv_lens.as<int32_t>();
            a3.num_requests       = 1;
            a3.q_len              = M;
            a3.num_kv_heads       = KVHr;
            a3.window             = d.window();
            a3.scale              = 0.125f;
            a3.k_scale            = m.k_scale;
            a3.v_scale            = m.v_scale;
            a3.num_splits         = attn_splits;
            a3.out                = {b.o.as<bf16_t>(),
                                     QHr * kHeadDim,
                                     b.qkvg.as<bf16_t>() + QkvOps::kGateCol,
                                     NQkvgR,
                                     1e-5f};
            if(b.attn_o.p == nullptr)
            {
                b.attn_o.alloc(r, size_t(Attn::PartialOFloats(a3)) * 4);
                b.attn_lse.alloc(r, size_t(Attn::PartialLseFloats(a3)) * 4);
                b.attn_cnt.alloc(r, size_t(Attn::NumTiles(a3)) * 4);
                b.attn_cnt.fill(0);
            }
            a3.o_partial      = b.attn_o.as<float>();
            a3.lse_partial    = b.attn_lse.as<float>();
            a3.split_counters = b.attn_cnt.as<uint32_t>();
            // 4 o_proj: this rank's K slice of every row of W_o
            auto a4 = gemm_args(tag_f32{},
                                b.o.p,
                                QHr * kHeadDim,
                                b.w_o.as<bf16_t>() + size_t(r) * QHr * kHeadDim,
                                S.QH * kHeadDim,
                                b.opart.p,
                                H,
                                H,
                                QHr * kHeadDim,
                                o_splits,
                                b,
                                1);
            // 5 all-reduce + post_attention_layernorm + residual gate + pre_feedforward norm
            typename PostAttn::Args n5{};
            n5.first_w         = nullptr;
            n5.first_eps       = 1e-8f;
            n5.gate_alpha      = b.a_attn.as<float>();
            n5.gate_beta       = b.b_attn.as<float>();
            n5.residual        = b.x.as<bf16_t>();
            n5.residual_stride = H;
            n5.x_out           = b.x1.as<bf16_t>();
            n5.x_stride        = H;
            n5.next_w          = b.w_preff.as<bf16_t>();
            n5.next_eps        = 1e-5f;
            n5.h_out           = b.h2.as<bf16_t>();
            n5.h_stride        = H;
            n5.M               = M;
            typename ArAttn::Args a5{
                ex_ar1.view(r, state, opt.spin), b.opart.as<float>(), H, M, H, n5};
            // 6 router GEMM, replicated (every expert on every rank)
            typename RouterG::Args a6{};
            a6.h           = b.h2.as<bf16_t>();
            a6.w           = b.w_router.as<float>();
            a6.logits      = b.logits.as<float>();
            a6.M           = M;
            a6.K           = H;
            a6.stride_h    = H;
            a6.ld_logits   = E;
            a6.num_experts = E;
            a6.k_splits    = r_splits;
            a6.rank        = 0;
            a6.world       = 1;
            a6.scale       = 0.17677669529663687f;
            if(r_splits > 1)
            {
                if(b.rpart.p == nullptr)
                {
                    b.rpart.alloc(r, size_t(RouterG::PartialSize(a6)) * 4);
                    b.rtick.alloc(r, size_t(RouterG::NumTickets(a6)) * 4);
                    b.rtick.fill(0);
                }
                a6.partial = b.rpart.as<float>();
                a6.tickets = b.rtick.as<uint32_t>();
            }
            // 7 router finalize
            const ck_tile::moe_route_table table{b.topk_ids.as<int32_t>(),
                                                 b.topk_w.as<float>(),
                                                 b.count.as<int32_t>(),
                                                 b.routes.as<int32_t>(),
                                                 b.active.as<int32_t>(),
                                                 b.num_active.as<int32_t>(),
                                                 M,
                                                 M,
                                                 kTopK,
                                                 E};
            typename RouterF::Args a7{
                b.logits.as<float>(), E, b.router_bias.as<float>(), 1.0f, 1e-15f, table};
            // 8 pooler, N-sharded: columns [r P / W, +P / W) of p, written in place
            auto a8 = gemm_args(tag_bf16{},
                                b.h2.p,
                                H,
                                b.w_pool.as<bf16_t>() + size_t(r) * (P / W) * H,
                                H,
                                b.p.as<bf16_t>() + r * (P / W),
                                P,
                                P / W,
                                H,
                                pool_splits,
                                b,
                                2);
            // 9 all-gather p (in place)
            typename AgBf16::Args a9{ex_p.view(r, state, opt.spin),
                                     b.p.as<bf16_t>() + r * (P / W),
                                     P * 2,
                                     M,
                                     P / W * 2,
                                     b.p.p,
                                     long_index_t(P) * 2,
                                     long_index_t(P / W) * 2};
            // 10 pre_expert_norm + MXFP8
            typename PreExp::Args a10{};
            a10.in           = b.p.as<bf16_t>();
            a10.in_stride    = P;
            a10.w            = b.w_preexp.as<bf16_t>();
            a10.eps          = 1e-5f;
            a10.p_out        = b.pn.as<bf16_t>();
            a10.p_stride     = P;
            a10.q_out        = b.pq.p;
            a10.q_stride     = P;
            a10.scale_out    = b.ps.p;
            a10.scale_stride = P / 32;
            a10.M            = M;
            // 11 gate/up: this rank's I / W columns of z
            typename GateUp::Args a11{table,
                                      b.pq.as<fp8_t>(),
                                      b.ps.as<uint8_t>(),
                                      b.gate_q.as<ck_tile::pk_fp4_t>(),
                                      b.gate_s.as<uint8_t>(),
                                      b.up_q.as<ck_tile::pk_fp4_t>(),
                                      b.up_s.as<uint8_t>(),
                                      b.z.as<fp8_t>(),
                                      b.zs.as<uint8_t>(),
                                      I,
                                      I,
                                      r * (I / W),
                                      r,
                                      W};
            // 12 / 13 all-gather z and its scales (raw bytes, in place)
            typename AgBytes::Args a12{ex_z.view(r, state, opt.spin),
                                       b.z.as<uint8_t>() + r * (I / W),
                                       I,
                                       R,
                                       I / W,
                                       b.z.p,
                                       long_index_t(I),
                                       long_index_t(I / W)};
            typename AgBytes::Args a13{ex_zs.view(r, state, opt.spin),
                                       b.zs.as<uint8_t>() + r * (I / W / 32),
                                       I / 32,
                                       R,
                                       I / W / 32,
                                       b.zs.p,
                                       long_index_t(I / 32),
                                       long_index_t(I / W / 32)};
            // 14 down: this rank's P / W columns of y, full K
            typename Down::Args a14{table,
                                    b.z.as<fp8_t>(),
                                    b.zs.as<uint8_t>(),
                                    b.down_q.as<ck_tile::pk_fp4_t>(),
                                    b.down_s.as<uint8_t>(),
                                    b.y.as<bf16_t>(),
                                    P,
                                    P,
                                    r * (P / W),
                                    r,
                                    W};
            // 15 all-gather y (in place)
            typename AgBf16::Args a15{ex_y.view(r, state, opt.spin),
                                      b.y.as<bf16_t>() + r * (P / W),
                                      P * 2,
                                      R,
                                      P / W * 2,
                                      b.y.p,
                                      long_index_t(P) * 2,
                                      long_index_t(P / W) * 2};
            // 16 combine (replicated)
            typename Combine::Args a16{
                table, b.y.as<bf16_t>(), b.w_postexp.as<bf16_t>(), b.acc.as<bf16_t>(), 1e-8f};
            // 17 unpooler: this rank's K slice
            auto a17 = gemm_args(tag_f32{},
                                 b.acc.as<bf16_t>() + r * (P / W),
                                 P,
                                 b.w_unpool.as<bf16_t>() + r * (P / W),
                                 P,
                                 b.upart.p,
                                 H,
                                 H,
                                 P / W,
                                 unp_splits,
                                 b,
                                 3);
            // 18 all-reduce + post_feedforward InputScaledRMSNorm + residual gate
            typename PostFf::Args n18{};
            n18.first_w         = b.w_postff.as<bf16_t>();
            n18.first_eps       = 1e-8f;
            n18.gate_alpha      = b.a_ffn.as<float>();
            n18.gate_beta       = b.b_ffn.as<float>();
            n18.residual        = b.x1.as<bf16_t>();
            n18.residual_stride = H;
            n18.x_out           = b.out.as<bf16_t>();
            n18.x_stride        = H;
            n18.next_w          = nullptr;
            n18.next_eps        = 1e-5f;
            n18.h_out           = nullptr;
            n18.h_stride        = H;
            n18.M               = M;
            typename ArFfn::Args a18{
                ex_ar2.view(r, state, opt.spin), b.upart.as<float>(), H, M, H, n18};

            return typename MK::ArgsTuple{a0,
                                          a1,
                                          a2,
                                          a3,
                                          a4,
                                          a5,
                                          a6,
                                          a7,
                                          a8,
                                          a9,
                                          a10,
                                          a11,
                                          a12,
                                          a13,
                                          a14,
                                          a15,
                                          a16,
                                          a17,
                                          a18};
        };

        std::vector<hipStream_t> streams(W);
        std::vector<typename MK::Kargs> kargs;
        index_t max_work = 1;
        for(int r = 0; r < W; ++r)
        {
            L67_HIP_CHECK(hipSetDevice(r));
            L67_HIP_CHECK(hipStreamCreateWithFlags(&streams[r], hipStreamNonBlocking));
            const auto probe = make_args(r, nullptr);
            rk[r].counters.alloc(r, size_t(MK::NumCounters(probe)) * 4);
            rk[r].counters.fill(0);
            const uint32_t* state = rk[r].counters.as<uint32_t>();
            const auto args       = make_args(r, state);
            kargs.push_back(MK::MakeKargs(args, rk[r].counters.as<uint32_t>()));
            bool ok = true;
            ck_tile::static_for<0, MK::kNumStages, 1>{}([&](auto i) {
                using Ph      = typename MK::template phase_t<i.value>;
                const auto& a = args.at(ck_tile::number<i.value>{});
                if(!Ph::IsSupportedArgument(a))
                {
                    std::printf("stage %d (%s): arguments not supported\n",
                                int(i.value),
                                kStageNames[i.value]);
                    ok = false;
                }
                max_work = std::max<index_t>(max_work, Ph::NumWork(a));
            });
            if(!ok || !MK::IsSupportedArgument(kargs.back()))
                return 2;
        }

        RunInfo& info      = out.info;
        info.grid          = grid;
        info.resident      = resident;
        info.lds           = MK::kSmemSize;
        info.stages        = MK::kNumStages;
        info.max_work      = max_work;
        info.qkvg_splits   = qkvg_splits;
        info.o_splits      = o_splits;
        info.pool_splits   = pool_splits;
        info.unpool_splits = unp_splits;
        info.router_splits = r_splits;
        info.attn_splits   = attn_splits;

        auto launch_all = [&]() {
            for(int r = 0; r < W; ++r)
            {
                L67_HIP_CHECK(hipSetDevice(r));
                ck_tile::launch_kernel(
                    ck_tile::stream_config{streams[r]},
                    ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs[r]));
            }
        };
        auto sync_all = [&]() {
            for(int r = 0; r < W; ++r)
            {
                L67_HIP_CHECK(hipSetDevice(r));
                L67_HIP_CHECK(hipStreamSynchronize(streams[r]));
            }
        };
        auto poison = [&]() {
            for(int r = 0; r < W; ++r)
            {
                Rank& b = rk[r];
                for(DevBuf* p :
                    {&b.h,      &b.qkvg,       &b.q,        &b.o,      &b.opart, &b.x1,
                     &b.h2,     &b.logits,     &b.topk_ids, &b.topk_w, &b.count, &b.routes,
                     &b.active, &b.num_active, &b.p,        &b.pn,     &b.pq,    &b.ps,
                     &b.z,      &b.zs,         &b.y,        &b.acc,    &b.upart, &b.out})
                    p->fill(0xFF);
            }
            for(int r = 0; r < W; ++r)
            {
                L67_HIP_CHECK(hipSetDevice(r));
                L67_HIP_CHECK(hipDeviceSynchronize());
            }
        };
        auto dump = [&](int r, RankDump& o) {
            Rank& b      = rk[r];
            o.h          = b.h.get<typename decltype(o.h)::value_type>(size_t(M) * H);
            o.qkvg       = b.qkvg.get<typename decltype(o.qkvg)::value_type>(size_t(M) * NQkvgR);
            o.q          = b.q.get<typename decltype(o.q)::value_type>(size_t(M) * QHr * kHeadDim);
            o.kcache     = b.kcache.get<typename decltype(o.kcache)::value_type>(kv_bytes);
            o.vcache     = b.vcache.get<typename decltype(o.vcache)::value_type>(kv_bytes);
            o.o          = b.o.get<typename decltype(o.o)::value_type>(size_t(M) * QHr * kHeadDim);
            o.opart      = b.opart.get<typename decltype(o.opart)::value_type>(size_t(M) * H);
            o.x1         = b.x1.get<typename decltype(o.x1)::value_type>(size_t(M) * H);
            o.h2         = b.h2.get<typename decltype(o.h2)::value_type>(size_t(M) * H);
            o.logits     = b.logits.get<typename decltype(o.logits)::value_type>(size_t(M) * E);
            o.topk_ids   = b.topk_ids.get<typename decltype(o.topk_ids)::value_type>(size_t(R));
            o.topk_w     = b.topk_w.get<typename decltype(o.topk_w)::value_type>(size_t(R));
            o.count      = b.count.get<typename decltype(o.count)::value_type>(size_t(E));
            o.routes     = b.routes.get<typename decltype(o.routes)::value_type>(size_t(E) * M);
            o.active     = b.active.get<typename decltype(o.active)::value_type>(size_t(E));
            o.num_active = b.num_active.get<typename decltype(o.num_active)::value_type>(1);
            o.p          = b.p.get<typename decltype(o.p)::value_type>(size_t(M) * P);
            o.pn         = b.pn.get<typename decltype(o.pn)::value_type>(size_t(M) * P);
            o.pq         = b.pq.get<typename decltype(o.pq)::value_type>(size_t(M) * P);
            o.ps         = b.ps.get<typename decltype(o.ps)::value_type>(size_t(M) * (P / 32));
            o.z          = b.z.get<typename decltype(o.z)::value_type>(size_t(R) * I);
            o.zs         = b.zs.get<typename decltype(o.zs)::value_type>(size_t(R) * (I / 32));
            o.y          = b.y.get<typename decltype(o.y)::value_type>(size_t(R) * P);
            o.acc        = b.acc.get<typename decltype(o.acc)::value_type>(size_t(M) * P);
            o.upart      = b.upart.get<typename decltype(o.upart)::value_type>(size_t(M) * H);
            o.out        = b.out.get<typename decltype(o.out)::value_type>(size_t(M) * H);
        };
        // Bytes a replay must reproduce: every activation, and the KV cache (new slots rewritten
        // with the same values; context untouched).
        auto same = [](const RankDump& a, const RankDump& b) -> bool {
            return a.h == b.h && a.qkvg == b.qkvg && a.q == b.q && a.kcache == b.kcache &&
                   a.vcache == b.vcache && a.o == b.o && a.opart == b.opart && a.x1 == b.x1 &&
                   a.h2 == b.h2 && a.logits == b.logits && a.topk_ids == b.topk_ids &&
                   a.topk_w == b.topk_w && a.count == b.count && a.active == b.active &&
                   a.num_active == b.num_active && a.p == b.p && a.pn == b.pn && a.pq == b.pq &&
                   a.ps == b.ps && a.z == b.z && a.zs == b.zs && a.y == b.y && a.acc == b.acc &&
                   a.upart == b.upart && a.out == b.out;
        };

        out.ranks.assign(W, RankDump{});
        for(int l = 0; l < opt.launches; ++l)
        {
            if(opt.poison)
                poison();
            const auto t0 = std::chrono::steady_clock::now();
            launch_all();
            sync_all();
            const double us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                    .count();
            for(int r = 0; r < W; ++r)
            {
                RankDump cur;
                dump(r, cur);
                if(l == 0)
                {
                    out.ranks[r]             = std::move(cur);
                    out.ranks[r].kcache_init = rk[r].kcache0;
                    out.ranks[r].vcache_init = rk[r].vcache0;
                }
                else if(!same(cur, out.ranks[r]))
                {
                    ++info.replay_mismatches;
                    std::printf("  launch %d rank %d: outputs differ from launch 0\n", l, r);
                }
            }
            std::printf("  launch %d done (%.0f us host wall incl. sync)\n", l, us);
        }

        // Timing: back-to-back launches, no host sync and no poisoning in between.
        if(opt.repeat > 0)
        {
            std::vector<hipEvent_t> e0(W), e1(W);
            for(int r = 0; r < W; ++r)
            {
                L67_HIP_CHECK(hipSetDevice(r));
                L67_HIP_CHECK(hipEventCreate(&e0[r]));
                L67_HIP_CHECK(hipEventCreate(&e1[r]));
            }
            for(int r = 0; r < W; ++r)
            {
                L67_HIP_CHECK(hipSetDevice(r));
                L67_HIP_CHECK(hipEventRecord(e0[r], streams[r]));
            }
            for(int i = 0; i < opt.repeat; ++i)
                launch_all();
            for(int r = 0; r < W; ++r)
            {
                L67_HIP_CHECK(hipSetDevice(r));
                L67_HIP_CHECK(hipEventRecord(e1[r], streams[r]));
            }
            float worst = 0;
            for(int r = 0; r < W; ++r)
            {
                L67_HIP_CHECK(hipSetDevice(r));
                L67_HIP_CHECK(hipEventSynchronize(e1[r]));
                float ms = 0;
                L67_HIP_CHECK(hipEventElapsedTime(&ms, e0[r], e1[r]));
                worst = std::max(worst, ms);
                L67_HIP_CHECK(hipEventDestroy(e0[r]));
                L67_HIP_CHECK(hipEventDestroy(e1[r]));
            }
            info.us_per_launch = 1e3 * worst / opt.repeat;
            // The replayed launches must still give launch 0's bytes.
            for(int r = 0; r < W; ++r)
            {
                RankDump cur;
                dump(r, cur);
                if(!same(cur, out.ranks[r]))
                {
                    ++info.replay_mismatches;
                    std::printf(
                        "  after %d timed launches rank %d: outputs differ\n", opt.repeat, r);
                }
            }
        }

        const uint32_t expect = static_cast<uint32_t>(opt.launches + std::max(0, opt.repeat));
        for(int r = 0; r < W; ++r)
        {
            uint32_t e = 0;
            L67_HIP_CHECK(hipSetDevice(r));
            L67_HIP_CHECK(hipMemcpy(&e, rk[r].counters.p, 4, hipMemcpyDeviceToHost));
            if(e != expect)
            {
                std::printf("  rank %d: device epoch %u, expected %u\n", r, e, expect);
                ++info.epoch_errors;
            }
            L67_HIP_CHECK(hipStreamDestroy(streams[r]));
        }
        return 0;
    }
};

} // namespace layer67

#define LAYER67_CAT2(a, b) a##b
#define LAYER67_CAT(a, b) LAYER67_CAT2(a, b)
