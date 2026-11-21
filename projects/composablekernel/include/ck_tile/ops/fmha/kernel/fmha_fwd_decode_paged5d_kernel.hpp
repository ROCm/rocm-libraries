// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/amd_cluster_load.hpp"

// gfx1250 fp8 decode attention over the SHUFFLE 5D paged KV cache (sglang layout):
//   K [num_blocks][nhead_k][hdim / 16][16 tokens][16 B]
//   V [num_blocks][nhead_k][1][hdim][16 tokens]
// One query token per sequence, hdim 128, page 16 tokens, fp8 e4m3 K/V with per-tensor
// descales, bf16 Q (quantized per row in the kernel) and bf16 output. The q heads of one kv head
// (at most 16) are the 16 columns of S^T = K Q^T.
//
// Grid (kNumSplits * groups, nhead_k, batch). Each kNumSplits consecutive splits are one workgroup
// cluster; each stores O / l in bf16 and (m, l), and after a cluster barrier every split merges
// its share of the rows (with several groups, see MergeGroups). K and V pages are copied to LDS
// byte for byte: in this layout the LDS bytes are WMMA operands as they are (S^T = K Q^T and O^T +=
// V^T P^T, no transposed reads).

namespace ck_tile {

struct FmhaDecodePaged5dHostArgs
{
    const void* q_ptr;           // [batch][nhead_q][hdim] bf16
    const void* k_ptr;           // 5D fp8
    const void* v_ptr;           // 5D fp8
    const int32_t* block_table;  // [batch][max_blocks_per_seq]
    const int32_t* seqlen_k_ptr; // [batch]
    const float* k_descale_ptr;  // one float
    const float* v_descale_ptr;  // one float
    void* o_ptr;                 // [batch][nhead_q][hdim] bf16
    // fmha_decode_paged5d_workspace_bytes; zero it once before first use (the counters of the
    // cluster groups reset themselves)
    void* workspace_ptr;
    index_t batch;
    index_t nhead_q;
    index_t nhead_k;
    index_t max_blocks_per_seq;
    index_t window; // keys attended, the query's own included; 0 = no window
    index_t pages_per_split;
    index_t num_groups; // clusters per (batch, kv head)
    float scale_s;
};

// num_splits workgroups per cluster, num_groups clusters per (batch, kv head)
struct FmhaDecodePaged5dConfig
{
    index_t num_splits;
    index_t pages_per_split;
    bool wave_split;
    index_t num_groups = 1;
};

// A range that fits one cluster (at most 15 splits: the cluster size is a 4-bit compile-time
// field) takes as many splits as it allows, each a whole number of 64-token steps, but at least 16
// pages per split once the grid would exceed about 640 workgroups. A longer range takes several
// clusters (groups, merged by the last split to finish a row block): the shortest of 8, 12 or 16
// pages per split that keeps at most 192 splits per (batch, kv head) and 512 workgroups, or else
// the wave-split layout (whose waves run their own pages without per-step barriers) with about
// 384 workgroups.
CK_TILE_HOST FmhaDecodePaged5dConfig fmha_decode_paged5d_config(index_t batch,
                                                                index_t nhead_k,
                                                                index_t max_seqlen_k,
                                                                index_t window)
{
    constexpr index_t kPage = 16;
    const index_t lo_pages  = window > 0 ? std::max<index_t>(0, max_seqlen_k - window) / kPage : 0;
    const index_t span      = std::max<index_t>(1, (max_seqlen_k + kPage - 1) / kPage - lo_pages);
    const auto round_up     = [](index_t x, index_t m) { return (x + m - 1) / m * m; };
    const auto ceil_div     = [](index_t x, index_t y) { return (x + y - 1) / y; };
    const index_t heads     = batch * nhead_k;
    const auto make         = [&](index_t pps, bool wave_split) {
        const index_t total  = ceil_div(span, pps);
        const index_t groups = ceil_div(total, 15);
        return FmhaDecodePaged5dConfig{ceil_div(total, groups), pps, wave_split, groups};
    };
    if(span <= 15 * 16)
    {
        index_t pps = round_up(ceil_div(span, std::min<index_t>(15, ceil_div(span, 4))), 4);
        if(heads * ceil_div(span, pps) > 640)
            pps = std::max<index_t>(pps, 16);
        return make(pps, false);
    }
    for(index_t pps : {8, 12, 16})
        if(ceil_div(span, pps) <= 192 && heads * ceil_div(span, pps) <= 512)
            return make(pps, false);
    return make(std::max<index_t>(32, round_up(ceil_div(span * heads, 384), 16)), true);
}

CK_TILE_HOST size_t fmha_decode_paged5d_workspace_bytes(index_t batch,
                                                        index_t nhead_k,
                                                        index_t num_splits,
                                                        index_t num_groups = 1)
{
    // per split: bf16 O / l [batch][nhead_k][groups][splits][16][128] and fp32 (m, l) [..][16][2];
    // per group the same with one split, then int32 counters [batch][nhead_k][splits]
    const size_t heads = static_cast<size_t>(batch) * nhead_k;
    const size_t rows  = heads * num_groups * num_splits * 16;
    const size_t grows = num_groups > 1 ? heads * num_groups * 16 : 0;
    const size_t cnt   = num_groups > 1 ? heads * num_splits : 0;
    return (rows + grows) * (128 * 2 + 2 * sizeof(float)) + cnt * sizeof(int32_t);
}

template <bool kWaveSplit_, index_t kNumSplits_, bool kGroups_ = false>
struct FmhaDecodePaged5dKernel
{
    static constexpr bool kWaveSplit    = kWaveSplit_;
    static constexpr index_t kNumSplits = kNumSplits_;
    // several clusters per (batch, kv head), merged by MergeGroups
    static constexpr bool kGroups       = kGroups_;
    static constexpr index_t kBlockSize = 128;
    static constexpr index_t kNumWaves  = 4;
    static constexpr index_t kHdim      = 128;
    static constexpr index_t kPage      = 16;
    static constexpr index_t kRows      = 16;
    static constexpr index_t kPageBytes = kPage * kHdim;
    // pages per wave per step: one (cooperative) or four (wave split)
    static constexpr index_t kWavePages = kWaveSplit ? 4 : 1;
    static constexpr index_t kStages    = kWaveSplit ? 2 : 3;
    static_assert(1 <= kNumSplits && kNumSplits <= 15);

    struct Kargs
    {
        const uint16_t* q;
        const uint8_t* k;
        const uint8_t* v;
        const int32_t* block_table;
        const int32_t* seqlen_k;
        const float* k_descale;
        const float* v_descale;
        uint16_t* o;
        uint16_t* ws_o;
        float* ws_ml;
        index_t nhead_q;
        index_t nhead_k;
        index_t gqa; // nhead_q / nhead_k, so the kernel does not divide
        index_t max_blocks_per_seq;
        index_t window;
        index_t pages_per_split;
        float scale_s;
        // used with kGroups only; last, so that the arguments the prologue needs come first
        index_t num_groups;
        uint16_t* gws_o;
        float* gws_ml;
        int32_t* counters; // per (batch, kv head, row block)
    };

    CK_TILE_HOST static Kargs MakeKargs(const FmhaDecodePaged5dHostArgs& h)
    {
        const size_t heads = static_cast<size_t>(h.batch) * h.nhead_k;
        const size_t rows  = heads * h.num_groups * kNumSplits * kRows;
        const size_t grows = heads * h.num_groups * kRows;
        auto* ws           = static_cast<uint8_t*>(h.workspace_ptr);
        auto* gws          = ws == nullptr ? nullptr : ws + rows * (kHdim * 2 + 8);
        return {static_cast<const uint16_t*>(h.q_ptr),
                static_cast<const uint8_t*>(h.k_ptr),
                static_cast<const uint8_t*>(h.v_ptr),
                h.block_table,
                h.seqlen_k_ptr,
                h.k_descale_ptr,
                h.v_descale_ptr,
                static_cast<uint16_t*>(h.o_ptr),
                reinterpret_cast<uint16_t*>(ws),
                ws == nullptr ? nullptr : reinterpret_cast<float*>(ws + rows * kHdim * 2),
                h.nhead_q,
                h.nhead_k,
                h.nhead_q / h.nhead_k,
                h.max_blocks_per_seq,
                h.window,
                h.pages_per_split,
                h.scale_s,
                h.num_groups,
                reinterpret_cast<uint16_t*>(gws),
                gws == nullptr ? nullptr : reinterpret_cast<float*>(gws + grows * kHdim * 2),
                gws == nullptr ? nullptr
                               : reinterpret_cast<int32_t*>(gws + grows * (kHdim * 2 + 8))};
    }

    CK_TILE_HOST static constexpr dim3 GridSize(index_t batch, index_t nhead_k, index_t groups = 1)
    {
        return dim3(kNumSplits * groups, nhead_k, batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

#if defined(__gfx125__)
    using v4i  = int32_t __attribute__((ext_vector_type(4)));
    using v8i  = int32_t __attribute__((ext_vector_type(8)));
    using v16i = int32_t __attribute__((ext_vector_type(16)));
    using v8f  = float __attribute__((ext_vector_type(8)));

    struct alignas(16) SmemCoop
    {
        alignas(16) uint8_t k[kStages][kNumWaves][kPageBytes];
        alignas(16) uint8_t v[kStages][kNumWaves][kPageBytes];
        alignas(16) uint8_t p[kRows][kNumWaves * kPage];
        alignas(16) float red[2][kNumWaves][kRows]; // per-wave row max / row sum
    };

    struct alignas(16) SmemWaveSplit
    {
        union
        {
            struct
            {
                alignas(16) uint8_t k[kStages][kNumWaves][kWavePages][kPageBytes];
                alignas(16) uint8_t v[kStages][kNumWaves][kWavePages][kPageBytes];
            } kv;
            // the waves' O for the intra-workgroup merge; padded rows keep the column-wise
            // stores of 16 rows off one bank
            alignas(16) float o[kNumWaves][kRows][kHdim + 4];
        };
        alignas(16) float ml[2][kNumWaves][kRows];
    };

    template <typename T, typename P>
    CK_TILE_DEVICE static T* ptr_as(P* p)
    {
        return static_cast<T*>(static_cast<void*>(p));
    }
    template <typename T, typename P>
    CK_TILE_DEVICE static const T* ptr_as(const P* p)
    {
        return static_cast<const T*>(static_cast<const void*>(p));
    }
    CK_TILE_DEVICE static bool is_neg_inf(float x) { return __builtin_isinf(x) && x < 0.f; }
    CK_TILE_DEVICE static float bf16_to_f32(uint32_t x) { return __uint_as_float(x << 16); }
    CK_TILE_DEVICE static uint32_t f32_to_bf16(float f)
    {
        const uint32_t u = __float_as_uint(f);
        return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    }
    CK_TILE_DEVICE static uint32_t pack_bf16(float a, float b)
    {
        return f32_to_bf16(a) | (f32_to_bf16(b) << 16);
    }
    CK_TILE_DEVICE static float exp2(float x) { return __builtin_amdgcn_exp2f(x); }

    CK_TILE_DEVICE void operator()(const Kargs& a) const
    {
        if constexpr(kWaveSplit)
        {
            __shared__ SmemWaveSplit sm;
            Run(a, sm);
        }
        else
        {
            __shared__ SmemCoop sm;
            Run(a, sm);
        }
    }

    template <typename Smem>
    CK_TILE_DEVICE void Run(const Kargs& a, Smem& sm) const
    {
        const index_t l = threadIdx.x % 32;
        const index_t w = __builtin_amdgcn_readfirstlane(threadIdx.x) / 32;
        const index_t n = l % 16, h = l / 16;
        // groups of kNumSplits along x, one cluster each
        const index_t gi = kGroups ? __builtin_amdgcn_cluster_id_x() : 0;
        const index_t s  = kGroups ? __builtin_amdgcn_cluster_workgroup_id_x() : blockIdx.x;
        const index_t g = blockIdx.y, b = blockIdx.z;
        const index_t sg  = gi * kNumSplits + s; // split index within the (batch, kv head)
        const index_t gqa = a.gqa;

        const index_t ctx = a.seqlen_k[b];
        const index_t lo  = a.window > 0 ? max(0, ctx - a.window) : 0;
        const index_t pg0 = lo / kPage + sg * a.pages_per_split;
        const index_t pend =
            min((ctx + kPage - 1) / kPage, lo / kPage + (sg + 1) * a.pages_per_split);
        constexpr index_t kStepPages = kNumWaves * kWavePages;
        const index_t steps          = pend > pg0 ? (pend - pg0 + kStepPages - 1) / kStepPages : 0;

        // block ids of the split's pages, 32 per register (lane = page), picked with readlane
        const int32_t* bt  = a.block_table + static_cast<long_index_t>(b) * a.max_blocks_per_seq;
        const auto load_bt = [&](index_t chunk) {
            return bt[min(pg0 + chunk * 32 + l, max(pend - 1, 0))];
        };
        int32_t bt_cur = load_bt(0), bt_next = load_bt(1);
        index_t cur_chunk = 0;

        // Q^T operand of row n: 16-byte d chunks 2i + h, after a per-row absmax quantization
        float qv[64];
        {
            const uint16_t* qrow =
                a.q +
                (static_cast<long_index_t>(b) * a.nhead_q + g * gqa + min(n, gqa - 1)) * kHdim;
            for(index_t i = 0; i < 4; ++i)
            {
                const index_t c = 2 * i + h;
                const v4i x0    = *ptr_as<v4i>(qrow + c * 16);
                const v4i x1    = *ptr_as<v4i>(qrow + c * 16 + 8);
                for(index_t e = 0; e < 4; ++e)
                {
                    qv[i * 16 + 2 * e]         = bf16_to_f32(x0[e] & 0xffff);
                    qv[i * 16 + 2 * e + 1]     = bf16_to_f32(static_cast<uint32_t>(x0[e]) >> 16);
                    qv[i * 16 + 8 + 2 * e]     = bf16_to_f32(x1[e] & 0xffff);
                    qv[i * 16 + 8 + 2 * e + 1] = bf16_to_f32(static_cast<uint32_t>(x1[e]) >> 16);
                }
            }
        }

        // LDS page buffers of wave w
        const auto kbuf = [&](index_t st, index_t q) -> uint8_t* {
            if constexpr(kWaveSplit)
                return &sm.kv.k[st][w][q][0];
            else
            {
                ignore = q;
                return &sm.k[st][w][0];
            }
        };
        const auto vbuf = [&](index_t st, index_t q) -> uint8_t* {
            if constexpr(kWaveSplit)
                return &sm.kv.v[st][w][q][0];
            else
            {
                ignore = q;
                return &sm.v[st][w][0];
            }
        };
        // page index (within the split) of wave w's q-th page of step t
        const auto page_idx = [&](index_t t, index_t q) {
            return (t * kNumWaves + w) * kWavePages + q;
        };
        const auto issue = [&](index_t t) {
            const index_t st = t % kStages;
            for(index_t q = 0; q < kWavePages; ++q)
            {
                // a page past the split copies the last one (its V is zeroed once it lands): every
                // step issues the same number of copies, which the asynccnt waits count on
                const index_t idx = min(page_idx(t, q), pend - 1 - pg0);
                uint8_t* kd       = kbuf(st, q);
                uint8_t* vd       = vbuf(st, q);
                const int32_t blk =
                    __builtin_amdgcn_readlane((idx >> 5) == cur_chunk ? bt_cur : bt_next, idx & 31);
                const long_index_t base =
                    (static_cast<long_index_t>(blk) * a.nhead_k + g) * kPageBytes;
                for(index_t i = 0; i < 4; ++i)
                {
                    const index_t off = (i * 32 + l) * 16;
                    __builtin_amdgcn_global_load_async_to_lds_b128(
                        to_global(ptr_as<v4i>(a.k + base + off)),
                        to_lds(ptr_as<v4i>(kd + off)),
                        0,
                        0);
                    __builtin_amdgcn_global_load_async_to_lds_b128(
                        to_global(ptr_as<v4i>(a.v + base + off)),
                        to_lds(ptr_as<v4i>(vd + off)),
                        0,
                        0);
                }
            }
        };
        index_t issued = min(steps, kStages);
        for(index_t t = 0; t < issued; ++t)
            issue(t);

        float qmax = 1e-20f;
        for(index_t i = 0; i < 64; ++i)
            qmax = fmaxf(qmax, fabsf(qv[i]));
        qmax             = fmaxf(qmax, __shfl_xor(qmax, 16, 32));
        const float qinv = 448.f / qmax;
        v16i qf{};
        for(index_t j = 0; j < 16; ++j)
        {
            const int32_t x =
                __builtin_amdgcn_cvt_pk_fp8_f32(qv[4 * j] * qinv, qv[4 * j + 1] * qinv, 0, false);
            qf[j] = __builtin_amdgcn_cvt_pk_fp8_f32(
                qv[4 * j + 2] * qinv, qv[4 * j + 3] * qinv, x, true);
        }
        // column n of S^T in log2 units
        const float scl = qmax / 448.f * *a.k_descale * a.scale_s * 1.4426950408889634f;
        const float vd  = *a.v_descale;

        // step t's copies of this wave are in LDS; zero the V bytes of tokens past the context
        const auto land = [&](index_t t) {
            if(issued > t + 2)
                __builtin_amdgcn_s_wait_asynccnt(16 * kWavePages);
            else if(issued > t + 1)
                __builtin_amdgcn_s_wait_asynccnt(8 * kWavePages);
            else
                __builtin_amdgcn_s_wait_asynccnt(0);
            for(index_t q = 0; q < kWavePages; ++q)
            {
                const index_t pg    = pg0 + page_idx(t, q);
                const index_t valid = __builtin_amdgcn_readfirstlane(
                    pg < pend ? min(max(ctx - pg * kPage, 0), kPage) : 0);
                uint8_t* vd8 = vbuf(t % kStages, q);
                if(valid == 0)
                    for(index_t i = 0; i < 4; ++i)
                        *ptr_as<v4i>(vd8 + (i * 32 + l) * 16) = v4i{0, 0, 0, 0};
                else if(valid < kPage)
                    for(index_t i = 0; i < 4; ++i)
                        for(index_t tok = valid; tok < kPage; ++tok)
                            vd8[(i * 32 + l) * 16 + tok] = 0;
            }
        };
        const auto advance_bt = [&](index_t t) {
            // the block-id chunk of step t's pages; the next one is loaded a chunk ahead
            const index_t chunk = (t * kStepPages) / 32;
            if(chunk != cur_chunk)
            {
                cur_chunk = chunk;
                bt_cur    = bt_next;
                bt_next   = load_bt(chunk + 1);
            }
        };

        float m_run = -numeric<float>::infinity(), l_run = 0.f;
        constexpr index_t kAcc = kWaveSplit ? 8 : 2; // 16-wide head-dim blocks per wave
        v8f acc[kAcc];
        for(index_t i = 0; i < kAcc; ++i)
            acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

        // S^T of one page: lane (n, h) gets tokens h*8 + r of row n, masked to the range
        const auto qk_page = [&](const uint8_t* kb, index_t pg, float(&sv)[8], float& mx) {
            v16i kf{};
            for(index_t i = 0; i < 4; ++i)
            {
                const v4i x = *ptr_as<v4i>(kb + (2 * i + h) * 256 + n * 16);
                for(index_t e = 0; e < 4; ++e)
                    kf[4 * i + e] = x[e];
            }
            const v8f st8 = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(
                kf, qf, 0, v8f{0, 0, 0, 0, 0, 0, 0, 0}, false, false);
            const index_t tok0 = pg * kPage + h * 8;
            for(index_t r = 0; r < 8; ++r)
            {
                const index_t pos = tok0 + r;
                const bool ok     = pos >= lo && pos < ctx && pos < pend * kPage;
                sv[r]             = ok ? st8[r] * scl : -numeric<float>::infinity();
                mx                = fmaxf(mx, sv[r]);
            }
        };

        if constexpr(kWaveSplit)
        {
            for(index_t t = 0; t < steps; ++t)
            {
                const index_t st = t % kStages;
                land(t);
                float sv[kWavePages][8];
                float mx = -numeric<float>::infinity();
                for(index_t q = 0; q < kWavePages; ++q)
                    qk_page(kbuf(st, q), pg0 + page_idx(t, q), sv[q], mx);
                mx                = fmaxf(mx, __shfl_xor(mx, 16, 32));
                const float m_new = fmaxf(m_run, mx);
                const float m_use = is_neg_inf(m_new) ? 0.f : m_new;
                const float alpha = exp2(m_run - m_use);
                // P scaled by 448: the row max lands on e4m3's largest normal; byte q*8 + r of
                // the P^T operand is token q*16 + h*8 + r, as on the V^T operand below
                float ps = 0.f;
                int32_t pk[8];
                for(index_t q = 0; q < kWavePages; ++q)
                    for(index_t r2 = 0; r2 < 4; ++r2)
                    {
                        const float p0 = exp2(sv[q][2 * r2] - m_use) * 448.f;
                        const float p1 = exp2(sv[q][2 * r2 + 1] - m_use) * 448.f;
                        ps += p0 + p1;
                        const index_t j = q * 8 + 2 * r2;
                        pk[j / 4]       = (j & 3) == 0
                                              ? __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, 0, false)
                                              : __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, pk[j / 4], true);
                    }
                ps    = ps + __shfl_xor(ps, 16, 32);
                l_run = l_run * alpha + ps;
                m_run = m_new;
                const v8i pf{pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6], pk[7]};
                for(index_t db = 0; db < 8; ++db)
                {
                    v8i vf{};
                    for(index_t q = 0; q < kWavePages; ++q)
                    {
                        const int2 x =
                            *ptr_as<int2>(&sm.kv.v[st][w][q][(db * 16 + n) * 16 + h * 8]);
                        vf[2 * q]     = x.x;
                        vf[2 * q + 1] = x.y;
                    }
                    acc[db] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                        vf, pf, 0, acc[db] * alpha, false, false);
                }
                // this wave is done with stage st: it takes step t + kStages
                if(t + kStages < steps)
                {
                    advance_bt(t + kStages);
                    issue(t + kStages);
                    ++issued;
                }
            }
        }
        else
        {
            if(steps > 0)
                land(0);
            __syncthreads();
            for(index_t t = 0; t < steps; ++t)
            {
                const index_t st = t % kStages;
                float sv[8];
                float mx = -numeric<float>::infinity();
                qk_page(kbuf(st, 0), pg0 + page_idx(t, 0), sv, mx);
                mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                if(h == 0)
                    sm.red[0][w][n] = mx;
                __syncthreads();
                // every wave is past step t-1's PV: its stage takes step t + 2
                if(t >= 1 && t + 2 < steps)
                {
                    advance_bt(t + 2);
                    issue(t + 2);
                    ++issued;
                }
                float m_step = sm.red[0][0][n];
                for(index_t i = 1; i < kNumWaves; ++i)
                    m_step = fmaxf(m_step, sm.red[0][i][n]);
                const float m_new = fmaxf(m_run, m_step);
                const float m_use = is_neg_inf(m_new) ? 0.f : m_new;
                const float alpha = exp2(m_run - m_use);
                float ps          = 0.f;
                float pv[8];
                for(index_t r = 0; r < 8; ++r)
                {
                    pv[r] = exp2(sv[r] - m_use) * 448.f;
                    ps += pv[r];
                }
                ps         = ps + __shfl_xor(ps, 16, 32);
                int32_t p0 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[0], pv[1], 0, false);
                p0         = __builtin_amdgcn_cvt_pk_fp8_f32(pv[2], pv[3], p0, true);
                int32_t p1 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[4], pv[5], 0, false);
                p1         = __builtin_amdgcn_cvt_pk_fp8_f32(pv[6], pv[7], p1, true);
                *ptr_as<int2>(&sm.p[n][w * kPage + h * 8]) = make_int2(p0, p1);
                if(h == 0)
                    sm.red[1][w][n] = ps;
                if(t + 1 < steps)
                    land(t + 1);
                __syncthreads();
                float psum = 0.f;
                for(index_t i = 0; i < kNumWaves; ++i)
                    psum += sm.red[1][i][n];
                l_run = l_run * alpha + psum;
                m_run = m_new;
                // O^T += V^T P^T for head-dim blocks w and w + 4; tokens of pages h and h + 2 on
                // both operands
                const v4i px = *ptr_as<v4i>(&sm.p[n][h * kPage]);
                const v4i py = *ptr_as<v4i>(&sm.p[n][(h + 2) * kPage]);
                const v8i pf{px[0], px[1], px[2], px[3], py[0], py[1], py[2], py[3]};
                for(index_t bk = 0; bk < 2; ++bk)
                {
                    const index_t dd = (bk * kNumWaves + w) * 16 + n;
                    const v4i x      = *ptr_as<v4i>(&sm.v[st][h][dd * 16]);
                    const v4i y      = *ptr_as<v4i>(&sm.v[st][h + 2][dd * 16]);
                    const v8i vf{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
                    acc[bk] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                        vf, pf, 0, acc[bk] * alpha, false, false);
                }
            }
        }

        // this split's result in registers of thread (row, 16 or 2 head dims): o[], m_row, l_row
        index_t row;
        index_t d0;
        float o[16];
        float m_row, l_row;
        if constexpr(kWaveSplit)
        {
            // merge the waves; lane (n, h) holds O[row n][d = db*16 + h*8 + r]
            if(h == 0)
            {
                sm.ml[0][w][n] = m_run;
                sm.ml[1][w][n] = l_run;
            }
            __syncthreads(); // also: every wave is done with the stages the merge buffer reuses
            float M = -numeric<float>::infinity();
            for(index_t i = 0; i < kNumWaves; ++i)
                M = fmaxf(M, sm.ml[0][i][n]);
            const float f = exp2(m_run - (is_neg_inf(M) ? 0.f : M));
            for(index_t db = 0; db < 8; ++db)
            {
                const v8f x = acc[db] * f;
                float4* dst = ptr_as<float4>(&sm.o[w][n][db * 16 + h * 8]);
                dst[0]      = make_float4(x[0], x[1], x[2], x[3]);
                dst[1]      = make_float4(x[4], x[5], x[6], x[7]);
            }
            __syncthreads();
            row   = threadIdx.x / 8;
            d0    = (threadIdx.x % 8) * 16;
            m_row = -numeric<float>::infinity();
            for(index_t i = 0; i < kNumWaves; ++i)
                m_row = fmaxf(m_row, sm.ml[0][i][row]);
            const float mu = is_neg_inf(m_row) ? 0.f : m_row;
            l_row          = 0.f;
            for(index_t i = 0; i < kNumWaves; ++i)
                l_row += sm.ml[1][i][row] * exp2(sm.ml[0][i][row] - mu);
            for(index_t e = 0; e < 16; ++e)
                o[e] = 0.f;
            for(index_t i = 0; i < kNumWaves; ++i)
                for(index_t e4 = 0; e4 < 4; ++e4)
                {
                    const float4 x = *ptr_as<float4>(&sm.o[i][row][d0 + 4 * e4]);
                    o[4 * e4] += x.x;
                    o[4 * e4 + 1] += x.y;
                    o[4 * e4 + 2] += x.z;
                    o[4 * e4 + 3] += x.w;
                }
        }
        else
        {
            row   = n;
            m_row = m_run;
            l_row = l_run;
            d0    = 0; // unused: the lane holds d = (bk*4 + w)*16 + h*8 + r
        }

        const auto store_rows = [&](uint16_t* base_ptr, float f) {
            // base_ptr points at row 0 of this (batch, kv head) block, rows hdim apart
            if constexpr(kWaveSplit)
            {
                uint32_t pk[8];
                for(index_t e = 0; e < 8; ++e)
                    pk[e] = pack_bf16(o[2 * e] * f, o[2 * e + 1] * f);
                v4i* dst = ptr_as<v4i>(base_ptr + row * kHdim + d0);
                dst[0]   = v4i{int32_t(pk[0]), int32_t(pk[1]), int32_t(pk[2]), int32_t(pk[3])};
                dst[1]   = v4i{int32_t(pk[4]), int32_t(pk[5]), int32_t(pk[6]), int32_t(pk[7])};
            }
            else
            {
                for(index_t bk = 0; bk < 2; ++bk)
                {
                    uint32_t pk[4];
                    for(index_t e = 0; e < 4; ++e)
                        pk[e] = pack_bf16(acc[bk][2 * e] * f, acc[bk][2 * e + 1] * f);
                    *ptr_as<v4i>(base_ptr + row * kHdim + (bk * kNumWaves + w) * 16 + h * 8) =
                        v4i{int32_t(pk[0]), int32_t(pk[1]), int32_t(pk[2]), int32_t(pk[3])};
                }
            }
        };
        const bool row_ok = row < gqa;

        if constexpr(kNumSplits == 1)
        {
            if(row_ok)
                store_rows(a.o + (static_cast<long_index_t>(b) * a.nhead_q + g * gqa) * kHdim,
                           l_row > 0.f ? vd / l_row : 0.f);
            return;
        }
        else
        {
            const long_index_t head = static_cast<long_index_t>(b) * a.nhead_k + g;
            const long_index_t part0 =
                ((head * (kGroups ? a.num_groups : 1) + gi) * kNumSplits) * kRows;
            store_rows(a.ws_o + (part0 + s * kRows) * kHdim, l_row > 0.f ? 1.f / l_row : 0.f);
            const bool ml_writer = kWaveSplit ? (threadIdx.x % 8 == 0) : (w == 0 && h == 0);
            if(ml_writer)
                *ptr_as<float2>(a.ws_ml + (part0 + s * kRows + row) * 2) =
                    make_float2(m_row, l_row);
            asm volatile("s_wait_storecnt 0x0" ::: "memory");
            __builtin_amdgcn_s_cluster_barrier();
            asm volatile("global_inv scope:SCOPE_SE" ::: "memory");

            // merge the splits: split s takes rows [s*kRowsPer, +kRowsPer); thread -> (row, 2 dims)
            constexpr index_t kRowsPer = (kRows + kNumSplits - 1) / kNumSplits;
            for(index_t rr = threadIdx.x / 64; rr < kRowsPer; rr += kBlockSize / 64)
            {
                const index_t mrow = s * kRowsPer + rr;
                if(mrow >= gqa)
                    break;
                const index_t d2 = (threadIdx.x % 64) * 2;
                float2 mlv[kNumSplits];
                uint32_t oi[kNumSplits];
                for(index_t i = 0; i < kNumSplits; ++i)
                {
                    mlv[i] = *ptr_as<float2>(a.ws_ml + (part0 + i * kRows + mrow) * 2);
                    oi[i]  = *ptr_as<uint32_t>(a.ws_o + (part0 + i * kRows + mrow) * kHdim + d2);
                }
                float mm = -numeric<float>::infinity();
                for(index_t i = 0; i < kNumSplits; ++i)
                    mm = fmaxf(mm, mlv[i].x);
                float ll = 0.f, o0 = 0.f, o1 = 0.f;
                for(index_t i = 0; i < kNumSplits; ++i)
                {
                    const float wgt = mlv[i].y > 0.f ? mlv[i].y * exp2(mlv[i].x - mm) : 0.f;
                    ll += wgt;
                    o0 += wgt * bf16_to_f32(oi[i] & 0xffff);
                    o1 += wgt * bf16_to_f32(oi[i] >> 16);
                }
                if constexpr(!kGroups)
                {
                    const float f = ll > 0.f ? vd / ll : 0.f;
                    *ptr_as<uint32_t>(
                        a.o + (static_cast<long_index_t>(b) * a.nhead_q + g * gqa + mrow) * kHdim +
                        d2) = pack_bf16(o0 * f, o1 * f);
                }
                else
                {
                    // this group's rows: O / L and (M, L), at device scope for the other groups
                    const long_index_t grow = (head * a.num_groups + gi) * kRows + mrow;
                    const float f           = ll > 0.f ? 1.f / ll : 0.f;
                    __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_o + grow * kHdim + d2),
                                            pack_bf16(o0 * f, o1 * f),
                                            __ATOMIC_RELAXED,
                                            __MEMORY_SCOPE_DEVICE);
                    if(d2 == 0)
                    {
                        __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_ml + grow * 2),
                                                __float_as_uint(mm),
                                                __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
                        __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_ml + grow * 2 + 1),
                                                __float_as_uint(ll),
                                                __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
                    }
                }
            }
            if constexpr(kGroups)
                MergeGroups(a, head, b, g, gqa, s, vd);
        }
    }

    // Split s of the last group to finish row block s merges that block over the groups. The
    // groups' rows were stored at device scope; once every wave's stores are acknowledged a
    // relaxed count hands them over, and the loads are device scope as well.
    CK_TILE_DEVICE void MergeGroups(const Kargs& a,
                                    long_index_t head,
                                    index_t b,
                                    index_t g,
                                    index_t gqa,
                                    index_t s,
                                    float vd) const
    {
        constexpr index_t kRowsPer = (kRows + kNumSplits - 1) / kNumSplits;
        __shared__ int32_t last;
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            int32_t* counter = a.counters + head * kNumSplits + s;
            const bool is_last =
                __scoped_atomic_fetch_add(counter, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) ==
                a.num_groups - 1;
            if(is_last) // zeroed again for the next launch; every group has counted
                __scoped_atomic_store_n(counter, 0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            last = is_last ? 1 : 0;
        }
        __syncthreads();
        if(last == 0)
            return;
        for(index_t rr = threadIdx.x / 64; rr < kRowsPer; rr += kBlockSize / 64)
        {
            const index_t mrow = s * kRowsPer + rr;
            if(mrow >= gqa)
                break;
            const index_t d2 = (threadIdx.x % 64) * 2;
            float mm = -numeric<float>::infinity(), ll = 0.f, o0 = 0.f, o1 = 0.f;
            // kChunk groups per round of loads; the device-scope buffer loads of a round go out
            // together (atomic loads would be issued one round trip at a time)
            constexpr index_t kChunk = 8;
            const long_index_t row0  = head * a.num_groups * kRows + mrow;
            const auto ml_res        = make_wave_buffer_resource(a.gws_ml, 0x7ffffffe);
            const auto o_res         = make_wave_buffer_resource(a.gws_o, 0x7ffffffe);
            for(index_t i0 = 0; i0 < a.num_groups; i0 += kChunk)
            {
                float2 mli[kChunk];
                uint32_t oi[kChunk];
                for(index_t j = 0; j < kChunk; ++j)
                {
                    const long_index_t grow = row0 + min(i0 + j, a.num_groups - 1) * kRows;
                    const auto ml2 =
                        amd_buffer_load_impl<float, 2, amd_buffer_coherence_enum::DEVICE>(
                            ml_res, static_cast<index_t>(grow * 2 * sizeof(float)), 0);
                    mli[j] = make_float2(ml2[number<0>{}], ml2[number<1>{}]);
                    oi[j]  = __float_as_uint(
                        amd_buffer_load_impl<float, 1, amd_buffer_coherence_enum::DEVICE>(
                            o_res, static_cast<index_t>((grow * kHdim + d2) * 2), 0)[number<0>{}]);
                }
                for(index_t j = 0; j < kChunk; ++j)
                {
                    const float mi = mli[j].x, li = mli[j].y;
                    if(i0 + j >= a.num_groups || !(li > 0.f))
                        continue;
                    // online merge: the running sums follow the largest max seen so far
                    const float mn = fmaxf(mm, mi);
                    const float c0 = exp2(mm - mn), c1 = exp2(mi - mn);
                    ll = ll * c0 + li * c1;
                    o0 = o0 * c0 + li * c1 * bf16_to_f32(oi[j] & 0xffff);
                    o1 = o1 * c0 + li * c1 * bf16_to_f32(oi[j] >> 16);
                    mm = mn;
                }
            }
            const float f = ll > 0.f ? vd / ll : 0.f;
            *ptr_as<uint32_t>(
                a.o + (static_cast<long_index_t>(b) * a.nhead_q + g * gqa + mrow) * kHdim + d2) =
                pack_bf16(o0 * f, o1 * f);
        }
    }
#else
    CK_TILE_DEVICE void operator()(const Kargs&) const {}
#endif
};

// The cluster size is a compile-time attribute of the entry, so the workgroup ids come from the
// trap temporaries without reading IB_STS2 (which waits for the kernel argument loads).
template <typename Kernel>
__global__ __launch_bounds__(Kernel::kBlockSize)
    __attribute__((cluster_dims(Kernel::kNumSplits, 1, 1))) void fmha_decode_paged5d_entry(
        typename Kernel::Kargs kargs)
{
    Kernel{}(kargs);
}

} // namespace ck_tile
