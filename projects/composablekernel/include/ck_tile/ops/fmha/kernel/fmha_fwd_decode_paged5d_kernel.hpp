// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"

// gfx125 fp8 decode attention over the SHUFFLE 5D paged KV cache (sglang layout), cache pages of
// page_size = 16, 64 or 128 tokens:
//   K [num_blocks][nhead_k][hdim / 16][page_size tokens][16 B]
//   V [num_blocks][nhead_k][page_size / 16][hdim][16 tokens]
// One query token per sequence, hdim 128, fp8 e4m3 K/V with per-tensor descales, bf16 or fp16 Q
// (quantized per row in the kernel) and output of the same type. The q heads of one kv head (at
// most 16) are the 16 columns of S^T = K Q^T. Below, a page is 16 tokens; a cache page holds
// page_size / 16.
//
// Grid (kNumSplits * groups, nhead_k, batch). Each kNumSplits consecutive splits are one workgroup
// cluster; each stores O / l in bf16 and (m, l), and after a cluster barrier every split merges
// its share of the rows (with several groups, see MergeGroups). The grid is sized for
// max_seqlen_k; each (batch, kv head) spreads its own key range over it, and clusters left
// without pages exit at once.
//
// K and V pages are copied to LDS with TDM gathers (rows of 16-byte vectors of one kv head): in
// this layout the LDS bytes are WMMA operands as they are (S^T = K Q^T and O^T += V^T P^T, no
// transposed reads).

namespace ck_tile {

struct FmhaDecodePaged5dHostArgs
{
    const void* q_ptr;           // [batch][nhead_q][hdim] bf16, or fp16 with qo_fp16
    const void* k_ptr;           // 5D fp8
    const void* v_ptr;           // 5D fp8
    const int32_t* block_table;  // [batch][max_blocks_per_seq]
    const int32_t* seqlen_k_ptr; // [batch]
    const float* k_descale_ptr;  // one float
    const float* v_descale_ptr;  // one float
    void* o_ptr;                 // [batch][nhead_q][hdim], the type of Q
    void* workspace_ptr;         // fmha_decode_paged5d_workspace_bytes, no initialization needed
    // fmha_decode_paged5d_num_counters int32; zero them once, every launch leaves them zero, so
    // launches of any shape can share them as long as they do not run concurrently
    int32_t* counters_ptr;
    index_t batch;
    index_t nhead_q;
    index_t nhead_k;
    index_t max_blocks_per_seq;
    index_t page_size;       // tokens per cache page: 16, 64 or 128
    index_t window;          // keys attended, the query's own included; 0 = no window
    index_t pages_per_split; // lengthened in the kernel if the grid would not cover a range
    index_t num_groups;      // clusters per (batch, kv head)
    float scale_s;
    bool qo_fp16 = false; // Q and O are fp16 instead of bf16
};

// workgroups of a cluster: a gfx1250 cluster holds up to 16
inline constexpr index_t kMaxClusterSplits = 16;

// num_splits workgroups per cluster, num_groups clusters per (batch, kv head)
struct FmhaDecodePaged5dConfig
{
    index_t num_splits;
    index_t pages_per_split;
    bool wave_split;
    index_t num_groups = 1;
};

// The most 16-token pages the key range of any context up to max_seqlen_k covers: a window of W
// keys touches up to ceil(W / 16) + 1 pages, and with larger cache pages the range starts on a
// 4-page boundary (see FmhaDecodePaged5dKernel::Run), up to 3 pages earlier.
CK_TILE_HOST index_t fmha_decode_paged5d_max_span(index_t max_seqlen_k,
                                                  index_t window,
                                                  index_t page_size = 16)
{
    index_t span = (max_seqlen_k + 15) / 16;
    if(window > 0)
        span = std::min<index_t>(span, (window + 15) / 16 + 1 + (page_size > 16 ? 3 : 0));
    return std::max<index_t>(span, 1);
}

// Up to 256 pages: one cluster of cooperative splits, as many as 4-page steps allow (at most 16),
// at least 8 pages each once the grid would exceed 256 workgroups, 16 beyond about 640. Up to 1024
// pages: one wave-split cluster while the grid has 64 to 512 workgroups. Longer: several clusters
// (groups) of the shortest 8-, 12- or 16-page cooperative splits within 512 workgroups, else
// wave-split splits for about 384 workgroups.
CK_TILE_HOST FmhaDecodePaged5dConfig fmha_decode_paged5d_config(
    index_t batch, index_t nhead_k, index_t max_seqlen_k, index_t window, index_t page_size = 16)
{
    const index_t span  = fmha_decode_paged5d_max_span(max_seqlen_k, window, page_size);
    const auto round_up = [](index_t x, index_t m) { return (x + m - 1) / m * m; };
    const auto ceil_div = [](index_t x, index_t y) { return (x + y - 1) / y; };
    const index_t heads = batch * nhead_k;
    const auto make     = [&](index_t pps, bool wave_split) {
        const index_t total  = ceil_div(span, pps);
        const index_t groups = ceil_div(total, kMaxClusterSplits);
        return FmhaDecodePaged5dConfig{ceil_div(total, groups), pps, wave_split, groups};
    };
    constexpr index_t kC = kMaxClusterSplits;
    if(span <= kC * 16)
    {
        index_t pps = round_up(ceil_div(span, std::min<index_t>(kC, ceil_div(span, 4))), 4);
        if(heads * ceil_div(span, pps) > 256)
            pps = std::max<index_t>(pps, 8);
        if(heads * ceil_div(span, pps) > 640)
            pps = std::max<index_t>(pps, 16);
        return make(pps, false);
    }
    const index_t wave_pps = round_up(ceil_div(span, kC), 16);
    const index_t wave_wgs = heads * ceil_div(span, wave_pps);
    if(span <= kC * 64 && wave_wgs <= 512 && (span <= 512 || wave_wgs >= 64))
        return make(wave_pps, true);
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
    // per group the same with one split
    const size_t heads = static_cast<size_t>(batch) * nhead_k;
    const size_t rows  = heads * num_groups * num_splits * 16;
    const size_t grows = num_groups > 1 ? heads * num_groups * 16 : 0;
    return (rows + grows) * (128 * 2 + 2 * sizeof(float));
}

// one per (batch, kv head, row block) for any configuration
CK_TILE_HOST size_t fmha_decode_paged5d_num_counters(index_t batch, index_t nhead_k)
{
    return static_cast<size_t>(batch) * nhead_k * kMaxClusterSplits;
}

template <bool kWaveSplit_,
          index_t kNumSplits_,
          bool kGroups_      = false,
          index_t kPageSize_ = 16,
          bool kQoFp16_      = false>
struct FmhaDecodePaged5dKernel
{
    static constexpr bool kQoFp16       = kQoFp16_; // Q and O are fp16 instead of bf16
    static constexpr bool kWaveSplit    = kWaveSplit_;
    static constexpr index_t kNumSplits = kNumSplits_;
    static constexpr index_t kPageSize  = kPageSize_; // tokens per cache page
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
    // 16-token pages per cache page
    static constexpr index_t kSubPages = kPageSize / kPage;
    static_assert(1 <= kNumSplits && kNumSplits <= kMaxClusterSplits);
    static_assert(kPageSize == 16 || kPageSize == 64 || kPageSize == 128);
    // LDS K of a wave-split step: four page-16 images, or with larger cache pages the 64 tokens as
    // they lie in the cache ([hdim / 16][64 tokens][16 B]); offsets of a page and a 16-byte chunk
    static constexpr index_t kKPageStride  = kWaveSplit && kSubPages > 1 ? 256 : kPageBytes;
    static constexpr index_t kKChunkStride = kWaveSplit && kSubPages > 1 ? 1024 : 256;

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
        index_t num_groups; // read with kGroups only
        // ceil(2^32 / (kNumSplits * num_groups)), for kGroups: the high half of a product with it
        // divides by the split count
        uint32_t splits_rcp;
        float scale_s;
        // used with kGroups only; last, so that the arguments the prologue needs come first
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
                h.num_groups,
                static_cast<uint32_t>(((uint64_t{1} << 32) + kNumSplits * h.num_groups - 1) /
                                      (kNumSplits * h.num_groups)),
                h.scale_s,
                reinterpret_cast<uint16_t*>(gws),
                gws == nullptr ? nullptr : reinterpret_cast<float*>(gws + grows * kHdim * 2),
                h.counters_ptr};
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
        // per wave, so that a wave's O for the intra-workgroup merge reuses only its own stages
        union alignas(16) PerWave
        {
            struct
            {
                alignas(16) uint8_t k[kStages][kWavePages][kPageBytes];
                alignas(16) uint8_t v[kStages][kWavePages][kPageBytes];
            } kv;
            // padded rows keep the column-wise stores of 16 rows off one bank
            alignas(16) float o[kRows][kHdim + 4];
        } wave[kNumWaves];
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
    CK_TILE_DEVICE static float fp16_to_f32(uint32_t x)
    {
        return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(x)));
    }
    CK_TILE_DEVICE static uint32_t pack_fp16(float a, float b)
    {
        return __builtin_bit_cast(uint16_t, static_cast<_Float16>(a)) |
               (static_cast<uint32_t>(__builtin_bit_cast(uint16_t, static_cast<_Float16>(b)))
                << 16);
    }
    CK_TILE_DEVICE static float exp2(float x) { return __builtin_amdgcn_exp2f(x); }

    // TDM gather of kCount rows of kRowBytes (row indices up to 2^31) to consecutive LDS rows
    template <index_t kRowBytes, index_t kCount>
    CK_TILE_DEVICE static void
    tdm_rows(const uint8_t* base, uint8_t* lds, const int32_t (&rows)[kCount])
    {
        static_assert(kCount <= 8 && kRowBytes % 8 == 0);
        TDM_GROUP0 g0{static_cast<uint32_t>(reinterpret_cast<uintptr_t>(to_lds(lds))),
                      reinterpret_cast<uintptr_t>(base),
                      static_cast<uint32_t>(TDMGatherIndexSize::Row32bit_Index),
                      1};
        TDM_GROUP1 g1;
        g1.data_size = 3; // 8-byte units
        g1.tensorDim0(kRowBytes / 8);
        g1.tensorDim1(0x7fffffff);
        g1.tensorDim0Stride(kRowBytes / 8);
        g1.tile_dim0 = kRowBytes / 8;
        g1.tile_dim1 = kCount; // row indices
        TDM_GROUP2 g2;
        TDM_GROUP3 g3;
        for(index_t i = 0; i < kCount; ++i)
        {
            if(i < 4)
                g2.mode1.setRowIndex32(i, static_cast<uint32_t>(rows[i]));
            else
                g3.mode1.setRowIndex32(i - 4, static_cast<uint32_t>(rows[i]));
        }
        __builtin_amdgcn_tensor_load_to_lds(
            bit_cast<uint32x4_t>(amd_wave_read_first_lane(g0.bitfield)),
            amd_wave_read_first_lane(g1.bitfield),
            amd_wave_read_first_lane(g2.bitfield),
            amd_wave_read_first_lane(g3.bitfield),
            int32x8_t{},
            0);
    }
    // pairwise: shorter dependency chains than a running max or sum
    template <index_t N>
    CK_TILE_DEVICE static float tree_max(const float* x)
    {
        if constexpr(N == 1)
            return x[0];
        else
            return fmaxf(tree_max<N / 2>(x), tree_max<N / 2>(x + N / 2));
    }
    template <index_t N>
    CK_TILE_DEVICE static float tree_sum(const float* x)
    {
        if constexpr(N == 1)
            return x[0];
        else
            return tree_sum<N / 2>(x) + tree_sum<N / 2>(x + N / 2);
    }

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
        const index_t gi = kGroups ? blockIdx.x / kNumSplits : 0;
        const index_t s  = kGroups ? blockIdx.x % kNumSplits : blockIdx.x;
        const index_t g = blockIdx.y, b = blockIdx.z;
        const index_t sg  = gi * kNumSplits + s; // split index within the (batch, kv head)
        const index_t gqa = a.gqa;

        constexpr index_t kStepPages = kNumWaves * kWavePages;
        const index_t ctx            = a.seqlen_k[b];
        const index_t lo             = a.window > 0 ? max(0, ctx - a.window) : 0;
        const index_t pg_hi          = (ctx + kPage - 1) / kPage;
        // with larger cache pages the range starts on a 4-page boundary, so that a wave-split
        // step's four pages of a wave lie in one cache page (positions before lo are masked)
        const index_t pg_lo = kSubPages > 1 ? lo / kPage / 4 * 4 : lo / kPage;
        const index_t span  = max(pg_hi - pg_lo, 1);
        // this range over all splits of the grid, in whole steps and at most the configured size
        constexpr index_t kMinPages = kWaveSplit ? 16 : (kGroups ? 8 : 4);
        index_t per_split;
        if constexpr(kGroups)
            per_split =
                __umulhi(static_cast<uint32_t>(span + kNumSplits * a.num_groups - 1), a.splits_rcp);
        else
            per_split = (span + kNumSplits - 1) / kNumSplits;
        // the configured size, but never too short to cover the range with this grid
        const index_t cover = (per_split + kStepPages - 1) / kStepPages * kStepPages;
        const index_t pps   = max(cover, min(a.pages_per_split, max(kMinPages, cover)));
        // groups without pages take no part in the merge
        if(gi > 0 && gi * kNumSplits * pps >= span)
            return;
        const index_t pg0   = pg_lo + sg * pps;
        const index_t pend  = min(pg_hi, pg_lo + (sg + 1) * pps);
        const index_t steps = pend > pg0 ? (pend - pg0 + kStepPages - 1) / kStepPages : 0;

        // block ids of the split's cache pages, 32 per register (lane = cache page relative to
        // the split's first), picked with readlane
        const int32_t* bt  = a.block_table + static_cast<long_index_t>(b) * a.max_blocks_per_seq;
        const index_t blk0 = pg0 / kSubPages;
        const auto load_bt = [&](index_t chunk) {
            return bt[min(blk0 + chunk * 32 + l, max(pend - 1, 0) / kSubPages)];
        };
        int32_t bt_cur = load_bt(0), bt_next = load_bt(1);
        index_t cur_chunk = 0;
        // the descales go out with the prologue loads instead of after the Q quantization
        const float kds = *a.k_descale, vd = *a.v_descale;
        // loads return in order: the block ids, which the first copies wait for, go out first
        __builtin_amdgcn_sched_barrier(0);

        // Q^T operand of row n: 16-byte d chunks 2i + h, quantized per row (absmax) below
        v4i qraw[8];
        {
            const uint16_t* qrow =
                a.q +
                (static_cast<long_index_t>(b) * a.nhead_q + g * gqa + min(n, gqa - 1)) * kHdim;
            for(index_t i = 0; i < 4; ++i)
            {
                const index_t c = 2 * i + h;
                qraw[2 * i]     = *ptr_as<v4i>(qrow + c * 16);
                qraw[2 * i + 1] = *ptr_as<v4i>(qrow + c * 16 + 8);
            }
        }

        // LDS page buffers of wave w
        const auto kbuf = [&](index_t st, index_t q) -> uint8_t* {
            if constexpr(kWaveSplit)
                return &sm.wave[w].kv.k[st][0][0] + q * kKPageStride;
            else
            {
                ignore = q;
                return &sm.k[st][w][0];
            }
        };
        const auto vbuf = [&](index_t st, index_t q) -> uint8_t* {
            if constexpr(kWaveSplit)
                return &sm.wave[w].kv.v[st][q][0];
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
        // block id of the cache page holding page idx (relative to pg0)
        const auto block_of = [&](index_t idx) {
            const index_t rb = (pg0 + idx) / kSubPages - blk0;
            return __builtin_amdgcn_readlane((rb >> 5) == cur_chunk ? bt_cur : bt_next, rb & 31);
        };
        // K and V of step t, one TDM gather each; a page past the split repeats the last one (its V
        // is zeroed once it lands)
        const auto issue = [&](index_t t) {
            const index_t last = pend - 1 - pg0;
            if constexpr(kSubPages == 1)
            {
                int32_t rows[kWavePages];
                for(index_t q = 0; q < kWavePages; ++q)
                    rows[q] = block_of(min(page_idx(t, q), last)) * a.nhead_k + g;
                tdm_rows<kPageBytes>(a.k, kbuf(t % kStages, 0), rows);
                tdm_rows<kPageBytes>(a.v, vbuf(t % kStages, 0), rows);
            }
            else if constexpr(kWaveSplit)
            {
                // the wave's four pages are a quarter of a 64-token cache page: K as eight rows
                // of 64 tokens x 16 B, V as one row of four page-16 images
                const index_t idx = min(page_idx(t, 0), last / 4 * 4);
                const index_t qtr = (pg0 + idx) % kSubPages / 4;
                const index_t row = block_of(idx) * a.nhead_k + g;
                int32_t krows[8];
                for(index_t c = 0; c < 8; ++c)
                    krows[c] = (row * 8 + c) * (kSubPages / 4) + qtr;
                const int32_t vrows[1] = {row * (kSubPages / 4) + qtr};
                tdm_rows<4 * 256>(a.k, kbuf(t % kStages, 0), krows);
                tdm_rows<4 * kPageBytes>(a.v, vbuf(t % kStages, 0), vrows);
            }
            else
            {
                // one page: K as eight rows of 16 tokens x 16 B (a page-16 image), V as one row
                const index_t idx = min(page_idx(t, 0), last);
                const index_t sub = (pg0 + idx) % kSubPages;
                const index_t row = block_of(idx) * a.nhead_k + g;
                int32_t krows[8];
                for(index_t c = 0; c < 8; ++c)
                    krows[c] = (row * 8 + c) * kSubPages + sub;
                const int32_t vrows[1] = {row * kSubPages + sub};
                tdm_rows<256>(a.k, kbuf(t % kStages, 0), krows);
                tdm_rows<kPageBytes>(a.v, vbuf(t % kStages, 0), vrows);
            }
        };
        index_t issued = min(steps, kStages);
        for(index_t t = 0; t < issued; ++t)
            issue(t);
        asm volatile("" ::"s"(kds), "s"(vd));

        float qv[64];
        const auto widen = [&](auto fp16) {
            for(index_t j = 0; j < 8; ++j)
                for(index_t e = 0; e < 4; ++e)
                {
                    const uint32_t u = static_cast<uint32_t>(qraw[j][e]);
                    if constexpr(decltype(fp16)::value)
                    {
                        qv[j * 8 + 2 * e]     = fp16_to_f32(u & 0xffff);
                        qv[j * 8 + 2 * e + 1] = fp16_to_f32(u >> 16);
                    }
                    else
                    {
                        qv[j * 8 + 2 * e]     = bf16_to_f32(u & 0xffff);
                        qv[j * 8 + 2 * e + 1] = bf16_to_f32(u >> 16);
                    }
                }
        };
        widen(bool_constant<kQoFp16>{});
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
        const float scl = qmax / 448.f * kds * a.scale_s * 1.4426950408889634f;

        // step t's copies of this wave are in LDS; zero the V bytes of tokens past the context
        const auto land = [&](index_t t) {
            // two gathers per step
            if(issued > t + 2)
                __builtin_amdgcn_s_wait_tensorcnt(4);
            else if(issued > t + 1)
                __builtin_amdgcn_s_wait_tensorcnt(2);
            else
                __builtin_amdgcn_s_wait_tensorcnt(0);
            // the step's pages are consecutive: if the last lies in the split and the context,
            // every token is valid (the common case, one scalar test instead of one per page)
            const index_t pg_end = pg0 + page_idx(t, kWavePages - 1) + 1;
            if(pg_end <= pend && pg_end * kPage <= ctx)
                return;
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
            const index_t chunk = ((pg0 + t * kStepPages) / kSubPages - blk0) / 32;
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

        // S^T of 16 key rows of stage st (row m: page kq(m), token kt(m) of this wave's pages);
        // lane (n, h) gets rows h*8 + r of column n, unscaled, -inf outside the range when
        // masked (first page pg_first)
        const auto qk_tile = [&](index_t st,
                                 index_t kq,
                                 index_t kt,
                                 index_t pg_first,
                                 index_t tok_h,
                                 float(&sv)[8],
                                 auto masked) {
            const uint8_t* kb = kbuf(st, kq) + kt * 16;
            v16i kf{};
            for(index_t i = 0; i < 4; ++i)
            {
                const v4i x = *ptr_as<v4i>(kb + (2 * i + h) * kKChunkStride);
                for(index_t e = 0; e < 4; ++e)
                    kf[4 * i + e] = x[e];
            }
            const v8f st8 = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(
                kf, qf, 0, v8f{0, 0, 0, 0, 0, 0, 0, 0}, false, false);
            for(index_t r = 0; r < 8; ++r)
            {
                sv[r] = st8[r];
                if constexpr(decltype(masked)::value)
                {
                    const index_t pos = pg_first * kPage + tok_h + r;
                    if(!(pos >= lo && pos < ctx && pos < pend * kPage))
                        sv[r] = -numeric<float>::infinity();
                }
            }
        };
        // pages [pg, pg + count) need no mask
        const auto inside = [&](index_t pg, index_t count) {
            return pg * kPage >= lo && (pg + count) * kPage <= ctx && pg + count <= pend;
        };
        // P scaled by 448 (the row max lands on e4m3's largest normal): exp2(s * scl + c)
        constexpr float kLog2P = 8.807354922057604f; // log2(448)

        if constexpr(kWaveSplit)
        {
            for(index_t t = 0; t < steps; ++t)
            {
                const index_t st = t % kStages;
                land(t);
                // tile j: rows 0-7 are tokens (j%2)*8 + m of page j/2, rows 8-15 those of page
                // 2 + j/2, so lane (n, h) gets all tokens of pages 2h and 2h + 1: byte j*8 + r of
                // its P^T operand is token (j%2)*8 + r of page 2h + j/2, as on the V^T operand
                const index_t pgw = pg0 + page_idx(t, 0);
                float sv[kWavePages][8];
                const auto qk4 = [&](auto masked) {
                    for(index_t j = 0; j < kWavePages; ++j)
                        qk_tile(st,
                                (n < 8 ? 0 : 2) + j / 2,
                                (j % 2) * 8 + (n & 7),
                                pgw + 2 * h + j / 2,
                                (j % 2) * 8,
                                sv[j],
                                masked);
                };
                if(inside(pgw, kWavePages))
                    qk4(bool_constant<false>{});
                else
                    qk4(bool_constant<true>{});
                float mx          = tree_max<kWavePages * 8>(&sv[0][0]);
                mx                = fmaxf(mx, __shfl_xor(mx, 16, 32)) * scl;
                const float m_new = fmaxf(m_run, mx);
                const float m_use = is_neg_inf(m_new) ? 0.f : m_new;
                const float c     = kLog2P - m_use;
                float pv[kWavePages][8];
                int32_t pk[8];
                for(index_t j = 0; j < kWavePages; ++j)
                    for(index_t r2 = 0; r2 < 4; ++r2)
                    {
                        const float p0    = exp2(fmaf(sv[j][2 * r2], scl, c));
                        const float p1    = exp2(fmaf(sv[j][2 * r2 + 1], scl, c));
                        pv[j][2 * r2]     = p0;
                        pv[j][2 * r2 + 1] = p1;
                        const index_t e   = j * 8 + 2 * r2; // operand byte
                        pk[e / 4]         = (e & 3) == 0
                                                ? __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, 0, false)
                                                : __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, pk[e / 4], true);
                    }
                float ps = tree_sum<kWavePages * 8>(&pv[0][0]);
                ps       = ps + __shfl_xor(ps, 16, 32);
                // rescale only when a row max grew
                if(__builtin_amdgcn_ballot_w32(m_new > m_run) != 0)
                {
                    const float alpha = exp2(m_run - m_use);
                    l_run *= alpha;
                    for(index_t db = 0; db < 8; ++db)
                        acc[db] *= alpha;
                }
                l_run += ps;
                m_run = m_new;
                const v8i pf{pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6], pk[7]};
                for(index_t db = 0; db < 8; ++db)
                {
                    const index_t off = (db * 16 + n) * 16;
                    const v4i x       = *ptr_as<v4i>(&sm.wave[w].kv.v[st][2 * h][off]);
                    const v4i y       = *ptr_as<v4i>(&sm.wave[w].kv.v[st][2 * h + 1][off]);
                    const v8i vf{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
                    acc[db] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                        vf, pf, 0, acc[db], false, false);
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
                const index_t pg = pg0 + page_idx(t, 0);
                float sv[8];
                if(inside(pg, 1))
                    qk_tile(st, 0, n, pg, h * 8, sv, bool_constant<false>{});
                else
                    qk_tile(st, 0, n, pg, h * 8, sv, bool_constant<true>{});
                float mx = tree_max<8>(sv);
                mx       = fmaxf(mx, __shfl_xor(mx, 16, 32)) * scl;
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
                const float c     = kLog2P - m_use;
                float pv[8];
                for(index_t r = 0; r < 8; ++r)
                    pv[r] = exp2(fmaf(sv[r], scl, c));
                float ps   = tree_sum<8>(pv);
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
                const float alpha = exp2(m_run - m_use);
                l_run             = l_run * alpha + psum;
                m_run             = m_new;
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
            // merge the waves; lane (n, h) holds O[row n][d = db*16 + h*8 + r]. Each wave stores
            // its O relative to its own max (in its own stages, which it is done with), so one
            // barrier suffices; the reader weights the waves.
            if(h == 0)
            {
                sm.ml[0][w][n] = m_run;
                sm.ml[1][w][n] = l_run;
            }
            for(index_t db = 0; db < 8; ++db)
            {
                const v8f x = acc[db];
                float4* dst = ptr_as<float4>(&sm.wave[w].o[n][db * 16 + h * 8]);
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
            float wi[kNumWaves];
            l_row = 0.f;
            for(index_t i = 0; i < kNumWaves; ++i)
            {
                wi[i] = exp2(sm.ml[0][i][row] - mu);
                l_row += sm.ml[1][i][row] * wi[i];
            }
            for(index_t e = 0; e < 16; ++e)
                o[e] = 0.f;
            for(index_t i = 0; i < kNumWaves; ++i)
                for(index_t e4 = 0; e4 < 4; ++e4)
                {
                    const float4 x = *ptr_as<float4>(&sm.wave[i].o[row][d0 + 4 * e4]);
                    o[4 * e4]      = fmaf(wi[i], x.x, o[4 * e4]);
                    o[4 * e4 + 1]  = fmaf(wi[i], x.y, o[4 * e4 + 1]);
                    o[4 * e4 + 2]  = fmaf(wi[i], x.z, o[4 * e4 + 2]);
                    o[4 * e4 + 3]  = fmaf(wi[i], x.w, o[4 * e4 + 3]);
                }
        }
        else
        {
            row   = n;
            m_row = m_run;
            l_row = l_run;
            d0    = 0; // unused: the lane holds d = (bk*4 + w)*16 + h*8 + r
        }

        // O in the output type; partials stay bf16
        const auto pack_out = [&](float x, float y) {
            return kQoFp16 ? pack_fp16(x, y) : pack_bf16(x, y);
        };
        const auto store_rows = [&](uint16_t* base_ptr, float f, auto pack) {
            // base_ptr points at row 0 of this (batch, kv head) block, rows hdim apart
            if constexpr(kWaveSplit)
            {
                uint32_t pk[8];
                for(index_t e = 0; e < 8; ++e)
                    pk[e] = pack(o[2 * e] * f, o[2 * e + 1] * f);
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
                        pk[e] = pack(acc[bk][2 * e] * f, acc[bk][2 * e + 1] * f);
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
                           l_row > 0.f ? vd / l_row : 0.f,
                           pack_out);
            return;
        }
        else
        {
            const long_index_t head = static_cast<long_index_t>(b) * a.nhead_k + g;
            const long_index_t part0 =
                ((head * (kGroups ? a.num_groups : 1) + gi) * kNumSplits) * kRows;
            store_rows(a.ws_o + (part0 + s * kRows) * kHdim,
                       l_row > 0.f ? 1.f / l_row : 0.f,
                       [](float x, float y) { return pack_bf16(x, y); });
            const bool ml_writer = kWaveSplit ? (threadIdx.x % 8 == 0) : (w == 0 && h == 0);
            if(ml_writer)
                *ptr_as<float2>(a.ws_ml + (part0 + s * kRows + row) * 2) =
                    make_float2(m_row, l_row);
            asm volatile("s_wait_storecnt 0x0" ::: "memory");
            cluster_barrier();
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
                        d2) = pack_out(o0 * f, o1 * f);
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
                MergeGroups(a, head, b, g, gqa, s, (span - 1) / (kNumSplits * pps) + 1, vd);
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
                                    index_t groups,
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
                groups - 1;
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
            for(index_t i0 = 0; i0 < groups; i0 += kChunk)
            {
                float2 mli[kChunk];
                uint32_t oi[kChunk];
                for(index_t j = 0; j < kChunk; ++j)
                {
                    const long_index_t grow = row0 + min(i0 + j, groups - 1) * kRows;
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
                    if(i0 + j >= groups || !(li > 0.f))
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
                kQoFp16 ? pack_fp16(o0 * f, o1 * f) : pack_bf16(o0 * f, o1 * f);
        }
    }
#else
    CK_TILE_DEVICE void operator()(const Kargs&) const {}
#endif
};

// Up to 15 splits the cluster size is a compile-time attribute: HIP stream capture drops a cluster
// size given to hipLaunchKernelEx only, so a captured graph would run without clusters. Only
// builds for gfx125 take the attribute; elsewhere the kernel is empty.
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
#define CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER(n) __attribute__((cluster_dims(n, 1, 1)))
#else
#define CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER(n)
#endif

template <typename Kernel>
__global__ __launch_bounds__(Kernel::kBlockSize) CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER(
    Kernel::kNumSplits) void fmha_decode_paged5d_entry(typename Kernel::Kargs kargs)
{
    Kernel{}(kargs);
}

// 16 splits: clang checks cluster_dims against 4 bits, so the cluster size comes with the launch,
// through hipDrvLaunchKernelEx, whose stream capture keeps it.
template <typename Kernel>
__global__ __launch_bounds__(Kernel::kBlockSize) void fmha_decode_paged5d_entry_launch_cluster(
    typename Kernel::Kargs kargs)
{
    Kernel{}(kargs);
}

#undef CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER

} // namespace ck_tile
