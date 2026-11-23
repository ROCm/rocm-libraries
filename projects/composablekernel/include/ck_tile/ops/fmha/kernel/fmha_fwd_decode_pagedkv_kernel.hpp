// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include "ck_tile/core/arch/cluster_barrier.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"

#include <algorithm>
#include <string>

// gfx125 decode attention (one query token per sequence) over a paged fp8 KV cache; the pipeline
// (BlockFmhaFwdDecodePagedKVPipelineTdm) describes the cache layout and what one workgroup
// computes.
//
// Grid (kNumSplits * groups, nhead_k, batch). The splits of one (batch, kv head) divide its key
// range among them. Each kNumSplits consecutive splits are one workgroup cluster: each stores its
// O / l in bf16 and (m, l) to the workspace, and after a cluster barrier every split merges its
// share of the rows. With several clusters per (batch, kv head) (groups), each cluster stores its
// merged rows and the last one to finish a row block merges the clusters (MergeGroups).
//
// The grid is sized for an upper bound of the context lengths; each (batch, kv head) spreads its
// own key range over it, and clusters left without pages exit at once. So a launch depends only
// on the bound and can be captured in a HIP graph.

namespace ck_tile {

// num_splits workgroups per cluster, num_groups clusters per (batch, kv head)
struct FmhaFwdDecodePagedKVConfig
{
    index_t num_splits;
    // 16-token pages per split for the longest range (fmha_fwd_decode_pagedkv_split_pages); the
    // kernel sizes splits from the grid and each sequence's own range
    index_t pages_per_split;
    bool wave_split;
    index_t num_groups = 1;
};

// The most 16-token pages the key range of any context up to max_seqlen_k covers: a window of W
// keys touches up to ceil(W / 16) + 1 pages, and with larger cache pages the range starts on a
// 4-page boundary (see FmhaFwdDecodePagedKVKernel), up to 3 pages earlier.
CK_TILE_HOST index_t fmha_fwd_decode_pagedkv_max_span(index_t max_seqlen_k,
                                                      index_t window,
                                                      index_t page_size = 16)
{
    index_t span = (max_seqlen_k + 15) / 16;
    if(window > 0)
        span = std::min<index_t>(span, (window + 15) / 16 + 1 + (page_size > 16 ? 3 : 0));
    return std::max<index_t>(span, 1);
}

// Pages per split the kernel uses for a range of span 16-token pages on this grid: the range
// spread over all splits in whole steps, at least 8 pages for cooperative splits of several
// clusters (FmhaFwdDecodePagedKVKernel computes the same on the device).
CK_TILE_HOST index_t fmha_fwd_decode_pagedkv_split_pages(index_t span,
                                                         index_t num_splits,
                                                         index_t num_groups,
                                                         bool wave_split)
{
    const index_t step  = wave_split ? 16 : 4;
    const index_t total = num_splits * num_groups;
    const index_t cover = ((span + total - 1) / total + step - 1) / step * step;
    return std::max<index_t>(cover, wave_split ? 16 : (num_groups > 1 ? 8 : 4));
}

// Layout for key ranges of up to span pages and heads = batch * nhead_k kv heads.
// - Up to 16 pages: one split, wave-split above 12 pages.
// - Up to 256 pages: one cluster of cooperative splits, as many as 4-page steps allow (at most
//   16), at least 8 pages each once the grid would exceed 256 workgroups, 16 beyond about 640.
//   With more heads than pages (128 at most; from 64 heads above 192 pages): one wave-split
//   split per started 64 pages within 480 workgroups, one instead of two up to 192 pages.
// - Up to 1024 pages: one wave-split cluster within 512 workgroups; above 512 pages only for 3
//   or more heads.
// - Otherwise several clusters (groups) of the shortest 8-, 12- or 16-page cooperative splits
//   within 192 splits and 512 workgroups, else wave-split splits for about 384 workgroups, at
//   most 480.
// The workgroup limits keep the grid resident at once: the workgroups of a cluster take one
// workgroup processor each, a processor's LDS holds two wave-split workgroups, and a shader
// engine runs at most 15 clusters.
// max_cluster_splits: workgroups a cluster holds on the device (ck_tile::get_max_cluster_size);
// a cluster takes no more splits than that, and below 2 every (batch, kv head) is one split.
CK_TILE_HOST FmhaFwdDecodePagedKVConfig
fmha_fwd_decode_pagedkv_config(index_t batch,
                               index_t nhead_k,
                               index_t max_seqlen_k,
                               index_t window,
                               index_t page_size          = 16,
                               index_t max_cluster_splits = kFmhaFwdDecodeMaxClusterSplits)
{
    const index_t span  = fmha_fwd_decode_pagedkv_max_span(max_seqlen_k, window, page_size);
    const auto round_up = [](index_t x, index_t m) { return (x + m - 1) / m * m; };
    const auto ceil_div = [](index_t x, index_t y) { return (x + y - 1) / y; };
    const index_t heads = batch * nhead_k;
    // splits one cluster takes on this device
    const index_t kC = std::clamp<index_t>(max_cluster_splits, 1, kFmhaFwdDecodeMaxClusterSplits);
    const auto make  = [&](index_t pps, bool wave_split) {
        const index_t total  = ceil_div(span, pps);
        const index_t groups = ceil_div(total, kC);
        const index_t splits = ceil_div(total, groups);
        return FmhaFwdDecodePagedKVConfig{
            splits,
            fmha_fwd_decode_pagedkv_split_pages(span, splits, groups, wave_split),
            wave_split,
            groups};
    };
    // the splits merge across a cluster: without clusters a (batch, kv head) is one split
    if(kC < 2)
        return make(round_up(span, span > 12 ? 16 : 4), span > 12);
    // the most wave-split splits per head that keep the grid within 480 workgroups
    const index_t wave_max = std::max<index_t>(1, 480 / std::max<index_t>(1, heads));
    if(span <= 16)
        return make(round_up(span, span > 12 ? 16 : 4), span > 12);
    if(span <= 256)
    {
        if(heads >= (span > 192 ? 64 : std::min<index_t>(span, 128) + 1))
        {
            index_t n = std::min({ceil_div(span, 64), wave_max, kC});
            // two splits this short take longer than one, which needs no merge
            if(n == 2 && span <= 192)
                n = 1;
            return make(round_up(ceil_div(span, n), 16), true);
        }
        index_t pps = round_up(ceil_div(span, std::min<index_t>(kC, ceil_div(span, 4))), 4);
        if(heads * ceil_div(span, pps) > 256)
            pps = std::max<index_t>(pps, 8);
        if(heads * ceil_div(span, pps) > 640)
            pps = std::max<index_t>(pps, 16);
        return make(pps, false);
    }
    const index_t wave_pps = round_up(ceil_div(span, kC), 16);
    const index_t wave_wgs = heads * ceil_div(span, wave_pps);
    if(span <= 1024 && wave_wgs <= 512 && (span <= 512 || heads >= 3))
        return make(wave_pps, true);
    for(index_t pps : {8, 12, 16})
        if(ceil_div(span, pps) <= 192 && heads * ceil_div(span, pps) <= 512)
            return make(pps, false);
    index_t pps = std::max<index_t>(32, round_up(ceil_div(span * heads, 384), 16));
    if(ceil_div(span, pps) > wave_max)
        pps = round_up(ceil_div(span, wave_max), 16);
    return make(pps, true);
}

CK_TILE_HOST size_t fmha_fwd_decode_pagedkv_workspace_bytes(index_t batch,
                                                            index_t nhead_k,
                                                            index_t num_splits,
                                                            index_t num_groups = 1)
{
    // per split: bf16 O / l [batch][nhead_k][groups][splits][16][128] and fp32 (m, l) [..][16][2];
    // per group the same with one split
    if(batch <= 0 || nhead_k <= 0 || num_splits <= 0 || num_groups <= 0)
        return 0;
    const size_t heads = static_cast<size_t>(batch) * nhead_k;
    const size_t rows  = heads * num_groups * num_splits * 16;
    const size_t grows = num_groups > 1 ? heads * num_groups * 16 : 0;
    return (rows + grows) * (128 * 2 + 2 * sizeof(float));
}

// At least fmha_fwd_decode_pagedkv_workspace_bytes of the selected configuration for any context
// length, window and page size, and non-decreasing in batch * nhead_k, so one buffer sized for
// the largest batch serves every call. Per kv head fmha_fwd_decode_pagedkv_config makes at most 16
// splits in one cluster (up to 128 heads), at most 192 and 512 / heads cooperative splits, or
// about 384 / heads and at most 480 / heads wave-split splits; a test checks the bound.
// max_cluster_splits as for fmha_fwd_decode_pagedkv_config.
CK_TILE_HOST size_t fmha_fwd_decode_pagedkv_max_workspace_bytes(
    index_t batch, index_t nhead_k, index_t max_cluster_splits = kFmhaFwdDecodeMaxClusterSplits)
{
    if(batch <= 0 || nhead_k <= 0)
        return 0;
    const size_t cluster =
        std::clamp<index_t>(max_cluster_splits, 1, kFmhaFwdDecodeMaxClusterSplits);
    if(cluster < 2)
        return fmha_fwd_decode_pagedkv_workspace_bytes(batch, nhead_k, 1);
    // split slots of all kv heads; with several groups splits * groups < total + groups, plus one
    // merge slot per group
    const auto slots = [cluster](size_t h) {
        const size_t one_cluster = h <= 128 ? 16 : 1;
        const size_t total       = std::max<size_t>(
            {one_cluster, std::min<size_t>(192, 512 / h), (384 + h - 1) / h, 480 / h});
        const size_t groups = (total + cluster - 1) / cluster;
        return h * (groups > 1 ? total + 2 * groups : total);
    };
    // fewer heads can need more in total (more splits per head)
    const size_t heads = static_cast<size_t>(batch) * nhead_k;
    size_t most        = slots(heads);
    for(size_t h = 1; h < std::min<size_t>(heads, 512); ++h)
        most = std::max(most, slots(h));
    return most * 16 * (128 * 2 + 2 * sizeof(float));
}

// one per (batch, kv head, row block) for any configuration
CK_TILE_HOST size_t fmha_fwd_decode_pagedkv_num_counters(index_t batch, index_t nhead_k)
{
    if(batch <= 0 || nhead_k <= 0)
        return 0;
    return static_cast<size_t>(batch) * nhead_k * kFmhaFwdDecodeMaxClusterSplits;
}

template <typename FmhaPipeline_>
struct FmhaFwdDecodePagedKVKernel
{
    using FmhaPipeline = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using Problem      = typename FmhaPipeline::Problem;
    using Traits       = typename Problem::Traits;
    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;

    static constexpr index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    static constexpr bool kWaveSplit        = FmhaPipeline::kWaveSplit;
    static constexpr index_t kPageBlockSize = FmhaPipeline::kPageBlockSize;
    static constexpr index_t kHeadDim       = FmhaPipeline::kHeadDim;
    static constexpr index_t kM0            = FmhaPipeline::kM0;
    static constexpr index_t kN0            = FmhaPipeline::kN0;
    static constexpr index_t kSubPages      = FmhaPipeline::kSubPages;
    static constexpr index_t kStepPages     = FmhaPipeline::kStepPages;
    // splits per cluster, and whether a (batch, kv head) takes several clusters
    static constexpr index_t kNumSplits = Traits::kNumSplits;
    static constexpr bool kHasGroups    = Traits::kHasGroups;
    // workgroups along x that form one cluster (make_cluster_kernel)
    static constexpr index_t kClusterSize = kNumSplits;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        return std::string("fmha_fwd_decode_pagedkv_d") + std::to_string(kHeadDim) + "_" +
               t2s<QDataType>::name + "_" + t2s<KDataType>::name + "kv_ps" +
               std::to_string(kPageBlockSize) + "_" + FmhaPipeline::name + "_s" +
               std::to_string(kNumSplits) + (kHasGroups ? "_groups" : "_ngroups");
    }

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const int32_t* block_table_ptr;
        const int32_t* seqlen_k_ptr;
        const float* k_descale_ptr;
        const float* v_descale_ptr;
        void* o_ptr;
        uint16_t* ws_o_ptr; // per split: O / l, bf16 [batch][nhead_k][groups][splits][kM0][hdim]
        float* ws_ml_ptr;   // per split: (m, l) [batch][nhead_k][groups][splits][kM0][2]
        index_t nhead_q;
        index_t nhead_k;
        index_t nhead_ratio_qk; // nhead_q / nhead_k, so the kernel does not divide
        index_t block_table_stride;
        index_t window_size;
        index_t num_groups; // read with kHasGroups only
        // ceil(2^32 / (kNumSplits * num_groups)), for kHasGroups: the high half of a product with
        // it divides by the split count
        uint32_t splits_rcp;
        float scale_s;
        // used with kHasGroups only; last, so that the arguments the prologue needs come first
        uint16_t* gws_o_ptr; // per group, as ws_o_ptr with one split
        float* gws_ml_ptr;
        int32_t* counters_ptr; // per (batch, kv head, row block)
    };

    // q_ptr, o_ptr: [batch][nhead_q][hdim]; k_ptr, v_ptr: the caches (see the pipeline);
    // block_table_ptr: [batch][block_table_stride] cache page ids; seqlen_k_ptr: [batch] context
    // lengths, read on the device; k_descale_ptr, v_descale_ptr: one float each.
    // window_size: keys attended, the query's own included; 0 = no window.
    // workspace_ptr: fmha_fwd_decode_pagedkv_workspace_bytes, needs no initialization.
    // counters_ptr: fmha_fwd_decode_pagedkv_num_counters int32; zero them once, every launch
    // leaves them zero, so launches of any shape can share them unless they run concurrently.
    // The kernel checks none of this: nhead_q / nhead_k must be a whole number of at most kM0.
    CK_TILE_HOST static Kargs MakeKargs(const void* q_ptr,
                                        const void* k_ptr,
                                        const void* v_ptr,
                                        void* o_ptr,
                                        const int32_t* block_table_ptr,
                                        index_t block_table_stride,
                                        const int32_t* seqlen_k_ptr,
                                        const float* k_descale_ptr,
                                        const float* v_descale_ptr,
                                        void* workspace_ptr,
                                        int32_t* counters_ptr,
                                        index_t batch,
                                        index_t nhead_q,
                                        index_t nhead_k,
                                        index_t window_size,
                                        index_t num_groups,
                                        float scale_s)
    {
        const size_t heads = static_cast<size_t>(batch) * nhead_k;
        const size_t rows  = heads * num_groups * kNumSplits * kM0;
        const size_t grows = heads * num_groups * kM0;
        auto* ws           = static_cast<uint8_t*>(workspace_ptr);
        auto* gws          = ws == nullptr ? nullptr : ws + rows * (kHeadDim * 2 + 8);
        return {q_ptr,
                k_ptr,
                v_ptr,
                block_table_ptr,
                seqlen_k_ptr,
                k_descale_ptr,
                v_descale_ptr,
                o_ptr,
                reinterpret_cast<uint16_t*>(ws),
                ws == nullptr ? nullptr : reinterpret_cast<float*>(ws + rows * kHeadDim * 2),
                nhead_q,
                nhead_k,
                nhead_q / nhead_k,
                block_table_stride,
                window_size,
                num_groups,
                static_cast<uint32_t>(((uint64_t{1} << 32) + kNumSplits * num_groups - 1) /
                                      (kNumSplits * num_groups)),
                scale_s,
                reinterpret_cast<uint16_t*>(gws),
                gws == nullptr ? nullptr : reinterpret_cast<float*>(gws + grows * kHeadDim * 2),
                counters_ptr};
    }

    CK_TILE_HOST static constexpr dim3
    GridSize(index_t batch, index_t nhead_k, index_t num_groups = 1)
    {
        return dim3(kNumSplits * num_groups, nhead_k, batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

#if defined(__gfx125__)
    using v4i = typename FmhaPipeline::v4i;

    template <typename T, typename P>
    CK_TILE_DEVICE static auto ptr_as(P* p)
    {
        return FmhaPipeline::template ptr_as<T>(p);
    }

    // row i_row of kv head g of sequence b in the output
    CK_TILE_DEVICE static uint16_t* ORow(const Kargs& a, index_t b, index_t g, index_t i_row)
    {
        return static_cast<uint16_t*>(a.o_ptr) +
               (static_cast<long_index_t>(b) * a.nhead_q + g * a.nhead_ratio_qk + i_row) * kHeadDim;
    }

    CK_TILE_DEVICE void operator()(const Kargs& a) const
    {
        alignas(16) __shared__ char smem_ptr[FmhaPipeline::GetSmemSize()];

        // groups of kNumSplits along x, one cluster each
        const index_t gi = kHasGroups ? blockIdx.x / kNumSplits : 0;
        const index_t s  = kHasGroups ? blockIdx.x % kNumSplits : blockIdx.x;
        const index_t g = blockIdx.y, b = blockIdx.z;
        const index_t sg  = gi * kNumSplits + s; // split index within the (batch, kv head)
        const index_t gqa = a.nhead_ratio_qk;

        const index_t ctx   = a.seqlen_k_ptr[b];
        const index_t lo    = a.window_size > 0 ? max(0, ctx - a.window_size) : 0;
        const index_t pg_hi = (ctx + kN0 - 1) / kN0;
        // with larger cache pages the range starts on a 4-page boundary, so that a wave-split
        // step's four pages of a wave lie in one cache page (positions before lo are masked)
        const index_t pg_lo = kSubPages > 1 ? lo / kN0 / 4 * 4 : lo / kN0;
        const index_t span  = max(pg_hi - pg_lo, 1);
        // this range over all splits of the grid in whole steps
        // (fmha_fwd_decode_pagedkv_split_pages); cooperative splits of several clusters take at
        // least 8 pages, so a range shorter than the grid was sized for leaves later clusters
        // empty, and they skip the merge
        constexpr index_t kMinPages = kWaveSplit ? 16 : (kHasGroups ? 8 : 4);
        index_t per_split;
        if constexpr(kHasGroups)
            per_split =
                __umulhi(static_cast<uint32_t>(span + kNumSplits * a.num_groups - 1), a.splits_rcp);
        else
            per_split = (span + kNumSplits - 1) / kNumSplits;
        const index_t cover = (per_split + kStepPages - 1) / kStepPages * kStepPages;
        const index_t pps   = max(cover, kMinPages);
        // groups without pages take no part in the merge
        if(gi > 0 && gi * kNumSplits * pps >= span)
            return;
        const typename FmhaPipeline::KeyRange range{
            ctx, lo, pg_lo + sg * pps, min(pg_hi, pg_lo + (sg + 1) * pps)};

        // this split's result in registers: each thread holds 16 head dims of one row
        const auto res =
            FmhaPipeline{}(static_cast<const QDataType*>(a.q_ptr),
                           b,
                           a.nhead_q,
                           gqa,
                           static_cast<const KDataType*>(a.k_ptr),
                           static_cast<const VDataType*>(a.v_ptr),
                           a.block_table_ptr + static_cast<long_index_t>(b) * a.block_table_stride,
                           a.nhead_k,
                           g,
                           range,
                           a.k_descale_ptr,
                           a.v_descale_ptr,
                           a.scale_s,
                           smem_ptr);
        const float vd = res.v_descale;

        if constexpr(kNumSplits == 1)
        {
            // one split: the output, no partials
            if(res.row < gqa)
                FmhaPipeline::StoreRows(
                    res, ORow(a, b, g, 0), res.l > 0.f ? vd / res.l : 0.f, [](float x, float y) {
                        return FmhaPipeline::pack_o(x, y);
                    });
            return;
        }
        else
        {
            // partials: O / l in bf16 and (m, l)
            const long_index_t head = static_cast<long_index_t>(b) * a.nhead_k + g;
            const long_index_t part0 =
                ((head * (kHasGroups ? a.num_groups : 1) + gi) * kNumSplits) * kM0;
            FmhaPipeline::StoreRows(res,
                                    a.ws_o_ptr + (part0 + s * kM0) * kHeadDim,
                                    res.l > 0.f ? 1.f / res.l : 0.f,
                                    [](float x, float y) { return FmhaPipeline::pack_bf16(x, y); });
            if(FmhaPipeline::IsRowOwner(res))
                *ptr_as<float2>(a.ws_ml_ptr + (part0 + s * kM0 + res.row) * 2) =
                    make_float2(res.m, res.l);
            asm volatile("s_wait_storecnt 0x0" ::: "memory");
            cluster_barrier();
            asm volatile("global_inv scope:SCOPE_SE" ::: "memory");

            // merge the splits: split s takes rows [s*kRowsPer, +kRowsPer); thread -> (row, 2 dims)
            constexpr index_t kRowsPer = (kM0 + kNumSplits - 1) / kNumSplits;
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
                    mlv[i] = *ptr_as<float2>(a.ws_ml_ptr + (part0 + i * kM0 + mrow) * 2);
                    oi[i] =
                        *ptr_as<uint32_t>(a.ws_o_ptr + (part0 + i * kM0 + mrow) * kHeadDim + d2);
                }
                float mm = -numeric<float>::infinity();
                for(index_t i = 0; i < kNumSplits; ++i)
                    mm = fmaxf(mm, mlv[i].x);
                float ll = 0.f, o0 = 0.f, o1 = 0.f;
                for(index_t i = 0; i < kNumSplits; ++i)
                {
                    const float wgt =
                        mlv[i].y > 0.f ? mlv[i].y * FmhaPipeline::exp2(mlv[i].x - mm) : 0.f;
                    ll += wgt;
                    o0 += wgt * FmhaPipeline::bf16_to_f32(oi[i] & 0xffff);
                    o1 += wgt * FmhaPipeline::bf16_to_f32(oi[i] >> 16);
                }
                if constexpr(!kHasGroups)
                {
                    const float f = ll > 0.f ? vd / ll : 0.f;
                    *ptr_as<uint32_t>(ORow(a, b, g, mrow) + d2) =
                        FmhaPipeline::pack_o(o0 * f, o1 * f);
                }
                else
                {
                    // this group's rows: O / L and (M, L), at device scope for the other groups
                    const long_index_t grow = (head * a.num_groups + gi) * kM0 + mrow;
                    const float f           = ll > 0.f ? 1.f / ll : 0.f;
                    __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_o_ptr + grow * kHeadDim + d2),
                                            FmhaPipeline::pack_bf16(o0 * f, o1 * f),
                                            __ATOMIC_RELAXED,
                                            __MEMORY_SCOPE_DEVICE);
                    if(d2 == 0)
                    {
                        __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_ml_ptr + grow * 2),
                                                __float_as_uint(mm),
                                                __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
                        __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_ml_ptr + grow * 2 + 1),
                                                __float_as_uint(ll),
                                                __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
                    }
                }
            }
            if constexpr(kHasGroups)
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
        constexpr index_t kRowsPer = (kM0 + kNumSplits - 1) / kNumSplits;
        __shared__ int32_t last;
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            int32_t* counter = a.counters_ptr + head * kNumSplits + s;
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
            const long_index_t row0  = head * a.num_groups * kM0 + mrow;
            const auto ml_res        = make_wave_buffer_resource(a.gws_ml_ptr, 0x7ffffffe);
            const auto o_res         = make_wave_buffer_resource(a.gws_o_ptr, 0x7ffffffe);
            for(index_t i0 = 0; i0 < groups; i0 += kChunk)
            {
                float2 mli[kChunk];
                uint32_t oi[kChunk];
                for(index_t j = 0; j < kChunk; ++j)
                {
                    const long_index_t grow = row0 + min(i0 + j, groups - 1) * kM0;
                    const auto ml2 =
                        amd_buffer_load_impl<float, 2, amd_buffer_coherence_enum::DEVICE>(
                            ml_res, static_cast<index_t>(grow * 2 * sizeof(float)), 0);
                    mli[j] = make_float2(ml2[number<0>{}], ml2[number<1>{}]);
                    oi[j]  = __float_as_uint(
                        amd_buffer_load_impl<float, 1, amd_buffer_coherence_enum::DEVICE>(
                            o_res,
                            static_cast<index_t>((grow * kHeadDim + d2) * 2),
                            0)[number<0>{}]);
                }
                for(index_t j = 0; j < kChunk; ++j)
                {
                    const float mi = mli[j].x, li = mli[j].y;
                    if(i0 + j >= groups || !(li > 0.f))
                        continue;
                    // online merge: the running sums follow the largest max seen so far
                    const float mn = fmaxf(mm, mi);
                    const float c0 = FmhaPipeline::exp2(mm - mn), c1 = FmhaPipeline::exp2(mi - mn);
                    ll = ll * c0 + li * c1;
                    o0 = o0 * c0 + li * c1 * FmhaPipeline::bf16_to_f32(oi[j] & 0xffff);
                    o1 = o1 * c0 + li * c1 * FmhaPipeline::bf16_to_f32(oi[j] >> 16);
                    mm = mn;
                }
            }
            const float f                               = ll > 0.f ? vd / ll : 0.f;
            *ptr_as<uint32_t>(ORow(a, b, g, mrow) + d2) = FmhaPipeline::pack_o(o0 * f, o1 * f);
        }
    }
#else
    CK_TILE_DEVICE void operator()(Kargs) const {}
#endif
};

} // namespace ck_tile
