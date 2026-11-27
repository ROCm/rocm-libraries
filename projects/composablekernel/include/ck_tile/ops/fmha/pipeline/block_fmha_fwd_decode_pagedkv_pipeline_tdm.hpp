// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_decode_pagedkv_pipeline_tdm_policy.hpp"

namespace ck_tile {

// gfx125 decode attention of one workgroup: the query heads of one kv head (one query token, at
// most kM0 heads) against a range of keys of a paged fp8 KV cache in the vectorized layout,
//   K [num_blocks][nhead_k][hdim / 16][page_block_size tokens][16 B]
//   V [num_blocks][nhead_k][page_block_size / 16][hdim][16 tokens]
// with a 2D block table. Q is quantized to fp8 per row here, K/V have per-tensor descales, and P
// is quantized with its row max on fp8's largest normal. The query heads are the 16 columns of
// S^T = K Q^T, so that the cache bytes copied to LDS are the WMMA operands of both products.
//
// The pipeline steps through 16-token pages (a cache page holds kSubPages of them), kStepPages
// per step, in one of two ways:
// - cooperative: the four waves share a step of four pages, one page each for S^T, and combine
//   the row max and row sum of the step through LDS (two barriers per step);
// - wave split: each wave runs four pages of its own per step, without barriers, and the waves
//   are merged once at the end.
// The result is unnormalized: O relative to the row max m, and the row sum l.
template <typename Problem_, typename Policy_ = BlockFmhaFwdDecodePagedKVPipelineTdmPolicy>
struct BlockFmhaFwdDecodePagedKVPipelineTdm
{
    using Problem   = remove_cvref_t<Problem_>;
    using Policy    = remove_cvref_t<Policy_>;
    using QDataType = remove_cvref_t<typename Problem::QDataType>;
    using KDataType = remove_cvref_t<typename Problem::KDataType>;
    using VDataType = remove_cvref_t<typename Problem::VDataType>;
    using ODataType = remove_cvref_t<typename Problem::ODataType>;

    static constexpr index_t kBlockSize  = Problem::kBlockSize;
    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kNumWarps   = Problem::kNumWarps;
    static constexpr index_t kHeadDim    = Problem::kHeadDim;
    static constexpr index_t kM0         = Problem::kM0;
    static constexpr index_t kN0         = Problem::kN0;

    static constexpr bool kWaveSplit        = Problem::kWaveSplit;
    static constexpr index_t kPageBlockSize = Problem::kPageBlockSize;
    static constexpr index_t kSubPages      = Policy::template kSubPages<Problem>;
    static constexpr index_t kWavePages     = Policy::template kWavePages<Problem>;
    static constexpr index_t kStages        = Policy::template kStages<Problem>;
    static constexpr index_t kPageBytes     = Policy::template kPageBytes<Problem>;
    // 16-token pages a workgroup takes per step
    static constexpr index_t kStepPages = kNumWarps * kWavePages;
    // Q and O are fp16 instead of bf16
    static constexpr bool kQoFp16 = std::is_same_v<QDataType, fp16_t>;

    static constexpr const char* name = kWaveSplit ? "tdm_wsplit" : "tdm_coop";

    // this pipeline with the cooperative layout
    using Cooperative =
        BlockFmhaFwdDecodePagedKVPipelineTdm<typename Problem::Cooperative, Policy_>;

    static_assert(kHeadDim == 128 && kM0 == 16 && kN0 == 16 && kNumWarps == 4);
    static_assert(std::is_same_v<KDataType, fp8_t> && std::is_same_v<VDataType, fp8_t>);
    static_assert(std::is_same_v<QDataType, bf16_t> || std::is_same_v<QDataType, fp16_t>);
    static_assert(std::is_same_v<ODataType, QDataType>);
    static_assert(kPageBlockSize == 16 || kPageBlockSize == 64 || kPageBlockSize == 128);
    static_assert(Problem::kKVMemoryLayout ==
                      BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT &&
                  Problem::kKVLookupTable ==
                      BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // The keys of one workgroup: positions [key_begin, seqlen_k) of the sequence that lie in
    // 16-token pages [page_begin, page_end). Positions of those pages outside the range are
    // masked. With cache pages above 16 tokens page_begin is a multiple of 4.
    struct KeyRange
    {
        index_t seqlen_k;
        index_t key_begin;
        index_t page_begin;
        index_t page_end;
    };

    using v4i  = int32_t __attribute__((ext_vector_type(4)));
    using v8i  = int32_t __attribute__((ext_vector_type(8)));
    using v16i = int32_t __attribute__((ext_vector_type(16)));
    using v8f  = float __attribute__((ext_vector_type(8)));

    // What a thread holds of the workgroup's result: 16 head dims of query head `row`, in two
    // blocks of 8. Wave split: head dims d0 + [0, 16). Cooperative: lane half `half` of wave
    // `wave` holds head dims (bk * 4 + wave) * 16 + half * 8 + [0, 8) for bk = 0, 1.
    struct Result
    {
        index_t row;
        index_t d0;
        index_t wave;
        index_t half;
        v8f o[2];        // sum of P V, with P relative to m
        float m;         // row max of S in log2 units; -inf without keys
        float l;         // row sum of P
        float v_descale; // *v_descale_ptr
    };

#if defined(__gfx125__)
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
    CK_TILE_DEVICE static float bf16_to_f32(uint32_t x)
    {
        return bf16_to_float_raw(static_cast<uint16_t>(x));
    }
    // round to nearest even; float_to_bf16_rtn_raw also tests for NaN, in every element stored
    CK_TILE_DEVICE static uint32_t f32_to_bf16(float f)
    {
        const uint32_t u = bit_cast<uint32_t>(f);
        return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    }
    CK_TILE_DEVICE static uint32_t pack_bf16(float a, float b)
    {
        return f32_to_bf16(a) | (f32_to_bf16(b) << 16);
    }
    CK_TILE_DEVICE static float fp16_to_f32(uint32_t x)
    {
        return type_convert<float>(bit_cast<fp16_t>(static_cast<uint16_t>(x)));
    }
    CK_TILE_DEVICE static uint32_t pack_fp16(float a, float b)
    {
        return bit_cast<uint16_t>(type_convert<fp16_t>(a)) |
               (static_cast<uint32_t>(bit_cast<uint16_t>(type_convert<fp16_t>(b))) << 16);
    }
    // two elements of O in the output type
    CK_TILE_DEVICE static uint32_t pack_o(float a, float b)
    {
        return kQoFp16 ? pack_fp16(a, b) : pack_bf16(a, b);
    }
    CK_TILE_DEVICE static float o_to_f32(uint32_t x)
    {
        return kQoFp16 ? fp16_to_f32(x) : bf16_to_f32(x);
    }
    CK_TILE_DEVICE static float exp2(float x) { return __builtin_amdgcn_exp2f(x); }
    // x of the lane 16 away, with a VALU permute: a bpermute would queue behind the LDS reads
    CK_TILE_DEVICE static float other_half(float x)
    {
        return bit_cast<float>(__builtin_amdgcn_permlanex16(
            0, bit_cast<int32_t>(x), 0x76543210, 0xfedcba98, false, true));
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

    // True on one thread of each row of the Result, the one holding head dim 0.
    CK_TILE_DEVICE static bool IsRowOwner(const Result& res)
    {
        if constexpr(kWaveSplit)
            return threadIdx.x % 8 == 0;
        else
            return res.wave == 0 && res.half == 0;
    }

    // Stores f * o of the thread's head dims of its row, two elements packed by pack(a, b), to
    // a block of rows: base_ptr points at row 0, rows are row_stride 16-bit elements apart.
    template <typename Pack>
    CK_TILE_DEVICE static void
    StoreRows(const Result& res, uint16_t* base_ptr, index_t row_stride, float f, Pack pack)
    {
        if constexpr(kWaveSplit)
        {
            uint32_t pk[8];
            for(index_t e = 0; e < 8; ++e)
                pk[e] = pack(res.o[e / 4][2 * (e % 4)] * f, res.o[e / 4][2 * (e % 4) + 1] * f);
            v4i* dst = ptr_as<v4i>(base_ptr + res.row * row_stride + res.d0);
            dst[0]   = v4i{int32_t(pk[0]), int32_t(pk[1]), int32_t(pk[2]), int32_t(pk[3])};
            dst[1]   = v4i{int32_t(pk[4]), int32_t(pk[5]), int32_t(pk[6]), int32_t(pk[7])};
        }
        else
        {
            for(index_t bk = 0; bk < 2; ++bk)
            {
                uint32_t pk[4];
                for(index_t e = 0; e < 4; ++e)
                    pk[e] = pack(res.o[bk][2 * e] * f, res.o[bk][2 * e + 1] * f);
                *ptr_as<v4i>(base_ptr + res.row * row_stride + (bk * kNumWarps + res.wave) * 16 +
                             res.half * 8) =
                    v4i{int32_t(pk[0]), int32_t(pk[1]), int32_t(pk[2]), int32_t(pk[3])};
            }
        }
    }

    // q_rows: the first of the workgroup's num_rows query heads (rows of kHeadDim), q_row_stride
    // elements apart
    // k_ptr, v_ptr: the caches; block_table: the sequence's row of the block table
    // scale_s: softmax scale; *k_descale_ptr is folded into it, *v_descale_ptr left to the caller
    CK_TILE_DEVICE Result operator()(const QDataType* q_rows,
                                     index_t q_row_stride,
                                     index_t num_rows,
                                     const KDataType* k_ptr,
                                     const VDataType* v_ptr,
                                     const int32_t* block_table,
                                     index_t nhead_k,
                                     index_t i_nhead_k,
                                     const KeyRange& range,
                                     const float* k_descale_ptr,
                                     const float* v_descale_ptr,
                                     float scale_s,
                                     void* smem_ptr) const
    {
        auto& sm = *static_cast<typename Policy::template Smem<Problem>*>(smem_ptr);

        const index_t l = threadIdx.x % 32;
        const index_t w = __builtin_amdgcn_readfirstlane(threadIdx.x) / 32;
        const index_t n = l % 16, h = l / 16;

        const index_t ctx   = range.seqlen_k;
        const index_t lo    = range.key_begin;
        const index_t pg0   = range.page_begin;
        const index_t pend  = range.page_end;
        const index_t steps = pend > pg0 ? (pend - pg0 + kStepPages - 1) / kStepPages : 0;

        // block ids of the range's cache pages, 32 per register (lane = cache page relative to
        // the first), picked with readlane
        const index_t blk0 = pg0 / kSubPages;
        const auto load_bt = [&](index_t chunk) {
            return block_table[min(blk0 + chunk * 32 + l, max(pend - 1, 0) / kSubPages)];
        };
        int32_t bt_cur = load_bt(0), bt_next = load_bt(1);
        index_t cur_chunk = 0;
        // the descales go out with the prologue loads instead of after the Q quantization
        const float kds = *k_descale_ptr, vd = *v_descale_ptr;
        // loads return in order: the block ids, which the first copies wait for, go out first
        __builtin_amdgcn_sched_barrier(0);

        // Q^T operand of row n: 16-byte d chunks 2i + h, quantized per row (absmax) below
        v4i qraw[8];
        {
            const QDataType* qrow = q_rows + min(n, num_rows - 1) * q_row_stride;
            for(index_t i = 0; i < 4; ++i)
            {
                const index_t c = 2 * i + h;
                qraw[2 * i]     = *ptr_as<v4i>(qrow + c * 16);
                qraw[2 * i + 1] = *ptr_as<v4i>(qrow + c * 16 + 8);
            }
        }

        // LDS page buffers of wave w
        const auto kbuf = [&](index_t st, index_t q) -> uint8_t* {
            return Policy::template GetKPage<Problem>(sm, w, st, q);
        };
        const auto vbuf = [&](index_t st, index_t q) -> uint8_t* {
            return Policy::template GetVPage<Problem>(sm, w, st, q);
        };
        // their LDS addresses, the destinations of the TDM gathers
        const auto kdst = [&](index_t st) {
            return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(to_lds(kbuf(st, 0))));
        };
        const auto vdst = [&](index_t st) {
            return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(to_lds(vbuf(st, 0))));
        };
        constexpr index_t kKChunkStride = Policy::template kKChunkStride<Problem>;
        const uint64_t k_addr           = reinterpret_cast<uintptr_t>(k_ptr);
        const uint64_t v_addr           = reinterpret_cast<uintptr_t>(v_ptr);
        // page index (within the range) of wave w's q-th page of step t
        const auto page_idx = [&](index_t t, index_t q) {
            return (t * kNumWarps + w) * kWavePages + q;
        };
        // block id of the cache page holding page idx (relative to pg0)
        const auto block_of = [&](index_t idx) {
            const index_t rb = (pg0 + idx) / kSubPages - blk0;
            return __builtin_amdgcn_readlane((rb >> 5) == cur_chunk ? bt_cur : bt_next, rb & 31);
        };
        // K and V of step t, one TDM gather each; a page past the range repeats the last one (its
        // V is zeroed once it lands)
        const auto issue = [&](index_t t) {
            const index_t last = pend - 1 - pg0;
            if constexpr(kSubPages == 1)
            {
                int32_t rows[kWavePages];
                for(index_t q = 0; q < kWavePages; ++q)
                    rows[q] = block_of(min(page_idx(t, q), last)) * nhead_k + i_nhead_k;
                amd_tdm_gather_rows<kPageBytes>(kdst(t % kStages), k_addr, rows);
                amd_tdm_gather_rows<kPageBytes>(vdst(t % kStages), v_addr, rows);
            }
            else if constexpr(kWaveSplit)
            {
                // the wave's four pages are a quarter of a 64-token cache page: K as eight rows
                // of 64 tokens x 16 B, V as one row of four page-16 images
                const index_t idx = min(page_idx(t, 0), last / 4 * 4);
                const index_t qtr = (pg0 + idx) % kSubPages / 4;
                const index_t row = block_of(idx) * nhead_k + i_nhead_k;
                int32_t krows[8];
                for(index_t c = 0; c < 8; ++c)
                    krows[c] = (row * 8 + c) * (kSubPages / 4) + qtr;
                const int32_t vrows[1] = {row * (kSubPages / 4) + qtr};
                amd_tdm_gather_rows<4 * 256>(kdst(t % kStages), k_addr, krows);
                amd_tdm_gather_rows<4 * kPageBytes>(vdst(t % kStages), v_addr, vrows);
            }
            else
            {
                // one page: K as eight rows of 16 tokens x 16 B (a page-16 image), V as one row
                const index_t idx = min(page_idx(t, 0), last);
                const index_t sub = (pg0 + idx) % kSubPages;
                const index_t row = block_of(idx) * nhead_k + i_nhead_k;
                int32_t krows[8];
                for(index_t c = 0; c < 8; ++c)
                    krows[c] = (row * 8 + c) * kSubPages + sub;
                const int32_t vrows[1] = {row * kSubPages + sub};
                amd_tdm_gather_rows<256>(kdst(t % kStages), k_addr, krows);
                amd_tdm_gather_rows<kPageBytes>(vdst(t % kStages), v_addr, vrows);
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
        qmax             = fmaxf(qmax, other_half(qmax));
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
        const float scl = qmax / 448.f * kds * scale_s * log2e_v<float>;

        // step t's copies of this wave are in LDS; zero the V bytes of the tokens before the
        // window and past the context: their P is 0, but their bytes may be NaN codes
        const auto land = [&](index_t t) {
            // two gathers per step
            if(issued > t + 2)
                s_wait_tensorcnt<4>();
            else if(issued > t + 1)
                s_wait_tensorcnt<2>();
            else
                s_wait_tensorcnt<0>();
            // the step's pages are consecutive: if the first starts inside the window and the
            // last lies in the range and the context, every token is valid (the common case, one
            // scalar test instead of one per page)
            const index_t pg_first = pg0 + page_idx(t, 0);
            const index_t pg_end   = pg0 + page_idx(t, kWavePages - 1) + 1;
            if(pg_first * kN0 >= lo && pg_end <= pend && pg_end * kN0 <= ctx)
                return;
            for(index_t q = 0; q < kWavePages; ++q)
            {
                const index_t pg = pg0 + page_idx(t, q);
                // the page's valid tokens are [begin, end)
                const index_t end = __builtin_amdgcn_readfirstlane(
                    pg < pend ? min(max(ctx - pg * kN0, 0), kN0) : 0);
                const index_t begin =
                    __builtin_amdgcn_readfirstlane(min(max(lo - pg * kN0, 0), end));
                uint8_t* vd8 = vbuf(t % kStages, q);
                if(begin == end)
                    for(index_t i = 0; i < 4; ++i)
                        *ptr_as<v4i>(vd8 + (i * 32 + l) * 16) = v4i{0, 0, 0, 0};
                else if(begin > 0 || end < kN0)
                    for(index_t i = 0; i < 4; ++i)
                        for(index_t tok = 0; tok < kN0; ++tok)
                            if(tok < begin || tok >= end)
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
                    const index_t pos = pg_first * kN0 + tok_h + r;
                    if(!(pos >= lo && pos < ctx && pos < pend * kN0))
                        sv[r] = -numeric<float>::infinity();
                }
            }
        };
        // pages [pg, pg + count) need no mask
        const auto inside = [&](index_t pg, index_t count) {
            return pg * kN0 >= lo && (pg + count) * kN0 <= ctx && pg + count <= pend;
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
                // the V^T operands of the step, read here so that the softmax covers the reads
                v8i vf[8];
                for(index_t db = 0; db < 8; ++db)
                {
                    const index_t off = (db * 16 + n) * 16;
                    const v4i x       = *ptr_as<v4i>(&sm.wave[w].kv.v[st][2 * h][off]);
                    const v4i y       = *ptr_as<v4i>(&sm.wave[w].kv.v[st][2 * h + 1][off]);
                    vf[db]            = v8i{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
                }
                float mx          = tree_max<kWavePages * 8>(&sv[0][0]);
                mx                = fmaxf(mx, other_half(mx)) * scl;
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
                ps       = ps + other_half(ps);
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
                    acc[db] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                        vf[db], pf, 0, acc[db], false, false);
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
                mx       = fmaxf(mx, other_half(mx)) * scl;
                if(h == 0)
                    sm.red[0][w][n] = mx;
                // the V^T operands of the step (every wave's V of this stage landed before the
                // last barrier), read here so that the barriers and the softmax cover the reads
                v8i vf[2];
                for(index_t bk = 0; bk < 2; ++bk)
                {
                    const index_t dd = (bk * kNumWarps + w) * 16 + n;
                    const v4i x      = *ptr_as<v4i>(&sm.v[st][h][dd * 16]);
                    const v4i y      = *ptr_as<v4i>(&sm.v[st][h + 2][dd * 16]);
                    vf[bk]           = v8i{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
                }
                __syncthreads();
                // every wave is past step t-1's PV: its stage takes step t + 2
                if(t >= 1 && t + 2 < steps)
                {
                    advance_bt(t + 2);
                    issue(t + 2);
                    ++issued;
                }
                float m_step = sm.red[0][0][n];
                for(index_t i = 1; i < kNumWarps; ++i)
                    m_step = fmaxf(m_step, sm.red[0][i][n]);
                const float m_new = fmaxf(m_run, m_step);
                const float m_use = is_neg_inf(m_new) ? 0.f : m_new;
                const float c     = kLog2P - m_use;
                float pv[8];
                for(index_t r = 0; r < 8; ++r)
                    pv[r] = exp2(fmaf(sv[r], scl, c));
                float ps   = tree_sum<8>(pv);
                ps         = ps + other_half(ps);
                int32_t p0 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[0], pv[1], 0, false);
                p0         = __builtin_amdgcn_cvt_pk_fp8_f32(pv[2], pv[3], p0, true);
                int32_t p1 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[4], pv[5], 0, false);
                p1         = __builtin_amdgcn_cvt_pk_fp8_f32(pv[6], pv[7], p1, true);
                *ptr_as<int2>(&sm.p[n][w * kN0 + h * 8]) = make_int2(p0, p1);
                if(h == 0)
                    sm.red[1][w][n] = ps;
                if(t + 1 < steps)
                    land(t + 1);
                __syncthreads();
                float psum = 0.f;
                for(index_t i = 0; i < kNumWarps; ++i)
                    psum += sm.red[1][i][n];
                const float alpha = exp2(m_run - m_use);
                l_run             = l_run * alpha + psum;
                m_run             = m_new;
                // O^T += V^T P^T for head-dim blocks w and w + 4; tokens of pages h and h + 2 on
                // both operands
                const v4i px = *ptr_as<v4i>(&sm.p[n][h * kN0]);
                const v4i py = *ptr_as<v4i>(&sm.p[n][(h + 2) * kN0]);
                const v8i pf{px[0], px[1], px[2], px[3], py[0], py[1], py[2], py[3]};
                for(index_t bk = 0; bk < 2; ++bk)
                    acc[bk] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                        vf[bk], pf, 0, acc[bk] * alpha, false, false);
            }
        }

        Result res;
        res.v_descale = vd;
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
            // thread -> (row, 16 head dims)
            const index_t row = threadIdx.x / 8;
            const index_t d0  = (threadIdx.x % 8) * 16;
            res.row           = row;
            res.d0            = d0;
            res.wave          = w;
            res.half          = h;
            res.m             = -numeric<float>::infinity();
            for(index_t i = 0; i < kNumWarps; ++i)
                res.m = fmaxf(res.m, sm.ml[0][i][row]);
            const float mu = is_neg_inf(res.m) ? 0.f : res.m;
            float wi[kNumWarps];
            res.l = 0.f;
            for(index_t i = 0; i < kNumWarps; ++i)
            {
                wi[i] = exp2(sm.ml[0][i][row] - mu);
                res.l += sm.ml[1][i][row] * wi[i];
            }
            float o[16];
            for(index_t e = 0; e < 16; ++e)
                o[e] = 0.f;
            for(index_t i = 0; i < kNumWarps; ++i)
                for(index_t e4 = 0; e4 < 4; ++e4)
                {
                    const float4 x = *ptr_as<float4>(&sm.wave[i].o[row][d0 + 4 * e4]);
                    o[4 * e4]      = fmaf(wi[i], x.x, o[4 * e4]);
                    o[4 * e4 + 1]  = fmaf(wi[i], x.y, o[4 * e4 + 1]);
                    o[4 * e4 + 2]  = fmaf(wi[i], x.z, o[4 * e4 + 2]);
                    o[4 * e4 + 3]  = fmaf(wi[i], x.w, o[4 * e4 + 3]);
                }
            for(index_t e = 0; e < 16; ++e)
                res.o[e / 8][e % 8] = o[e];
        }
        else
        {
            // lane (n, h) of wave w holds row n, head-dim blocks w and w + 4
            res.row  = n;
            res.d0   = 0;
            res.wave = w;
            res.half = h;
            res.m    = m_run;
            res.l    = l_run;
            for(index_t bk = 0; bk < 2; ++bk)
                res.o[bk] = acc[bk];
        }
        return res;
    }
#endif
};

} // namespace ck_tile
