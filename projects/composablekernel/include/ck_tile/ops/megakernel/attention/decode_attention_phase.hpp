// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/megakernel/attention/attention_output.hpp"

namespace ck_tile {

/// @brief Split-KV decode / verify attention for head_dim 64 and 8 q heads per kv head, as one
/// megakernel phase, with the split combine and the output epilogue fused into the last split.
///
/// Work item w = tile * num_splits + split, tile = (request b, q tile t, kv head g) ordered
/// ((b * QTiles + t) * kv_heads + g). A q tile packs the 8 q heads of kv head g times 16 query
/// rows into one 128-row Q block (row r = head (r / 16), query row 16 t + r % 16), so a 16-row
/// verify window fills the WMMA M dimension and K/V are read once per kv head.
///
/// A split walks its share of the tile's 64-key KV blocks: S = Q K^T (WMMA), causal and sliding
/// window mask, softmax, O += P V (WMMA), and writes O / l and lse (log2 domain) to the
/// workspace. It then counts itself on the tile's split counter; the last split to arrive merges
/// all splits (one thread per Q row), scales by v_scale and calls OutputFunctor with the row.
///
/// Inputs
/// - Q (bf16): element (token, q_head, d) at q[token * stride_q + q_head * 64 + d]. Request b owns
///   tokens [b * q_len, (b + 1) * q_len); token i of request b sits at position
///   kv_lens[b] - q_len + i, i.e. its K/V are already in the cache (kv_lens counts them).
/// - KV cache (KVDataType = bf16 or fp8 e4m3, K and V separate, same layout): element
///   (page p, kv head g, slot s, d) at cache[p * page_stride + g * head_stride +
///   s * token_stride + d]; d contiguous. Position j of request b lives in page
///   block_table[b * block_table_stride + j / kPageSize], slot j % kPageSize.
///   kPageSize: 16, 32 or a multiple of 64. The qkv epilogue writes cache[page][kv head][slot][64]:
///   page_stride = kv_heads * kPageSize * 64, head_stride = kPageSize * 64, token_stride = 64.
///   fp8: dequantised value = fp8 * k_scale (v_scale), per tensor.
/// - window > 0: key j is visible to query position p iff p - window < j <= p (sliding);
///   window == 0: iff j <= p (global).
///
/// Workspace (zeroed counters once; partials need no init): o_partial NumWork * 128 * 64 floats,
/// lse_partial NumWork * 128 floats, split_counters NumTiles words. The last split resets its
/// counter, so launches (and graph replays) reuse the buffer.
///
/// kRunningMax = false drops the running row max: p = exp2(s * scale * log2e). Legal only while
/// |q.k| * scale * k_scale stays well below 88 (fp32 exp overflow, with the row sum on top);
/// The target model's weightless QK norm bounds it by 43.8. kRunningMax = true is the usual online
/// softmax.
///
/// TODO(perf): K/V go global -> VGPR -> LDS (needed for the fp8 -> bf16 conversion); use TDM
/// for bf16 and an fp8 WMMA (Q quantised) path for fp8. Partials could be bf16. The combine reads
/// num_splits * 32 KiB per tile from one workgroup; a separate combine phase spread over
/// workgroups would shorten the tail at large split counts.
template <typename KVDataType_,
          typename OutputFunctor_,
          index_t kPageSize_ = 16,
          bool kRunningMax_  = false>
struct DecodeAttentionPhase
{
    using QDataType     = bf16_t;
    using KVDataType    = remove_cvref_t<KVDataType_>;
    using OutputFunctor = remove_cvref_t<OutputFunctor_>;
    using CompDataType  = bf16_t; // WMMA operand type: fp8 K/V are converted when staged in LDS
    using AccDataType   = float;

    static_assert(std::is_same_v<KVDataType, bf16_t> || std::is_same_v<KVDataType, fp8_t>,
                  "KV cache must be bf16 or fp8 e4m3");

    static constexpr bool kRunningMax  = kRunningMax_;
    static constexpr index_t kPageSize = kPageSize_;

    static constexpr index_t kHeadDim     = 64;
    static constexpr index_t kGroup       = 8;                     // q heads per kv head
    static constexpr index_t kRowsPerHead = 16;                    // query rows per q tile
    static constexpr index_t kM0          = kGroup * kRowsPerHead; // Q block rows
    static constexpr index_t kN0          = 64;                    // keys per KV block
    static constexpr index_t kK0          = 32;                    // gemm0 K step
    static constexpr index_t kN1          = kHeadDim;
    static constexpr index_t kK1          = 32; // gemm1 K step

    static constexpr index_t kNumWarps = 4;
    static constexpr index_t kWarpSize = 32; // gfx1250; host get_warp_size() would say 64
    static constexpr index_t kThreads  = kNumWarps * kWarpSize; // threads on the device
    static_assert(kThreads == kM0, "the combine gives each thread one Q row");
    /// Megakernel contract: waves * get_warp_size() of the compile pass (the launch size is
    /// kBlockSize / get_warp_size() waves). Device code uses kThreads.
    static constexpr index_t kBlockSize = kNumWarps * get_warp_size();

    /// Output rows per counter slot: one q tile of one request.
    static constexpr index_t kSlotRows = kRowsPerHead;

    using barrier_pipeline = void;

    // LDS rows are padded by 16 B (Q, K) and 32 B (V, read with ds_load_tr) against bank
    // conflicts, as in the qr_tdm fmha policy.
    static constexpr index_t kQStride = kHeadDim + 8;
    static constexpr index_t kKStride = kHeadDim + 8;
    static constexpr index_t kVStride = kHeadDim + 16;
    static constexpr index_t kQBytes  = kM0 * kQStride * sizeof(CompDataType);
    static constexpr index_t kKBytes  = kN0 * kKStride * sizeof(CompDataType);
    static constexpr index_t kVBytes  = kN0 * kVStride * sizeof(CompDataType);
    static constexpr index_t kFlagOff = kQBytes + kKBytes + kVBytes;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kFlagOff + 16; }

    struct Args
    {
        const QDataType* q;
        index_t stride_q;

        const KVDataType* k_cache;
        const KVDataType* v_cache;
        long_index_t page_stride;
        index_t head_stride;
        index_t token_stride;
        const int32_t* block_table;
        index_t block_table_stride;
        const int32_t* kv_lens; // [num_requests], incl. the q_len new tokens

        index_t num_requests;
        index_t q_len;
        index_t num_kv_heads;
        index_t window; // 0: global
        float scale;    // softmax scale (1/8)
        float k_scale;  // fp8 dequant scales, 1 for bf16
        float v_scale;

        index_t num_splits;
        float* o_partial;
        float* lse_partial;
        uint32_t* split_counters;

        typename OutputFunctor::Args out;
    };

    // ---------------------------------------------------------------- work decomposition

    CK_TILE_HOST_DEVICE static index_t QTiles(const Args& a)
    {
        return integer_divide_ceil(a.q_len, kRowsPerHead);
    }
    CK_TILE_HOST_DEVICE static index_t NumTiles(const Args& a)
    {
        return a.num_requests * QTiles(a) * a.num_kv_heads;
    }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return NumTiles(a) * a.num_splits; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a)
    {
        return a.num_requests * QTiles(a);
    }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t /*slot*/)
    {
        return a.num_kv_heads * a.num_splits;
    }

    /// Where one work item sits, and the KV blocks [kv_block_begin, kv_block_end) it walks.
    struct Item
    {
        index_t tile, split, b, t, g;
        index_t row0; // first query row of the tile within the request
        index_t rows; // valid query rows, <= 16
        index_t pos0; // position of row0
        index_t kv_block_begin, kv_block_end;
    };

    CK_TILE_HOST_DEVICE static Item Locate(const Args& a, index_t work, index_t kv_len)
    {
        Item it{};
        it.split               = work % a.num_splits;
        it.tile                = work / a.num_splits;
        it.g                   = it.tile % a.num_kv_heads;
        const index_t bt       = it.tile / a.num_kv_heads;
        it.t                   = bt % QTiles(a);
        it.b                   = bt / QTiles(a);
        it.row0                = it.t * kRowsPerHead;
        it.rows                = min(kRowsPerHead, a.q_len - it.row0);
        it.pos0                = kv_len - a.q_len + it.row0;
        const index_t pos_last = it.pos0 + it.rows - 1;
        const index_t key_lo   = a.window > 0 ? max(index_t{0}, it.pos0 - a.window + 1) : 0;
        const index_t blk_lo   = key_lo / kN0;
        const index_t blk_hi   = integer_divide_ceil(pos_last + 1, kN0);
        const index_t per      = integer_divide_ceil(blk_hi - blk_lo, a.num_splits);
        it.kv_block_begin      = min(blk_hi, blk_lo + it.split * per);
        it.kv_block_end        = min(blk_hi, it.kv_block_begin + per);
        return it;
    }

    /// True when Q block row r (head r / 16) must not see key position j.
    CK_TILE_HOST_DEVICE static bool IsMasked(const Args& a, const Item& it, index_t r, index_t j)
    {
        // Padding rows (r % 16 >= rows) borrow the last valid row's position; never stored.
        const index_t p = it.pos0 + min(r % kRowsPerHead, it.rows - 1);
        return j > p || (a.window > 0 && j <= p - a.window);
    }

    CK_TILE_DEVICE static index_t SlotOf(const Args& a, index_t work)
    {
        return amd_wave_read_first_lane(work / a.num_splits / a.num_kv_heads);
    }

    /// Token rows this item reads: its Q rows and the request's new K/V rows before them.
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t bt = work / a.num_splits / a.num_kv_heads;
        const index_t t  = bt % QTiles(a);
        const index_t b  = bt / QTiles(a);
        const index_t r0 = b * a.q_len;
        return make_tuple(r0, r0 + min(a.q_len, (t + 1) * kRowsPerHead));
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        const bool kv_vec = (a.token_stride * sizeof(KVDataType)) % 16 == 0 &&
                            (a.head_stride * sizeof(KVDataType)) % 16 == 0 &&
                            (a.page_stride * sizeof(KVDataType)) % 16 == 0;
        // Counter slots are kSlotRows token rows; with several requests that needs q_len % 16 == 0.
        const bool slots = a.q_len % kRowsPerHead == 0 || a.num_requests == 1;
        return a.q != nullptr && a.k_cache != nullptr && a.v_cache != nullptr &&
               a.block_table != nullptr && a.kv_lens != nullptr && a.o_partial != nullptr &&
               a.lse_partial != nullptr && a.split_counters != nullptr && a.q_len > 0 &&
               a.num_requests > 0 && a.num_kv_heads > 0 && a.num_splits > 0 &&
               a.stride_q % 8 == 0 && kv_vec && slots && a.window >= 0;
    }

    CK_TILE_DEVICE static auto InitBarriers() { return 0; }

    // ---------------------------------------------------------------- host helpers

    static constexpr index_t kMinBlocksPerSplit = 4; // 256 keys: partial O traffic <= 1/2 of KV
    static constexpr index_t kMaxSplits         = 64;

    /// Keys one q tile attends at most: window + q_len - 1 (sliding) or the context (global).
    CK_TILE_HOST_DEVICE static index_t
    MaxSpanKeys(index_t max_kv_len, index_t q_len, index_t window)
    {
        return window > 0 ? min(max_kv_len, window + min(q_len, kRowsPerHead) - 1) : max_kv_len;
    }

    /// Split count: fill the grid once (num_tiles * splits <= grid), at least kMinBlocksPerSplit
    /// KV blocks per split, at most kMaxSplits. SWA layers come out at <= 8 splits whatever the
    /// context; global layers use the grid.
    CK_TILE_HOST static index_t
    ChooseNumSplits(index_t num_tiles, index_t max_span_keys, index_t grid)
    {
        const index_t by_grid = max(index_t{1}, grid / max(index_t{1}, num_tiles));
        const index_t by_len =
            max(index_t{1}, integer_divide_ceil(max_span_keys, kN0) / kMinBlocksPerSplit);
        return min(min(by_grid, by_len), kMaxSplits);
    }

    CK_TILE_HOST static long_index_t PartialOFloats(const Args& a)
    {
        return static_cast<long_index_t>(NumWork(a)) * kM0 * kHeadDim;
    }
    CK_TILE_HOST static long_index_t PartialLseFloats(const Args& a)
    {
        return static_cast<long_index_t>(NumWork(a)) * kM0;
    }

    // ---------------------------------------------------------------- block policy
    // Same WMMA configuration as the gfx1250 qr_tdm fmha pipeline for hdim 64 (4x1 warps,
    // 16x16x32 bf16, transposed C so P feeds gemm1 without a lane shuffle), with kM0 = 128.

    using WarpGemm0 = WarpGemmDispatcher<CompDataType, CompDataType, AccDataType, 16, 16, 32, true>;
    using WarpGemm1 = WarpGemmDispatcher<CompDataType,
                                         CompDataType,
                                         AccDataType,
                                         16,
                                         16,
                                         32,
                                         true,
                                         false,
                                         false,
                                         WGAttrNumAccessEnum::Default>;
    using BlockWarps = sequence<kNumWarps, 1, 1>;

    using Gemm0 = BlockGemmARegBRegCRegV2<
        BlockGemmProblem<CompDataType,
                         CompDataType,
                         AccDataType,
                         kThreads,
                         TileGemmShape<sequence<kM0, kN0, kK0>, BlockWarps, sequence<16, 16, 32>>>,
        BlockGemmARegBRegCRegV2CustomPolicy<CompDataType,
                                            CompDataType,
                                            AccDataType,
                                            BlockWarps,
                                            WarpGemm0,
                                            GemmLoopOrder::MNK>>;
    using Gemm1 = BlockGemmARegBRegCRegV2<
        BlockGemmProblem<CompDataType,
                         CompDataType,
                         AccDataType,
                         kThreads,
                         TileGemmShape<sequence<kM0, kN1, kK1>, BlockWarps, sequence<16, 16, 32>>>,
        BlockGemmARegBRegCRegV2CustomPolicy<CompDataType,
                                            CompDataType,
                                            AccDataType,
                                            BlockWarps,
                                            WarpGemm1,
                                            GemmLoopOrder::KMN>>;

    CK_TILE_DEVICE static constexpr auto MakeQRegDistribution()
    {
        constexpr index_t MIter = kM0 / (kNumWarps * WarpGemm0::kM);
        constexpr index_t KIter = kHeadDim / WarpGemm0::kK;
        constexpr auto outer =
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MIter, kNumWarps>, sequence<KIter>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        return make_static_tile_distribution(detail::make_embed_tile_distribution_encoding(
            outer, typename WarpGemm0::AWarpDstrEncoding{}));
    }

    CK_TILE_DEVICE static constexpr auto MakeKRegDistribution()
    {
        constexpr index_t NIter = kN0 / WarpGemm0::kN;
        constexpr index_t KIter = kK0 / WarpGemm0::kK;
        constexpr auto outer =
            tile_distribution_encoding<sequence<kNumWarps>,
                                       tuple<sequence<NIter, 1>, sequence<KIter>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        return make_static_tile_distribution(detail::make_embed_tile_distribution_encoding(
            outer, typename WarpGemm0::BWarpDstrEncoding{}));
    }

    CK_TILE_DEVICE static constexpr auto MakePRegDistribution()
    {
        constexpr index_t MIter = kM0 / (kNumWarps * WarpGemm1::kM);
        constexpr index_t KIter = kN0 / WarpGemm1::kK;
        constexpr auto outer =
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MIter, kNumWarps>, sequence<KIter>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};
        return make_static_tile_distribution(detail::make_embed_tile_distribution_encoding(
            outer, typename WarpGemm1::AWarpDstrEncoding{}));
    }

    // V is read from its row-major (key, d) LDS tile with ds_load_tr.
    CK_TILE_DEVICE static constexpr auto MakeVRegDistribution()
    {
        constexpr index_t NIter = kN1 / WarpGemm1::kN;
        constexpr index_t KIter = kK1 / WarpGemm1::kK;
        constexpr auto outer =
            tile_distribution_encoding<sequence<kNumWarps>,
                                       tuple<sequence<NIter, 1>, sequence<KIter>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};
        constexpr auto enc = detail::make_embed_tile_distribution_encoding(
            outer, typename WarpGemm1::BWarpDstrEncoding{});
        return make_static_tile_distribution(
            typename InputTileDistributionTraits<decltype(enc),
                                                 CompDataType>::TransposedDstrEncode{});
    }

    // A KV block is loaded as kNumSub sub-blocks of kSubRows keys, each inside one page.
    static_assert(kPageSize == 16 || kPageSize == 32 || kPageSize % kN0 == 0,
                  "page size must be 16, 32 or a multiple of 64");
    static constexpr index_t kSubRows = min(kPageSize, kN0);
    static constexpr index_t kNumSub  = kN0 / kSubRows;

    // Global -> VGPR distribution of one (kSubRows x 64) sub-block: vectors along d, at most
    // 16 bytes, every thread busy.
    static constexpr index_t kKVVec =
        min(static_cast<index_t>(16 / sizeof(KVDataType)), kSubRows* kHeadDim / kThreads);

    CK_TILE_DEVICE static constexpr auto MakeKVDramDistribution()
    {
        constexpr index_t kLaneCols = kHeadDim / kKVVec;
        constexpr index_t kLaneRows = kWarpSize / kLaneCols;
        constexpr index_t kRep      = kSubRows / (kNumWarps * kLaneRows);
        static_assert(kRep >= 1 && kRep * kNumWarps * kLaneRows == kSubRows);
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kRep, kNumWarps, kLaneRows>, sequence<kLaneCols, kKVVec>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
    }

    // KV elements to the WMMA type. Not cast_tile: with native fp8 types, type_convert<bf16_t>
    // of an fp8_t is an integer conversion; only type_convert<float> decodes e4m3.
    // TODO(perf): packed v_cvt_pk_f32_fp8 or a bit-level e4m3 -> bf16 expansion.
    template <typename Tile>
    CK_TILE_DEVICE static auto ToCompType(const Tile& t)
    {
        if constexpr(std::is_same_v<KVDataType, CompDataType>)
            return t;
        else
            return tile_elementwise_in(
                [](const KVDataType& x) {
                    return float_to_bf16<bf16_rounding_mode::standard>(type_convert<float>(x));
                },
                t);
    }

    // gemm0's C (M-outer) to gemm1's A (K-outer): an in-thread reorder (see qr_tdm).
    template <typename PCompute>
    CK_TILE_DEVICE static auto MakePForGemm1(const PCompute& p_compute)
    {
        constexpr index_t kPMI = Gemm1::MIterPerWarp;
        constexpr index_t kPKI = kN0 / Gemm1::WarpGemm::kK;
        auto p_tile      = make_static_distributed_tensor<CompDataType>(MakePRegDistribution());
        const auto p_src = cast_tile<CompDataType>(p_compute);
        constexpr index_t kPBuf = decltype(p_src)::get_thread_buffer_size();
        constexpr index_t kPSub = kPBuf / (kPMI * kPKI);
        using p_bulk_t          = array<CompDataType, kPSub>;
        static_for<0, kPMI, 1>{}([&](auto mi) {
            static_for<0, kPKI, 1>{}([&](auto ki) {
                p_tile.get_thread_buffer().template set_as<p_bulk_t>(
                    number<ki * kPMI + mi>{},
                    p_src.get_thread_buffer().template get_as<p_bulk_t>(number<mi * kPKI + ki>{}));
            });
        });
        return p_tile;
    }

    // ---------------------------------------------------------------- device

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token&)
    {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(get_warp_size() == kWarpSize, "wave32 only");
#endif
        const index_t b_of_work =
            amd_wave_read_first_lane(work / a.num_splits / a.num_kv_heads / QTiles(a));
        const index_t kv_len = amd_wave_read_first_lane(a.kv_lens[b_of_work]);
        const Item it        = Locate(a, work, kv_len);

        float* o_part   = a.o_partial + static_cast<long_index_t>(work) * kM0 * kHeadDim;
        float* lse_part = a.lse_partial + static_cast<long_index_t>(work) * kM0;

        RunSplit(a, it, kv_len, o_part, lse_part, smem);

        auto* flag = reinterpret_cast<uint32_t*>(static_cast<char*>(smem) + kFlagOff);
        if(LastArrival(a.split_counters + it.tile, a.num_splits, flag))
            Combine(a, it);
    }

    /// Publishes this workgroup's partials and counts it on @p counter. True in the workgroup
    /// that completes the count, which also resets it and acquires the other splits' partials.
    CK_TILE_DEVICE static bool LastArrival(uint32_t* counter, index_t splits, uint32_t* flag)
    {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            const uint32_t prev =
                __scoped_atomic_fetch_add(counter, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE);
            const bool last = prev + 1 == static_cast<uint32_t>(splits);
            if(last)
                __scoped_atomic_store_n(counter, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            *flag = last ? 1u : 0u;
        }
        __syncthreads();
        const bool last = __builtin_amdgcn_readfirstlane(*flag) != 0;
        if(last)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        return last;
    }

    CK_TILE_DEVICE static void RunSplit(
        const Args& a, const Item& it, index_t kv_len, float* o_part, float* lse_part, void* smem)
    {
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};

        if(it.kv_block_begin >= it.kv_block_end)
        {
            // No keys here: an empty partial that the combine skips.
            if(threadIdx.x < kM0)
                lse_part[threadIdx.x] = -numeric<float>::infinity();
            return;
        }

        auto* q_lds = static_cast<CompDataType*>(smem);
        auto* k_lds = reinterpret_cast<CompDataType*>(static_cast<char*>(smem) + kQBytes);
        auto* v_lds = reinterpret_cast<CompDataType*>(static_cast<char*>(smem) + kQBytes + kKBytes);

        // Q -> LDS, one row per thread. TODO(perf): TDM, or a 3D view over (head, row, d).
        {
            using vec_t     = ext_vector_t<uint32_t, 4>;
            const index_t r = threadIdx.x;
            const index_t i = r % kRowsPerHead;
            const index_t h = r / kRowsPerHead;
            vec_t v[kHeadDim / 8];
            if(i < it.rows)
            {
                const QDataType* src =
                    a.q + static_cast<long_index_t>(it.b * a.q_len + it.row0 + i) * a.stride_q +
                    (it.g * kGroup + h) * kHeadDim;
                static_for<0, kHeadDim / 8, 1>{}([&](auto c) {
                    v[c] = *static_cast<const vec_t*>(static_cast<const void*>(src + c * 8));
                });
            }
            else
            {
                static_for<0, kHeadDim / 8, 1>{}([&](auto c) { v[c] = vec_t{0, 0, 0, 0}; });
            }
            static_for<0, kHeadDim / 8, 1>{}([&](auto c) {
                *static_cast<vec_t*>(static_cast<void*>(q_lds + r * kQStride + c * 8)) = v[c];
            });
        }

        const auto q_lds_desc =
            make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kHeadDim>{}),
                                         make_tuple(number<kQStride>{}, number<1>{}),
                                         number<8>{},
                                         number<1>{});
        const auto k_lds_desc =
            make_naive_tensor_descriptor(make_tuple(number<kN0>{}, number<kHeadDim>{}),
                                         make_tuple(number<kKStride>{}, number<1>{}),
                                         number<8>{},
                                         number<1>{});
        const auto v_lds_desc =
            make_naive_tensor_descriptor(make_tuple(number<kN0>{}, number<kHeadDim>{}),
                                         make_tuple(number<kVStride>{}, number<1>{}),
                                         number<8>{},
                                         number<1>{});

        auto q_lds_view = make_tensor_view<address_space_enum::lds>(q_lds, q_lds_desc);
        auto k_lds_view = make_tensor_view<address_space_enum::lds>(k_lds, k_lds_desc);
        auto v_lds_view = make_tensor_view<address_space_enum::lds>(v_lds, v_lds_desc);

        auto k_lds_read = make_tile_window(
            k_lds_view, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0}, MakeKRegDistribution());
        auto v_lds_read = make_tile_window(
            v_lds_view, make_tuple(number<kK1>{}, number<kN1>{}), {0, 0}, MakeVRegDistribution());

        // Sub-block u of KV block blk of K or V: its page, then a (kSubRows x 64) window.
        const index_t bt_row = it.b * a.block_table_stride;
        auto kv_window       = [&](const KVDataType* cache, index_t blk, index_t u) {
            const index_t j0    = blk * kN0 + u * kSubRows;
            const index_t page  = amd_wave_read_first_lane(a.block_table[bt_row + j0 / kPageSize]);
            const KVDataType* p = cache + page * a.page_stride + it.g * a.head_stride +
                                  (j0 % kPageSize) * a.token_stride;
            auto view = make_naive_tensor_view<address_space_enum::global>(
                p,
                make_tuple(number<kSubRows>{}, number<kHeadDim>{}),
                make_tuple(a.token_stride, number<1>{}),
                number<kKVVec>{},
                number<1>{});
            return make_tile_window(view,
                                    make_tuple(number<kSubRows>{}, number<kHeadDim>{}),
                                          {0, 0},
                                    MakeKVDramDistribution());
        };
        using KVRegTile = decltype(load_tile(kv_window(a.k_cache, 0, 0)));
        statically_indexed_array<KVRegTile, kNumSub> k_reg;
        statically_indexed_array<KVRegTile, kNumSub> v_reg;
        auto load_kv = [&](index_t blk) {
            static_for<0, kNumSub, 1>{}([&](auto u) {
                k_reg(u) = load_tile(kv_window(a.k_cache, blk, u));
                v_reg(u) = load_tile(kv_window(a.v_cache, blk, u));
            });
        };
        load_kv(it.kv_block_begin);

        block_sync_lds();
        auto q_tile = load_tile(make_tile_window(q_lds_view,
                                                 make_tuple(number<kM0>{}, number<kHeadDim>{}),
                                                 {0, 0},
                                                 MakeQRegDistribution()));

        constexpr auto gemm_0 = Gemm0{};
        constexpr auto gemm_1 = Gemm1{};

        auto o_acc = gemm_1.MakeCBlockTile();
        clear_tile(o_acc);
        using STile  = decltype(gemm_0.MakeCBlockTile());
        using MLTile = decltype(block_tile_reduce<AccDataType>(
            STile{}, sequence<1>{}, [](auto x, auto y) { return x + y; }, AccDataType{0}));
        auto m       = MLTile{};
        auto l       = MLTile{};
        set_tile(m, -numeric<AccDataType>::infinity());
        clear_tile(l);

        const float scale_log2 = a.scale * a.k_scale * log2e_v<float>;
        const index_t pos_last = it.pos0 + it.rows - 1;
        const auto f_sum       = [](auto x, auto y) { return x + y; };
        const auto f_max       = [](auto x, auto y) { return max(x, y); };

        for(index_t blk = it.kv_block_begin; blk < it.kv_block_end; ++blk)
        {
            const index_t j0 = blk * kN0;

            // Stage this block: the previous block's gemm1 is done with V and gemm0 with K.
            block_sync_lds();
            static_for<0, kNumSub, 1>{}([&](auto u) {
                auto k_win = make_tile_window(k_lds_view,
                                              make_tuple(number<kSubRows>{}, number<kHeadDim>{}),
                                              {u * kSubRows, 0});
                auto v_win = make_tile_window(v_lds_view,
                                              make_tuple(number<kSubRows>{}, number<kHeadDim>{}),
                                              {u * kSubRows, 0});
                store_tile(k_win, ToCompType(k_reg[u]));
                auto v_c = ToCompType(v_reg[u]);
                // Slots past the context may hold anything (NaN): P is 0 there, V must be too.
                const index_t r0 = j0 + u * kSubRows;
                if(r0 + kSubRows > kv_len)
                {
                    set_tile_if(v_c, type_convert<CompDataType>(0.f), [&](auto idx) {
                        return r0 + idx.at(I0) >= kv_len;
                    });
                }
                store_tile(v_win, v_c);
            });
            block_sync_lds();

            // Prefetch the next block into VGPRs while this one computes.
            if(blk + 1 < it.kv_block_end)
                load_kv(blk + 1);

            // S = Q K^T
            auto s_acc = gemm_0.MakeCBlockTile();
            clear_tile(s_acc);
            static_for<0, kHeadDim / kK0, 1>{}([&](auto i_k0) {
                const auto k_tile = load_tile(k_lds_read);
                gemm_0(s_acc,
                       get_slice_tile(
                           q_tile, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
                       k_tile);
                move_tile_window(k_lds_read, {0, kK0});
            });
            move_tile_window(k_lds_read, {0, -kHeadDim});

            // Causal / window mask, also covers the slots past the context (j > p).
            const bool edge =
                (j0 + kN0 - 1 > it.pos0) || (a.window > 0 && j0 <= pos_last - a.window);
            if(edge)
            {
                set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto idx) {
                    return IsMasked(a, it, idx.at(I0), j0 + idx.at(I1));
                });
            }

            auto p_compute =
                make_static_distributed_tensor<AccDataType>(s_acc.get_tile_distribution());
            if constexpr(kRunningMax)
            {
                auto m_local = block_tile_reduce<AccDataType>(
                    s_acc, sequence<1>{}, f_max, -numeric<AccDataType>::infinity());
                block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});
                const auto m_old = m;
                tile_elementwise_inout(
                    [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    // A row with no visible key yet keeps m = -inf; use 0 so exp2 gives 0.
                    const float m_v     = __builtin_isinf_sign(m[i_idx]) < 0 ? 0.f : m[i_idx];
                    const float row_max = scale_log2 * m_v;
                    sweep_tile_span(s_spans[I1], [&](auto idx1) {
                        constexpr auto ij = make_tuple(idx0, idx1);
                        p_compute(ij)     = exp2(scale_log2 * s_acc[ij] - row_max);
                    });
                });
                auto rowsum =
                    block_tile_reduce<AccDataType>(p_compute, sequence<1>{}, f_sum, AccDataType{0});
                block_tile_reduce_sync(rowsum, f_sum, bool_constant<false>{});

                constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    const float m_v      = __builtin_isinf_sign(m[i_idx]) < 0 ? 0.f : m[i_idx];
                    const float corr     = exp2(scale_log2 * (m_old[i_idx] - m_v));
                    l(i_idx)             = corr * l[i_idx] + rowsum[i_idx];
                    sweep_tile_span(o_spans[I1],
                                    [&](auto idx1) { o_acc(make_tuple(idx0, idx1)) *= corr; });
                });
            }
            else
            {
                // |score| <= 43.8: exp2 of the raw scaled score cannot overflow.
                tile_elementwise_inout(
                    [&](auto& p, auto s) { p = exp2(scale_log2 * s); }, p_compute, s_acc);
                auto rowsum =
                    block_tile_reduce<AccDataType>(p_compute, sequence<1>{}, f_sum, AccDataType{0});
                block_tile_reduce_sync(rowsum, f_sum, bool_constant<false>{});
                tile_elementwise_inout([](auto& x, auto y) { x += y; }, l, rowsum);
            }

            // O += P V
            const auto p_tile = MakePForGemm1(p_compute);
            static_for<0, kN0 / kK1, 1>{}([&](auto i_k1) {
                const auto v_tile = load_tile_transpose(v_lds_read);
                gemm_1(o_acc,
                       get_slice_tile(
                           p_tile, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                       v_tile);
                move_tile_window(v_lds_read, {kK1, 0});
            });
            move_tile_window(v_lds_read, {-kN0, 0});
        }

        // Partial: O / l and lse = m * scale_log2 + log2(l), log2 domain. l == 0 (every key of
        // this split masked for the row) gives O = 0, lse = -inf.
        auto lse = make_static_distributed_tensor<AccDataType>(l.get_tile_distribution());
        {
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const float lv       = l[i_idx];
                const float inv      = lv > 0.f ? 1.f / lv : 0.f;
                float m_v            = 0.f;
                if constexpr(kRunningMax)
                    m_v = __builtin_isinf_sign(m[i_idx]) < 0 ? 0.f : scale_log2 * m[i_idx];
                lse(i_idx) = lv > 0.f ? m_v + log2(lv) : -numeric<AccDataType>::infinity();
                sweep_tile_span(o_spans[I1],
                                [&](auto idx1) { o_acc(make_tuple(idx0, idx1)) *= inv; });
            });
        }

        auto o_view = make_naive_tensor_view<address_space_enum::global>(
            o_part,
            make_tuple(number<kM0>{}, number<kHeadDim>{}),
            make_tuple(number<kHeadDim>{}, number<1>{}),
            number<4>{},
            number<1>{});
        auto o_win =
            make_tile_window(o_view, make_tuple(number<kM0>{}, number<kHeadDim>{}), {0, 0});
        store_tile(o_win, o_acc);

        auto lse_view = make_naive_tensor_view<address_space_enum::global>(
            lse_part, make_tuple(number<kM0>{}), make_tuple(number<1>{}), number<1>{}, number<1>{});
        auto lse_win = make_tile_window(lse_view, make_tuple(number<kM0>{}), {0});
        store_tile(lse_win, lse);
    }

    /// Merges the tile's splits, one thread per Q row, and hands each row to OutputFunctor.
    CK_TILE_DEVICE static void Combine(const Args& a, const Item& it)
    {
        using f4_t      = ext_vector_t<float, 4>;
        const index_t r = threadIdx.x;
        const index_t i = r % kRowsPerHead;
        if(i >= it.rows)
            return;

        const index_t S       = a.num_splits;
        const long_index_t w0 = static_cast<long_index_t>(it.tile) * S;
        const float* lse_base = a.lse_partial + w0 * kM0 + r;
        const float* o_base   = a.o_partial + (w0 * kM0 + r) * kHeadDim;

        float mx = -numeric<float>::infinity();
        for(index_t s = 0; s < S; ++s)
            mx = max(mx, lse_base[s * kM0]);

        array<float, kHeadDim> o;
        static_for<0, kHeadDim, 1>{}([&](auto d) { o[d] = 0.f; });
        float sum = 0.f;
        for(index_t s = 0; s < S; ++s)
        {
            const float lse_s = lse_base[s * kM0];
            if(__builtin_isinf_sign(lse_s) < 0)
                continue;
            const float w = exp2(lse_s - mx);
            sum += w;
            const float* src = o_base + static_cast<long_index_t>(s) * kM0 * kHeadDim;
            static_for<0, kHeadDim / 4, 1>{}([&](auto c) {
                const f4_t v = *static_cast<const f4_t*>(static_cast<const void*>(src + c * 4));
                static_for<0, 4, 1>{}([&](auto k) { o[c * 4 + k] += w * v[k.value]; });
            });
        }
        const float inv = a.v_scale / sum;
        static_for<0, kHeadDim, 1>{}([&](auto d) { o[d] *= inv; });

        OutputFunctor::template Apply<kHeadDim>(
            a.out, it.b * a.q_len + it.row0 + i, it.g * kGroup + r / kRowsPerHead, o);
    }
};

} // namespace ck_tile
