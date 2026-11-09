// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_batch_prefill_qr_tdm_page_tile_window.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_tdm_sched_page_tile.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include <stdexcept>
#include <string>

namespace ck_tile {

// Batch prefill over a paged KV cache with a physical page size independent of the logical N128
// KV tile; reads page IDs from a vLLM 2D block table or an SGLang 1D page list while preserving
// the QR M128/N128 compute layout.
//
// Experimental, gfx1250 only: BF16 or FP16, group mode, linear [page, token, kv_head, dim] K/V
// pools, D64/V64 or D128/V128, no mask or a left/right window aligned to the top-left or
// bottom-right corner (causal is left unbounded, right 0; a bounded left is a sliding window),
// optional sink tokens (sink_size) and per-head sink logits (sink_ptr), optional tanh logits soft
// cap (no bias), optional elementwise bias or ALiBi, optional LSE output, no dropout. Build each
// instance in its own translation unit with the scheduled-TDM FMHA flags (-mllvm
// -amdgpu-expert-scheduling-mode -fgpu-flush-denormals-to-zero): the launch contract below rests on
// measurements of that generated code, and the generated code was observed to change when several
// instances share a translation unit.
template <typename FmhaPipeline_,
          typename EpiloguePipeline_,
          index_t KvSplits_      = 1,
          index_t QueryTailRows_ = 0>
struct FmhaBatchPrefillQrTdmPageTileKernel
    : private FmhaBatchPrefillWithPagedKVCacheKernel<FmhaPipeline_, EpiloguePipeline_>
{
    using Base         = FmhaBatchPrefillWithPagedKVCacheKernel<FmhaPipeline_, EpiloguePipeline_>;
    using FmhaPipeline = remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline    = remove_cvref_t<EpiloguePipeline_>;
    using Kargs               = typename Base::Kargs;
    using PageBlockTableKargs = typename Base::PageBlockTableKargs;
    using QDataType           = typename FmhaPipeline::QDataType;
    using KDataType           = typename FmhaPipeline::KDataType;
    using VDataType           = typename FmhaPipeline::VDataType;
    using ODataType           = typename FmhaPipeline::ODataType;
    using SaccDataType        = typename FmhaPipeline::SaccDataType;
    using LSEDataType         = typename FmhaPipeline::LSEDataType;
    using BiasDataType        = remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using FmhaMask            = typename FmhaPipeline::FmhaMask;
    using Policy              = typename FmhaPipeline::Policy;

    using Base::BlockSize;
    using Base::GetTileIndex;
    // KV split (KvSplits_ 2): the grid's batch dimension also enumerates the splits. Split s
    // attends to its N0-aligned half of the KV tiles and writes an FP32 partial output and its
    // LSE to plane s of the workspace (O [split][total_q][nhead][dim], LSE
    // [split][nhead][total_q]: the split stride is nhead_stride_lse rows); a merge kernel
    // combines the two planes by their LSE. KvSplits_ 1 is the ordinary kernel.
    // Query tail (QueryTailRows_ R > 0, KV split only): each sequence's rows past its last full
    // R-row tile, as a sequence of their own (a bottom-right mask keeps its diagonal), for a main
    // kernel with R-row tiles that runs the full tiles; the grid covers R rows per sequence.
    CK_TILE_HOST static constexpr auto
    GridSize(index_t batch, index_t heads, index_t queries, index_t dim)
    {
        return Base::GridSize(
            batch * KvSplits_, heads, QueryTailRows_ > 0 ? QueryTailRows_ : queries, dim);
    }
    static constexpr index_t kKvSplits      = KvSplits_;
    static constexpr index_t kQueryTailRows = QueryTailRows_;
    static_assert(KvSplits_ == 1 || KvSplits_ == 2);
    static_assert(QueryTailRows_ == 0 ||
                  (KvSplits_ == 2 && QueryTailRows_ % FmhaPipeline_::kM0 == 0));
    static_assert(KvSplits_ == 1 ||
                  ((FmhaPipeline::kPageBlockSize == 64 || FmhaPipeline::kPageBlockSize == 128) &&
                   (FmhaPipeline::kQKHeaddim == 64 || FmhaPipeline::kQKHeaddim == 128) &&
                   FmhaPipeline::kN0 == 128 && FmhaPipeline::kStoreLSE &&
                   FmhaPipeline::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                   !FmhaPipeline::kHasSink && !FmhaPipeline::kHasLogitsSoftCap &&
                   std::is_same_v<typename FmhaPipeline::ODataType, float>));
    // The page-tile policy takes an FP32 output for a KV-split partial
    // (detail::fmha_tdm_is_split_partial_v), so only a KV split has one.
    static_assert(KvSplits_ == 2 || !std::is_same_v<typename FmhaPipeline::ODataType, float>,
                  "an FP32 output is the partial output of a KV split");
    using Base::MakeKargs;

    static constexpr bool kIsGroupMode = Base::kIsGroupMode;
    static constexpr bool kStoreLSE    = Base::kStoreLSE;
    // Sink tokens (sink_size, StreamingLLM) and the per-head sink logit (sink_ptr) share one
    // instance, as in the existing batch prefill kernel; sink tokens need a mask.
    static constexpr bool kHasSink = FmhaPipeline::kHasSink;
    // Tanh logits soft cap (no bias), as in the existing batch prefill kernel.
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    // Elementwise bias [nhead, total_q, stride_bias] or ALiBi slopes, as in the existing kernel.
    static constexpr auto BiasEnum = FmhaPipeline::BiasEnum;
    static_assert(!kHasSink || FmhaMask::IsMasking, "sink tokens need a masked instance");
    static constexpr auto kKVLookupTable = Base::kKVLookupTable;
    static constexpr index_t kBlockPerCu = Base::kBlockPerCu;
    static constexpr index_t kBlockSize  = Base::kBlockSize;

    static_assert(kIsGroupMode);
    // Vectorized KV cache: K page-head [D/8][P][8], V page-head [P/8][D][8] (16-bit, 8 wide).
    static constexpr bool kVectorizedKV =
        FmhaPipeline::kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    static constexpr index_t kKWindowKind =
        kVectorizedKV ? FmhaPageTileKVKind::kVectorizedK : FmhaPageTileKVKind::kLinear;
    static constexpr index_t kVWindowKind =
        kVectorizedKV ? FmhaPageTileKVKind::kVectorizedV : FmhaPageTileKVKind::kLinear;
    static constexpr bool kSglangTable =
        kKVLookupTable == BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    static_assert(kSglangTable ||
                  kKVLookupTable == BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D);
    // The physical allocation page may differ from the logical N128 tile.
    static_assert(FmhaPipeline::kPageBlockSize == 16 || FmhaPipeline::kPageBlockSize == 64 ||
                  FmhaPipeline::kPageBlockSize == 128 || FmhaPipeline::kPageBlockSize == 256 ||
                  FmhaPipeline::kPageBlockSize == 1024);
    static_assert(FmhaPipeline::kPageBlockSize % FmhaPipeline::kN0 == 0 ||
                  FmhaPipeline::kN0 % FmhaPipeline::kPageBlockSize == 0);
    using PageTileTraits =
        FmhaBatchPrefillPageTileTraits<FmhaPipeline::kPageBlockSize, FmhaPipeline::kQKHeaddim>;

    // The page tile writer is a workgroup collective over exactly kWaves waves (counted in
    // waves, not threads: the host pass sees a different warp size than the device).
    static_assert(kBlockSize / get_warp_size() == PageTileTraits::kWaves &&
                      kBlockSize % get_warp_size() == 0,
                  "block must be exactly the page tile writer's waves");

    // Metadata byte budget comes from the page size (32 B for P16, else 16 B).
    static constexpr index_t kPageMetadataBytes = PageTileTraits::kMetadataBytes;

    static_assert(Policy::GetLdsOffsetK0() == 0);
    static_assert(Policy::GetLdsOffsetK1() >= Policy::GetLdsOffsetK0() + Policy::kKFootprintBytes);
    static_assert(Policy::GetLdsOffsetV0() >= Policy::GetLdsOffsetK1() + Policy::kKFootprintBytes);
    static_assert(Policy::GetLdsOffsetV1() >= Policy::GetLdsOffsetV0() + Policy::kVFootprintBytes);
    static_assert(Policy::GetLdsArenaSize() >= Policy::GetLdsOffsetV1() + Policy::kVFootprintBytes);
    static_assert(Policy::GetLdsOffsetK0() % 16 == 0 && Policy::GetLdsOffsetK1() % 16 == 0 &&
                  Policy::GetLdsOffsetV0() % 16 == 0 && Policy::GetLdsOffsetV1() % 16 == 0);
    // P16 needs a 32-byte pending slot; every other page size keeps 16 bytes.
    static_assert(kPageMetadataBytes % 16 == 0 && kPageMetadataBytes >= 16);
    static_assert(Policy::GetLdsArenaSize() % 16 == 0,
                  "the pending metadata slot must start 16-byte aligned");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(FmhaPipeline::GetSmemSize() ==
                      Policy::GetLdsArenaSize() + kPageMetadataBytes);
        return max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    // The LDS base is aligned explicitly: the arena sub-buffers sit at
    // 16-byte-aligned offsets and the pending metadata slot (32 B for P16) follows the
    // arena, so the arena size must be a multiple of the slot and the base must be at
    // least as aligned as the slot.
    static constexpr index_t kLdsBaseAlignmentBytes = 32;
    static_assert(kLdsBaseAlignmentBytes % kPageMetadataBytes == 0);
    static_assert(Policy::GetLdsArenaSize() % kPageMetadataBytes == 0,
                  "the pending metadata slot must start aligned to its own size");

    // ---- LDS residency launch contract ---------------------------------------------------
    // KNOWN LIMITATION, AVOIDED AT LAUNCH, NOT FIXED. On gfx1250 a page tile that issues two
    // TDM fragments per wave and operand (P16) was measured nondeterministic when three
    // workgroups are resident per WGP; the root cause is unknown. A kernel cannot observe
    // its own residency, so the limit is a launch contract: every launcher must request
    // GetLaunchDynamicSmemSize() bytes of dynamic LDS (CheckLaunch and MakeCheckedKernel
    // refuse less). Under the assumptions below this bounds LDS residency to
    // kMaxResidentWorkgroups without changing GetSmemSize(), the logical LDS recorded in the
    // code object; the bound is derived, not measured.
    //
    // Derived from resources, not from an instance name: nonzero only when the tile issues more
    // than one fragment per wave AND the static LDS alone would admit more than
    // kMaxResidentWorkgroups (so P16/D128, already LDS-bound at two, needs nothing). Both
    // masks are covered: P16/D64/causal is currently limited to two by its registers, but the
    // contract must not depend on that margin.
    //
    // Assumptions: 320 KiB of LDS per WGP, allocated in 2048-byte granules. The request
    // fills the reservation to a whole number of granules, so a finer granule cannot
    // re-admit a third workgroup.
    static constexpr index_t kLdsBytesPerWgp        = 320 * 1024;
    static constexpr index_t kLdsAllocGranuleBytes  = 2048;
    static constexpr index_t kMaxResidentWorkgroups = 2;
    static constexpr bool kNeedsResidencyCap        = PageTileTraits::kFragmentsPerWave > 1;
    // Smallest granule-aligned reservation for which kMaxResidentWorkgroups + 1 do not fit.
    static constexpr index_t kCappedReservationBytes =
        (kLdsBytesPerWgp / ((kMaxResidentWorkgroups + 1) * kLdsAllocGranuleBytes) + 1) *
        kLdsAllocGranuleBytes;

    CK_TILE_HOST_DEVICE static constexpr index_t GetLaunchDynamicSmemSize()
    {
        if constexpr(kNeedsResidencyCap)
            return GetSmemSize() >= kCappedReservationBytes
                       ? index_t{0}
                       : kCappedReservationBytes - GetSmemSize();
        else
            return 0;
    }

    static_assert(!kNeedsResidencyCap ||
                      (kMaxResidentWorkgroups + 1) *
                              integer_divide_ceil(GetSmemSize() + GetLaunchDynamicSmemSize(),
                                                  kLdsAllocGranuleBytes) *
                              kLdsAllocGranuleBytes >
                          kLdsBytesPerWgp,
                  "the launch contract must exclude a third resident workgroup");

    // Host-side launch contract: the dynamic LDS request above and the runtime arguments
    // the kernel specialises on but does not re-check on the device (page size, head
    // dimensions, mask direction, page/token strides, table pointers). Call before every
    // launch; throws std::invalid_argument and launches nothing on a violation.
    CK_TILE_HOST static void CheckLaunch(const Kargs& kargs, index_t dynamic_smem_bytes)
    {
        auto require = [](bool ok, const char* what) {
            if(!ok)
                throw std::invalid_argument(std::string("qr_tdm_page_tile launch contract: ") +
                                            what);
        };
        require(dynamic_smem_bytes >= GetLaunchDynamicSmemSize(),
                "dynamic LDS request below GetLaunchDynamicSmemSize() (residency limit)");
        require(kargs.page_block_size == FmhaPipeline::kPageBlockSize,
                "page_block_size differs from the compiled page size");
        require(kargs.hdim_q == FmhaPipeline::kQKHeaddim, "hdim_q differs from the compiled D");
        require(kargs.hdim_v == FmhaPipeline::kN1, "hdim_v differs from the compiled V width");
        require(kargs.nhead_ratio_qk > 0, "nhead_ratio_qk must be positive");
        if constexpr(!kVectorizedKV)
            require(kargs.stride_k >= kargs.hdim_q && kargs.stride_v >= kargs.hdim_v,
                    "K/V token stride smaller than the head dimension");
        if constexpr(kSglangTable)
        {
            require(kargs.page_table.kv_indptr != nullptr, "kv_indptr is null");
            require(kargs.page_table.kv_page_indices != nullptr, "kv_page_indices is null");
            require(kargs.page_table.kv_last_page_lens != nullptr, "kv_last_page_lens is null");
        }
        else
        {
            require(kargs.page_table.seqlen_k_ptr != nullptr, "seqlen_k_ptr is null");
            require(kargs.page_table.block_table_ptr != nullptr, "block_table_ptr is null");
        }
        require(kargs.seqstart_q_ptr != nullptr, "seqstart_q_ptr is null");
        if constexpr(kStoreLSE)
            require(kargs.lse_ptr != nullptr, "lse_ptr is null");
        if constexpr(kVectorizedKV)
        {
            // The token stride is implied by the layout (stride_k/stride_v are not read, but
            // must stay positive); a page-head holds P x D elements.
            require(kargs.stride_k > 0 && kargs.stride_v > 0, "K/V token stride not positive");
            require(static_cast<long_index_t>(kargs.batch_stride_k) >=
                            static_cast<long_index_t>(FmhaPipeline::kPageBlockSize) *
                                kargs.hdim_q &&
                        static_cast<long_index_t>(kargs.batch_stride_v) >=
                            static_cast<long_index_t>(FmhaPipeline::kPageBlockSize) * kargs.hdim_v,
                    "page stride smaller than one vectorized page-head");
            require(kargs.batch_stride_k % 8 == 0 && kargs.batch_stride_v % 8 == 0 &&
                        kargs.nhead_stride_k % 8 == 0 && kargs.nhead_stride_v % 8 == 0,
                    "vectorized K/V page or head stride not a multiple of 8 elements");
        }
        else
        {
            require(
                static_cast<long_index_t>(kargs.batch_stride_k) >=
                        static_cast<long_index_t>(FmhaPipeline::kPageBlockSize) * kargs.stride_k &&
                    static_cast<long_index_t>(kargs.batch_stride_v) >=
                        static_cast<long_index_t>(FmhaPipeline::kPageBlockSize) * kargs.stride_v,
                "page stride smaller than page_block_size token rows");
            require(kargs.stride_k % FmhaPipeline::kAlignmentK == 0 &&
                        kargs.stride_v % FmhaPipeline::kAlignmentV == 0,
                    "K/V token stride not a multiple of the vector alignment");
        }
        if constexpr(FmhaMask::IsMasking)
        {
            // Any left/right window (a negative size is unbounded); the corner follows
            // mask_type as in the existing batch prefill kernel (top-left, else bottom-right).
            require(kargs.mask_type != GenericAttentionMaskEnum::NO_MASK,
                    "a masked instance needs a mask type");
            if constexpr(kHasSink)
                require(kargs.sink_size >= 0, "sink_size must not be negative");
            else
                require(kargs.sink_size == 0, "sink tokens need a sink instance");
            // Each KV split shifts a bottom-right window to its key range; a top-left corner or
            // a left bound cannot follow (the device traps on both).
            if constexpr(KvSplits_ == 2)
                require(kargs.mask_type != GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT &&
                            kargs.window_size_left < 0,
                        "the KV split takes bottom-right masks without a left window");
        }
        if constexpr(!kHasSink)
            require(kargs.sink_ptr == nullptr, "sink logits need a sink instance");
        if constexpr(kHasLogitsSoftCap)
            require(kargs.logits_soft_cap > 0.f, "a soft-cap instance needs logits_soft_cap > 0");
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            require(kargs.bias_ptr != nullptr && kargs.stride_bias >= kargs.seqlen_k,
                    "an elementwise-bias instance needs bias_ptr and stride_bias >= seqlen_k");
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            require(kargs.alibi_slope_ptr != nullptr, "an ALiBi instance needs the slopes");
    }

    // Checked launch path: owns the dynamic LDS request, so a launcher cannot issue a value
    // other than GetLaunchDynamicSmemSize(), and runs CheckLaunch before anything is issued.
    // Returns the ck_tile launch functor for launch_kernel().
    CK_TILE_HOST static auto MakeCheckedKernel(const Kargs& kargs, dim3 grids)
    {
        CheckLaunch(kargs, GetLaunchDynamicSmemSize());
        return make_kernel<kBlockPerCu, gfx125_t>(FmhaBatchPrefillQrTdmPageTileKernel{},
                                                  grids,
                                                  BlockSize(),
                                                  GetLaunchDynamicSmemSize(),
                                                  kargs);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // Arena at offset 0 (16-byte-aligned sub-buffers), then the pending metadata slot.
        alignas(kLdsBaseAlignmentBytes) __shared__ char smem_ptr[GetSmemSize()];

        const auto [i_tile_m, i_tile_n, i_nhead, i_split_batch] = GetTileIndex(kargs);
        const index_t i_batch =
            KvSplits_ == 1 ? i_split_batch : amd_wave_read_first_lane(i_split_batch / KvSplits_);
        const index_t split_id =
            KvSplits_ == 1 ? 0 : amd_wave_read_first_lane(i_split_batch % KvSplits_);
        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN1);

        long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
        kargs.seqlen_q           = kargs.seqstart_q_ptr[i_batch + 1] - query_start;
        if constexpr(QueryTailRows_ > 0)
        {
            const index_t tail_begin = static_cast<index_t>(static_cast<uint32_t>(kargs.seqlen_q) /
                                                            QueryTailRows_ * QueryTailRows_);
            query_start += tail_begin;
            kargs.seqlen_q -= tail_begin;
        }
        if(kargs.seqlen_q <= i_m0)
            return;

        // This sequence's page ids (page_row) and, for the SGLang list, how many it has. The vLLM
        // path keeps its original statement order (its code objects are compared bit for bit).
        const int32_t* page_row = nullptr;
        index_t sglang_pages    = 0;
        if constexpr(kSglangTable)
        {
            // kv_page_indices[kv_indptr[b], kv_indptr[b + 1]) are the pages of sequence b; its last
            // page holds kv_last_page_lens[b] tokens.
            const int32_t page_start = kargs.page_table.kv_indptr[i_batch];
            sglang_pages             = kargs.page_table.kv_indptr[i_batch + 1] - page_start;
            kargs.seqlen_k           = 0;
            if(sglang_pages > 0)
            {
                const int32_t last_page_len = kargs.page_table.kv_last_page_lens[i_batch];
                if(page_start < 0 || last_page_len <= 0 ||
                   last_page_len > FmhaPipeline::kPageBlockSize || kargs.stride_k <= 0 ||
                   kargs.stride_v <= 0 || kargs.batch_stride_k <= 0 || kargs.batch_stride_v <= 0 ||
                   kargs.num_total_pages <= 0)
                    __builtin_trap();
                kargs.seqlen_k = (sglang_pages - 1) * FmhaPipeline::kPageBlockSize + last_page_len;
            }
            else if(sglang_pages < 0)
                __builtin_trap();
            page_row = kargs.page_table.kv_page_indices + page_start;
        }
        else
        {
            kargs.seqlen_k = kargs.page_table.seqlen_k_ptr[i_batch];
        }
        const long_index_t kv_head = i_nhead / kargs.nhead_ratio_qk;
        if constexpr(!kSglangTable)
        {
            const long_index_t first_page =
                static_cast<long_index_t>(i_batch) * kargs.page_table.batch_stride_block_table;
            page_row = kargs.page_table.block_table_ptr + first_page;
            if(kargs.seqlen_k > 0)
            {
                const index_t page_count = (kargs.seqlen_k - 1) / FmhaPipeline::kPageBlockSize + 1;
                if(kargs.stride_k <= 0 || kargs.stride_v <= 0 || kargs.batch_stride_k <= 0 ||
                   kargs.batch_stride_v <= 0 || kargs.page_table.batch_stride_block_table <= 0 ||
                   kargs.num_total_pages <= 0 ||
                   page_count > kargs.page_table.batch_stride_block_table || first_page < 0)
                    __builtin_trap();
            }
        }
        // Table entries the window may index for this sequence.
        index_t page_capacity = [&]() {
            if constexpr(kSglangTable)
                return sglang_pages;
            else
                return kargs.page_table.batch_stride_block_table;
        }();
        if constexpr(KvSplits_ == 2)
        {
            static_assert(!kSglangTable);
            if(kargs.seqlen_k < 0)
                __builtin_trap();
            const long_index_t orig_seqlen_k = kargs.seqlen_k;
            // N0-aligned split starts preserve all original shuffled physical page IDs.
            const long_index_t span =
                ((static_cast<long_index_t>(kargs.seqlen_k) + 2 * FmhaPipeline::kN0 - 1) /
                 (2 * FmhaPipeline::kN0)) *
                FmhaPipeline::kN0;
            const long_index_t begin   = split_id * span;
            const index_t split_length = static_cast<index_t>(
                max(long_index_t{0}, min(span, static_cast<long_index_t>(kargs.seqlen_k) - begin)));
            // An empty second split of a short KV can start beyond the table (pages smaller than
            // the N0 tile): use its one-past cursor only.
            const index_t raw_skip_pages = begin / FmhaPipeline::kPageBlockSize;
            const index_t skip_pages =
                split_length == 0 ? min(raw_skip_pages, page_capacity) : raw_skip_pages;
            if(skip_pages > page_capacity)
                __builtin_trap();
            kargs.seqlen_k = split_length;
            if constexpr(FmhaMask::IsMasking)
            {
                // The split sees KV [begin, begin + seqlen_k) of the original keys: shift the
                // window so that the mask stays in the original coordinates (bottom-right only).
                if(kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT)
                    __builtin_trap();
                const index_t shift = static_cast<index_t>(orig_seqlen_k - begin - kargs.seqlen_k);
                if(kargs.window_size_right >= 0)
                    kargs.window_size_right += shift;
                if(kargs.window_size_left >= 0)
                {
                    if(kargs.window_size_left < shift)
                        __builtin_trap();
                    kargs.window_size_left -= shift;
                }
            }
            page_row += skip_pages;
            page_capacity -= skip_pages;
        }
        const auto* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                            query_start * kargs.stride_q +
                            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q;
        const auto* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) + kv_head * kargs.nhead_stride_k;
        const auto* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) + kv_head * kargs.nhead_stride_v;
        auto* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) + query_start * kargs.stride_o +
                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o;
        if constexpr(KvSplits_ == 2)
            o_ptr += static_cast<long_index_t>(split_id) * kargs.nhead_stride_lse * kargs.stride_o;

        const auto q_naive = make_naive_tensor_view<address_space_enum::global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            number<FmhaPipeline::kAlignmentQ>{},
            number<1>{});
        const auto q_view = pad_tensor_view(
            q_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
            sequence<true, false>{});
        const auto q_window = make_tile_window(
            q_view,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
            {i_m0, 0});

        using KLengths =
            decltype(make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}));
        using VLengths =
            decltype(make_tuple(number<FmhaPipeline::kN1>{}, number<FmhaPipeline::kK1>{}));
        const auto k_window = FmhaBatchPrefillQrTdmPageTileWindow<KDataType,
                                                                  KLengths,
                                                                  void,
                                                                  FmhaPipeline::kAlignmentK,
                                                                  FmhaPipeline::kPageBlockSize,
                                                                  FmhaPipeline::kQKHeaddim,
                                                                  kKWindowKind,
                                                                  Policy::Options::kScalarRowClamp,
                                                                  Policy::kDrainLdsBeforeTdmIssue>{
            k_ptr,
            kargs.seqlen_k,
            kargs.stride_k,
            page_row,
            static_cast<long_index_t>(kargs.batch_stride_k),
            page_capacity,
            kargs.num_total_pages,
            {},
            {0, 0}};
        const auto v_window = FmhaBatchPrefillQrTdmPageTileWindow<VDataType,
                                                                  VLengths,
                                                                  void,
                                                                  FmhaPipeline::kAlignmentV,
                                                                  FmhaPipeline::kPageBlockSize,
                                                                  FmhaPipeline::kN1,
                                                                  kVWindowKind,
                                                                  Policy::Options::kScalarRowClamp,
                                                                  Policy::kDrainLdsBeforeTdmIssue>{
            v_ptr,
            kargs.seqlen_k,
            kargs.stride_v,
            page_row,
            static_cast<long_index_t>(kargs.batch_stride_v),
            page_capacity,
            kargs.num_total_pages,
            {},
            {0, 0}};

        // Elementwise bias [nhead, total_q, stride_bias], bounded by seqlen_q x seqlen_k.
        auto bias_window = [&]() {
            constexpr auto bias_lengths =
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_bias +
                    query_start * kargs.stride_bias;
                const auto bias_naive = make_naive_tensor_view<address_space_enum::global>(
                    bias_ptr,
                    make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                    make_tuple(kargs.stride_bias, 1),
                    number<FmhaPipeline::kAlignmentBias>{},
                    number<1>{});
                const auto bias_view =
                    pad_tensor_view(bias_naive, bias_lengths, sequence<true, true>{});
                return make_tile_window(bias_view, bias_lengths, {i_m0, 0});
            }
            else
            {
                return make_null_tile_window(bias_lengths);
            }
        }();
        // LSE [nhead, total_q] (group mode), one value per query row, bounded by seqlen_q like O.
        auto lse_window = [&]() {
            constexpr auto lse_lengths = make_tuple(number<FmhaPipeline::kM0>{});
            if constexpr(kStoreLSE)
            {
                auto* lse_ptr = reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lse +
                                query_start;
                if constexpr(KvSplits_ == 2)
                    lse_ptr += static_cast<long_index_t>(split_id) * kargs.nhead_stride_lse *
                               kargs.num_head_q;
                const auto lse_naive = make_naive_tensor_view<address_space_enum::global>(
                    lse_ptr, make_tuple(kargs.seqlen_q), make_tuple(1), number<1>{}, number<1>{});
                const auto lse_view = pad_tensor_view(lse_naive, lse_lengths, sequence<true>{});
                return make_tile_window(lse_view, lse_lengths, {i_m0});
            }
            else
            {
                return make_null_tile_window(lse_lengths);
            }
        }();
        // FmhaFwdMaskKargs is only part of Kargs when the problem actually masks.
        // A dense instance therefore must not read the window from Kargs, exactly as
        // the generic batch-prefill kernel guards this with kHasMask.
        const FmhaMask mask = [&]() {
            if constexpr(FmhaMask::IsMasking)
            {
                if constexpr(kHasSink)
                    return make_generic_attention_mask_from_lr_window<FmhaMask>(
                        kargs.window_size_left,
                        kargs.window_size_right,
                        kargs.sink_size,
                        kargs.seqlen_q,
                        kargs.seqlen_k,
                        kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
                else
                    return make_generic_attention_mask_from_lr_window<FmhaMask>(
                        kargs.window_size_left,
                        kargs.window_size_right,
                        kargs.seqlen_q,
                        kargs.seqlen_k,
                        kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            }
            else
            {
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
            }
        }();

        // Per-head sink logit in the score domain, as in the existing batch prefill kernel.
        // With bias the core seeds the row max with sink_v * log2(e), so the logit enters as is.
        const float sink_value = [&]() {
            if constexpr(kHasSink && BiasEnum != BlockAttentionBiasEnum::NO_BIAS)
                return kargs.sink_ptr != nullptr
                           ? *(static_cast<const float*>(kargs.sink_ptr) + i_nhead)
                           : -numeric<float>::infinity();
            else if constexpr(kHasSink)
                return kargs.sink_ptr != nullptr
                           ? *(static_cast<const float*>(kargs.sink_ptr) + i_nhead) / kargs.scale_s
                           : -numeric<float>::infinity();
            else
                return -numeric<float>::infinity();
        }();
        // ALiBi with the slope folded into the exp2 domain, as in the existing kernel.
        const auto position_encoding = [&]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const SaccDataType slope =
                    *(reinterpret_cast<const SaccDataType*>(kargs.alibi_slope_ptr) +
                      i_batch * kargs.alibi_slope_stride + i_nhead) *
                    log2e_v<SaccDataType>;
                if constexpr(FmhaMask::IsMasking)
                    return make_alibi_from_lr_mask<SaccDataType, true>(slope,
                                                                       kargs.window_size_left,
                                                                       kargs.window_size_right,
                                                                       kargs.seqlen_q,
                                                                       kargs.seqlen_k,
                                                                       kargs.mask_type);
                else
                    return Alibi<SaccDataType, true>{
                        slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
            }
            else
            {
                return EmptyPositionEncoding<SaccDataType>{};
            }
        }();
        const float logits_soft_cap = [&]() {
            if constexpr(kHasLogitsSoftCap)
                return kargs.logits_soft_cap;
            else
                return 0.f;
        }();
        auto o_acc_tile = FmhaPipeline{}(q_window,
                                         k_window,
                                         v_window,
                                         bias_window,
                                         lse_window,
                                         mask,
                                         position_encoding,
                                         kargs.scale_s,
                                         sink_value,
                                         logits_soft_cap,
                                         smem_ptr + Policy::GetLdsOffsetK0(),
                                         smem_ptr + Policy::GetLdsOffsetK1(),
                                         smem_ptr + Policy::GetLdsOffsetV0(),
                                         smem_ptr + Policy::GetLdsOffsetV1());

        const auto o_naive = make_naive_tensor_view<address_space_enum::global>(
            o_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_o, 1),
            number<FmhaPipeline::kAlignmentO>{},
            number<1>{});
        const auto o_view =
            pad_tensor_view(o_naive,
                            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                            sequence<true, false>{});
        auto o_window =
            make_tile_window(o_view,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});
        EpiloguePipeline{}(o_window, o_acc_tile, nullptr);
    }
};

} // namespace ck_tile
