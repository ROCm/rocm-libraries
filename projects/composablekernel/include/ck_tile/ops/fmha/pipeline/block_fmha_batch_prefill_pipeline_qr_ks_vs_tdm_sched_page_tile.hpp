// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_batch_prefill_page_tile.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_paged.hpp"

namespace ck_tile {

// Owns its page-ID schedule and keeps the M128/N128 compute geometry of the scheduled TDM pipeline.
// The physical allocation page may differ from the logical N128 tile: P>=128
// uses a slice of one physical page, P64 two pages (one per pair of waves), P16 assembles
// eight page-local fragments. The per-head sink logit (virtual sink) is merged at normalization in
// sink instances only.
template <typename Problem_>
struct BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTile
    : BlockFmhaPipelineQRKSVSTdmSchedPaged<Problem_,
                                           FmhaTdmSchedPolicyFor<Problem_>,
                                           Problem_::kHasSink>
{
    using Base    = BlockFmhaPipelineQRKSVSTdmSchedPaged<Problem_,
                                                         FmhaTdmSchedPolicyFor<Problem_>,
                                                         Problem_::kHasSink>;
    using Problem = typename Base::Problem;
    static_assert(Problem::kIsGroupMode);
    // Physical page support. The supported set is P % N == 0 or N % P == 0:
    // P256/P1024 slice one page, P128 maps one page to one tile, P64 maps two pages to one
    // tile, P16 bundles eight independently mapped fragments.
    static_assert(Problem::kPageBlockSize == 16 || Problem::kPageBlockSize == 64 ||
                      Problem::kPageBlockSize == 128 || Problem::kPageBlockSize == 256 ||
                      Problem::kPageBlockSize == 1024,
                  "unsupported physical page size for the batch-prefill page-tile core");
    static_assert(Problem::kPageBlockSize % Problem::BlockFmhaShape::kN0 == 0 ||
                      Problem::BlockFmhaShape::kN0 % Problem::kPageBlockSize == 0,
                  "supported pages satisfy P % N == 0 or N % P == 0");
    // Linear pages, or the vectorized layout with 8 x 16-bit vectors (the window issue lands both
    // in the same LDS image).
    static_assert(Problem::kKVMemoryLayout ==
                      BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT ||
                  (Problem::kKVMemoryLayout ==
                       BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT &&
                   Problem::kVectorSize == 8));
    static_assert(Problem::kKVLookupTable ==
                      BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D ||
                  Problem::kKVLookupTable ==
                      BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D);
    static_assert(Problem::kPadSeqLenQ && Problem::kPadSeqLenK);
    static_assert(!Problem::kPadHeadDimQ && !Problem::kPadHeadDimV);
    // BF16 or FP16, the same type for Q, K and V.
    static_assert((std::is_same_v<typename Base::QDataType, bf16_t> ||
                   std::is_same_v<typename Base::QDataType, half_t>) &&
                  std::is_same_v<typename Base::KDataType, typename Base::QDataType> &&
                  std::is_same_v<typename Base::VDataType, typename Base::QDataType>);
    static_assert(Base::kM0 == 128 && Base::kN0 == 128);
    // D64/V64 and D128/V128 are the supported head dimensions; the physical WMMA
    // shape stays 16x16x32 and is not the head width.
    static_assert((Base::kQKHeaddim == 64 && Base::kN1 == 64) ||
                      (Base::kQKHeaddim == 128 && Base::kN1 == 128),
                  "the batch-prefill page tile supports D64/V64 or D128/V128");
    static_assert(Base::kBlockPerCu == 1);
    // Not supported: sinks, and dropout, bias and logits soft cap (rejected by the core).
    static_assert(Base::Policy::AsyncCopy);

    using PageTileTraits =
        FmhaBatchPrefillPageTileTraits<Problem::kPageBlockSize, Base::kQKHeaddim>;

    static constexpr index_t kPageBlockSize = Problem::kPageBlockSize;
    static constexpr index_t kVectorSize    = Problem::kVectorSize;
    static constexpr index_t kHeadDim       = Base::kQKHeaddim;
    static constexpr auto kKVMemoryLayout   = Problem::kKVMemoryLayout;
    static constexpr auto kKVLookupTable    = Problem::kKVLookupTable;
    // The per-wave tensor issue budget has one definition: Base::PagedTransfer.
    static constexpr const char* name = "qr_tdm_sched_page_tile";
};

} // namespace ck_tile
