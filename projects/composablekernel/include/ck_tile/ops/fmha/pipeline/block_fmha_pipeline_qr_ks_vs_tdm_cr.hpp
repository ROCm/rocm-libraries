// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_cr_compute.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"

namespace ck_tile {

// CR means conditional rescaling. A partial-sum guard avoids maximum reductions
// and output rescaling on the common softmax path.
// The measured B32/H8 square specializations retain runtime softmax scale.
// Generic short/medium/long variants use the same schedule without fixed strides.
template <int Sequence = 0, bool Short = false, bool FirstScore = false, int GridGroup = -1>
struct FmhaTdmCrTuning
{
    static constexpr int kSequence       = Sequence;
    static constexpr bool kShort         = Sequence ? Sequence <= 2048 : Short;
    static constexpr bool kFirstScore    = Sequence ? Sequence <= 2048 : FirstScore;
    static constexpr int kQueryTiles     = 0;
    static constexpr int kKeyTiles       = 0;
    static constexpr int kGridGroup      = GridGroup;
    static constexpr bool kTailFragments = false;
};

struct FmhaTdmCrGeneralTuning : FmhaTdmCrTuning<>
{
};

// Reuse dense schedules across batch sizes. With KeyTiles == 0, only the
// query grid and head count are fixed; padded sequence lengths stay dynamic.
// The launcher checks the geometry; tensor pointers and scale stay dynamic.
template <int QueryTiles, int KeyTiles = QueryTiles, bool TailFragments = false>
struct FmhaTdmCrTileTuning : FmhaTdmCrTuning<0, QueryTiles <= 16, QueryTiles <= 16>
{
    static constexpr int kQueryTiles     = QueryTiles;
    static constexpr int kKeyTiles       = KeyTiles;
    static constexpr bool kTailFragments = TailFragments;
};

template <typename Problem_>
struct BlockFmhaPipelineQRKSVSTdmCr
{
    using Problem                        = remove_cvref_t<Problem_>;
    using BlockFmhaShape                 = typename Problem::BlockFmhaShape;
    using FmhaMask                       = typename Problem::FmhaMask;
    using Arguments                      = detail::fmha_cr::Arguments;
    static constexpr const char* name    = "qr_tdm_cr";
    static constexpr index_t kM0         = BlockFmhaShape::kM0;
    static constexpr index_t kN0         = BlockFmhaShape::kN0;
    static constexpr index_t kBlockSize  = 128;
    static constexpr index_t kBlockPerCu = 1;
    static constexpr bool kStoreLSE      = Problem::kStoreLSE;
    static constexpr bool kCausal        = FmhaMask::IsMasking;

    using QDataType                = typename Problem::QDataType;
    static constexpr int kHeadDim  = BlockFmhaShape::kQKHeaddim;
    static constexpr int kValueDim = BlockFmhaShape::kN1;
    static constexpr bool kCompact = std::is_same_v<QDataType, bf16_t> && kHeadDim == 128 &&
                                     kValueDim == 128 && !Problem::kIsGroupMode;
    static_assert(std::is_same_v<QDataType, bf16_t> || std::is_same_v<QDataType, half_t>);
    static_assert(std::is_same_v<typename Problem::KDataType, QDataType> &&
                  std::is_same_v<typename Problem::VDataType, QDataType> &&
                  std::is_same_v<typename Problem::PDataType, QDataType> &&
                  std::is_same_v<typename Problem::ODataType, QDataType>);
    static_assert((kHeadDim == 64 && kValueDim == 64) || (kHeadDim == 128 && kValueDim == 128) ||
                  (kHeadDim == 192 && kValueDim == 128));
    static_assert((kM0 == kN0 && (kM0 == 64 || kM0 == 128)) ||
                  (kHeadDim == 192 && kM0 == 128 && kN0 == 64));
    static_assert(BlockFmhaShape::kK0 == 32 && BlockFmhaShape::kK1 == 32 &&
                  BlockFmhaShape::NumWarps == 4);
    static_assert(std::is_same_v<typename Problem::SaccDataType, float> &&
                  std::is_same_v<typename Problem::OaccDataType, float> &&
                  std::is_same_v<typename Problem::LSEDataType, float>);
    static_assert(Problem::kBlockPerCu == kBlockPerCu);
    static_assert(!Problem::kPadHeadDimQ && !Problem::kPadHeadDimV);
    static_assert(!Problem::kHasDropout && !Problem::kHasLogitsSoftCap && !Problem::kHasSink);
    static_assert(!Problem::kSkipMinSeqlenQ);
    static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                  Problem::QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE);
    static_assert(std::is_same_v<typename BlockFmhaShape::VLayout, tensor_layout::gemm::RowMajor>);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 2 * kN0 * (kHeadDim + (kHeadDim == 192 ? 24 : 8) + kValueDim + 16) * 2;
    }

    // The pipeline includes the LDS shuffle and TDM O epilogue. Keeping
    // them in one scheduling region preserves the standalone kernel's allocation.
    template <typename Tuning, typename Args>
    CK_TILE_DEVICE static void Run(Args a)
    {
        if constexpr(!std::is_same_v<Args, Arguments>)
        {
            a.group      = Problem::kIsGroupMode;
            using Native = std::conditional_t<std::is_same_v<QDataType, bf16_t>, __bf16, _Float16>;
            constexpr bool rectangular = kM0 > kN0;
            constexpr int unroll = rectangular || (kHeadDim == 64 && kM0 == 64 && kCausal) ? 2 : 1;
            constexpr int qk_priority = rectangular ? 2 : 1;
            detail::fmha_cr::run<kCausal,
                                 true,
                                 false,
                                 kStoreLSE,
                                 false,
                                 0,
                                 0,
                                 0,
                                 unroll,
                                 false,
                                 qk_priority,
                                 1,
                                 true,
                                 Native,
                                 kHeadDim,
                                 kValueDim,
                                 kM0,
                                 Args,
                                 0,
                                 0,
                                 kN0>(a);
        }
        else
        {
            constexpr int S              = Tuning::kSequence;
            constexpr int query_tiles    = Tuning::kQueryTiles;
            constexpr bool aligned_tiles = query_tiles && Tuning::kKeyTiles;
            constexpr bool causal_tiles  = aligned_tiles && kCausal;
            constexpr int group          = Tuning::kGridGroup >= 0 ? Tuning::kGridGroup
                                           : causal_tiles          ? 32
                                           : aligned_tiles && query_tiles == 4 && !kStoreLSE ? 32
                                           : S == 32768 && kCausal                           ? 1
                                           : Tuning::kShort
                                               ? (kM0 == 128 && Problem::kPadSeqLenK && !kCausal ? 32 : 16)
                                               : 0;
            constexpr int padding        = S == 32768 ? (kCausal ? 16 : 12)
                                           : S        ? (kCausal || S <= 2048 ? 0 : 4)
                                                      : (Tuning::kShort ? 0 : 4);
            constexpr int unroll =
                causal_tiles                                                                 ? 1
                : aligned_tiles && query_tiles == 16 && Tuning::kKeyTiles == 8 && !kStoreLSE ? 8
                : aligned_tiles || Tuning::kTailFragments                                    ? 4
                : S == 32768 && !kCausal                                                     ? 8
                : S ? (kCausal ? (S == 512 ? 1 : (S == 32768 ? 4 : 2)) : 4)
                    : 1;
            constexpr int qk_priority      = S && kCausal && S == 512 ? 2 : 1;
            constexpr int pv_priority      = aligned_tiles
                                                 ? (query_tiles <= 5 || query_tiles == 16 ? 1 : 0)
                                             : !S || S == 512 || (kCausal && S <= 4096) ? 1
                                                                                        : 0;
            constexpr bool set_pv_priority = aligned_tiles || !S || kCausal || S != 2048;
            constexpr bool peel_causal     = kCausal && !Problem::kPadSeqLenK && kM0 == kN0 &&
                                         ((S && S != 512) || (!S && Tuning::kFirstScore));
            detail::fmha_cr::run<kCausal,
                                 Problem::kPadSeqLenK,
                                 !Problem::kPadSeqLenQ,
                                 kStoreLSE,
                                 Tuning::kFirstScore || (!kCausal && S >= 16384),
                                 group,
                                 padding,
                                 S,
                                 unroll,
                                 peel_causal,
                                 qk_priority,
                                 pv_priority,
                                 set_pv_priority,
                                 __bf16,
                                 128,
                                 128,
                                 kM0,
                                 Args,
                                 query_tiles,
                                 Tuning::kKeyTiles,
                                 kN0,
                                 Tuning::kTailFragments>(a);
        }
    }
};

} // namespace ck_tile
