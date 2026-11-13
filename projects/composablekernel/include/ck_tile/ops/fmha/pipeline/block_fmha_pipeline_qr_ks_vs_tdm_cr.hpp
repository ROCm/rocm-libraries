// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
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
    static constexpr bool kVirtualSink   = false;
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

struct FmhaTdmCrSinkTuning : FmhaTdmCrGeneralTuning
{
    static constexpr bool kVirtualSink = true;
};

// Causal virtual-sink rows favor keeping the empty-tile path out of line.
struct FmhaTdmCrCausalSinkTuning : FmhaTdmCrSinkTuning
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

    using QDataType                             = typename Problem::QDataType;
    using KDataType                             = typename Problem::KDataType;
    using VDataType                             = typename Problem::VDataType;
    using PDataType                             = typename Problem::PDataType;
    using ODataType                             = typename Problem::ODataType;
    using SaccDataType                          = typename Problem::SaccDataType;
    using BiasDataType                          = typename Problem::BiasDataType;
    using LSEDataType                           = typename Problem::LSEDataType;
    using RandValOutputDataType                 = typename Problem::RandValOutputDataType;
    using AttentionVariant                      = typename Problem::AttentionVariant;
    using VLayout                               = typename BlockFmhaShape::VLayout;
    using CrBasePipeline                        = BlockFmhaPipelineQRKSVSTdmCr;
    static constexpr bool kUsesCompactArguments = false;
    using Tuning                                = FmhaTdmCrTuning<>;
    struct Policy
    {
        static constexpr bool AsyncCopy = false;
    };
    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kIgnoreFastExp2              = true;
    static constexpr int kN1                           = BlockFmhaShape::kN1;
    static constexpr bool kIsGroupMode                 = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ                  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK                  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ                 = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV                 = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap            = Problem::kHasLogitsSoftCap;
    static constexpr bool kHasDropout                  = Problem::kHasDropout;
    static constexpr bool kHasSink                     = Problem::kHasSink;
    static constexpr auto BiasEnum                     = Problem::BiasEnum;
    static constexpr bool kScoreTransform =
        Problem::kHasLogitsSoftCap || Problem::BiasEnum != BlockAttentionBiasEnum::NO_BIAS;
    static constexpr bool kSequenceFeatures = Problem::kHasSink;
    static constexpr bool kGeneralArguments = kScoreTransform || kSequenceFeatures;
    static constexpr int kHeadDim           = BlockFmhaShape::kQKHeaddim;
    static constexpr int kValueDim          = BlockFmhaShape::kN1;
    static constexpr bool kCompact = std::is_same_v<QDataType, bf16_t> && kHeadDim == 128 &&
                                     kValueDim == 128 && !Problem::kIsGroupMode &&
                                     !Problem::kPadHeadDimQ && !Problem::kPadHeadDimV &&
                                     !kGeneralArguments;
    static_assert(std::is_same_v<QDataType, bf16_t> || std::is_same_v<QDataType, half_t>);
    static_assert(std::is_same_v<typename Problem::KDataType, QDataType> &&
                  std::is_same_v<typename Problem::VDataType, QDataType> &&
                  std::is_same_v<typename Problem::PDataType, QDataType> &&
                  std::is_same_v<typename Problem::ODataType, QDataType>);
    static_assert((kHeadDim == kValueDim && kHeadDim >= 32 && kHeadDim <= 160 &&
                   kHeadDim % 32 == 0) ||
                  (kHeadDim == 192 && kValueDim == 128));
    static_assert((kM0 == kN0 && (kM0 == 64 || kM0 == 128)) || (kM0 == 128 && kN0 == 64) ||
                  (kM0 == 64 && kN0 == 128 && kHeadDim <= 64));
    static_assert(BlockFmhaShape::kK0 == 32 && BlockFmhaShape::kK1 == 32 &&
                  BlockFmhaShape::NumWarps == 4);
    static_assert(std::is_same_v<typename Problem::SaccDataType, float> &&
                  std::is_same_v<typename Problem::OaccDataType, float> &&
                  std::is_same_v<typename Problem::LSEDataType, float>);
    static_assert(Problem::kBlockPerCu == kBlockPerCu);
    static_assert(!Problem::kHasDropout);
    static_assert(!Problem::kSkipMinSeqlenQ || Problem::kIsGroupMode);
    static_assert(Problem::QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE);
    static_assert(!Problem::kHasLogitsSoftCap ||
                  Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS);
    static_assert(Problem::BiasEnum != BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                  std::is_same_v<typename Problem::BiasDataType, QDataType>);
    static_assert(std::is_same_v<typename BlockFmhaShape::VLayout, tensor_layout::gemm::RowMajor>);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 2 * kN0 *
                   (detail::fmha_cr::LdsLayout<kHeadDim, 16>::kPitch +
                    detail::fmha_cr::LdsLayout<kValueDim, 32>::kPitch) *
                   2 +
               (BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS
                    ? kM0 * detail::fmha_cr::LdsLayout<kN0, 16>::kPitch * sizeof(QDataType)
                    : 0);
    }

    template <typename Kargs>
    CK_TILE_DEVICE static auto GetTileIndex(const Kargs& a)
    {
        const int tiles = gridDim.x;
        int linear      = blockIdx.x + tiles * blockIdx.y;
        int bh, tile;
        if constexpr(kCausal)
        {
            const int per_batch = tiles * a.num_head_q;
            const int within    = linear % per_batch;
            bh                  = linear / per_batch * a.num_head_q + within % a.num_head_q;
            tile                = tiles - 1 - within / a.num_head_q;
        }
        else
        {
            const int total = tiles * gridDim.y, xcd = linear % 8;
            linear = xcd * (total / 8) + min(xcd, total % 8) + linear / 8;
            bh     = linear / tiles;
            tile   = linear % tiles;
        }
        constexpr bool grouped_heads =
            std::is_same_v<QDataType, bf16_t> && kCausal && kM0 == 128 &&
            (kPadHeadDimQ || kPadHeadDimV || kHeadDim == 32 || kHeadDim == 96 || kHeadDim == 160);
        if constexpr(grouped_heads)
        {
            if(gridDim.y % 16 == 0)
            {
                const unsigned physical = blockIdx.x + tiles * blockIdx.y;
                const unsigned slot = physical / 8, depth = slot / 2;
                const unsigned chunk = depth / tiles;
                bh                   = (physical % 8) * (gridDim.y / 8) + chunk * 2 + slot % 2;
                tile                 = tiles - 1 - (depth - chunk * tiles);
            }
        }
        bh   = amd_wave_read_first_lane(bh);
        tile = amd_wave_read_first_lane(tile);
        return make_tuple(tile, 0, bh % a.num_head_q, bh / a.num_head_q);
    }

    template <typename NativeTuning = Tuning, typename Kargs>
    CK_TILE_DEVICE static void RunPrepared(const Kargs& a,
                                           int tile,
                                           int head,
                                           int batch,
                                           const QDataType* q,
                                           const KDataType* k,
                                           const VDataType* v,
                                           ODataType* o,
                                           long_index_t bias_offset,
                                           long_index_t lse_offset)
    {
        constexpr bool general = kGeneralArguments || NativeTuning::kVirtualSink ||
                                 std::is_same_v<NativeTuning, FmhaTdmCrGeneralTuning>;
        using Base = std::conditional_t<kPadHeadDimQ || kPadHeadDimV,
                                        detail::fmha_cr::HeadPaddedArguments,
                                        std::conditional_t<general,
                                                           detail::fmha_cr::GeneralArguments,
                                                           detail::fmha_cr::StridedArguments>>;
        using Scores =
            std::conditional_t<kScoreTransform,
                               detail::fmha_cr::ScoreArguments<Base, BiasEnum, kHasLogitsSoftCap>,
                               Base>;
        using Sequence = std::conditional_t<
            kSequenceFeatures || NativeTuning::kVirtualSink,
            detail::fmha_cr::SequenceArguments<Scores, kHasSink, NativeTuning::kVirtualSink>,
            Scores>;
        using Prepared = std::conditional_t<kIsGroupMode && kHeadDim == 64,
                                            detail::fmha_cr::WideStrideArguments<Sequence>,
                                            Sequence>;
        Prepared x{};
        x.q = q;
        x.k = k;
        x.v = v;
        x.o = o;
        // The shared kernel has already selected the batch and Q/KV heads.
        x.heads      = 1;
        x.query_tile = tile;
        x.sq         = a.seqlen_q;
        x.sk         = a.seqlen_k;
        x.scale      = a.scale_s;
        x.stride_q   = a.stride_q;
        x.stride_k   = a.stride_k;
        x.stride_v   = a.stride_v;
        x.stride_o   = a.stride_o;
        if constexpr(kStoreLSE)
            x.lse = static_cast<float*>(a.lse_ptr) + lse_offset +
                    long_index_t{head} * a.nhead_stride_lse;
        if constexpr(kPadHeadDimQ || kPadHeadDimV)
        {
            x.hdim_q = a.hdim_q;
            x.hdim_v = a.hdim_v;
        }
        if constexpr(kCausal)
        {
            x.mask_type = static_cast<int>(a.mask_type);
            x.left      = a.window_size_left;
            x.right     = a.window_size_right;
        }
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            x.bias = static_cast<const BiasDataType*>(a.bias_ptr) + bias_offset +
                     long_index_t{head} * a.nhead_stride_bias;
            x.stride_bias = a.stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            x.bias = static_cast<const float*>(a.alibi_slope_ptr) +
                     long_index_t{batch} * a.alibi_slope_stride + head;
        if constexpr(kHasLogitsSoftCap)
            x.logits_soft_cap = a.logits_soft_cap;
        if constexpr(kHasSink && kCausal)
            x.sink_size = a.sink_size;
        if constexpr(NativeTuning::kVirtualSink)
            x.sink = static_cast<const float*>(a.sink_ptr) + head;
        Run<NativeTuning>(x);
    }

    // The pipeline includes the LDS shuffle and TDM O epilogue. Keeping
    // them in one scheduling region preserves the standalone kernel's allocation.
    template <typename Tuning, typename Args>
    CK_TILE_DEVICE static void Run(Args a)
    {
        if constexpr(!std::is_same_v<Args, Arguments>)
        {
            using Native = std::conditional_t<std::is_same_v<QDataType, bf16_t>, __bf16, _Float16>;
            constexpr bool rectangular = kM0 > kN0;
            constexpr int unroll = rectangular || (kHeadDim == 64 && kM0 == 64 && kCausal) ? 2 : 1;
            constexpr int qk_priority = rectangular && !(kHeadDim == 192 && kPadHeadDimQ) ? 2 : 1;
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
                                 kN0,
                                 false,
                                 std::is_same_v<Tuning, FmhaTdmCrCausalSinkTuning>>(a);
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

// Native output and sink variants share the common kernel ABI and preparation.
template <typename Pipeline, typename Tuning_>
struct FmhaTdmCrPipelineVariant : Pipeline
{
    using Tuning                         = Tuning_;
    static constexpr index_t kBlockPerCu = Pipeline::kHeadDim == 192 && !Pipeline::kCausal &&
                                                   std::is_same_v<Tuning, FmhaTdmCrGeneralTuning>
                                               ? 3
                                               : Pipeline::kBlockPerCu;
    static_assert(std::is_same_v<Tuning, FmhaTdmCrGeneralTuning> ||
                  std::is_same_v<Tuning, FmhaTdmCrSinkTuning> ||
                  std::is_same_v<Tuning, FmhaTdmCrCausalSinkTuning>);
    template <typename... Args>
    CK_TILE_DEVICE static void RunPrepared(Args&&... args)
    {
        Pipeline::template RunPrepared<Tuning>(static_cast<Args&&>(args)...);
    }
};

// Compact BSHD variants retain their small device ABI through the shared kernel.
template <typename Pipeline, typename Tuning_>
struct FmhaTdmCrCompactPipelineVariant : Pipeline
{
    using Tuning                                = Tuning_;
    static constexpr bool kUsesCompactArguments = true;
    static_assert(Pipeline::kCompact);
    static_assert(!std::is_base_of_v<FmhaTdmCrGeneralTuning, Tuning>);
    CK_TILE_DEVICE static void RunCompact(typename Pipeline::Arguments a)
    {
        Pipeline::template Run<Tuning>(a);
    }
};

} // namespace ck_tile
