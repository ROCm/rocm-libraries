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
template <int Sequence = 0, bool Short = false, bool FirstScore = false>
struct FmhaTdmCrTuning
{
    static constexpr int kSequence    = Sequence;
    static constexpr bool kShort      = Sequence ? Sequence <= 2048 : Short;
    static constexpr bool kFirstScore = Sequence ? Sequence <= 2048 : FirstScore;
};

template <typename Problem_>
struct BlockFmhaPipelineQRKSVSTdmCr
{
    using Problem                        = remove_cvref_t<Problem_>;
    using BlockFmhaShape                 = typename Problem::BlockFmhaShape;
    using FmhaMask                       = typename Problem::FmhaMask;
    using Arguments                      = detail::fmha_cr::Arguments;
    static constexpr const char* name    = "qr_tdm_cr";
    static constexpr index_t kM0         = 128;
    static constexpr index_t kN0         = 128;
    static constexpr index_t kBlockSize  = 128;
    static constexpr index_t kBlockPerCu = 1;
    static constexpr bool kStoreLSE      = Problem::kStoreLSE;
    static constexpr bool kCausal        = FmhaMask::IsMasking;

    static_assert(std::is_same_v<typename Problem::QDataType, bf16_t> &&
                  std::is_same_v<typename Problem::KDataType, bf16_t> &&
                  std::is_same_v<typename Problem::VDataType, bf16_t> &&
                  std::is_same_v<typename Problem::PDataType, bf16_t> &&
                  std::is_same_v<typename Problem::ODataType, bf16_t>);
    static_assert(BlockFmhaShape::kM0 == 128 && BlockFmhaShape::kN0 == 128 &&
                  BlockFmhaShape::kQKHeaddim == 128 && BlockFmhaShape::kN1 == 128);
    static_assert(BlockFmhaShape::kK0 == 32 && BlockFmhaShape::kK1 == 32 &&
                  BlockFmhaShape::NumWarps == 4);
    static_assert(std::is_same_v<typename BlockFmhaShape::Gemm0BlockWarps, sequence<4, 1, 1>> &&
                  std::is_same_v<typename BlockFmhaShape::Gemm1BlockWarps, sequence<4, 1, 1>> &&
                  std::is_same_v<typename BlockFmhaShape::Gemm0WarpTile, sequence<16, 16, 32>> &&
                  std::is_same_v<typename BlockFmhaShape::Gemm1WarpTile, sequence<16, 16, 32>>);
    static_assert(std::is_same_v<typename Problem::SaccDataType, float> &&
                  std::is_same_v<typename Problem::OaccDataType, float> &&
                  std::is_same_v<typename Problem::LSEDataType, float>);
    static_assert(!Problem::kIsGroupMode && !Problem::kPadHeadDimQ && !Problem::kPadHeadDimV);
    static_assert(!Problem::kHasDropout && !Problem::kHasLogitsSoftCap && !Problem::kHasSink);
    static_assert(!Problem::kSkipMinSeqlenQ);
    static_assert(Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS &&
                  Problem::QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE);
    static_assert(std::is_same_v<typename BlockFmhaShape::VLayout, tensor_layout::gemm::RowMajor>);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 2 * 128 * (136 + 144) * 2; }

    // The pipeline includes the LDS shuffle and asynchronous O epilogue. Keeping
    // them in one scheduling region preserves the standalone kernel's allocation.
    template <typename Tuning>
    CK_TILE_DEVICE static void Run(Arguments a)
    {
        constexpr int S           = Tuning::kSequence;
        constexpr int group       = Tuning::kShort ? 16 : 0;
        constexpr int padding     = S ? (kCausal || S <= 2048 ? 0 : 4) : (Tuning::kShort ? 0 : 4);
        constexpr int unroll      = S ? (kCausal ? (S == 512 ? 1 : (S == 32768 ? 4 : 2)) : 4) : 1;
        constexpr int qk_priority = S && kCausal && S == 512 ? 2 : 1;
        constexpr int pv_priority = !S || S == 512 || (kCausal && S <= 4096) ? 1 : 0;
        constexpr bool set_pv_priority = !S || kCausal || S != 2048;
        detail::fmha_cr::run<kCausal,
                             Problem::kPadSeqLenK,
                             !Problem::kPadSeqLenQ,
                             kStoreLSE,
                             Tuning::kFirstScore,
                             group,
                             padding,
                             S,
                             unroll,
                             S && kCausal && S != 512,
                             qk_priority,
                             pv_priority,
                             set_pv_priority>(a);
    }
};

} // namespace ck_tile
