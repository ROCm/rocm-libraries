// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"

namespace ck_tile {

// Where a blockscale K descale reaches the scores.
enum class FmhaTdmSchedFp8KDescale
{
    None,       // not blockscale
    PerStage,   // K64 QK steps: each QK stage's scores times its K descale
    WithScaleS, // K128 with bias: each QK stage's scores times its K descale and scale_s
    InSoftmax,  // K128 without bias: a per-stage column scale of the softmax
};

// Switches the FP8 sched pipeline derives from its problem and policy.
template <typename Problem, typename Policy, bool EnableVirtualSink>
struct FmhaTdmSchedFp8Features
{
    using Geometry               = typename Policy::Geometry;
    using FmhaMask               = remove_cvref_t<typename Problem::FmhaMask>;
    static constexpr auto QScale = Problem::QScaleEnum;
    static constexpr auto Bias   = Problem::BiasEnum;

    static constexpr bool kQuantized  = QScale != BlockAttentionQuantScaleEnum::NO_SCALE;
    static constexpr bool kBlockScale = QScale == BlockAttentionQuantScaleEnum::BLOCKSCALE;
    static constexpr bool kMx         = QScale == BlockAttentionQuantScaleEnum::MX;
    static constexpr bool kD256       = Geometry::kHeadDimQK == 256;

    // Numerics: these change the computed values.
    // Merged PERHEAD/BLOCKSCALE rows fold their own head's q descale into the softmax scale.
    static constexpr bool kRowScale =
        Problem::kMergeNumHeadGroupsSeqLenQ &&
        (QScale == BlockAttentionQuantScaleEnum::PERHEAD || kBlockScale);
    // Quantized modes cast P to fp8 against this fixed scale and undo it in the epilogue.
    static constexpr float kPScale = 256.0f;
    // Normalize O by the row sum of the fp8 P gemm_1 consumed, so a common rounding gain cancels.
    static constexpr bool kNormalizeQuantizedP = kQuantized && !kMx;
    // P is rounded stochastically where O is normalized by the sum of the same quantized P.
    static constexpr bool kStochasticP = kNormalizeQuantizedP;
    // Without LSE or sink nothing needs the unquantized row sum, so l sums q(P) directly.
    static constexpr bool kQuantizedRowSum =
        kNormalizeQuantizedP && !Problem::kStoreLSE && !Problem::kHasSink && !EnableVirtualSink;
    // Where the blockscale K descale is applied.
    static constexpr FmhaTdmSchedFp8KDescale kKDescale =
        !kBlockScale                              ? FmhaTdmSchedFp8KDescale::None
        : Geometry::kQkWmmaK != 128               ? FmhaTdmSchedFp8KDescale::PerStage
        : Bias == BlockAttentionBiasEnum::NO_BIAS ? FmhaTdmSchedFp8KDescale::InSoftmax
                                                  : FmhaTdmSchedFp8KDescale::WithScaleS;

    // Loop shape: control flow and LDS layout.
    // Dense loops count down and skip the per-tile mask and sink bookkeeping.
    static constexpr bool kUseCountdownLoop = Policy::template UseCountdownLoop<Problem>();
    // Segmented LDS layout: odd wavefronts read the K/V halves placed in another LDS segment.
    static constexpr bool kLdsSegSplit = Policy::kLdsSegLayout;
    // Band masks need lane-linear rows; masked merged tiles are not, so they check per element.
    static constexpr bool kBandMask =
        std::is_same_v<FmhaMask, SimplifiedGenericAttentionMask<FmhaMask::IsMasking>> &&
        !kUseCountdownLoop && !(Problem::kMergeNumHeadGroupsSeqLenQ && FmhaMask::IsMasking);

    // The elementwise bias tile reaches LDS by TDM and is read back right before its add.
    static constexpr bool kBiasViaTdm = Bias == BlockAttentionBiasEnum::ELEMENTWISE_BIAS;

    // Two bias workgroups per CU cannot afford two loop copies; every tile carries the escape.
    static constexpr bool kSinkEscapeEveryTile = Problem::kHasSink && FmhaMask::IsMasking &&
                                                 kBandMask && kBiasViaTdm &&
                                                 Policy::kBlockPerCu > 1;
    // Otherwise sink tiles with a mask run apart, so only they carry the per-lane sink escape.
    static constexpr bool kSinkEscapeLoop =
        Problem::kHasSink && FmhaMask::IsMasking && kBandMask && !kSinkEscapeEveryTile;

    // ISA tuning only: same values, chosen for register use or the instruction schedule.
    // Merged D192 at two workgroups per CU cannot afford IsOutOfBoundUnsigned's per-row VGPRs.
    static constexpr bool kUnpinnedMaskWidth = Problem::kMergeNumHeadGroupsSeqLenQ &&
                                               Geometry::kHeadDimQK == 192 &&
                                               Policy::kBlockPerCu > 1;
    // Only D256 MX folds each QK stage into a lane row max as it lands; others use one tree.
    static constexpr bool kProgressiveMax = kD256 && kUseCountdownLoop && kMx;
    // Blockscale keeps the per-stage K read order; other modes issue the next K group early.
    static constexpr bool kQkLead = !kBlockScale;
    // Every-tile sink escape: one scalar P seed per wavefront (blockscale has no SGPR to spare).
    static constexpr bool kScalarPSeed = kSinkEscapeEveryTile && !kBlockScale;
    // One predicate pass keeps D64 sink edge tiles from holding tail and pad indices.
    static constexpr bool kFusedEdgePredicate = Geometry::kHeadDimQK == 64 &&
                                                !FmhaMask::IsMasking &&
                                                Bias == BlockAttentionBiasEnum::NO_BIAS;
};

} // namespace ck_tile
