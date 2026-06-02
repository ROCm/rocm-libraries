// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <ck_tile/dispatcher/fmha_kernel_key.hpp>
#include <functional>
#include <string>
#include <vector>

#include "adapters/sdpa/SdpaCandidateSelector.hpp"
#include "adapters/sdpa/SdpaPerfKnobs.hpp"

namespace {

using ck_dsl_provider::enumerateCandidates;
using ck_dsl_provider::knobsToKernelKey;
using ck_dsl_provider::pipelineForKnobs;
using ck_dsl_provider::problemToFmhaProblem;
using ck_dsl_provider::SdpaPerfKnobs;
using ck_dsl_provider::SdpaSelectionProblem;
using ck_dsl_provider::selectAnalyticFallback;
using ck_dsl_provider::selectArgmax;
using ck_dsl_provider::supportsTiled2d;
using ck_tile::dispatcher::FmhaKernelKey;

/// In-family reference problem (head 64, block 32, GQA 8, bf16). All
/// fields valid for supportsTiled2d.
SdpaSelectionProblem makeReferenceProblem() {
    SdpaSelectionProblem p;
    p.batch = 2;
    p.num_query_heads = 64;
    p.num_kv_heads = 8;
    p.seqlen_q = 1024;
    p.seqlen_k = 1024;
    p.head_size = 64;
    p.block_size = 32;
    p.dtype = "bf16";
    p.use_paged_kv = true;
    p.mask_type = 1;  // top-left causal
    p.bias_type = 0;
    return p;
}

// ---------------------------------------------------------------------------
// Enumerator: every emitted combo passes the supports() mirror.
// ---------------------------------------------------------------------------

TEST(SdpaCandidateSelectorEnumerator, EveryEmittedComboIsSupported) {
    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);

    ASSERT_FALSE(combos.empty());
    for (const SdpaPerfKnobs& k : combos) {
        const auto result = supportsTiled2d(problem, k);
        EXPECT_TRUE(result.supported)
            << "enumerated combo nw=" << k.num_warps << " mw=" << k.block_m_per_warp
            << " t=" << k.tile_size << " rejected: " << result.reason;
    }
}

TEST(SdpaCandidateSelectorEnumerator, EmittedCombosCarryProblemVariantLanes) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.use_sinks = true;
    problem.sliding_window = 128;

    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_FALSE(combos.empty());
    for (const SdpaPerfKnobs& k : combos) {
        EXPECT_TRUE(k.use_sinks);
        EXPECT_EQ(k.sliding_window, 128);
    }
}

TEST(SdpaCandidateSelectorEnumerator, BlockMPerWarp32NeverPairsWithEightWarps) {
    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    for (const SdpaPerfKnobs& k : combos) {
        if (k.block_m_per_warp == 32) {
            EXPECT_NE(k.num_warps, 8);
        }
    }
}

// ---------------------------------------------------------------------------
// supports() mirror: spot-check known-bad shapes are rejected.
// ---------------------------------------------------------------------------

TEST(SdpaCandidateSelectorSupports, TileSizeNotMultipleOfBlockRejected) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.block_size = 32;
    SdpaPerfKnobs knobs;
    knobs.num_warps = 4;
    knobs.block_m_per_warp = 16;
    knobs.tile_size = 48;  // not a multiple of 32
    const auto result = supportsTiled2d(problem, knobs);
    EXPECT_FALSE(result.supported);
}

TEST(SdpaCandidateSelectorSupports, DmaFloorRejectsTinyTileSize) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.block_size = 16;
    problem.head_size = 64;
    SdpaPerfKnobs knobs;
    knobs.num_warps = 8;  // threads = 512 -> need tile*hd >= 4096
    knobs.block_m_per_warp = 16;
    knobs.tile_size = 16;  // 16*64 = 1024 < 4096 -> rejected
    const auto result = supportsTiled2d(problem, knobs);
    EXPECT_FALSE(result.supported);
}

TEST(SdpaCandidateSelectorSupports, GqaNonDivisibleRejected) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    // num_queries_per_kv = 64 / 5 ... use a value that does not divide
    // 16*num_warps. With num_warps=1 (block_m_base=16) a qpk of 3 does
    // not divide 16.
    problem.num_query_heads = 24;
    problem.num_kv_heads = 8;  // qpk = 3
    SdpaPerfKnobs knobs;
    knobs.num_warps = 1;  // base BLOCK_M = 16; 16 % 3 != 0
    knobs.block_m_per_warp = 16;
    knobs.tile_size = 64;
    const auto result = supportsTiled2d(problem, knobs);
    EXPECT_FALSE(result.supported);
}

TEST(SdpaCandidateSelectorSupports, GqaDivisibleAccepted) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.num_query_heads = 32;
    problem.num_kv_heads = 8;  // qpk = 4 divides 16
    SdpaPerfKnobs knobs;
    knobs.num_warps = 1;
    knobs.block_m_per_warp = 16;
    knobs.tile_size = 64;
    const auto result = supportsTiled2d(problem, knobs);
    EXPECT_TRUE(result.supported) << result.reason;
}

TEST(SdpaCandidateSelectorSupports, UnsupportedHeadSizeRejected) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.head_size = 192;  // not in {64,128,256}
    SdpaPerfKnobs knobs;
    knobs.num_warps = 4;
    knobs.block_m_per_warp = 16;
    knobs.tile_size = 64;
    const auto result = supportsTiled2d(problem, knobs);
    EXPECT_FALSE(result.supported);
}

TEST(SdpaCandidateSelectorSupports, Mfma32WithNonMultipleOf32TileRejected) {
    // __post_init__: use_mfma_32x32 requires tile_size_eff % 32 == 0. A
    // block_size=16 problem with tile_size=16 passes supports_tiled_2d proper
    // but the mfma32 atom needs a 32-multiple tile, so the combo is unbuildable.
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.block_size = 16;
    problem.head_size = 64;
    SdpaPerfKnobs knobs;
    knobs.num_warps = 1;
    knobs.block_m_per_warp = 32;
    knobs.tile_size = 16;  // multiple of block_size=16 but not of 32
    knobs.use_mfma_32x32 = true;
    knobs.use_transposed_qk_32x32 = true;
    const auto result = supportsTiled2d(problem, knobs);
    EXPECT_FALSE(result.supported);
}

TEST(SdpaCandidateSelectorEnumerator, BlockSize16EmitsOnlyBuildableMfma32Combos) {
    // Regression guard: for a block_size=16 problem, every emitted mfma32 combo
    // must carry a 32-multiple tile_size (the __post_init__ mfma32 rule).
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.block_size = 16;
    problem.head_size = 64;
    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_FALSE(combos.empty());
    for (const SdpaPerfKnobs& k : combos) {
        if (k.use_mfma_32x32) {
            EXPECT_EQ(k.tile_size % 32, 0)
                << "emitted mfma32 combo with non-32-multiple tile_size=" << k.tile_size;
        }
    }
}

// ---------------------------------------------------------------------------
// Selection: argmax over a stubbed score returns the peak combo;
// determinism and stable tie-break.
// ---------------------------------------------------------------------------

TEST(SdpaCandidateSelectorSelection, ArgmaxReturnsScorePeak) {
    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_GE(combos.size(), 2u);

    // Pick a target combo from the middle of the set and make the stub
    // score peak exactly on its kernel-key tile_shape.
    const SdpaPerfKnobs target = combos[combos.size() / 2];
    const FmhaKernelKey targetKey = knobsToKernelKey(problem, target);

    auto stubScore = [&](const FmhaKernelKey& key) -> double {
        const bool match = key.algorithm.tile_shape.m0 == targetKey.algorithm.tile_shape.m0 &&
                           key.algorithm.tile_shape.n0 == targetKey.algorithm.tile_shape.n0 &&
                           key.algorithm.tile_shape.k0 == targetKey.algorithm.tile_shape.k0 &&
                           key.algorithm.pipeline == targetKey.algorithm.pipeline;
        return match ? 100.0 : 1.0;
    };

    const SdpaPerfKnobs picked = selectArgmax(problem, combos, stubScore);
    EXPECT_EQ(picked.block_m(), target.block_m());
    EXPECT_EQ(picked.tile_size, target.tile_size);
    EXPECT_EQ(picked.use_mfma_32x32, target.use_mfma_32x32);
}

TEST(SdpaCandidateSelectorSelection, ConstantScoreYieldsStableFirstCombo) {
    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_FALSE(combos.empty());

    auto constScore = [](const FmhaKernelKey&) -> double { return 7.0; };

    const SdpaPerfKnobs first = selectArgmax(problem, combos, constScore);
    const SdpaPerfKnobs second = selectArgmax(problem, combos, constScore);
    // Stable tie-break: the first enumerated combo wins, repeatably.
    EXPECT_EQ(first.num_warps, combos.front().num_warps);
    EXPECT_EQ(first.block_m_per_warp, combos.front().block_m_per_warp);
    EXPECT_EQ(first.tile_size, combos.front().tile_size);
    EXPECT_EQ(first.num_warps, second.num_warps);
    EXPECT_EQ(first.tile_size, second.tile_size);
}

// ---------------------------------------------------------------------------
// Mapping: knobs -> FmhaKernelKey scored + non-scored fields.
// ---------------------------------------------------------------------------

TEST(SdpaCandidateSelectorMapping, ScoredTileShapeMatchesKnobs) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    SdpaPerfKnobs knobs;
    knobs.num_warps = 4;
    knobs.block_m_per_warp = 16;
    knobs.tile_size = 64;

    const FmhaKernelKey key = knobsToKernelKey(problem, knobs);
    // m0 = launch BLOCK_M = num_warps * block_m_per_warp = 64.
    EXPECT_EQ(key.algorithm.tile_shape.m0, 64);
    EXPECT_EQ(key.algorithm.tile_shape.n0, 64);
    // Default 16x16x32 atom -> k0/k1 = 32.
    EXPECT_EQ(key.algorithm.tile_shape.k0, 32);
    EXPECT_EQ(key.algorithm.tile_shape.k1, 32);
}

TEST(SdpaCandidateSelectorMapping, Mfma32AtomChangesK0K1) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    SdpaPerfKnobs knobs;
    knobs.num_warps = 4;
    knobs.block_m_per_warp = 32;
    knobs.tile_size = 64;
    knobs.use_mfma_32x32 = true;
    knobs.use_transposed_qk_32x32 = true;

    const FmhaKernelKey key = knobsToKernelKey(problem, knobs);
    EXPECT_EQ(key.algorithm.tile_shape.m0, 128);  // 4 * 32
    EXPECT_EQ(key.algorithm.tile_shape.k0, 16);
    EXPECT_EQ(key.algorithm.tile_shape.k1, 16);
}

TEST(SdpaCandidateSelectorMapping, PipelineFollowsScheduleFlags) {
    SdpaSelectionProblem dense = makeReferenceProblem();
    dense.use_paged_kv = false;

    SdpaPerfKnobs plain;
    plain.num_warps = 4;
    plain.tile_size = 64;
    EXPECT_EQ(pipelineForKnobs(dense, plain), "qr_async");

    SdpaPerfKnobs earlyv = plain;
    earlyv.use_early_v_schedule = true;
    EXPECT_EQ(pipelineForKnobs(dense, earlyv), "qr_async_trload");

    SdpaPerfKnobs mfma32 = plain;
    mfma32.block_m_per_warp = 32;
    mfma32.use_mfma_32x32 = true;
    mfma32.use_transposed_qk_32x32 = true;
    EXPECT_EQ(pipelineForKnobs(dense, mfma32), "qr_async_trload_v3");

    // Paged KV dominates the pipeline choice.
    SdpaSelectionProblem paged = makeReferenceProblem();
    paged.use_paged_kv = true;
    EXPECT_EQ(pipelineForKnobs(paged, plain), "qr_pagedkv");
}

TEST(SdpaCandidateSelectorMapping, SignatureMatchesProblem) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.use_sinks = true;
    problem.use_paged_kv = true;
    problem.mask_type = 1;
    problem.bias_type = 0;
    problem.skip_min_seqlen_q = true;

    SdpaPerfKnobs knobs;
    knobs.num_warps = 4;
    knobs.tile_size = 64;

    const FmhaKernelKey key = knobsToKernelKey(problem, knobs);
    EXPECT_EQ(key.signature.data_type, "bf16");
    EXPECT_EQ(key.signature.mask_type, 1);
    EXPECT_EQ(key.signature.bias_type, 0);
    EXPECT_TRUE(key.signature.has_sink);
    EXPECT_TRUE(key.signature.use_paged_kv);
    EXPECT_TRUE(key.signature.skip_min_seqlen_q);
    EXPECT_FALSE(key.signature.has_lse);  // paged kernel has no LSE
    EXPECT_EQ(key.signature.hdim_q, 64);
    EXPECT_EQ(key.signature.hdim_v, 64);
}

TEST(SdpaCandidateSelectorMapping, NonScoredFieldsAtDocumentedDefaults) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    SdpaPerfKnobs knobs;
    knobs.num_warps = 2;
    knobs.block_m_per_warp = 16;
    knobs.tile_size = 64;

    const FmhaKernelKey key = knobsToKernelKey(problem, knobs);
    // pad_* default-filled true.
    EXPECT_TRUE(key.algorithm.pad_s);
    EXPECT_TRUE(key.algorithm.pad_sk);
    EXPECT_TRUE(key.algorithm.pad_d);
    EXPECT_TRUE(key.algorithm.pad_dv);
    // wave_shape / warp_tile_shape at struct defaults (not scored).
    EXPECT_EQ(key.algorithm.wave_shape.m0, 1);
    EXPECT_EQ(key.algorithm.warp_tile_shape.m0, 0);
    EXPECT_EQ(key.algorithm.selection_rank, 0);
    // block_per_cu mapped from num_warps for identifier distinctness.
    EXPECT_EQ(key.algorithm.block_per_cu, 2);
    EXPECT_EQ(key.gfx_arch, "gfx950");
}

TEST(SdpaCandidateSelectorMapping, IsDeterministic) {
    const SdpaSelectionProblem problem = makeReferenceProblem();
    SdpaPerfKnobs knobs;
    knobs.num_warps = 4;
    knobs.tile_size = 64;

    const FmhaKernelKey a = knobsToKernelKey(problem, knobs);
    const FmhaKernelKey b = knobsToKernelKey(problem, knobs);
    EXPECT_EQ(a, b);
}

TEST(SdpaCandidateSelectorMapping, FmhaProblemFromSpecIsConsistent) {
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.use_sinks = true;
    problem.use_paged_kv = true;

    const auto fp = problemToFmhaProblem(problem);
    EXPECT_EQ(fp.data_type, "bf16");
    EXPECT_EQ(fp.batch, 2);
    EXPECT_EQ(fp.nhead_q, 64);
    EXPECT_EQ(fp.nhead_k, 8);
    EXPECT_EQ(fp.hdim_q, 64);
    EXPECT_EQ(fp.hdim_v, 64);
    EXPECT_EQ(fp.mask_type, 1);
    EXPECT_TRUE(fp.has_sink);
    EXPECT_TRUE(fp.use_paged_kv);
    EXPECT_EQ(fp.gfx_arch, "gfx950");
}

// ---------------------------------------------------------------------------
// Analytic fallback: returns the production-policy-aligned combo.
// ---------------------------------------------------------------------------

TEST(SdpaCandidateSelectorFallback, LongPrefillPicksPolicyCombo) {
    // head 64, block 32, GQA 8, bf16, seqlen_q=1024 (> 256). Policy:
    // tile_size = 2*block = 64; num_warps thresholds give 2 for >256;
    // block_m_per_warp = 16; plain pipeline.
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.seqlen_q = 1024;
    problem.sliding_window = 0;

    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_FALSE(combos.empty());
    const SdpaPerfKnobs pick = selectAnalyticFallback(problem, combos);

    EXPECT_EQ(pick.num_warps, 2);
    EXPECT_EQ(pick.block_m_per_warp, 16);
    EXPECT_EQ(pick.tile_size, 64);  // 2 * block_size
    EXPECT_FALSE(pick.use_mfma_32x32);
    EXPECT_FALSE(pick.use_early_v_schedule);
}

TEST(SdpaCandidateSelectorFallback, MediumPrefillPicksFourWarps) {
    // seqlen_q in (128, 256] -> num_warps target 4.
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.seqlen_q = 256;
    problem.sliding_window = 0;

    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_FALSE(combos.empty());
    const SdpaPerfKnobs pick = selectAnalyticFallback(problem, combos);

    EXPECT_EQ(pick.num_warps, 4);
    EXPECT_EQ(pick.block_m_per_warp, 16);
    EXPECT_EQ(pick.tile_size, 64);
    EXPECT_FALSE(pick.use_mfma_32x32);
}

TEST(SdpaCandidateSelectorFallback, SlidingWindowShrinksTileSize) {
    // sliding_window > 0 -> tile_size policy = block_size (= 32 here).
    SdpaSelectionProblem problem = makeReferenceProblem();
    problem.seqlen_q = 1024;
    problem.sliding_window = 128;

    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_FALSE(combos.empty());
    const SdpaPerfKnobs pick = selectAnalyticFallback(problem, combos);

    EXPECT_EQ(pick.tile_size, 32);  // block_size
}

TEST(SdpaCandidateSelectorFallback, IsDeterministic) {
    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> combos = enumerateCandidates(problem);
    ASSERT_FALSE(combos.empty());
    const SdpaPerfKnobs a = selectAnalyticFallback(problem, combos);
    const SdpaPerfKnobs b = selectAnalyticFallback(problem, combos);
    EXPECT_EQ(a.num_warps, b.num_warps);
    EXPECT_EQ(a.block_m_per_warp, b.block_m_per_warp);
    EXPECT_EQ(a.tile_size, b.tile_size);
    EXPECT_EQ(a.use_mfma_32x32, b.use_mfma_32x32);
}

}  // namespace
