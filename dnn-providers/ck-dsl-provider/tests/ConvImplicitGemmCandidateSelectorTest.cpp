// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// CPU-only unit tests for the implicit-GEMM conv candidate selector.
// Mirrors the structure of SdpaCandidateSelectorTest.cpp: enumeration,
// validity-gate spot checks, analytic-fallback policy, argmax tie-break.
// No GPU required -- the selector is pure CPU logic.

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "adapters/conv_implicit_gemm/ConvImplicitGemmCandidateSelector.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmPerfKnobs.hpp"

namespace {

using ck_dsl_provider::buildSelectionProblem;
using ck_dsl_provider::ConvImplicitGemmPerfKnobs;
using ck_dsl_provider::ConvImplicitGemmSpec;
using ck_dsl_provider::ConvSelectionProblem;
using ck_dsl_provider::enumerateCandidates;
using ck_dsl_provider::selectAnalyticFallback;
using ck_dsl_provider::selectArgmax;
using ck_dsl_provider::supportsImplicitGemm;

/// In-family reference problem: N=8, C=128, K=128, 56x56 input, 3x3 filter,
/// stride 1, pad 1, bf16. The 64x64x64 tile + mem pipeline lands well in
/// the trained envelope for this shape.
ConvSelectionProblem makeReferenceProblem() {
    ConvSelectionProblem p;
    p.N = 8;
    p.C = 128;
    p.K = 128;
    p.G = 1;
    p.Hi = 56;
    p.Wi = 56;
    p.R = 3;
    p.S = 3;
    p.sH = 1;
    p.sW = 1;
    p.pH = 1;
    p.pW = 1;
    p.dH = 1;
    p.dW = 1;
    p.dtype = "bf16";
    return p;
}

/// Structural equality over the knob fields the selection varies.
bool sameKnobs(const ConvImplicitGemmPerfKnobs& a, const ConvImplicitGemmPerfKnobs& b) {
    return a.tile_m == b.tile_m && a.tile_n == b.tile_n && a.tile_k == b.tile_k &&
           a.warp_m == b.warp_m && a.warp_n == b.warp_n && a.warp_tile_m == b.warp_tile_m &&
           a.warp_tile_n == b.warp_tile_n && a.warp_tile_k == b.warp_tile_k &&
           a.pipeline == b.pipeline && a.wave_size == b.wave_size;
}

}  // namespace

// ---------------------------------------------------------------------------
// Enumerator: every emitted combo is supported; expected cardinality.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmEnumerator, EveryEmittedComboIsSupported) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(combos.empty());
    for (const auto& k : combos) {
        const auto result = supportsImplicitGemm(problem, k, "gfx950");
        EXPECT_TRUE(result.supported)
            << "enumerated combo tile=(" << k.tile_m << "," << k.tile_n << "," << k.tile_k
            << ") pipeline=" << k.pipeline << " rejected: " << result.reason;
    }
}

TEST(ConvImplicitGemmEnumerator, EmitsAllForwardPipelinesForCompatibleTiles) {
    // 10 tiles * 8 forward pipelines = 80 raw candidates. compv4 is the
    // only pipeline with a tile restriction: it excludes the (128,128,64)
    // [32,32,16-atom] tile and the (128,64,128) [wave 8x2x1] tile. So we
    // expect 80 - 2 = 78 buildable candidates.
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    EXPECT_EQ(combos.size(), 78u);
}

TEST(ConvImplicitGemmEnumerator, Compv4NeverPairsWithIncompatibleTiles) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    for (const auto& k : combos) {
        if (k.pipeline == "compv4") {
            // COMPV4_COMPATIBLE_TILES: warp_tile [16,16,*] only -- the
            // 32x32x16-atom tiles and the 128-wide-M tile must NOT appear.
            const bool incompat =
                (k.tile_m == 128 && k.tile_n == 128 && k.tile_k == 64) ||
                (k.tile_m == 32 && k.tile_n == 128 && k.tile_k == 64) ||
                (k.tile_m == 64 && k.tile_n == 128 && k.tile_k == 64) ||
                (k.tile_m == 128 && k.tile_n == 64 && k.tile_k == 128);
            EXPECT_FALSE(incompat) << "compv4 emitted with incompatible tile (" << k.tile_m << ","
                                   << k.tile_n << "," << k.tile_k << ")";
        }
    }
}

TEST(ConvImplicitGemmEnumerator, EnumerationOrderIsDeterministic) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto a = enumerateCandidates(problem, "gfx950");
    const auto b = enumerateCandidates(problem, "gfx950");
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_TRUE(sameKnobs(a[i], b[i])) << "enumeration drift at index " << i;
    }
}

TEST(ConvImplicitGemmEnumerator, TileTripleMatchesTableEntry) {
    // Wave grid + MFMA atom must be the TILE_TO_WAVE/WARP entry for the
    // tile -- the enumerator pins these from the table, the validity gate
    // re-checks them. Cross-validate against the table by sampling a
    // 32x32x16-atom tile (64,128,64) -> (warp_m=2, warp_n=2,
    // warp_tile=32,32,16) and a 16x16x32-atom tile (64,64,128) ->
    // (warp_m=4, warp_n=1, warp_tile=16,16,32).
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    bool sawA = false, sawB = false;
    for (const auto& k : combos) {
        if (k.tile_m == 64 && k.tile_n == 128 && k.tile_k == 64) {
            EXPECT_EQ(k.warp_m, 2);
            EXPECT_EQ(k.warp_n, 2);
            EXPECT_EQ(k.warp_tile_m, 32);
            EXPECT_EQ(k.warp_tile_n, 32);
            EXPECT_EQ(k.warp_tile_k, 16);
            sawA = true;
        }
        if (k.tile_m == 64 && k.tile_n == 64 && k.tile_k == 128) {
            EXPECT_EQ(k.warp_m, 4);
            EXPECT_EQ(k.warp_n, 1);
            EXPECT_EQ(k.warp_tile_m, 16);
            EXPECT_EQ(k.warp_tile_n, 16);
            EXPECT_EQ(k.warp_tile_k, 32);
            sawB = true;
        }
    }
    EXPECT_TRUE(sawA);
    EXPECT_TRUE(sawB);
}

// ---------------------------------------------------------------------------
// supports() gate: spot-check rejection paths.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmSupports, OffTableTileRejected) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    ConvImplicitGemmPerfKnobs k;
    k.tile_m = 256;  // not in TILE_TO_WAVE
    k.tile_n = 256;
    k.tile_k = 256;
    k.warp_m = 4;
    k.warp_n = 4;
    k.warp_tile_m = 32;
    k.warp_tile_n = 32;
    k.warp_tile_k = 16;
    k.pipeline = "mem";
    const auto r = supportsImplicitGemm(problem, k, "gfx950");
    EXPECT_FALSE(r.supported);
}

TEST(ConvImplicitGemmSupports, MismatchedWaveGridRejected) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    ConvImplicitGemmPerfKnobs k;
    k.tile_m = 64;
    k.tile_n = 64;
    k.tile_k = 64;
    k.warp_m = 1;  // table says (4,1) for (64,64,64); wrong on purpose
    k.warp_n = 1;
    k.warp_tile_m = 16;
    k.warp_tile_n = 16;
    k.warp_tile_k = 16;
    k.pipeline = "mem";
    const auto r = supportsImplicitGemm(problem, k, "gfx950");
    EXPECT_FALSE(r.supported);
}

TEST(ConvImplicitGemmSupports, UnknownPipelineRejected) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    ConvImplicitGemmPerfKnobs k;
    k.tile_m = 64;
    k.tile_n = 64;
    k.tile_k = 64;
    k.warp_m = 4;
    k.warp_n = 1;
    k.warp_tile_m = 16;
    k.warp_tile_n = 16;
    k.warp_tile_k = 16;
    k.pipeline = "not_a_real_pipeline";
    const auto r = supportsImplicitGemm(problem, k, "gfx950");
    EXPECT_FALSE(r.supported);
}

TEST(ConvImplicitGemmSupports, Compv4WithIncompatibleTileRejected) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    ConvImplicitGemmPerfKnobs k;
    k.tile_m = 128;  // (128,128,64) is in TILE_TO_WAVE but NOT in COMPV4_COMPATIBLE_TILES
    k.tile_n = 128;
    k.tile_k = 64;
    k.warp_m = 4;
    k.warp_n = 4;
    k.warp_tile_m = 32;
    k.warp_tile_n = 32;
    k.warp_tile_k = 16;
    k.pipeline = "compv4";
    const auto r = supportsImplicitGemm(problem, k, "gfx950");
    EXPECT_FALSE(r.supported);
}

// ---------------------------------------------------------------------------
// buildSelectionProblem: spec -> selection problem round-trip.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmSupports, BuildSelectionProblemCopiesShapeFields) {
    ConvImplicitGemmSpec spec;
    spec.problem.N = 16;
    spec.problem.C = 64;
    spec.problem.K = 256;
    spec.problem.Hi = 28;
    spec.problem.Wi = 28;
    spec.problem.R = 1;
    spec.problem.S = 1;
    spec.problem.sH = 2;
    spec.problem.sW = 2;
    spec.problem.pH = 0;
    spec.problem.pW = 0;
    spec.problem.dH = 1;
    spec.problem.dW = 1;

    const ConvSelectionProblem sp = buildSelectionProblem(spec, "bf16");
    EXPECT_EQ(sp.N, 16);
    EXPECT_EQ(sp.C, 64);
    EXPECT_EQ(sp.K, 256);
    EXPECT_EQ(sp.G, 1);  // ck_dsl conv-fwd is ungrouped today
    EXPECT_EQ(sp.Hi, 28);
    EXPECT_EQ(sp.Wi, 28);
    EXPECT_EQ(sp.R, 1);
    EXPECT_EQ(sp.S, 1);
    EXPECT_EQ(sp.sH, 2);
    EXPECT_EQ(sp.dtype, "bf16");
    // Ho/Wo derived: (28 + 0 - 1)/2 + 1 = 14.
    EXPECT_EQ(sp.Ho(), 14);
    EXPECT_EQ(sp.Wo(), 14);
}

// ---------------------------------------------------------------------------
// Argmax tie-break: with a flat score, larger MFMA atom wins.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmSelection, FlatScorePrefersLargerAtomTieBreak) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(combos.empty());

    auto flat = [](const ConvImplicitGemmPerfKnobs&) { return 5.0; };
    const ConvImplicitGemmPerfKnobs first = selectArgmax(problem, combos, flat);
    const ConvImplicitGemmPerfKnobs second = selectArgmax(problem, combos, flat);

    // Deterministic + repeatable.
    EXPECT_TRUE(sameKnobs(first, second));

    // With flat scoring, the tie-break prefers the largest atom area
    // present in the candidate set. The 32x32x16 atom (area 1024) beats
    // the 16x16x* atom (area 256).
    const std::int64_t maxAtomArea = [&]() {
        std::int64_t m = 0;
        for (const auto& k : combos) {
            m = std::max<std::int64_t>(m, static_cast<std::int64_t>(k.warp_tile_m) * k.warp_tile_n);
        }
        return m;
    }();
    const std::int64_t pickedArea =
        static_cast<std::int64_t>(first.warp_tile_m) * first.warp_tile_n;
    EXPECT_EQ(pickedArea, maxAtomArea);
}

TEST(ConvImplicitGemmSelection, ArgmaxReturnsScorePeak) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    ASSERT_GE(combos.size(), 2u);

    // Reward the middle combo specifically.
    const ConvImplicitGemmPerfKnobs target = combos[combos.size() / 2];
    auto score = [&target](const ConvImplicitGemmPerfKnobs& k) {
        const bool match = k.tile_m == target.tile_m && k.tile_n == target.tile_n &&
                           k.tile_k == target.tile_k && k.pipeline == target.pipeline;
        return match ? 100.0 : 1.0;
    };

    const ConvImplicitGemmPerfKnobs picked = selectArgmax(problem, combos, score);
    EXPECT_EQ(picked.tile_m, target.tile_m);
    EXPECT_EQ(picked.tile_n, target.tile_n);
    EXPECT_EQ(picked.tile_k, target.tile_k);
    EXPECT_EQ(picked.pipeline, target.pipeline);
}

TEST(ConvImplicitGemmSelection, DistinctShapesCanProduceDistinctPicks) {
    // Shape-sensitive injected score: reward the tile_m closest to a
    // gemm_m-derived target. Two very different shapes must be able to
    // produce different picks (i.e. selection is NOT shape-invariant).
    ConvSelectionProblem small = makeReferenceProblem();
    small.N = 1;
    small.Hi = 8;
    small.Wi = 8;
    small.K = 32;

    ConvSelectionProblem big = makeReferenceProblem();
    big.N = 32;
    big.Hi = 112;
    big.Wi = 112;
    big.K = 512;

    const auto combosSmall = enumerateCandidates(small, "gfx950");
    const auto combosBig = enumerateCandidates(big, "gfx950");
    ASSERT_FALSE(combosSmall.empty());
    ASSERT_FALSE(combosBig.empty());

    auto targetTileM = [](int target) {
        return [target](const ConvImplicitGemmPerfKnobs& k) {
            return -static_cast<double>(std::abs(k.tile_m - target));
        };
    };

    const auto pickSmall = selectArgmax(small, combosSmall, targetTileM(16));
    const auto pickBig = selectArgmax(big, combosBig, targetTileM(128));
    EXPECT_NE(pickSmall.tile_m, pickBig.tile_m);
}

// ---------------------------------------------------------------------------
// Analytic fallback policy.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmFallback, MidSizedShapePicksDefaultTileAndMemPipeline) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(combos.empty());
    const auto pick = selectAnalyticFallback(problem, combos);
    // analyticTarget returns (64,64,64,mem) for this gemm_m / gemm_n window.
    EXPECT_EQ(pick.tile_m, 64);
    EXPECT_EQ(pick.tile_n, 64);
    EXPECT_EQ(pick.tile_k, 64);
    EXPECT_EQ(pick.pipeline, "mem");
}

TEST(ConvImplicitGemmFallback, TinyShapeShrinksTileM) {
    ConvSelectionProblem problem = makeReferenceProblem();
    problem.N = 1;
    problem.Hi = 4;
    problem.Wi = 4;
    problem.R = 1;
    problem.S = 1;
    problem.pH = 0;
    problem.pW = 0;
    // gemm_m = 1 * 4 * 4 = 16 -> the "very small M" branch in analyticTarget.
    const auto combos = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(combos.empty());
    const auto pick = selectAnalyticFallback(problem, combos);
    EXPECT_EQ(pick.tile_m, 16);
}

TEST(ConvImplicitGemmFallback, LargeShapeOpensTileN) {
    ConvSelectionProblem problem = makeReferenceProblem();
    problem.N = 32;
    problem.Hi = 112;
    problem.Wi = 112;
    problem.R = 3;
    problem.S = 3;
    problem.pH = 1;
    problem.pW = 1;
    problem.K = 256;
    // gemm_m >> 4096 and gemm_n >= 128 -> opens to tile_n=128.
    const auto combos = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(combos.empty());
    const auto pick = selectAnalyticFallback(problem, combos);
    EXPECT_EQ(pick.tile_n, 128);
}

TEST(ConvImplicitGemmFallback, IsDeterministic) {
    const ConvSelectionProblem problem = makeReferenceProblem();
    const auto combos = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(combos.empty());
    const auto a = selectAnalyticFallback(problem, combos);
    const auto b = selectAnalyticFallback(problem, combos);
    EXPECT_TRUE(sameKnobs(a, b));
}
