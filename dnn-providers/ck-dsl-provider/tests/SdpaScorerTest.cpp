// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Scorer + selection tests. This TU includes ONLY SdpaScorer.hpp (the
// HIP-free pimpl header) and SdpaCandidateSelector.hpp, so it compiles
// as plain CXX -- it never sees fmha_ml_heuristic.hpp (HIP). The scorer
// loads the in-tree gfx950 model via its default constructor.

#include <gtest/gtest.h>

#include <ck_tile/dispatcher/fmha_kernel_key.hpp>
#include <ck_tile/dispatcher/fmha_problem.hpp>
#include <cmath>
#include <vector>

#include "adapters/sdpa/SdpaCandidateSelector.hpp"
#include "adapters/sdpa/SdpaPerfKnobs.hpp"
#include "adapters/sdpa/SdpaScorer.hpp"

namespace {

using ck_dsl_provider::enumerateCandidates;
using ck_dsl_provider::knobsToKernelKey;
using ck_dsl_provider::problemToFmhaProblem;
using ck_dsl_provider::SdpaPerfKnobs;
using ck_dsl_provider::SdpaScorer;
using ck_dsl_provider::SdpaSelectionProblem;
using ck_dsl_provider::selectAnalyticFallback;
using ck_dsl_provider::selectPerfKnobs;
using ck_tile::dispatcher::FmhaKernelKey;
using ck_tile::dispatcher::FmhaProblem;

/// In-family reference problem (head 64, block 32, GQA 8, bf16). Mirrors
/// the candidate-selector suite's reference shape so the two test files
/// agree on a known-buildable problem.
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

/// fp16 variant of the reference problem (same shape, fp16 dtype).
SdpaSelectionProblem makeFp16Problem() {
    SdpaSelectionProblem p = makeReferenceProblem();
    p.dtype = "fp16";
    return p;
}

/// Structural equality over the knob fields that distinguish a combo.
/// SdpaPerfKnobs has no operator==, so the tests compare the axes the
/// selection actually varies.
bool sameKnobs(const SdpaPerfKnobs& a, const SdpaPerfKnobs& b) {
    return a.num_warps == b.num_warps && a.block_m_per_warp == b.block_m_per_warp &&
           a.tile_size == b.tile_size && a.waves_per_eu == b.waves_per_eu &&
           a.use_mfma_32x32 == b.use_mfma_32x32 &&
           a.use_transposed_qk_32x32 == b.use_transposed_qk_32x32 &&
           a.use_register_pv == b.use_register_pv &&
           a.use_early_v_schedule == b.use_early_v_schedule &&
           a.use_fast_paged_kv_desc == b.use_fast_paged_kv_desc && a.use_sinks == b.use_sinks &&
           a.sliding_window == b.sliding_window;
}

}  // namespace

// ---------------------------------------------------------------------------
// Real model loads from the in-tree path.
// ---------------------------------------------------------------------------

TEST(SdpaScorer, DefaultConstructorLoadsInTreeModel) {
    SdpaScorer scorer;
    EXPECT_TRUE(scorer.isLoaded()) << "default SdpaScorer should load the in-tree gfx950 model "
                                      "baked into CK_DSL_FMHA_FWD_MODEL_PATH";
}

TEST(SdpaScorer, PredictReturnsFiniteValueForSampleCandidate) {
    SdpaScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> candidates = enumerateCandidates(problem);
    ASSERT_FALSE(candidates.empty());

    const FmhaProblem fmhaProblem = problemToFmhaProblem(problem);
    const FmhaKernelKey key = knobsToKernelKey(problem, candidates.front());
    const double pred = scorer.predict(fmhaProblem, key);
    EXPECT_TRUE(std::isfinite(pred)) << "predicted TFLOPS must be finite (got " << pred << ")";
}

// ---------------------------------------------------------------------------
// Loaded path: selection is deterministic across repeated calls.
// ---------------------------------------------------------------------------

TEST(SdpaScorer, LoadedSelectionIsDeterministic) {
    SdpaScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> candidates = enumerateCandidates(problem);
    ASSERT_FALSE(candidates.empty());

    const SdpaPerfKnobs first = selectPerfKnobs(problem, candidates, scorer);
    const SdpaPerfKnobs second = selectPerfKnobs(problem, candidates, scorer);
    EXPECT_TRUE(sameKnobs(first, second))
        << "selectPerfKnobs over the same candidates must return the same combo twice";
}

TEST(SdpaScorer, LoadedSelectionPicksAnEnumeratedCandidate) {
    SdpaScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const SdpaSelectionProblem problem = makeFp16Problem();
    const std::vector<SdpaPerfKnobs> candidates = enumerateCandidates(problem);
    ASSERT_FALSE(candidates.empty());

    const SdpaPerfKnobs chosen = selectPerfKnobs(problem, candidates, scorer);
    bool found = false;
    for (const SdpaPerfKnobs& c : candidates) {
        if (sameKnobs(c, chosen)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "the scored pick must be one of the enumerated candidates";
}

// ---------------------------------------------------------------------------
// Model-missing -> analytic fallback (NOT first-fit).
// ---------------------------------------------------------------------------

TEST(SdpaScorer, MissingModelIsNotLoaded) {
    SdpaScorer bad{"/nonexistent/path/to/model.lgbm"};
    EXPECT_FALSE(bad.isLoaded());
}

TEST(SdpaScorer, MissingModelFallsBackToAnalyticPolicy) {
    SdpaScorer bad{"/nonexistent/path/to/model.lgbm"};
    ASSERT_FALSE(bad.isLoaded());

    const SdpaSelectionProblem problem = makeReferenceProblem();
    const std::vector<SdpaPerfKnobs> candidates = enumerateCandidates(problem);
    ASSERT_FALSE(candidates.empty());

    const SdpaPerfKnobs viaSelect = selectPerfKnobs(problem, candidates, bad);
    const SdpaPerfKnobs viaAnalytic = selectAnalyticFallback(problem, candidates);
    EXPECT_TRUE(sameKnobs(viaSelect, viaAnalytic))
        << "with no model, selectPerfKnobs must equal the analytic fallback, not a first-fit";
}

TEST(SdpaScorer, MissingModelFallbackHoldsForFp16) {
    SdpaScorer bad{"/nonexistent/path/to/model.lgbm"};
    ASSERT_FALSE(bad.isLoaded());

    const SdpaSelectionProblem problem = makeFp16Problem();
    const std::vector<SdpaPerfKnobs> candidates = enumerateCandidates(problem);
    ASSERT_FALSE(candidates.empty());

    const SdpaPerfKnobs viaSelect = selectPerfKnobs(problem, candidates, bad);
    const SdpaPerfKnobs viaAnalytic = selectAnalyticFallback(problem, candidates);
    EXPECT_TRUE(sameKnobs(viaSelect, viaAnalytic));
}
