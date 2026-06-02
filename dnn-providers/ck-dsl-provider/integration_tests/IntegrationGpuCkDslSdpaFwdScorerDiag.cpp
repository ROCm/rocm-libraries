// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Scorer wiring diagnostic. Pure CPU (LightGBM inference only -- no device,
// no JIT, no launch). Answers ONE question: is the dispatcher heuristic
// actually WIRED correctly (features reach the model, predictions vary with
// the config), independent of whether the model's ranking is GOOD for our
// kernel?
//
// Three probes, all at an in-family problem (bf16 / d64 / Hq64-Hkv8 -- the
// regime the gfx950 fwd model was trained on):
//   1. OUR DSL CANDIDATES   -- predicted TFLOPS for every enumerated knob
//                              combo (via knobsToKernelKey). Shows whether
//                              the predictions vary across our candidates
//                              and which combo the model scores highest.
//   2. TILE-SPACE SWEEP     -- predicted TFLOPS over a grid of tile_shape
//                              (m0, n0) values set DIRECTLY on the key,
//                              spanning the trained block-tile range. Shows
//                              the model's response surface and where it
//                              peaks when queried across the trained space.
//   3. m0 MONOTONICITY      -- vary ONLY tile_shape.m0 at a fixed n0. If the
//                              prediction is flat the feature is not
//                              reaching the model (a wiring bug); if it
//                              varies the model is wired and simply has a
//                              preference (a model/mapping-quality matter).
//
// Prints everything to stdout so the wiring verdict survives with plugin
// logging off. No EXPECT on the model's preference -- this is a probe; the
// only assertion is that the model loaded and predictions are not all
// identical (the wiring check).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "adapters/sdpa/SdpaCandidateSelector.hpp"
#include "adapters/sdpa/SdpaPerfKnobs.hpp"
#include "adapters/sdpa/SdpaScorer.hpp"

namespace {

using ck_dsl_provider::SdpaPerfKnobs;
using ck_dsl_provider::SdpaScorer;
using ck_dsl_provider::SdpaSelectionProblem;
using ck_tile::dispatcher::FmhaKernelKey;
using ck_tile::dispatcher::FmhaProblem;

// In-family selection problem: bf16, D64, Hq64 / Hkv8 (GQA ratio 8),
// S2048, block_size 32 -- the model's trained regime. mask=causal,
// use_paged_kv mirrors the runtime (the scoring key forces it false
// internally; here we set the problem the same way the plan builder does).
SdpaSelectionProblem inFamilyProblem() {
    SdpaSelectionProblem p;
    p.batch = 2;
    p.num_query_heads = 64;
    p.num_kv_heads = 8;
    p.seqlen_q = 2048;
    p.seqlen_k = 2048;
    p.head_size = 64;
    p.block_size = 32;
    p.dtype = "bf16";
    p.use_paged_kv = true;
    p.use_sinks = false;
    p.sliding_window = 0;
    p.mask_type = 1;
    p.bias_type = 0;
    p.skip_min_seqlen_q = false;
    return p;
}

std::string knobStr(const SdpaPerfKnobs& k) {
    return "nw=" + std::to_string(k.num_warps) + ",mw=" + std::to_string(k.block_m_per_warp) +
           ",t=" + std::to_string(k.tile_size) + ",mfma32=" + (k.use_mfma_32x32 ? "1" : "0");
}

class SdpaScorerDiag : public ::testing::Test {};

TEST_F(SdpaScorerDiag, PredictionsVaryAndPeakLocation) {
    SdpaScorer scorer;
    if (!scorer.isLoaded()) {
        GTEST_SKIP() << "FMHA fwd model not loaded; cannot run the scorer wiring diagnostic";
    }

    const SdpaSelectionProblem prob = inFamilyProblem();
    const FmhaProblem fmhaProb = ck_dsl_provider::problemToFmhaProblem(prob);

    // ---- Probe 1: our enumerated DSL candidates -------------------------
    const std::vector<SdpaPerfKnobs> candidates = ck_dsl_provider::enumerateCandidates(prob);
    ASSERT_FALSE(candidates.empty());

    struct Scored {
        SdpaPerfKnobs knobs;
        std::uint16_t m0, n0;
        double tflops;
    };
    std::vector<Scored> scored;
    scored.reserve(candidates.size());
    double minPred = 1e30;
    double maxPred = -1e30;
    for (const SdpaPerfKnobs& c : candidates) {
        const FmhaKernelKey key = ck_dsl_provider::knobsToKernelKey(prob, c);
        const double t = scorer.predict(fmhaProb, key);
        scored.push_back({c, key.algorithm.tile_shape.m0, key.algorithm.tile_shape.n0, t});
        minPred = std::min(minPred, t);
        maxPred = std::max(maxPred, t);
    }
    std::sort(scored.begin(), scored.end(),
              [](const Scored& a, const Scored& b) { return a.tflops > b.tflops; });

    std::cout
        << "\n[ScorerDiag] ===== Probe 1: OUR DSL CANDIDATES (in-family bf16/d64/gqa8 S2048), "
        << candidates.size() << " combos, sorted by predicted TFLOPS =====\n";
    for (const Scored& s : scored) {
        std::cout << "[ScorerDiag]   pred=" << s.tflops << "  m0=" << s.m0 << " n0=" << s.n0
                  << "  (" << knobStr(s.knobs) << ")\n";
    }
    std::cout << "[ScorerDiag]   --> BEST DSL candidate: pred=" << scored.front().tflops << " ("
              << knobStr(scored.front().knobs) << ")  WORST: pred=" << scored.back().tflops << " ("
              << knobStr(scored.back().knobs) << ")\n";

    // ---- Probe 2: direct tile-shape (m0, n0) sweep on the key -----------
    // Take a base key and override ONLY tile_shape.m0 / n0 across the
    // trained block-tile range; everything else (pipeline, k0/k1, n1,
    // k0max, paged, dtype, problem) is held fixed. This queries the model's
    // response surface over the tile dims independent of our enumerator.
    const FmhaKernelKey baseKey = ck_dsl_provider::knobsToKernelKey(prob, candidates.front());
    const std::vector<std::uint16_t> m0s{16, 32, 64, 128, 256};
    const std::vector<std::uint16_t> n0s{64, 128, 256};
    std::cout << "\n[ScorerDiag] ===== Probe 2: direct tile (m0 x n0) sweep, base key held fixed "
                 "=====\n";
    double bestSweep = -1e30;
    std::uint16_t bestM0 = 0, bestN0 = 0;
    for (std::uint16_t m0 : m0s) {
        std::string row = "[ScorerDiag]   m0=" + std::to_string(m0) + " :";
        for (std::uint16_t n0 : n0s) {
            FmhaKernelKey key = baseKey;
            key.algorithm.tile_shape.m0 = m0;
            key.algorithm.tile_shape.n0 = n0;
            const double t = scorer.predict(fmhaProb, key);
            row += "  n0=" + std::to_string(n0) + ":" + std::to_string(t);
            if (t > bestSweep) {
                bestSweep = t;
                bestM0 = m0;
                bestN0 = n0;
            }
        }
        std::cout << row << "\n";
    }
    std::cout << "[ScorerDiag]   --> sweep PEAK at m0=" << bestM0 << " n0=" << bestN0
              << " pred=" << bestSweep << "\n";

    // ---- Probe 3: m0 monotonicity at fixed n0=128 -----------------------
    std::cout << "\n[ScorerDiag] ===== Probe 3: vary ONLY m0 (n0=128 fixed) =====\n";
    double mMin = 1e30;
    double mMax = -1e30;
    for (std::uint16_t m0 : m0s) {
        FmhaKernelKey key = baseKey;
        key.algorithm.tile_shape.m0 = m0;
        key.algorithm.tile_shape.n0 = 128;
        const double t = scorer.predict(fmhaProb, key);
        std::cout << "[ScorerDiag]   m0=" << m0 << " -> pred=" << t << "\n";
        mMin = std::min(mMin, t);
        mMax = std::max(mMax, t);
    }
    std::cout << "[ScorerDiag]   --> m0 response range: [" << mMin << ", " << mMax << "]\n";

    // ---- Wiring verdict -------------------------------------------------
    // The ONLY hard assertion: the model's predictions are not all
    // identical. Distinct predictions across configs prove the features
    // reach the model and predict() is wired -- regardless of whether the
    // model's preference is good for our kernel.
    const bool varies = (maxPred - minPred) > 1e-6 || (mMax - mMin) > 1e-6;
    std::cout << "[ScorerDiag] WIRING VERDICT: predictions "
              << (varies ? "VARY across configs (model is wired + features reach it)"
                         : "are CONSTANT (FEATURES NOT REACHING THE MODEL -- wiring bug)")
              << "\n"
              << std::endl;
    EXPECT_TRUE(varies) << "predict() returned identical TFLOPS for every config -- the config "
                           "features are not reaching the model (a wiring bug, not a model-quality "
                           "issue)";
}

}  // namespace
