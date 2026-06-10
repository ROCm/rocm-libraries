// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvImplicitGemmCandidateSelector.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_set>

namespace ck_dsl_provider {

namespace {

// Trained tile triples (TILE_TO_WAVE / TILE_TO_WARP keys in
// grouped_config_rules.py). Each entry: (tile_m, tile_n, tile_k,
// warp_m, warp_n, warp_tile_m, warp_tile_n, warp_tile_k). The wave grid
// (warp_m/n) and MFMA atom (warp_tile_*) are pinned by the table -- the
// enumerator does NOT vary them independently. Insertion order matches
// the Python dict so the enumeration is byte-stable across the two
// implementations.
struct TileEntry {
    std::int32_t tile_m;
    std::int32_t tile_n;
    std::int32_t tile_k;
    std::int32_t warp_m;
    std::int32_t warp_n;
    std::int32_t warp_tile_m;
    std::int32_t warp_tile_n;
    std::int32_t warp_tile_k;
};

constexpr std::array<TileEntry, 10> kTileTable{{
    // warp_tile [16,16,16]
    {16, 64, 64, 1, 4, 16, 16, 16},
    {32, 64, 64, 2, 2, 16, 16, 16},
    {64, 64, 64, 4, 1, 16, 16, 16},
    // warp_tile [32,32,16]
    {32, 128, 64, 1, 4, 32, 32, 16},
    {64, 128, 64, 2, 2, 32, 32, 16},
    {128, 128, 64, 4, 4, 32, 32, 16},
    // warp_tile [16,16,32]
    {16, 64, 128, 1, 4, 16, 16, 32},
    {32, 64, 128, 2, 2, 16, 16, 32},
    {64, 64, 128, 4, 1, 16, 16, 32},
    {128, 64, 128, 8, 2, 16, 16, 32},
}};

// VARIANT_PIPELINES["forward"] -- the 8 forward pipelines the codegen
// reports as universally buildable. comp_async / basic_async_v1 collapse
// onto compv3 in the scoring (PIPELINE_MAP default = 0), so they share
// the score with compv3; the deterministic tie-break picks between them.
constexpr std::array<const char*, 8> kForwardPipelines{
    "basic_v1", "mem",     "compv3",     "compv4",
    "compv5",   "compv6",  "comp_async", "basic_async_v1",
};

// COMPV4_COMPATIBLE_TILES (grouped_config_rules.py). compv4 cannot run on
// the wave=8x2x1 (128,64,128) tile or any warp_tile=[32,32,16] tile.
bool isCompv4CompatibleTile(std::int32_t tm, std::int32_t tn, std::int32_t tk) {
    // warp_tile [16,16,16] subset: (16,64,64) (32,64,64) (64,64,64)
    if (tk == 64 && tn == 64 && (tm == 16 || tm == 32 || tm == 64)) {
        return true;
    }
    // warp_tile [16,16,32] subset: (16,64,128) (32,64,128) (64,64,128)
    if (tk == 128 && tn == 64 && (tm == 16 || tm == 32 || tm == 64)) {
        return true;
    }
    return false;
}

bool isForwardPipeline(const std::string& p) {
    for (const char* known : kForwardPipelines) {
        if (p == known) {
            return true;
        }
    }
    return false;
}

const TileEntry* findTileEntry(std::int32_t tm, std::int32_t tn, std::int32_t tk) {
    for (const auto& e : kTileTable) {
        if (e.tile_m == tm && e.tile_n == tn && e.tile_k == tk) {
            return &e;
        }
    }
    return nullptr;
}

// MFMA atom area used by the deterministic tie-break in selectArgmax.
std::int64_t atomArea(const ConvImplicitGemmPerfKnobs& k) {
    return static_cast<std::int64_t>(k.warp_tile_m) * k.warp_tile_n;
}

}  // namespace

ConvSelectionProblem buildSelectionProblem(const ConvImplicitGemmSpec& spec,
                                           const std::string& dtype) {
    ConvSelectionProblem out;
    out.N = spec.problem.N;
    out.C = spec.problem.C;
    out.K = spec.problem.K;
    out.G = 1;  // ck_dsl conv-fwd path is ungrouped today
    out.Hi = spec.problem.Hi;
    out.Wi = spec.problem.Wi;
    out.R = spec.problem.R;
    out.S = spec.problem.S;
    out.sH = spec.problem.sH;
    out.sW = spec.problem.sW;
    out.pH = spec.problem.pH;
    out.pW = spec.problem.pW;
    out.dH = spec.problem.dH;
    out.dW = spec.problem.dW;
    out.dtype = dtype;
    return out;
}

ConvSupportsResult supportsImplicitGemm(const ConvSelectionProblem& problem,
                                        const ConvImplicitGemmPerfKnobs& knobs) {
    // Structural: dtype gate -- trained envelopes exist for bf16/gfx950
    // and fp16/gfx942. fp32 passes through here (the kernel itself may
    // build), but the scorer-driven path bails to the analytic fallback
    // for any dtype/arch pair without an oracle (selectPerfKnobs). We do
    // NOT reject fp32 candidates here so enumeration stays uniform; the
    // scoring step is what discriminates.
    (void)problem;

    // Tile triple must be in the trained TILE_TO_WAVE table.
    const TileEntry* entry = findTileEntry(knobs.tile_m, knobs.tile_n, knobs.tile_k);
    if (!entry) {
        return {false, "implicit-GEMM: tile (" + std::to_string(knobs.tile_m) + "," +
                           std::to_string(knobs.tile_n) + "," + std::to_string(knobs.tile_k) +
                           ") not in the trained TILE_TO_WAVE table"};
    }

    // Wave grid + MFMA atom must match the TILE_TO_WAVE/WARP entry. The
    // enumerator pins these from the table, but a hand-constructed knob
    // set (e.g. from a test) may pass through here.
    if (knobs.warp_m != entry->warp_m || knobs.warp_n != entry->warp_n) {
        return {false, "implicit-GEMM: wave grid (" + std::to_string(knobs.warp_m) + "," +
                           std::to_string(knobs.warp_n) + ") does not match TILE_TO_WAVE for tile"};
    }
    if (knobs.warp_tile_m != entry->warp_tile_m || knobs.warp_tile_n != entry->warp_tile_n ||
        knobs.warp_tile_k != entry->warp_tile_k) {
        return {false, "implicit-GEMM: MFMA atom does not match TILE_TO_WARP for tile"};
    }

    // Pipeline must be in VARIANT_PIPELINES["forward"].
    if (!isForwardPipeline(knobs.pipeline)) {
        return {false,
                "implicit-GEMM: pipeline '" + knobs.pipeline + "' not in forward variant set"};
    }
    // compv4 has stricter tile requirements (COMPV4_COMPATIBLE_TILES).
    if (knobs.pipeline == "compv4" &&
        !isCompv4CompatibleTile(knobs.tile_m, knobs.tile_n, knobs.tile_k)) {
        return {false, "implicit-GEMM: compv4 not compatible with tile (" +
                           std::to_string(knobs.tile_m) + "," + std::to_string(knobs.tile_n) +
                           "," + std::to_string(knobs.tile_k) + ")"};
    }
    // wave_size must be 64 on all oracle arches (gfx950, gfx942 are CDNA).
    if (knobs.wave_size != 64) {
        return {false, "implicit-GEMM: wave_size=" + std::to_string(knobs.wave_size) +
                           " unsupported (oracle arches gfx950/gfx942 require wave64)"};
    }
    return {true, ""};
}

std::vector<ConvImplicitGemmPerfKnobs> enumerateCandidates(const ConvSelectionProblem& problem) {
    std::vector<ConvImplicitGemmPerfKnobs> out;
    out.reserve(kTileTable.size() * kForwardPipelines.size());

    for (const auto& tile : kTileTable) {
        for (const char* pipeline : kForwardPipelines) {
            ConvImplicitGemmPerfKnobs k;
            k.tile_m = tile.tile_m;
            k.tile_n = tile.tile_n;
            k.tile_k = tile.tile_k;
            k.warp_m = tile.warp_m;
            k.warp_n = tile.warp_n;
            k.warp_tile_m = tile.warp_tile_m;
            k.warp_tile_n = tile.warp_tile_n;
            k.warp_tile_k = tile.warp_tile_k;
            k.pipeline = pipeline;
            k.wave_size = 64;

            if (supportsImplicitGemm(problem, k).supported) {
                out.push_back(std::move(k));
            }
        }
    }
    return out;
}

ConvImplicitGemmPerfKnobs selectArgmax(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates,
    const std::function<double(const ConvImplicitGemmPerfKnobs&)>& score) {
    (void)problem;
    // Caller guarantees non-empty.
    std::size_t bestIdx = 0;
    double bestScore = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double s = score(candidates[i]);
        // Strict > preserves enumeration order on ties (the first combo
        // with the maximal score wins) -> stable, deterministic.
        if (i == 0 || s > bestScore) {
            bestScore = s;
            bestIdx = i;
        } else if (s == bestScore && atomArea(candidates[i]) > atomArea(candidates[bestIdx])) {
            // MFMA-atom tie-break. comp_async / basic_async_v1 collapse to
            // PIPELINE_MAP=0 (== compv3) in the feature encoding, so for the
            // same tile they score identically to the compv3 candidate. The
            // 32x32x16 atom is the oracle-best on the bulk of the bf16/gfx950
            // training distribution, so break the tie toward the larger atom.
            bestIdx = i;
        }
    }
    return candidates[bestIdx];
}

namespace {

// Analytic target. The CK conv codegen's default-config tuner picks the
// 64x64x64 tile + ``mem`` pipeline (matches grouped_conv_forward
// codegen defaults on gfx950 bf16). For very small problems we shrink
// the tile so it doesn't dwarf the work; for very large GEMM-M we open
// it up to (64,128,64) for better wave utilisation.
struct AnalyticTarget {
    std::int32_t tile_m;
    std::int32_t tile_n;
    std::int32_t tile_k;
    std::string pipeline;
};

AnalyticTarget analyticTarget(const ConvSelectionProblem& problem) {
    AnalyticTarget t{64, 64, 64, "mem"};

    const std::int64_t gemmM = static_cast<std::int64_t>(problem.N) * problem.Ho() * problem.Wo();
    const std::int64_t gemmN = problem.K;

    if (gemmM <= 64) {
        // Very small M -- shrink tile_m to avoid heavy under-utilisation.
        t.tile_m = 16;
        t.tile_n = 64;
        t.tile_k = 64;
    } else if (gemmM >= 4096 && gemmN >= 128) {
        // Large M and N -- open the N tile for higher waves-per-CTA.
        t.tile_m = 64;
        t.tile_n = 128;
        t.tile_k = 64;
    }
    return t;
}

double analyticCloseness(const ConvImplicitGemmPerfKnobs& cand, const AnalyticTarget& target) {
    double dist = 0.0;
    // Tile-shape distance dominates; log-ratio so 16 vs 32 weighs the
    // same as 64 vs 128.
    auto logRatio = [](std::int32_t a, std::int32_t b) {
        const double x = static_cast<double>(std::max(a, 1));
        const double y = static_cast<double>(std::max(b, 1));
        const double r = x / y;
        return r >= 1.0 ? r : 1.0 / r;
    };
    dist += 2.0 * (logRatio(cand.tile_m, target.tile_m) - 1.0);
    dist += 2.0 * (logRatio(cand.tile_n, target.tile_n) - 1.0);
    dist += 1.0 * (logRatio(cand.tile_k, target.tile_k) - 1.0);
    // Pipeline match: prefer the target pipeline; modest penalty for
    // any other forward pipeline. The async variants share a feature
    // encoding with compv3 so they are not OOV but they aren't the
    // codegen default either.
    if (cand.pipeline != target.pipeline) {
        dist += 1.0;
    }
    return -dist;
}

}  // namespace

ConvImplicitGemmPerfKnobs selectAnalyticFallback(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates) {
    // Caller guarantees non-empty.
    const AnalyticTarget target = analyticTarget(problem);

    std::size_t bestIdx = 0;
    double bestScore = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double s = analyticCloseness(candidates[i], target);
        // Strict > -> first (enumeration-order) candidate wins ties.
        if (i == 0 || s > bestScore) {
            bestScore = s;
            bestIdx = i;
        }
    }
    return candidates[bestIdx];
}

ConvImplicitGemmPerfKnobs selectPerfKnobs(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates,
    const ConvImplicitGemmScorer* scorer) {
    return rankPerfKnobs(problem, candidates, scorer).front();
}

std::vector<ConvImplicitGemmPerfKnobs> rankPerfKnobs(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates,
    const ConvImplicitGemmScorer* scorer) {
    // Score function: ML when the caller supplied a scorer (non-null) and
    // its model loaded; analytic-closeness proxy otherwise. The caller
    // (plan builder registry) is responsible for passing a scorer whose
    // (dtype, arch) matches the current problem -- passing nullptr is the
    // correct signal for "no oracle for this combination".
    const bool useScorer = scorer && scorer->isLoaded();
    const AnalyticTarget analyticTgt = useScorer ? AnalyticTarget{} : analyticTarget(problem);

    // Build (score, original-index) pairs so the tie-break can read the
    // candidate at sort time without re-scoring.
    std::vector<std::pair<double, std::size_t>> scored;
    scored.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double s = useScorer ? scorer->predict(problem, candidates[i])
                                   : analyticCloseness(candidates[i], analyticTgt);
        scored.emplace_back(s, i);
    }

    // Sort descending by score; on EXACT score tie prefer the larger
    // MFMA atom (matches selectArgmax); on remaining tie preserve
    // enumeration order via stable_sort.
    std::stable_sort(scored.begin(), scored.end(),
                     [&candidates](const std::pair<double, std::size_t>& a,
                                   const std::pair<double, std::size_t>& b) {
                         if (a.first != b.first) {
                             return a.first > b.first;
                         }
                         return atomArea(candidates[a.second]) > atomArea(candidates[b.second]);
                     });

    std::vector<ConvImplicitGemmPerfKnobs> ranked;
    ranked.reserve(scored.size());
    for (const auto& [_, idx] : scored) {
        ranked.push_back(candidates[idx]);
    }
    return ranked;
}

}  // namespace ck_dsl_provider
