// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "ConvImplicitGemmPerfKnobs.hpp"
#include "ConvImplicitGemmScorer.hpp"
#include "ConvImplicitGemmSpec.hpp"

namespace ck_dsl_provider {

/// Problem-shape inputs the candidate enumerator and scoring mapping
/// need. Mirrors ``ConvProblem`` (the 13 shape fields the adapter
/// extracts) plus the dtype string and group count, which the feature
/// engine needs but the ck_dsl conv-fwd path treats as fixed (G=1).
///
/// ``dtype`` follows the model's spelling (e.g. "bf16", "fp16"). The
/// plan builder selects the oracle whose (dtype, arch) pair matches; if
/// none exists the analytic policy is used as a fallback.
///
/// The fields here are NOT a separate copy of ``ConvProblem`` --
/// ``buildSelectionProblem`` derives this struct from a ``ConvProblem``
/// and a dtype string so the adapter need only fill in the spec it
/// already builds.
///
/// **2D-only by design.** No depth (D, Z, Do, stride_d, pad_d) fields
/// appear here because the ck_dsl conv-fwd path is 2D-only today. The
/// LightGBM scorer's feature extractor (``extractConvFeatures``) PINS
/// the 3D feature slots to their 2D values; growing this struct with
/// depth fields without updating the extractor would silently feed the
/// booster 2D values for what is now a 3D problem. If 3D conv lands,
/// flip ``kConvSelectionDim`` in the .cpp and add the depth fields
/// here in the same change.
struct ConvSelectionProblem {
    // --- 2D conv shape (group dim folded in via G) -------------------
    std::int32_t N{0};
    std::int32_t C{0};   // channels per ConvProblem (== input C)
    std::int32_t K{0};   // output channels
    std::int32_t G{1};   // group count; ck_dsl conv-fwd is ungrouped today
    std::int32_t Hi{0};
    std::int32_t Wi{0};
    std::int32_t R{0};   // filter H (feature engine's "Y")
    std::int32_t S{0};   // filter W (feature engine's "X")

    std::int32_t sH{1};
    std::int32_t sW{1};
    std::int32_t pH{0};
    std::int32_t pW{0};
    std::int32_t dH{1};
    std::int32_t dW{1};

    std::string dtype{"bf16"};

    /// Derived output height (mirror of ConvProblem::Ho).
    [[nodiscard]] std::int32_t Ho() const {
        const std::int32_t effY = (R - 1) * dH + 1;
        return (Hi + 2 * pH - effY) / sH + 1;
    }

    /// Derived output width (mirror of ConvProblem::Wo).
    [[nodiscard]] std::int32_t Wo() const {
        const std::int32_t effX = (S - 1) * dW + 1;
        return (Wi + 2 * pW - effX) / sW + 1;
    }
};

/// Convenience: build a ``ConvSelectionProblem`` from the spec the
/// adapter has already constructed. Pulls the 13 shape fields off
/// ``spec.problem`` and the dtype off the caller.
[[nodiscard]] ConvSelectionProblem buildSelectionProblem(const ConvImplicitGemmSpec& spec,
                                                         const std::string& dtype);

/// Result of ``supportsImplicitGemm``: a verdict plus a structured
/// reason (empty when supported).
struct ConvSupportsResult {
    bool supported{false};
    std::string reason;
};

/// Validity gate for a (problem, knobs, arch) triple against the ck_dsl
/// implicit-GEMM kernel and the trained-data envelope.
///
/// Mirrors the ck_dsl ``is_valid_spec`` predicate (the Python bridge
/// the plan builder calls today) for the structural constraints, plus a
/// trained-table membership check: ``(tile_m, tile_n, tile_k)`` must be
/// one of the ten triples in ``TILE_TO_WAVE`` / ``TILE_TO_WARP``, and
/// (warp_m, warp_n) + (warp_tile_m, warp_tile_n, warp_tile_k) must
/// match that triple's table entry. Pipelines outside
/// ``VARIANT_PIPELINES["forward"]`` are rejected. ``compv4`` additionally
/// requires the tile triple to be in ``COMPV4_COMPATIBLE_TILES``.
/// The warp_tile atom must be present in the f16 MMA catalog for ``arch``
/// (same check as ``is_valid_spec`` in the Python DSL).
///
/// Used by ``enumerateCandidates`` to keep the emitted set buildable; a
/// post-enum re-validation can use the same predicate as a guard.
[[nodiscard]] ConvSupportsResult supportsImplicitGemm(const ConvSelectionProblem& problem,
                                                      const ConvImplicitGemmPerfKnobs& knobs,
                                                      const std::string& arch);

/// Enumerate kernel-knob combos for the problem on ``arch``, pre-filtered
/// by ``supportsImplicitGemm``. Every returned candidate is guaranteed to
/// pass ``isApplicable`` on that arch -- the contract that makes the
/// ranked list's top pick always buildable.
///
/// The enumeration walks the 10 trained tile triples (TILE_TO_WAVE keys)
/// and the 8 forward pipelines (VARIANT_PIPELINES["forward"]) for a max
/// of 80 raw candidates, drops tiles whose warp_tile atom is absent from
/// the arch's f16 MMA catalog, drops the compv4 tiles that aren't in
/// COMPV4_COMPATIBLE_TILES, and pins the wave grid + MFMA atom from the
/// TILE_TO_WAVE / TILE_TO_WARP tables.
///
/// The enumeration order is deterministic (tiles in TILE_TO_WAVE
/// insertion order, then pipelines in VARIANT_PIPELINES["forward"]
/// order), which fixes the tie-break order used by ``selectArgmax``.
[[nodiscard]] std::vector<ConvImplicitGemmPerfKnobs> enumerateCandidates(
    const ConvSelectionProblem& problem, const std::string& arch);

/// Argmax selection over a set of candidate knob combos using an
/// injected score callable. The callable receives the candidate's
/// ``ConvImplicitGemmPerfKnobs`` and returns a scalar score (higher is
/// better). The highest-scoring combo is returned.
///
/// Deterministic tie-break: on an EXACT score tie, the candidate with
/// the larger MFMA atom (``warp_tile_m * warp_tile_n``) wins. The model
/// has no atom feature beyond what is captured in the existing tile /
/// pipeline features, so two configurations that differ ONLY in the
/// atom can predict identical TFLOPS. The 32x32x16 atom is the
/// oracle-best on the bulk of the training distribution, so we break
/// the tie toward it. Any remaining tie falls to enumeration order.
///
/// ``candidates`` must be non-empty (the caller guarantees at least one
/// buildable combo, or falls back to the analytic pick).
[[nodiscard]] ConvImplicitGemmPerfKnobs selectArgmax(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates,
    const std::function<double(const ConvImplicitGemmPerfKnobs&)>& score);

/// Analytic fallback pick over the SAME enumerated+filtered combo set.
/// Approximates the production rule-of-thumb the CK conv codegen uses
/// when no measurement is available: prefer the 64x64x64 tile (the
/// universally-buildable middle of the table) and the ``mem`` pipeline
/// (the codegen default that dominates the training set). This is the
/// MODEL-LOAD-FAILURE fallback -- it is NOT a trivial first-fit; it
/// scores each candidate by distance to the analytic target and returns
/// the best. ``candidates`` must be non-empty.
[[nodiscard]] ConvImplicitGemmPerfKnobs selectAnalyticFallback(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates);

/// Top-level knob selection for the plan builder. Picks the best combo
/// from ``candidates`` (which the caller enumerated for ``problem``):
///
///   * ``scorer != nullptr`` and its model loaded -> ML argmax over the
///     candidates, scoring each by ``scorer->predict(problem, cand)``;
///   * ``scorer == nullptr`` (no registry entry for this dtype/arch) OR
///     model failed to load -> the analytic fallback (NOT a trivial
///     first-fit).
///
/// The caller (plan builder) is responsible for supplying the scorer
/// that matches the current (dtype, arch) pair. Passing nullptr is the
/// correct signal for "no oracle for this combination"; this function
/// never inspects dtype or arch directly.
///
/// HIP-free: ``ConvImplicitGemmScorer`` wraps LightGBM through a plain
/// C-API declaration, so this header stays plain CXX. ``candidates``
/// must be non-empty (the caller hard-errors on an empty enumeration).
[[nodiscard]] ConvImplicitGemmPerfKnobs selectPerfKnobs(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates,
    const ConvImplicitGemmScorer* scorer);

/// Rank all candidates from best to worst under the same scoring rule as
/// ``selectPerfKnobs`` (ML when ``scorer`` is non-null and loaded,
/// analytic-closeness otherwise). Tie-break matches ``selectArgmax``:
/// equal scores resolve toward the larger MFMA atom; remaining ties fall
/// to enumeration order. ``selectPerfKnobs`` is equivalent to
/// ``rankPerfKnobs(...).front()`` -- use this overload when the caller
/// needs to walk the ranking (e.g. the plan builder falling through to
/// the next-best combo when the DSL rejects the top pick after the
/// overlay is applied).
[[nodiscard]] std::vector<ConvImplicitGemmPerfKnobs> rankPerfKnobs(
    const ConvSelectionProblem& problem,
    const std::vector<ConvImplicitGemmPerfKnobs>& candidates,
    const ConvImplicitGemmScorer* scorer);

}  // namespace ck_dsl_provider
