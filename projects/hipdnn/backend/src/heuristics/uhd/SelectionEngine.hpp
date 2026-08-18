// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "EngineRegistry.hpp"
#include "FeatureExtractor.hpp"
#include "ScoreTransform.hpp"
#include "adapters/IUhdAdapter.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

/// Device-property key holding the GPU architecture (e.g. "gfx942"), used for
/// RFC 0019 §8.3 arch-keyed UHD resolution (not for feature extraction).
///
/// NOTE: Per RFC 0019 §6.1, architecture is NOT a device feature in the
/// `$device.*` namespace for feature extraction. It is a KDP property used only
/// for selecting which arch-keyed UHD to apply. The DeviceProperties FlatBuffer
/// field is `architecture_name`; this key is shortened to `arch` for the internal
/// device-vars map that drives arch resolution.
inline constexpr const char* DEVICE_ARCH_KEY = "arch";

/// @brief Result of scoring a single kernel candidate.
struct ScoredCandidate
{
    int64_t kernelId;
    int64_t priority;
    double score;
    bool scoreValid = true; // false if scoring failed (fallback to priority)
};

/// @brief Selection trace for observability (RFC 0019 §13).
///
/// Contains model provenance, selection path details, and diagnostic information.
struct SelectionTrace
{
    // Model provenance
    std::string uhdId;
    std::string modelVersion;
    std::vector<std::string> trainingArches;

    // Selection path
    std::string adapterType; // "tree_data", "static_order", etc.
    bool usedModel = false; // true if model scored, false if fallback
    std::string fallbackReason; // if usedModel==false, why

    // Arch validation (RFC 0019 §9.3)
    bool archWasTrained = true; // false if device arch not in training set
    std::string deviceArch; // the device arch being checked

    // Feature contract
    std::string featuresHashModel; // hash from model artifact
    std::string featuresHashConfig; // hash from UHD config
    bool featuresHashMatch = true;
};

/// @brief Result of the UHD selection process.
struct SelectionResult
{
    /// True when selection ran to completion without degrading. Note this includes the
    /// engine that registered no candidates: nothing to rank is a trivially complete
    /// selection, not a failure. Use `trace.usedModel` to ask the narrower question of
    /// whether the model actually scored anything.
    bool applied = false;
    std::vector<int64_t> sortedKernelIds; // kernels sorted by score (best first)
    std::vector<ScoredCandidate> scoredCandidates; // full scoring details

    // Best candidate info (for score-only mode, RFC §12.1)
    std::optional<double> bestScore;
    std::optional<int64_t> bestKernelId;

    // Fallback reason (for diagnostics)
    std::string fallbackReason;

    // Observability trace (RFC 0019 §13)
    SelectionTrace trace;

    /// True when selection completed without ruling the engine out — either the model
    /// ran, or the static_order fallback produced a ranking.
    ///
    /// RFC 0019 §6 step 6 requires failing *open*: a model that is absent, mismatched,
    /// or throwing degrades to priority+id ordering, and the engine stays in play.
    /// Callers must gate on this rather than on `applied`, which is narrower and only
    /// reports whether the model itself ran.
    ///
    /// The `applied` disjunct covers the engine that registered no candidates: there
    /// is nothing to rank, but selection did not fail, so the engine is not dropped.
    bool hasOrdering() const
    {
        return applied || !sortedKernelIds.empty();
    }
};

// score_transform now lives in ScoreTransform.hpp so EngineRegistry can validate a
// descriptor's declared transform at load without depending on the selection engine.

/// @brief UHD selection engine.
///
/// Implements the selection flow from RFC §6:
/// 1. Extract features from device/kernel/query context
/// 2. Score each candidate using the UHD adapter
/// 3. Sort by objective (max/min)
/// 4. Apply deterministic tie-breaking (priority, then id)
/// 5. Fall back to static ordering on any error
class SelectionEngine
{
public:
    /// Perform kernel selection for an engine.
    ///
    /// @param engineId Engine to select kernels for.
    /// @param deviceVars Device properties ($device.*).
    /// @param queryVars Problem/query properties ($q.*).
    /// @returns Selection result with sorted kernel IDs.
    ///
    /// The kernel metadata ($kernel.*) is obtained from the engine's
    /// registered candidates. Device and query vars must be provided
    /// by the caller (from handle and graph).
    static SelectionResult select(int64_t engineId,
                                  const FeatureExtractionContext::ValueMap& deviceVars,
                                  const FeatureExtractionContext::ValueMap& queryVars);

    /// Score-only mode: get best candidate and score without full sorting.
    /// Used for engine-level comparison (RFC §12.1 model A).
    static SelectionResult scoreOnly(int64_t engineId,
                                     const FeatureExtractionContext::ValueMap& deviceVars,
                                     const FeatureExtractionContext::ValueMap& queryVars);

private:
    /// Build the $kernel.* bindings for a candidate from its KMD metadata, plus the
    /// implicit `priority` and `id` fields every candidate carries.
    static FeatureExtractionContext::ValueMap buildKernelVars(const KernelCandidate& candidate);

    /// Rank candidates by a declared field order (RFC 0019 §5 `static_order`).
    ///
    /// A comparator, not a scorer. Fields are compared pairwise in declaration order,
    /// so there is no packed key to overflow and no features_signature to resolve —
    /// `static_order` carries "no features, no model, no hash" per §5. Callers set
    /// `applied`/`usedModel`: this is the model for a static_order UHD, and the
    /// fail-open path for everything else.
    ///
    /// @param orderFields Field names, most significant first. Empty means
    ///        priority then id, the §6 step 6 default.
    static SelectionResult applyDeclaredOrder(const std::vector<KernelCandidate>& candidates,
                                              const std::vector<std::string>& orderFields);

    /// Resolve one order-field for a candidate. `priority` and `id` are implicit on
    /// every candidate; any other name comes from its KMD metadata. Accepts both the
    /// bare (`tile_m`) and namespaced (`$kernel.tile_m`) spelling.
    /// @returns nullopt when the candidate does not carry the field.
    static std::optional<double> lookupOrderField(const KernelCandidate& candidate,
                                                  const std::string& field);

    /// Apply static ordering fallback (RFC 0019 §6 step 6): priority then id.
    static SelectionResult applyStaticOrdering(const std::vector<KernelCandidate>& candidates,
                                               const std::string& reason);

    /// Sort scored candidates by objective and tie-break.
    static void sortByObjective(std::vector<ScoredCandidate>& candidates,
                                const std::string& objective);
};

} // namespace hipdnn_backend::heuristics::uhd
