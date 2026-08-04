// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "EngineRegistry.hpp"
#include "FeatureExtractor.hpp"
#include "adapters/IUhdAdapter.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Result of scoring a single kernel candidate.
struct ScoredCandidate
{
    int64_t kernelId;
    int64_t priority;
    double score;
    bool scoreValid = true; // false if scoring failed (fallback to priority)
};

/// @brief Result of the UHD selection process.
struct SelectionResult
{
    bool applied = false;                          // true if UHD successfully ranked
    std::vector<int64_t> sortedKernelIds;          // kernels sorted by score (best first)
    std::vector<ScoredCandidate> scoredCandidates; // full scoring details

    // Best candidate info (for score-only mode, RFC §12.1)
    std::optional<double> bestScore;
    std::optional<int64_t> bestKernelId;

    // Fallback reason (for diagnostics)
    std::string fallbackReason;
};

/// @brief Score transform utilities (RFC §5, §12.3).
///
/// Models may be trained on transformed targets (e.g., log1p(tflops)).
/// These utilities recover the original scale for cross-engine comparison.
namespace ScoreTransform
{

/// Apply inverse transform to recover original scale.
/// @param rawScore Score from the model.
/// @param transform Transform name from UhdConfig::scoreTransform.
/// @returns Transformed score in original units.
inline double applyInverse(double rawScore, const std::string& transform)
{
    if(transform == "log1p")
    {
        // Inverse of log1p is expm1
        return std::expm1(rawScore);
    }
    if(transform == "log")
    {
        return std::exp(rawScore);
    }
    if(transform == "sqrt")
    {
        return rawScore * rawScore;
    }
    // "identity" or unknown: no transform
    return rawScore;
}

/// Apply forward transform (for training/debugging).
inline double applyForward(double value, const std::string& transform)
{
    if(transform == "log1p")
    {
        return std::log1p(value);
    }
    if(transform == "log")
    {
        return std::log(value);
    }
    if(transform == "sqrt")
    {
        return std::sqrt(value);
    }
    return value;
}

} // namespace ScoreTransform

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
    /// Build feature vector for a candidate.
    static std::vector<double> buildFeatureVector(const FeatureExtractor& extractor,
                                                  const FeatureExtractionContext& ctx,
                                                  const KernelCandidate& candidate);

    /// Apply static ordering fallback.
    static SelectionResult applyStaticOrdering(const std::vector<KernelCandidate>& candidates,
                                               const std::string& reason);

    /// Sort scored candidates by objective and tie-break.
    static void sortByObjective(std::vector<ScoredCandidate>& candidates,
                                const std::string& objective);
};

} // namespace hipdnn_backend::heuristics::uhd
