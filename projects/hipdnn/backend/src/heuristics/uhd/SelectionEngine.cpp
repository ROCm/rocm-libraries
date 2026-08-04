// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SelectionEngine.hpp"

#include <algorithm>
#include <limits>

namespace hipdnn_backend::heuristics::uhd
{

SelectionResult SelectionEngine::select(int64_t engineId,
                                        const FeatureExtractionContext::ValueMap& deviceVars,
                                        const FeatureExtractionContext::ValueMap& queryVars)
{
    SelectionResult result;

    // Look up engine in registry
    const auto engineOpt = EngineRegistry::instance().getEngine(engineId);
    if(!engineOpt.has_value())
    {
        result.fallbackReason = "engine not found in registry";
        return result;
    }

    const EngineEntry& engine = engineOpt->get();
    const UhdConfig& cfg = engine.uhdConfig;

    // No candidates? Nothing to select.
    if(engine.candidates.empty())
    {
        result.applied = true;
        result.fallbackReason = "no candidates";
        return result;
    }

    // Get or create adapter
    auto adapter = EngineRegistry::instance().getOrCreateAdapter(engineId);
    if(adapter == nullptr)
    {
        return applyStaticOrdering(engine.candidates, "adapter creation failed");
    }

    // Build feature extractor from signature
    FeatureExtractor extractor(cfg.featuresSignature);

    // Validate features hash if adapter provides one
    const std::string& adapterHash = adapter->getFeaturesHash();
    if(!adapterHash.empty() && !cfg.featuresHash.empty() && adapterHash != cfg.featuresHash)
    {
        return applyStaticOrdering(engine.candidates,
                                   "features hash mismatch: adapter=" + adapterHash +
                                       " config=" + cfg.featuresHash);
    }

    // Build base context with device and query vars
    FeatureExtractionContext baseCtx;
    baseCtx.bindDeviceVars(deviceVars);
    baseCtx.bindQueryVars(queryVars);

    // Score each candidate
    std::vector<ScoredCandidate> scored;
    scored.reserve(engine.candidates.size());

    for(const auto& candidate : engine.candidates)
    {
        ScoredCandidate sc;
        sc.kernelId = candidate.kernelId;
        sc.priority = candidate.priority;

        try
        {
            // Build feature vector with this candidate's kernel metadata
            auto features = buildFeatureVector(extractor, baseCtx, candidate);

            // Validate feature count
            if(!adapter->validateFeatureCount(features.size()))
            {
                sc.scoreValid = false;
                sc.score = 0.0;
            }
            else
            {
                sc.score = adapter->score(features);
                sc.scoreValid = true;
            }
        }
        catch(const std::exception&)
        {
            // Scoring failed for this candidate; mark invalid
            sc.scoreValid = false;
            sc.score = 0.0;
        }

        scored.push_back(sc);
    }

    // Check if any candidates were scored successfully
    const bool anyValid =
        std::any_of(scored.begin(), scored.end(), [](const ScoredCandidate& s) { return s.scoreValid; });

    if(!anyValid)
    {
        return applyStaticOrdering(engine.candidates, "all candidates failed scoring");
    }

    // Sort by objective (max or min) with tie-breaking
    sortByObjective(scored, cfg.objective);

    // Build result
    result.applied = true;
    result.scoredCandidates = scored;
    result.sortedKernelIds.reserve(scored.size());
    for(const auto& s : scored)
    {
        result.sortedKernelIds.push_back(s.kernelId);
    }

    // Best candidate info
    if(!scored.empty() && scored[0].scoreValid)
    {
        result.bestKernelId = scored[0].kernelId;
        result.bestScore = scored[0].score;

        // Apply inverse transform if configured (for cross-engine comparison)
        if(cfg.scoreCalibrated && !cfg.scoreTransform.empty())
        {
            result.bestScore = ScoreTransform::applyInverse(*result.bestScore, cfg.scoreTransform);
        }
    }

    return result;
}

SelectionResult SelectionEngine::scoreOnly(int64_t engineId,
                                           const FeatureExtractionContext::ValueMap& deviceVars,
                                           const FeatureExtractionContext::ValueMap& queryVars)
{
    // For now, score-only uses the same path as full selection.
    // A future optimization could skip sorting and just find the best.
    return select(engineId, deviceVars, queryVars);
}

std::vector<double> SelectionEngine::buildFeatureVector(const FeatureExtractor& extractor,
                                                        const FeatureExtractionContext& baseCtx,
                                                        const KernelCandidate& candidate)
{
    // Create a copy of the base context and add kernel metadata
    FeatureExtractionContext ctx;

    // Re-bind device and query vars (copying from base)
    // Note: This is a workaround since VariableContext doesn't expose iteration.
    // In production, we'd want a more efficient approach.
    ctx = baseCtx;

    // Bind kernel metadata
    FeatureExtractionContext::ValueMap kernelVars;
    for(const auto& [key, value] : candidate.metadata)
    {
        kernelVars[key] = value;
    }
    // Also add priority and id as implicit kernel fields
    kernelVars["priority"] = static_cast<double>(candidate.priority);
    kernelVars["id"] = static_cast<double>(candidate.kernelId);

    ctx.bindKernelVars(kernelVars);

    // Extract features
    return extractor.extract(ctx);
}

SelectionResult
    SelectionEngine::applyStaticOrdering(const std::vector<KernelCandidate>& candidates,
                                         const std::string& reason)
{
    SelectionResult result;
    result.applied = false;
    result.fallbackReason = reason;

    // Sort by priority (lower is better), then by id (lower is better)
    std::vector<ScoredCandidate> scored;
    scored.reserve(candidates.size());
    for(const auto& c : candidates)
    {
        ScoredCandidate sc;
        sc.kernelId = c.kernelId;
        sc.priority = c.priority;
        sc.score = 0.0;
        sc.scoreValid = false;
        scored.push_back(sc);
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredCandidate& a, const ScoredCandidate& b) {
        if(a.priority != b.priority)
        {
            return a.priority < b.priority;
        }
        return a.kernelId < b.kernelId;
    });

    result.scoredCandidates = scored;
    result.sortedKernelIds.reserve(scored.size());
    for(const auto& s : scored)
    {
        result.sortedKernelIds.push_back(s.kernelId);
    }

    if(!scored.empty())
    {
        result.bestKernelId = scored[0].kernelId;
    }

    return result;
}

void SelectionEngine::sortByObjective(std::vector<ScoredCandidate>& candidates,
                                      const std::string& objective)
{
    const bool maximize = (objective != "min");

    std::sort(candidates.begin(), candidates.end(),
              [maximize](const ScoredCandidate& a, const ScoredCandidate& b) {
                  // Invalid scores sort to the end
                  if(a.scoreValid != b.scoreValid)
                  {
                      return a.scoreValid; // valid before invalid
                  }
                  if(!a.scoreValid && !b.scoreValid)
                  {
                      // Both invalid: fall through to tie-break
                  }
                  else if(a.score != b.score)
                  {
                      // Sort by score (higher first for max, lower first for min)
                      return maximize ? (a.score > b.score) : (a.score < b.score);
                  }

                  // Tie-break: lower priority first, then lower id
                  if(a.priority != b.priority)
                  {
                      return a.priority < b.priority;
                  }
                  return a.kernelId < b.kernelId;
              });
}

} // namespace hipdnn_backend::heuristics::uhd
