// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SelectionEngine.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace hipdnn_backend::heuristics::uhd
{

SelectionResult SelectionEngine::select(int64_t engineId,
                                        const FeatureExtractionContext::ValueMap& deviceVars,
                                        const FeatureExtractionContext::ValueMap& queryVars)
{
    SelectionResult result;

    // Look up engine in registry. Hold the snapshot by shared_ptr for the whole
    // selection: a concurrent re-registration (RFC 0019 §9.2 descriptor replacement)
    // swaps the registry's slot, and this keeps the entry we started with alive and
    // internally consistent rather than reading one that is being replaced.
    const auto enginePtr = EngineRegistry::instance().getEngine(engineId);
    if(enginePtr == nullptr)
    {
        result.fallbackReason = "engine not found in registry";
        result.trace.fallbackReason = result.fallbackReason;
        return result;
    }

    const EngineEntry& engine = *enginePtr;

    // RFC 0019 §3.1 + §8.3: Resolve arch-keyed UHD for sort_kernel_catalog role
    // Extract arch from deviceVars (try "arch", fallback to "default")
    std::string arch = "default";
    auto archIt = deviceVars.find(DEVICE_ARCH_KEY);
    if(archIt != deviceVars.end())
    {
        if(const auto* archStr = std::get_if<std::string>(&archIt->second))
        {
            arch = *archStr;
        }
    }

    // Resolve UHD by arch (tries exact match, then "default", then nullopt)
    auto uhdOpt = engine.resolveSortKernelCatalog(arch);
    if(!uhdOpt.has_value())
    {
        // Fall back to legacy uhdConfig if no role-scoped UHD found
        if(!engine.uhdConfig.uhdId.empty())
        {
            uhdOpt = engine.uhdConfig;
        }
        else
        {
            result.fallbackReason = "no sort_kernel_catalog UHD for arch '" + arch + "'";
            result.trace.fallbackReason = result.fallbackReason;
            return result;
        }
    }

    const UhdConfig& cfg = uhdOpt.value();

    // Populate trace with UHD config (RFC 0019 §13)
    result.trace.uhdId = cfg.uhdId;
    result.trace.adapterType = cfg.adapterType;
    result.trace.featuresHashConfig = cfg.featuresHash;

    // No candidates? Nothing to select.
    if(engine.candidates.empty())
    {
        result.applied = true;
        result.fallbackReason = "no candidates";
        result.trace.fallbackReason = result.fallbackReason;
        return result;
    }

    // static_order is a declared precedence, not a model (RFC 0019 §5: "no features,
    // no model, no hash"). Rank with the comparator directly — no adapter, no feature
    // extraction, no signature. Building a scorer for it forced a packed scalar key
    // that could not represent a lexicographic order, and required a
    // features_signature the RFC says static_order does not carry.
    if(cfg.adapterType == "static_order")
    {
        auto ordered = applyDeclaredOrder(engine.candidates, cfg.staticOrderFields);
        ordered.trace = result.trace;
        ordered.applied = true;
        ordered.trace.usedModel = true; // the declared order is the model
        return ordered;
    }

    // Every fallback below routes through here so the provenance gathered so far
    // (uhdId, adapterType, hashes, arch) survives into the returned trace. Returning
    // applyStaticOrdering() directly would hand back a default-constructed trace and
    // report an empty uhdId for exactly the failures worth diagnosing.
    const auto degrade = [&result, &engine](const std::string& reason) {
        auto fallback = applyStaticOrdering(engine.candidates, reason);
        fallback.trace = result.trace;
        fallback.trace.usedModel = false;
        fallback.trace.fallbackReason = reason;
        return fallback;
    };

    // Get or create adapter for this role+arch
    // Resolve from the snapshot, not by ID. Going back through the map would let a
    // re-registration landing mid-selection pair a new model with this snapshot's
    // config and candidates — and the mismatch is silent, since `objective` and
    // `score.transform` are read from the old config while the score comes from the
    // new model.
    auto adapter = EngineRegistry::instance().getOrCreateAdapter(enginePtr, cfg,
                                                                   "sort_kernel_catalog", arch);
    if(adapter == nullptr)
    {
        HIPDNN_SDK_LOG_WARN("UHD: engine " << engineId << " uhd='" << cfg.uhdId << "' adapter '"
                                           << cfg.adapterType
                                           << "' could not be created; using static order");
        return degrade("adapter creation failed");
    }

    // Populate trace with adapter info (RFC 0019 §9.3, §13)
    result.trace.featuresHashModel = adapter->getFeaturesHash();
    result.trace.modelVersion = adapter->getModelVersion();
    result.trace.trainingArches = adapter->getTrainingArches();

    // Check if device arch was seen during training (RFC 0019 §9.3).
    // A model trained only on gfx942 has no basis for ranking gfx950, so an
    // out-of-distribution device degrades rather than extrapolating.
    // Note: arch already extracted above, reuse the variable
    if(!arch.empty() && arch != "default")
    {
        result.trace.deviceArch = arch;
        result.trace.archWasTrained = adapter->isTrainedForArch(arch);

        if(!result.trace.archWasTrained)
        {
            HIPDNN_SDK_LOG_WARN("UHD: engine "
                                << engineId << " uhd='" << cfg.uhdId << "' model version='"
                                << adapter->getModelVersion() << "' was not trained for arch '"
                                << arch << "'; using static order");
            return degrade("device arch '" + arch + "' not in training set");
        }
    }

    // Get cached feature extractor for this role+arch (or create on first use)
    auto extractor = EngineRegistry::instance().getOrCreateExtractor(enginePtr, cfg,
                                                                       "sort_kernel_catalog", arch);
    if(extractor == nullptr)
    {
        HIPDNN_SDK_LOG_WARN("UHD: engine " << engineId << " uhd='" << cfg.uhdId
                                           << "' feature extractor could not be built; "
                                              "using static order");
        return degrade("failed to create feature extractor");
    }

    // Validate features hash if adapter provides one
    const std::string& adapterHash = adapter->getFeaturesHash();
    if(!adapterHash.empty() && !cfg.featuresHash.empty() && adapterHash != cfg.featuresHash)
    {
        result.trace.featuresHashMatch = false;
        return degrade("features hash mismatch: adapter=" + adapterHash
                       + " config=" + cfg.featuresHash);
    }

    // Build the context once and reuse it. Device and query bindings are shared by
    // every candidate; only the $kernel.* namespace is rebound per candidate.
    FeatureExtractionContext ctx;
    ctx.bindDeviceVars(deviceVars);
    ctx.bindQueryVars(queryVars);

    // RFC 0019 §6 step 2: extract shared features once, not once per candidate.
    // sharedRow holds the $device.*/$q.* slots; each candidate overwrites only the
    // $kernel.*-dependent slots.
    std::vector<double> sharedRow;
    try
    {
        sharedRow = extractor->extractSharedRow(ctx);
    }
    catch(const std::exception& e)
    {
        HIPDNN_SDK_LOG_WARN("UHD: engine " << engineId << " uhd='" << cfg.uhdId
                                           << "' shared feature extraction failed: " << e.what()
                                           << "; using static order");
        return degrade(std::string("shared feature extraction failed: ") + e.what());
    }

    // The feature vector width is fixed by the signature, so the adapter's arity
    // contract (RFC 0019 §7.3c) can be checked once rather than per candidate. Doing
    // it here also means a mismatch reports itself instead of masquerading as
    // "all candidates failed scoring".
    if(!adapter->validateFeatureCount(sharedRow.size()))
    {
        return degrade("feature count mismatch: signature has " + std::to_string(sharedRow.size())
                       + " features, model expects "
                       + std::to_string(adapter->expectedFeatureCount()));
    }

    // Score each candidate
    std::vector<ScoredCandidate> scored;
    scored.reserve(engine.candidates.size());

    std::vector<double> features;
    for(const auto& candidate : engine.candidates)
    {
        ScoredCandidate sc;
        sc.kernelId = candidate.kernelId;
        sc.priority = candidate.priority;

        try
        {
            // Rebind only this candidate's metadata, then refresh the kernel slots.
            // clearKernelVars() matters: without it a candidate whose metadata omits a
            // field would silently inherit the previous candidate's value.
            ctx.clearKernelVars();
            ctx.bindKernelVars(buildKernelVars(candidate));

            features = sharedRow;
            extractor->extractKernelInto(ctx, features);

            const double score = adapter->score(features);

            // A non-finite score is not a ranking. Treating it as one is worse than
            // dropping the candidate: NaN compares false against everything, so the
            // sort comparator would find NaN "equivalent" to two values that are not
            // equivalent to each other, breaking strict weak ordering and making
            // std::sort undefined. NaN can propagate from ops like pow(), though
            // string-valued bindings now throw (RFC 0019 §7.2 type-error enforcement)
            // rather than yielding quiet_NaN.
            if(!std::isfinite(score))
            {
                sc.scoreValid = false;
                sc.score = 0.0;
            }
            else
            {
                sc.score = score;
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
    const bool anyValid = std::any_of(
        scored.begin(), scored.end(), [](const ScoredCandidate& s) { return s.scoreValid; });

    if(!anyValid)
    {
        return degrade("all candidates failed scoring");
    }

    // Sort by objective (max or min) with tie-breaking
    sortByObjective(scored, cfg.objective);

    // Build result
    result.applied = true;
    result.scoredCandidates = scored;
    // TODO(RFC-0019): sortedKernelIds duplicates the kernelId column of
    // scoredCandidates. Keep both while the consumer API is in flux (§12.1 score-only
    // mode wants only the best; §6 wants the full ranking); collapse to one once
    // RFC 0007 fixes the shape it needs.
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

        // Recover the original scale (RFC 0019 §5, §12.3). `transform` and `calibrated`
        // are orthogonal: `transform` says the model was trained on a transformed
        // target and must be inverted to report the declared `units`, while
        // `calibrated` says the recovered value is comparable across engines. Gating
        // the inverse on `calibrated` would report a log-space number as if it were
        // tflops whenever a model is transformed but uncalibrated.
        if(!cfg.scoreTransform.empty())
        {
            result.bestScore = score_transform::applyInverse(*result.bestScore, cfg.scoreTransform);
        }
    }

    // Mark trace as using model (RFC 0019 §13)
    result.trace.usedModel = true;

    return result;
}

SelectionResult SelectionEngine::scoreOnly(int64_t engineId,
                                           const FeatureExtractionContext::ValueMap& deviceVars,
                                           const FeatureExtractionContext::ValueMap& queryVars)
{
    // TODO(RFC-0019 §12.1): score-only mode is meant to be the *cheap* engine estimate
    // (model A) — a single f(graph) prediction that never enumerates candidates. This
    // delegates to full selection, so it still scores every candidate and sorts them.
    // Correct results, wrong cost profile. Replacing this needs the distinct model A
    // artifact, which lands with Phase 7 alongside RFC 0007.
    return select(engineId, deviceVars, queryVars);
}

FeatureExtractionContext::ValueMap
    SelectionEngine::buildKernelVars(const KernelCandidate& candidate)
{
    FeatureExtractionContext::ValueMap kernelVars;
    kernelVars.reserve(candidate.metadata.size() + 2);

    for(const auto& [key, value] : candidate.metadata)
    {
        kernelVars[key] = value;
    }

    // priority and id are implicit kernel fields every candidate carries, so
    // static_order signatures can reference them without KMD declarations.
    kernelVars["priority"] = static_cast<double>(candidate.priority);
    kernelVars["id"] = static_cast<double>(candidate.kernelId);

    return kernelVars;
}

std::optional<double> SelectionEngine::lookupOrderField(const KernelCandidate& candidate,
                                                        const std::string& field)
{
    // Order fields may be written bare or namespaced; both name the same thing.
    const std::string bare
        = field.rfind("$kernel.", 0) == 0 ? field.substr(std::string("$kernel.").size()) : field;

    // priority and id are implicit on every candidate, so a static_order UHD can name
    // them without the engine declaring any KMD metadata.
    if(bare == "priority")
    {
        return static_cast<double>(candidate.priority);
    }
    if(bare == "id")
    {
        return static_cast<double>(candidate.kernelId);
    }

    const auto it = candidate.metadata.find(bare);
    if(it == candidate.metadata.end())
    {
        return std::nullopt;
    }
    return it->second;
}

SelectionResult SelectionEngine::applyDeclaredOrder(const std::vector<KernelCandidate>& candidates,
                                                    const std::vector<std::string>& orderFields)
{
    SelectionResult result;

    static const std::vector<std::string> s_defaultOrder = {"priority", "id"};
    const std::vector<std::string>& fields = orderFields.empty() ? s_defaultOrder : orderFields;

    // Pre-resolve each candidate's key so the comparator does no lookups. A missing
    // field sorts after a present one, which keeps the order total and deterministic
    // without dropping the candidate.
    struct Keyed
    {
        const KernelCandidate* candidate;
        std::vector<std::optional<double>> key;
    };

    std::vector<Keyed> keyed;
    keyed.reserve(candidates.size());
    for(const auto& candidate : candidates)
    {
        Keyed k{&candidate, {}};
        k.key.reserve(fields.size());
        for(const auto& field : fields)
        {
            k.key.push_back(lookupOrderField(candidate, field));
        }
        keyed.push_back(std::move(k));
    }

    std::sort(keyed.begin(), keyed.end(), [](const Keyed& a, const Keyed& b) {
        for(size_t i = 0; i < a.key.size(); ++i)
        {
            const auto& lhs = a.key[i];
            const auto& rhs = b.key[i];
            if(lhs.has_value() != rhs.has_value())
            {
                return lhs.has_value(); // present before missing
            }
            if(lhs.has_value() && *lhs != *rhs)
            {
                return *lhs < *rhs; // lower value ranks first
            }
        }
        // Stable final arbitration; RFC 0019 §6 step 5 forbids declaration order.
        return a.candidate->kernelId < b.candidate->kernelId;
    });

    result.scoredCandidates.reserve(keyed.size());
    result.sortedKernelIds.reserve(keyed.size());
    for(const auto& k : keyed)
    {
        ScoredCandidate sc;
        sc.kernelId = k.candidate->kernelId;
        sc.priority = k.candidate->priority;
        // A declared order produces no score. Leaving scoreValid false keeps
        // bestScore unset, so a static_order engine correctly declines to take part
        // in the §12.3 cross-engine score comparison rather than inventing a number.
        sc.score = 0.0;
        sc.scoreValid = false;
        result.scoredCandidates.push_back(sc);
        result.sortedKernelIds.push_back(sc.kernelId);
    }

    if(!result.sortedKernelIds.empty())
    {
        result.bestKernelId = result.sortedKernelIds.front();
    }

    return result;
}

SelectionResult SelectionEngine::applyStaticOrdering(const std::vector<KernelCandidate>& candidates,
                                                     const std::string& reason)
{
    // The fail-open default is the same comparator with the default field order, so
    // §6 step 6 and the static_order adapter are literally one code path.
    //
    // Unrelated to the `SelectionHeuristic::StaticOrdering` policy despite the name:
    // that one ranks engines by a fixed vendor precedence (RFC 0007). This ranks
    // kernels within a single engine. See the default-chain comment in
    // EngineHeuristicDescriptor::resolveHeuristicPolicyOrder.
    auto result = applyDeclaredOrder(candidates, {});
    result.applied = false;
    result.fallbackReason = reason;
    return result;
}

void SelectionEngine::sortByObjective(std::vector<ScoredCandidate>& candidates,
                                      const std::string& objective)
{
    const bool maximize = (objective != "min");

    std::sort(candidates.begin(),
              candidates.end(),
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
