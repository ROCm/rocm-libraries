// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "EngineRegistry.hpp"

#include "FeatureExtractor.hpp"
#include "ScoreTransform.hpp"
#include "adapters/TreeDataAdapter.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace hipdnn_backend::heuristics::uhd
{

EngineRegistry& EngineRegistry::instance()
{
    static EngineRegistry s_instance;
    return s_instance;
}

void EngineRegistry::registerEngine(EngineEntry entry)
{
    // Validate KMD field coverage: every $kernel.* in features_signature
    // must exist in at least one candidate's metadata (RFC 0019 §7.3).
    if(!entry.uhdConfig.featuresSignature.empty() && !entry.candidates.empty())
    {
        // Collect all metadata keys across candidates (the KMD field space)
        std::unordered_set<std::string> kmdFields;
        for(const auto& candidate : entry.candidates)
        {
            for(const auto& [key, _] : candidate.metadata)
            {
                kmdFields.insert(key);
            }
        }
        // Also include implicit kernel fields that SelectionEngine adds
        kmdFields.insert("priority");
        kmdFields.insert("id");

        // Build extractor to parse signature and extract $kernel.* refs.
        // A malformed entry throws JsonLogicError; re-raise as invalid_argument so the
        // exception type matches what registerEngine documents.
        std::vector<std::string> missingFields;
        try
        {
            const FeatureExtractor extractor(entry.uhdConfig.featuresSignature);
            missingFields = extractor.getMissingKmdFields(kmdFields);
        }
        catch(const JsonLogicError& e)
        {
            std::ostringstream oss;
            oss << "UHD features_signature contains an entry that is neither a bare $ref "
                << "nor valid JsonLogic. Engine ID: " << entry.engineId << ", uhd='"
                << entry.uhdConfig.uhdId << "': " << e.what();
            throw std::invalid_argument(oss.str());
        }

        if(!missingFields.empty())
        {
            std::ostringstream oss;
            oss << "UHD features_signature references $kernel.* fields not present in "
                << "candidate metadata: ";
            for(size_t i = 0; i < missingFields.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << missingFields[i];
            }
            oss << ". Engine ID: " << entry.engineId
                << ". Ensure all $kernel.* fields in features_signature are present in candidate "
                   "metadata.";
            throw std::invalid_argument(oss.str());
        }
    }

    validateFeaturesHash(entry);
    validateScoreTransform(entry);
    validateObjective(entry);

    const int64_t engineId = entry.engineId;

    const std::lock_guard<std::mutex> lock(_mutex);
    // Replace the slot rather than assigning through it: a selection already holding
    // the previous snapshot keeps reading a consistent entry until it finishes.
    _engines[engineId] = std::make_shared<EngineEntry>(std::move(entry));
}

void EngineRegistry::validateObjective(const EngineEntry& entry)
{
    const UhdConfig& cfg = entry.uhdConfig;

    // RFC 0019 §5 and §6 step 4 name exactly two values. sortByObjective computes
    // `maximize = (objective != "min")`, so anything unrecognized — a typo, a case
    // variant, "minimize" — silently maximizes. For a min-objective model that
    // inverts the ranking with no diagnostic, which is the same class of silent
    // corruption the score.transform check exists to prevent.
    if(!cfg.objective.empty() && cfg.objective != "max" && cfg.objective != "min")
    {
        std::ostringstream oss;
        oss << "UHD declares an unrecognized objective '" << cfg.objective
            << "'. Engine ID: " << entry.engineId << ", uhd='" << cfg.uhdId
            << "'. Supported: \"max\", \"min\" (empty defaults to max).";
        throw std::invalid_argument(oss.str());
    }
}

void EngineRegistry::validateScoreTransform(const EngineEntry& entry)
{
    const UhdConfig& cfg = entry.uhdConfig;

    // A transform this runtime cannot invert means the score cannot be reported in the
    // units the descriptor declares. Rejecting at load is the only honest option:
    // score_transform::applyInverse has no way to signal "unknown" at the point of
    // use, so an unrecognized name passes the model's transformed output straight
    // through as if it were already in the declared units. RFC 0019 §12.3 feeds
    // exactly that number into cross-engine comparison, where a wrong scale silently
    // corrupts engine selection.
    if(!score_transform::isSupported(cfg.scoreTransform))
    {
        std::ostringstream oss;
        oss << "UHD declares an unsupported score.transform '" << cfg.scoreTransform
            << "'. Engine ID: " << entry.engineId << ", uhd='" << cfg.uhdId
            << "'. Supported: " << score_transform::supportedTransformList() << ".";
        throw std::invalid_argument(oss.str());
    }
}

void EngineRegistry::validateFeaturesHash(const EngineEntry& entry)
{
    const UhdConfig& cfg = entry.uhdConfig;

    // A declared hash must describe the signature it ships with (RFC 0019 §7.3). The
    // load-time check in SelectionEngine only compares the model's embedded hash to
    // this one; without this check both could agree while neither matches the
    // signature actually being evaluated, which defeats the point of fingerprinting
    // the feature contract.
    //
    // Checked even for an empty signature: a descriptor declaring a hash over no
    // features is self-inconsistent, and skipping it here would let that through.
    if(!cfg.featuresHash.empty())
    {
        std::string actual;
        try
        {
            actual = FeatureExtractor::computeHash(cfg.featuresSignature);
        }
        catch(const JsonLogicError& e)
        {
            // Surface a malformed signature entry as invalid_argument, matching what
            // registerEngine documents, rather than leaking JsonLogicError to a caller
            // that only catches invalid_argument.
            std::ostringstream oss;
            oss << "UHD features_signature contains an entry that is neither a bare $ref "
                << "nor valid JsonLogic. Engine ID: " << entry.engineId << ", uhd='" << cfg.uhdId
                << "': " << e.what();
            throw std::invalid_argument(oss.str());
        }

        if(actual != cfg.featuresHash)
        {
            std::ostringstream oss;
            oss << "UHD features_hash does not describe its own features_signature. "
                << "Engine ID: " << entry.engineId << ", uhd='" << cfg.uhdId
                << "', declared=" << cfg.featuresHash << ", computed=" << actual
                << ". The signature and hash must be emitted together — regenerate the "
                   "descriptor rather than editing either by hand.";
            throw std::invalid_argument(oss.str());
        }
        return;
    }

    if(cfg.featuresSignature.empty())
    {
        return;
    }

    // Feature-bearing adapters are required to carry a hash (RFC 0019 §7.3). Absent
    // one there is nothing to check the model artifact against, so the mismatch guard
    // in SelectionEngine silently passes.
    if(cfg.adapterType == "tree_data" || cfg.adapterType == "onnx" || cfg.adapterType == "table")
    {
        HIPDNN_SDK_LOG_WARN("UHD: engine "
                            << entry.engineId << " uhd='" << cfg.uhdId << "' adapter '"
                            << cfg.adapterType
                            << "' declares a features_signature but no features_hash; the "
                               "feature contract with the model artifact cannot be enforced");
    }
}

std::shared_ptr<const EngineEntry> EngineRegistry::getEngine(int64_t engineId) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _engines.find(engineId);
    if(it == _engines.end())
    {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<IUhdAdapter>
    EngineRegistry::getOrCreateAdapter(const std::shared_ptr<const EngineEntry>& entry) const
{
    if(entry == nullptr)
    {
        return nullptr;
    }

    // The cache members are `mutable`, so they can be filled through a const entry.
    // The registry mutex still guards them: every read and write of cachedAdapter
    // goes through this function or its by-ID overload.
    const std::lock_guard<std::mutex> lock(_mutex);

    // Return cached adapter if available
    if(entry->cachedAdapter != nullptr)
    {
        return entry->cachedAdapter;
    }

    // Create adapter based on type
    const auto& cfg = entry->uhdConfig;

    // static_order never reaches here: SelectionEngine ranks it with the declared-order
    // comparator instead of building a scorer.
    if(cfg.adapterType == "tree_data")
    {
        // TreeDataAdapter loads from model file
        if(!cfg.modelArtifactPath.empty())
        {
            auto adapter = TreeDataAdapter::load(cfg.modelArtifactPath, cfg.featuresHash);
            entry->cachedAdapter = std::move(adapter);
        }
    }
    // TODO: Add table, onnx, custom_library adapters when implemented

    return entry->cachedAdapter;
}

std::shared_ptr<IUhdAdapter> EngineRegistry::getOrCreateAdapter(int64_t engineId) const
{
    return getOrCreateAdapter(getEngine(engineId));
}

bool EngineRegistry::hasEngine(int64_t engineId) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _engines.find(engineId) != _engines.end();
}

size_t EngineRegistry::size() const
{
    // Was reading _engines unlocked, which races registerEngine's insert and clear().
    const std::lock_guard<std::mutex> lock(_mutex);
    return _engines.size();
}

std::vector<int64_t> EngineRegistry::getAllEngineIds() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    std::vector<int64_t> ids;
    ids.reserve(_engines.size());
    for(const auto& [id, _] : _engines)
    {
        ids.push_back(id);
    }
    return ids;
}

void EngineRegistry::clear()
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _engines.clear();
}

std::shared_ptr<FeatureExtractor>
    EngineRegistry::getOrCreateExtractor(const std::shared_ptr<const EngineEntry>& entry) const
{
    if(entry == nullptr)
    {
        return nullptr;
    }

    const std::lock_guard<std::mutex> lock(_mutex);

    // Return cached extractor if available
    if(entry->cachedExtractor != nullptr)
    {
        return entry->cachedExtractor;
    }

    // Create extractor from features signature if non-empty
    if(!entry->uhdConfig.featuresSignature.empty())
    {
        entry->cachedExtractor
            = std::make_shared<FeatureExtractor>(entry->uhdConfig.featuresSignature);
    }

    return entry->cachedExtractor;
}

std::shared_ptr<FeatureExtractor> EngineRegistry::getOrCreateExtractor(int64_t engineId) const
{
    return getOrCreateExtractor(getEngine(engineId));
}

} // namespace hipdnn_backend::heuristics::uhd
